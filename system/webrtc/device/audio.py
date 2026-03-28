import asyncio
import fractions
import logging
import threading
import time
from collections import deque

import numpy as np
from av import AudioFrame
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError

from cereal import car, messaging

AUDIO_PTIME = 0.020
MIC_SAMPLE_RATE = 16000
SPEAKER_SAMPLE_RATE = 48000

AudibleAlert = car.CarControl.HUDControl.AudibleAlert
BODY_SOUND_ALERTS = {
  "engage": AudibleAlert.engage,
  "disengage": AudibleAlert.disengage,
  "prompt": AudibleAlert.prompt,
  "warning": AudibleAlert.warningImmediate,
}
BODY_SOUND_NAMES = frozenset(BODY_SOUND_ALERTS)


class PcmBuffer:
  def __init__(self, dtype=np.int16):
    self._chunks: deque[np.ndarray] = deque()
    self._offset = 0
    self._size = 0
    self._dtype = dtype

  def push(self, samples: np.ndarray):
    if samples.size == 0:
      return
    chunk = np.ascontiguousarray(samples, dtype=self._dtype)
    self._chunks.append(chunk)
    self._size += chunk.size

  def available(self) -> int:
    return self._size

  def pop(self, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=self._dtype)
    written = 0

    while written < size and self._chunks:
      chunk = self._chunks[0]
      remaining = chunk.size - self._offset
      take = min(size - written, remaining)
      out[written:written + take] = chunk[self._offset:self._offset + take]
      written += take
      self._offset += take

      if self._offset >= chunk.size:
        self._chunks.popleft()
        self._offset = 0

    self._size -= written
    return out


class BodyMicAudioTrack(AudioStreamTrack):
  def __init__(self):
    super().__init__()
    self._loop = asyncio.get_running_loop()
    self._buffer = PcmBuffer()
    self._buffer_event = asyncio.Event()
    self._sample_rate = MIC_SAMPLE_RATE
    self._samples_per_frame = int(self._sample_rate * AUDIO_PTIME)
    self._lock = threading.Lock()
    self._running = True
    self._thread = threading.Thread(target=self._poll_cereal, daemon=True)
    self._thread.start()

  def _poll_cereal(self):
    sm = messaging.SubMaster(['rawAudioData'])
    while self._running:
      sm.update(20)
      if sm.updated['rawAudioData']:
        raw_bytes = sm['rawAudioData'].data
        if len(raw_bytes) > 0:
          # .copy() required: frombuffer is a view over the cereal message buffer, invalidated by next sm.update()
          pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16).copy()

          def _push(samples=pcm_int16):
            with self._lock:
              self._buffer.push(samples)
            self._buffer_event.set()

          self._loop.call_soon_threadsafe(_push)

  async def recv(self):
    if self.readyState != "live":
      raise MediaStreamError

    while True:
      with self._lock:
        if self._buffer.available() >= self._samples_per_frame:
          frame_samples = self._buffer.pop(self._samples_per_frame)
          break
        self._buffer_event.clear()
      if self.readyState != "live":
        raise MediaStreamError
      await self._buffer_event.wait()

    if hasattr(self, "_timestamp"):
      self._timestamp += self._samples_per_frame
      wait = self._start + (self._timestamp / self._sample_rate) - time.monotonic()
      await asyncio.sleep(wait)
    else:
      self._start = time.monotonic()
      self._timestamp = 0

    frame = AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
    frame.planes[0].update(frame_samples.tobytes())
    frame.pts = self._timestamp
    frame.sample_rate = self._sample_rate
    frame.time_base = fractions.Fraction(1, self._sample_rate)
    return frame

  def stop(self):
    super().stop()
    self._running = False
    self._buffer_event.set()


class BodySpeaker:
  def __init__(self):
    self._pm = messaging.PubMaster(['soundRequest', 'webrtcAudioData'])
    self._task: asyncio.Task | None = None

  def play_sound(self, sound_name: str):
    msg = messaging.new_message('soundRequest')
    msg.soundRequest.sound = BODY_SOUND_ALERTS[sound_name]
    self._pm.send('soundRequest', msg)

  def start_track(self, track):
    self._task = asyncio.ensure_future(self._consume_track(track))

  async def _consume_track(self, track):
    from av import AudioResampler

    logger = logging.getLogger("webrtcd")
    resampler = AudioResampler(format='s16', layout='mono', rate=SPEAKER_SAMPLE_RATE)
    try:
      while True:
        frame = await track.recv()
        for resampled in resampler.resample(frame):
          msg = messaging.new_message('webrtcAudioData')
          pcm = resampled.to_ndarray()
          if pcm.ndim > 1:
            pcm = pcm.reshape(-1)
          msg.webrtcAudioData.data = np.ascontiguousarray(pcm).tobytes()
          msg.webrtcAudioData.sampleRate = SPEAKER_SAMPLE_RATE
          self._pm.send('webrtcAudioData', msg)
    except MediaStreamError:
      logger.info("Incoming browser audio track ended")
    except asyncio.CancelledError:
      raise
    except Exception:
      logger.exception("BodySpeaker track consumption error")

  async def stop(self):
    if self._task is not None and not self._task.done():
      self._task.cancel()
      try:
        await self._task
      except asyncio.CancelledError:
        pass
    self._task = None
