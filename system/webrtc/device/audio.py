import asyncio
import fractions
import logging
import threading
import time
from collections import deque
from typing import Callable

import numpy as np
from av import AudioFrame
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError

from cereal import car, messaging

AUDIO_PTIME = 0.020
MIC_SAMPLE_RATE = 16000
SPEAKER_SAMPLE_RATE = 48000

# Separate msgq publisher from ``webrtcAudioData`` (exclusive to webrtcd) so tools can play PCM without stealing the socket.
BODY_REALTIME_PCM_SERVICE = "bodyRealtimeAudioData"

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
          if not self._running:
            break
          # .copy() required: frombuffer is a view over the cereal message buffer, invalidated by next sm.update()
          pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16).copy()

          def _push(samples=pcm_int16):
            with self._lock:
              self._buffer.push(samples)
            self._buffer_event.set()

          if self._running:
            try:
              self._loop.call_soon_threadsafe(_push)
            except RuntimeError:
              # Event loop already closed (process teardown); drop sample.
              pass

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

  def join_poll_thread(self, timeout: float = 2.0) -> None:
    if self._thread.is_alive():
      self._thread.join(timeout=timeout)


class BodySpeaker:
  def __init__(
    self,
    pcm_service: str = "webrtcAudioData",
    pcm_gain: float = 1.0,
    on_publish_sample_rate: Callable[[int], None] | None = None,
    on_downlink_pcm: Callable[[np.ndarray], None] | None = None,
  ):
    self._pcm_service = pcm_service
    self._pcm_gain = float(pcm_gain)
    self._on_publish_sample_rate = on_publish_sample_rate
    self._on_downlink_pcm = on_downlink_pcm
    self._bad_sr_logged = False
    self._pm = messaging.PubMaster(["soundRequest", pcm_service])
    self._task: asyncio.Task | None = None

  def play_sound(self, sound_name: str):
    msg = messaging.new_message('soundRequest')
    msg.soundRequest.sound = BODY_SOUND_ALERTS[sound_name]
    self._pm.send('soundRequest', msg)

  def start_track(self, track):
    if self._task is not None and not self._task.done():
      self._task.cancel()
    self._task = asyncio.ensure_future(self._consume_track(track))

  async def _consume_track(self, track):
    from av import AudioResampler

    logger = logging.getLogger("webrtcd")
    resampler = AudioResampler(format='s16', layout='mono', rate=SPEAKER_SAMPLE_RATE)
    svc = self._pcm_service
    try:
      while True:
        frame = await track.recv()
        for resampled in resampler.resample(frame):
          msg = messaging.new_message(svc)
          ad = getattr(msg, svc)
          pcm = resampled.to_ndarray()
          if pcm.ndim > 1:
            # Downmix to mono: planar is (2, samples); packed/interleaved is (samples, 2).
            # reshape(-1) on interleaved stereo would treat L,R,L,R as one channel → harsh comb/filtered distortion.
            if pcm.shape[1] == 2:
              pcm = pcm.mean(axis=1).astype(np.int16)
            elif pcm.shape[0] == 2:
              pcm = pcm.mean(axis=0).astype(np.int16)
            else:
              pcm = pcm.reshape(-1)
          if self._pcm_gain != 1.0:
            v = pcm.astype(np.float32) * self._pcm_gain
            np.clip(v, -32768, 32767, out=v)
            pcm = v.astype(np.int16)
          out_sr = int(resampled.sample_rate)
          ad.data = np.ascontiguousarray(pcm).tobytes()
          ad.sampleRate = out_sr
          if self._on_downlink_pcm is not None:
            self._on_downlink_pcm(pcm)
          if self._on_publish_sample_rate is not None:
            self._on_publish_sample_rate(out_sr)
          elif out_sr != SPEAKER_SAMPLE_RATE and not self._bad_sr_logged:
            logger.warning(
              "BodySpeaker: resampled sample_rate=%s != soundd expected %s; feed_webrtc_pcm will ignore audio",
              out_sr,
              SPEAKER_SAMPLE_RATE,
            )
            self._bad_sr_logged = True
          self._pm.send(svc, msg)
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
