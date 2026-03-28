import math
import threading
import time
from collections import deque

import numpy as np
import wave

from cereal import car, messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import Ratekeeper
from openpilot.common.utils import retry
from openpilot.common.swaglog import cloudlog

from openpilot.system import micd
from openpilot.system.hardware import HARDWARE

SAMPLE_RATE = 48000
SAMPLE_BUFFER = 4096 # (approx 100ms)
# Cap queued WebRTC/bodyRealtime PCM. Larger queue trades latency for smooth playback: dropping
# oldest audio (old cap ~0.5s) causes clicks/glitches; a multi-second cap absorbs jitter and brief
# callback stalls. Still bounded so a dead consumer cannot grow RAM without limit (~30s float32).
MAX_WEBRTC_QUEUED_SAMPLES = SAMPLE_RATE * 30
MAX_VOLUME = 1.0
MIN_VOLUME = 0.1
SELFDRIVE_STATE_TIMEOUT = 5 # 5 seconds
FILTER_DT = 1. / (micd.SAMPLE_RATE / micd.FFT_SAMPLES)

AMBIENT_DB = 24 # DB where MIN_VOLUME is applied
DB_SCALE = 30 # AMBIENT_DB + DB_SCALE is where MAX_VOLUME is applied

VOLUME_BASE = 20
if HARDWARE.get_device_type() == "tizi":
  AMBIENT_DB = 30
  VOLUME_BASE = 10

AudibleAlert = car.CarControl.HUDControl.AudibleAlert


sound_list: dict[int, tuple[str, int | None, float]] = {
  # AudibleAlert, file name, play count (none for infinite)
  AudibleAlert.engage: ("engage.wav", 1, MAX_VOLUME),
  AudibleAlert.disengage: ("disengage.wav", 1, MAX_VOLUME),
  AudibleAlert.refuse: ("refuse.wav", 1, MAX_VOLUME),

  AudibleAlert.prompt: ("prompt.wav", 1, MAX_VOLUME),
  AudibleAlert.promptRepeat: ("prompt.wav", None, MAX_VOLUME),
  AudibleAlert.promptDistracted: ("prompt_distracted.wav", None, MAX_VOLUME),

  AudibleAlert.warningSoft: ("warning_soft.wav", None, MAX_VOLUME),
  AudibleAlert.warningImmediate: ("warning_immediate.wav", None, MAX_VOLUME),
}
if HARDWARE.get_device_type() == "tizi":
  sound_list.update({
    AudibleAlert.engage: ("engage_tizi.wav", 1, MAX_VOLUME),
    AudibleAlert.disengage: ("disengage_tizi.wav", 1, MAX_VOLUME),
  })

def check_selfdrive_timeout_alert(sm):
  ss_missing = time.monotonic() - sm.recv_time['selfdriveState']

  if ss_missing > SELFDRIVE_STATE_TIMEOUT:
    if sm['selfdriveState'].enabled and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < 10:
      return True

  return False


class Soundd:
  def __init__(self):
    self.load_sounds()

    self.current_alert = AudibleAlert.none
    self.current_volume = MIN_VOLUME
    self.current_sound_frame = 0

    self.selfdrive_timeout_alert = False

    self.spl_filter_weighted = FirstOrderFilter(0, 2.5, FILTER_DT, initialized=False)

    self._webrtc_lock = threading.Lock()
    # (source, float32 mono chunk). source "body" = bodyRealtimeAudioData; "webrtc" = webrtcAudioData.
    self._webrtc_chunks: deque[tuple[str, np.ndarray]] = deque()
    self._webrtc_sr_warned = False
    self._pm_queue = messaging.PubMaster(["sounddWebrtcQueueState"])

  def _webrtc_queued_samples(self) -> int:
    return sum(c.shape[0] for _, c in self._webrtc_chunks)

  def _webrtc_body_queued_samples(self) -> int:
    return sum(c.shape[0] for t, c in self._webrtc_chunks if t == "body")

  def feed_webrtc_pcm(self, pcm_int16: np.ndarray, sample_rate: int, *, source: str = "webrtc") -> None:
    if pcm_int16.size == 0:
      return
    if sample_rate != SAMPLE_RATE:
      if not self._webrtc_sr_warned:
        cloudlog.warning("webrtcAudioData sampleRate %s != soundd %s; ignoring webrtc audio", sample_rate, SAMPLE_RATE)
        self._webrtc_sr_warned = True
      return
    fl = pcm_int16.astype(np.float32) / 32768.0
    with self._webrtc_lock:
      while self._webrtc_queued_samples() > MAX_WEBRTC_QUEUED_SAMPLES and self._webrtc_chunks:
        self._webrtc_chunks.popleft()
      self._webrtc_chunks.append((source, fl))

  def take_webrtc(self, frames: int) -> np.ndarray:
    out = np.zeros(frames, dtype=np.float32)
    taken = 0
    with self._webrtc_lock:
      while taken < frames and self._webrtc_chunks:
        tag, chunk = self._webrtc_chunks[0]
        need = frames - taken
        if chunk.shape[0] <= need:
          out[taken:taken + chunk.shape[0]] = chunk
          taken += chunk.shape[0]
          self._webrtc_chunks.popleft()
        else:
          out[taken:taken + need] = chunk[:need]
          self._webrtc_chunks[0] = (tag, chunk[need:])
          taken = frames
    return out

  def load_sounds(self):
    self.loaded_sounds: dict[int, np.ndarray] = {}

    # Load all sounds
    for sound in sound_list:
      filename, play_count, volume = sound_list[sound]

      with wave.open(BASEDIR + "/selfdrive/assets/sounds/" + filename, 'r') as wavefile:
        assert wavefile.getnchannels() == 1
        assert wavefile.getsampwidth() == 2
        assert wavefile.getframerate() == SAMPLE_RATE

        length = wavefile.getnframes()
        self.loaded_sounds[sound] = np.frombuffer(wavefile.readframes(length), dtype=np.int16).astype(np.float32) / (2**16/2)

  def get_sound_data(self, frames): # get "frames" worth of data from the current alert sound, looping when required

    ret = np.zeros(frames, dtype=np.float32)

    if self.current_alert != AudibleAlert.none:
      num_loops = sound_list[self.current_alert][1]
      sound_data = self.loaded_sounds[self.current_alert]
      written_frames = 0

      current_sound_frame = self.current_sound_frame % len(sound_data)
      loops = self.current_sound_frame // len(sound_data)

      while written_frames < frames and (num_loops is None or loops < num_loops):
        available_frames = sound_data.shape[0] - current_sound_frame
        frames_to_write = min(available_frames, frames - written_frames)
        ret[written_frames:written_frames+frames_to_write] = sound_data[current_sound_frame:current_sound_frame+frames_to_write]
        written_frames += frames_to_write
        self.current_sound_frame += frames_to_write

    return ret * self.current_volume

  def callback(self, data_out: np.ndarray, frames: int, time, status) -> None:
    if status:
      cloudlog.warning(f"soundd stream over/underflow: {status}")
    sound = self.get_sound_data(frames)
    webrtc = self.take_webrtc(frames)
    mixed = sound + webrtc
    np.clip(mixed, -1.0, 1.0, out=mixed)
    data_out[:frames, 0] = mixed

  def update_alert(self, new_alert):
    current_alert_played_once = self.current_alert == AudibleAlert.none or self.current_sound_frame > len(self.loaded_sounds[self.current_alert])
    if self.current_alert != new_alert and (new_alert != AudibleAlert.none or current_alert_played_once):
      self.current_alert = new_alert
      self.current_sound_frame = 0

  def _drain_live_pcm(self, webrtc_sock, body_sock) -> None:
    """Move every pending PCM chunk into the playback deque.

    SubMaster uses conflate=True on all services, which keeps only the *latest* message per socket
    between polls — fine for state, but it drops most audio frames when the producer outruns our
    poll rate. Live PCM uses dedicated subscribers with conflate=False plus drain_sock so chunks
    are fed in order and the PortAudio callback sees a steadier queue.
    """
    for ev in messaging.drain_sock(webrtc_sock, wait_for_one=False):
      wa = ev.webrtcAudioData
      raw = wa.data
      if len(raw) > 0:
        pcm = np.frombuffer(raw, dtype=np.int16).copy()
        self.feed_webrtc_pcm(pcm, int(wa.sampleRate), source="webrtc")
    for ev in messaging.drain_sock(body_sock, wait_for_one=False):
      br = ev.bodyRealtimeAudioData
      raw = br.data
      if len(raw) > 0:
        pcm = np.frombuffer(raw, dtype=np.int16).copy()
        self.feed_webrtc_pcm(pcm, int(br.sampleRate), source="body")

  def get_audible_alert(self, sm):
    if sm.updated['soundRequest']:
      new_alert = sm['soundRequest'].sound.raw
      if new_alert != AudibleAlert.none:
        self.update_alert(new_alert)

    if sm.updated['selfdriveState']:
      new_alert = sm['selfdriveState'].alertSound.raw
      self.update_alert(new_alert)
    elif check_selfdrive_timeout_alert(sm):
      self.update_alert(AudibleAlert.warningImmediate)
      self.selfdrive_timeout_alert = True
    elif self.selfdrive_timeout_alert:
      self.update_alert(AudibleAlert.none)
      self.selfdrive_timeout_alert = False

  def calculate_volume(self, weighted_db):
    volume = ((weighted_db - AMBIENT_DB) / DB_SCALE) * (MAX_VOLUME - MIN_VOLUME) + MIN_VOLUME
    return math.pow(VOLUME_BASE, (np.clip(volume, MIN_VOLUME, MAX_VOLUME) - 1))

  @retry(attempts=10, delay=3)
  def get_stream(self, sd):
    # reload sounddevice to reinitialize portaudio
    sd._terminate()
    sd._initialize()
    return sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, callback=self.callback, blocksize=SAMPLE_BUFFER)

  def soundd_thread(self):
    # sounddevice must be imported after forking processes
    import sounddevice as sd

    sm = messaging.SubMaster(['selfdriveState', 'soundPressure', 'soundRequest'], poll='selfdriveState')
    webrtc_pcm_sock = messaging.sub_sock('webrtcAudioData', conflate=False)
    body_pcm_sock = messaging.sub_sock('bodyRealtimeAudioData', conflate=False)

    with self.get_stream(sd) as stream:
      # Faster than 20 Hz so we pull msgq → deque closer to real time; callback still drains at a
      # fixed ~blocksize / SAMPLE_RATE, but feeding was ~50 ms apart and conflate hid dropped frames.
      rk = Ratekeeper(100)

      cloudlog.info(f"soundd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}, {stream.blocksize=}")
      while True:
        sm.update(0)
        self._drain_live_pcm(webrtc_pcm_sock, body_pcm_sock)

        q = self._webrtc_queued_samples()
        qb = self._webrtc_body_queued_samples()
        msg = messaging.new_message("sounddWebrtcQueueState", valid=True)
        msg.sounddWebrtcQueueState.queuedSamples = int(min(q, 2**31 - 1))
        msg.sounddWebrtcQueueState.bodyRealtimeQueuedSamples = int(min(qb, 2**31 - 1))
        msg.sounddWebrtcQueueState.hasSplitTelemetry = True
        self._pm_queue.send("sounddWebrtcQueueState", msg)

        if sm.updated['soundPressure'] and self.current_alert == AudibleAlert.none: # only update volume filter when not playing alert
          self.spl_filter_weighted.update(sm["soundPressure"].soundPressureWeightedDb)
          self.current_volume = self.calculate_volume(float(self.spl_filter_weighted.x))

        self.get_audible_alert(sm)

        rk.keep_time()

        assert stream.active


def main():
  s = Soundd()
  s.soundd_thread()


if __name__ == "__main__":
  main()
