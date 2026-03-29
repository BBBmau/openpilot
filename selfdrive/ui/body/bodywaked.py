#!/usr/bin/env python3
"""
Wake-word listener for comma body.

Detects "hey body" (or a test wake phrase) from rawAudioData and sets
BodyWakeIgnition to bring the body onroad — starting the voice assistant,
walk cycle, and full body stack.

Uses a 3-stage openWakeWord ONNX pipeline (melspectrogram → embedding →
wake-word classifier) running on onnxruntime, with no openwakeword dependency.

Enable with param BodyWakeWordEnabled. To use a custom wake-word ONNX model,
set BodyWakeWordModel to its path; otherwise the bundled hey_jarvis model is
used for pipeline testing.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog


def _log(msg: str):
  """Print to terminal for standalone debugging, and cloudlog for swaglog."""
  print(msg, flush=True)

SAMPLE_RATE = 16_000
OWW_CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz — openWakeWord standard frame
MEL_FRAMES_PER_EMBEDDING = 76
EMBEDDINGS_PER_PREDICTION = 16

COOLDOWN_S = 4.0
DEFAULT_THRESHOLD = 0.5

# Scores below this are silence/noise — not worth logging
SCORE_LOG_FLOOR = 0.05
# How often (seconds) to emit a periodic summary even when nothing interesting happens
SUMMARY_INTERVAL_S = 10.0

MODEL_DIR = Path(__file__).parent / "models"
MELSPEC_MODEL = MODEL_DIR / "melspectrogram.onnx"
EMBEDDING_MODEL = MODEL_DIR / "embedding_model.onnx"
DEFAULT_WW_MODEL = MODEL_DIR / "hey_jarvis_v0.1.onnx"


def _param_str(params: Params, key: str) -> str:
  raw = params.get(key)
  if raw is None:
    return ""
  if isinstance(raw, bytes):
    return raw.decode("utf-8").strip()
  return str(raw).strip()


class WakeWordDetector:
  """3-stage openWakeWord ONNX pipeline: mel → embedding → classifier."""

  def __init__(self, ww_model_path: str | Path):
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1

    self.mel_sess = ort.InferenceSession(str(MELSPEC_MODEL), opts)
    self.emb_sess = ort.InferenceSession(str(EMBEDDING_MODEL), opts)
    self.ww_sess = ort.InferenceSession(str(ww_model_path), opts)

    self.mel_buffer = np.zeros((0, 32), dtype=np.float32)
    self.emb_buffer = np.zeros((0, 96), dtype=np.float32)

  def reset(self):
    self.mel_buffer = np.zeros((0, 32), dtype=np.float32)
    self.emb_buffer = np.zeros((0, 96), dtype=np.float32)

  def process_audio(self, audio_int16: np.ndarray) -> float | None:
    """Feed int16 audio; returns wake-word probability or None if not enough data yet."""
    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    audio_f32 = audio_f32.reshape(1, -1)

    mel_out = self.mel_sess.run(None, {"input": audio_f32})[0]
    # mel_out shape: (1, 1, time_steps, 32) → squeeze to (time_steps, 32)
    mel_frames = mel_out.squeeze()
    if mel_frames.ndim == 1:
      mel_frames = mel_frames.reshape(1, -1)
    self.mel_buffer = np.vstack((self.mel_buffer, mel_frames))

    if self.mel_buffer.shape[0] < MEL_FRAMES_PER_EMBEDDING:
      return None

    # consume oldest 76 frames for embedding
    mel_input = self.mel_buffer[:MEL_FRAMES_PER_EMBEDDING]
    self.mel_buffer = self.mel_buffer[MEL_FRAMES_PER_EMBEDDING:]

    mel_input = mel_input.reshape(1, MEL_FRAMES_PER_EMBEDDING, 32, 1)
    emb_out = self.emb_sess.run(None, {"input_1": mel_input})[0]
    embedding = emb_out.reshape(1, 96)
    self.emb_buffer = np.vstack((self.emb_buffer, embedding))

    # keep only the last N embeddings
    if self.emb_buffer.shape[0] > EMBEDDINGS_PER_PREDICTION:
      self.emb_buffer = self.emb_buffer[-EMBEDDINGS_PER_PREDICTION:]

    if self.emb_buffer.shape[0] < EMBEDDINGS_PER_PREDICTION:
      return None

    ww_input = self.emb_buffer.reshape(1, EMBEDDINGS_PER_PREDICTION, 96)
    prob = float(self.ww_sess.run(None, {"x.1": ww_input})[0].squeeze())
    return prob


def main():
  config_realtime_process(0, 5)
  cloudlog.bind(daemon="bodywaked")

  params = Params()

  custom = _param_str(params, "BodyWakeWordModel")
  if custom and Path(custom).is_file():
    ww_path = custom
    cloudlog.event("bodywaked: using custom model", path=custom)
  else:
    if custom:
      cloudlog.error(f"bodywaked: BodyWakeWordModel path not found: {custom!r}, falling back to default")
    ww_path = str(DEFAULT_WW_MODEL)
    cloudlog.warning("bodywaked: using hey_jarvis test model — train a 'hey body' model for production")

  try:
    detector = WakeWordDetector(ww_path)
  except Exception:
    cloudlog.exception("bodywaked: failed to load ONNX models")
    return

  _log("bodywaked: detector ready, listening for wake word (waiting for rawAudioData from micd...)")

  sm = messaging.SubMaster(["rawAudioData"])
  pcm_buf = np.empty(0, dtype=np.int16)
  last_fire = 0.0
  warned_bad_sr = False

  # analysis / accuracy tracking
  inference_count = 0
  max_score = 0.0
  max_score_above_floor = 0.0
  detections = 0
  last_summary = time.monotonic()
  audio_rms_max = 0.0
  audio_chunks_received = 0

  while True:
    sm.update(1000)
    if not sm.updated["rawAudioData"]:
      continue

    msg = sm["rawAudioData"]
    if msg.sampleRate != SAMPLE_RATE:
      if not warned_bad_sr:
        cloudlog.error(f"bodywaked: expected {SAMPLE_RATE=} got {msg.sampleRate=}")
        warned_bad_sr = True
      continue

    chunk = np.frombuffer(msg.data, dtype=np.int16)
    if chunk.size == 0:
      continue
    audio_chunks_received += 1
    rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
    audio_rms_max = max(audio_rms_max, rms)
    pcm_buf = np.concatenate((pcm_buf, chunk))

    thresh_s = _param_str(params, "BodyWakeWordThreshold") or str(DEFAULT_THRESHOLD)
    try:
      threshold = float(thresh_s)
    except ValueError:
      threshold = DEFAULT_THRESHOLD

    while pcm_buf.size >= OWW_CHUNK_SAMPLES:
      frame = pcm_buf[:OWW_CHUNK_SAMPLES]
      pcm_buf = pcm_buf[OWW_CHUNK_SAMPLES:]

      prob = detector.process_audio(frame)
      if prob is None:
        continue

      inference_count += 1
      max_score = max(max_score, prob)

      if prob >= SCORE_LOG_FLOOR:
        max_score_above_floor = max(max_score_above_floor, prob)
        _log(f"bodywaked: score={prob:.4f} threshold={threshold:.2f} inference#{inference_count}")

      now = time.monotonic()
      if prob >= threshold and now - last_fire >= COOLDOWN_S:
        detections += 1
        _log(f"bodywaked: WAKE WORD DETECTED | score={prob:.4f} threshold={threshold:.2f} "
             f"detection#{detections} inference#{inference_count}")
        params.put_bool("BodyWakeIgnition", True)
        last_fire = now
        detector.reset()

      # periodic summary for analysis even during silence
      if now - last_summary >= SUMMARY_INTERVAL_S:
        _log(f"bodywaked: summary | inferences={inference_count} detections={detections} "
             f"max_score={max_score:.4f} max_notable={max_score_above_floor:.4f} "
             f"threshold={threshold:.2f} audio_rms_max={audio_rms_max:.0f} chunks={audio_chunks_received}")
        max_score = 0.0
        max_score_above_floor = 0.0
        audio_rms_max = 0.0
        last_summary = now


if __name__ == "__main__":
  main()
