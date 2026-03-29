#!/usr/bin/env python3
"""
Wake-word listener for comma body.

Detects a wake phrase from rawAudioData and sets BodyWakeIgnition to bring the
body onroad — starting the voice assistant, walk cycle, and full body stack.

Uses a 3-stage openWakeWord ONNX pipeline (melspectrogram → embedding →
wake-word classifier) running on onnxruntime.

Enable with param BodyWakeWordEnabled. Set BodyWakeWordModel to a custom ONNX
path, or the bundled hey_jarvis model is used for testing.
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

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz
MEL_FRAMES_PER_EMBEDDING = 76
EMBEDDINGS_PER_PREDICTION = 16
COOLDOWN_S = 4.0
DEFAULT_THRESHOLD = 0.5
SUMMARY_INTERVAL_S = 10.0

MODEL_DIR = Path(__file__).parent / "models"
MELSPEC_MODEL = MODEL_DIR / "melspectrogram.onnx"
EMBEDDING_MODEL = MODEL_DIR / "embedding_model.onnx"
DEFAULT_WW_MODEL = MODEL_DIR / "hey_comma_v17_big.onnx"


def _log(msg: str):
  print(msg, flush=True)


class WakeWordDetector:
  """3-stage openWakeWord ONNX pipeline: mel → embedding → classifier.

  Each 80 ms audio chunk produces new mel frames that slide into a 76-frame
  window.  A new embedding is computed every chunk (~93 % overlap), giving
  one wake-word probability per chunk once the pipeline is warmed up (~1.3 s).
  """

  def __init__(self, ww_model_path: str | Path):
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1

    self.mel_sess = ort.InferenceSession(str(MELSPEC_MODEL), opts)
    self.emb_sess = ort.InferenceSession(str(EMBEDDING_MODEL), opts)
    self.ww_sess = ort.InferenceSession(str(ww_model_path), opts)
    self.ww_input_name = self.ww_sess.get_inputs()[0].name
    self.reset()

  def reset(self):
    self.mel_buffer = np.zeros((0, 32), dtype=np.float32)
    self.emb_buffer = np.zeros((0, 96), dtype=np.float32)

  def process_audio(self, audio_int16: np.ndarray) -> float | None:
    """Feed 1280 int16 samples; returns wake-word probability or None during warmup."""
    audio_f32 = audio_int16.astype(np.float32).reshape(1, -1)

    mel_out = self.mel_sess.run(None, {"input": audio_f32})[0]
    # Transform to match Google's speech_embedding expected input range
    mel_frames = np.squeeze(mel_out) / 10.0 + 2.0
    if mel_frames.ndim == 1:
      mel_frames = mel_frames.reshape(1, -1)

    self.mel_buffer = np.vstack((self.mel_buffer, mel_frames))
    if self.mel_buffer.shape[0] > MEL_FRAMES_PER_EMBEDDING * 4:
      self.mel_buffer = self.mel_buffer[-MEL_FRAMES_PER_EMBEDDING * 4:]

    if self.mel_buffer.shape[0] < MEL_FRAMES_PER_EMBEDDING:
      return None

    # Sliding window: always use the most recent 76 mel frames
    mel_input = self.mel_buffer[-MEL_FRAMES_PER_EMBEDDING:].reshape(1, MEL_FRAMES_PER_EMBEDDING, 32, 1)
    embedding = self.emb_sess.run(None, {"input_1": mel_input})[0].reshape(1, 96)
    self.emb_buffer = np.vstack((self.emb_buffer, embedding))
    if self.emb_buffer.shape[0] > EMBEDDINGS_PER_PREDICTION:
      self.emb_buffer = self.emb_buffer[-EMBEDDINGS_PER_PREDICTION:]

    if self.emb_buffer.shape[0] < EMBEDDINGS_PER_PREDICTION:
      return None

    ww_input = self.emb_buffer.reshape(1, EMBEDDINGS_PER_PREDICTION, 96)
    return float(self.ww_sess.run(None, {self.ww_input_name: ww_input})[0].squeeze())


def main():
  config_realtime_process(0, 5)
  cloudlog.bind(daemon="bodywaked")
  params = Params()

  custom_path = (params.get("BodyWakeWordModel") or "").strip()
  if custom_path and Path(custom_path).is_file():
    ww_path = custom_path
  else:
    ww_path = str(DEFAULT_WW_MODEL)
    cloudlog.warning("bodywaked: using hey_jarvis test model")

  try:
    detector = WakeWordDetector(ww_path)
  except Exception:
    cloudlog.exception("bodywaked: failed to load ONNX models")
    return

  _log("bodywaked: listening for wake word")

  sm = messaging.SubMaster(["rawAudioData"])
  pcm_buf = np.empty(0, dtype=np.int16)
  last_fire = 0.0

  inference_count = 0
  detections = 0
  max_score = 0.0
  last_summary = time.monotonic()

  threshold_raw = (params.get("BodyWakeWordThreshold") or "").strip()
  try:
    threshold = float(threshold_raw) if threshold_raw else DEFAULT_THRESHOLD
  except ValueError:
    threshold = DEFAULT_THRESHOLD

  while True:
    sm.update(1000)
    if not sm.updated["rawAudioData"]:
      continue

    if params.get_bool("BodyWakeIgnition"):
      continue

    msg = sm["rawAudioData"]
    if msg.sampleRate != SAMPLE_RATE:
      continue

    chunk = np.frombuffer(msg.data, dtype=np.int16)
    if chunk.size == 0:
      continue
    pcm_buf = np.concatenate((pcm_buf, chunk))

    while pcm_buf.size >= CHUNK_SAMPLES:
      frame = pcm_buf[:CHUNK_SAMPLES]
      pcm_buf = pcm_buf[CHUNK_SAMPLES:]

      prob = detector.process_audio(frame)
      if prob is None:
        continue

      inference_count += 1
      max_score = max(max_score, prob)

      now = time.monotonic()
      if prob >= threshold and now - last_fire >= COOLDOWN_S:
        detections += 1
        _log(f"bodywaked: DETECTED score={prob:.4f} detection#{detections}")
        params.put_bool("BodyWakeIgnition", True)
        last_fire = now
        detector.reset()

      if now - last_summary >= SUMMARY_INTERVAL_S:
        _log(f"bodywaked: inferences={inference_count} detections={detections} "
             f"max_score={max_score:.4f} threshold={threshold:.2f}")
        max_score = 0.0
        last_summary = now


if __name__ == "__main__":
  main()
