#!/usr/bin/env python3
"""
Wake-word listener for comma body: sets BodyVoiceAssistantActive when the phrase is heard.

For a true "hey comma" detector, train an openWakeWord ONNX model and set param BodyWakeWordModel
to its absolute path. If unset, a built-in model (hey_jarvis) is used only to validate the pipeline;
it will not respond to "hey comma".
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog

# Must match system/micd.py (rawAudioData)
SAMPLE_RATE = 16_000

# openWakeWord expects 80 ms frames at 16 kHz
_OWW_FRAME_SAMPLES = 1280
_COOLDOWN_S = 4.0
# Require several consecutive frames above threshold (model name -> frames)
_OWW_PATIENCE_FRAMES = 3


def _param_str(params: Params, key: str) -> str:
  raw = params.get(key)
  if raw is None:
    return ""
  if isinstance(raw, bytes):
    return raw.decode("utf-8").strip()
  return str(raw).strip()


def _load_model(params: Params):
  import openwakeword
  from openwakeword.model import Model

  custom = _param_str(params, "BodyWakeWordModel")
  if custom and Path(custom).is_file():
    paths = [custom.strip()]
    cloudlog.event("bodywaked_custom_model", path=paths[0])
  else:
    if custom:
      cloudlog.error(f"bodywaked: BodyWakeWordModel path not found: {custom!r}")
    paths_dict = dict(zip(openwakeword.models.keys(), openwakeword.get_pretrained_model_paths(), strict=True))
    paths = [paths_dict["hey_jarvis"]]
    cloudlog.warning(
      "bodywaked: using built-in hey_jarvis model for testing only; "
      + "train a 'hey comma' openWakeWord model and set BodyWakeWordModel to that .onnx path"
    )
  return Model(wakeword_model_paths=paths)


def main():
  config_realtime_process(0, 5)
  cloudlog.bind(daemon="bodywaked")

  params = Params()
  try:
    model = _load_model(params)
  except Exception:
    cloudlog.exception("bodywaked: failed to load wake models")
    raise

  model_names = list(model.models.keys())
  if len(model_names) != 1:
    cloudlog.warning(f"bodywaked: expected one wake model, got {model_names}")
  patience_mdl = model_names[0]

  sm = messaging.SubMaster(["rawAudioData"])
  pcm = np.empty(0, dtype=np.int16)
  last_fire = 0.0
  warned_bad_sr = False

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

    pcm = np.concatenate((pcm, chunk))

    thresh_s = _param_str(params, "BodyWakeWordThreshold") or "0.35"
    try:
      threshold = float(thresh_s)
    except ValueError:
      threshold = 0.35

    while pcm.size >= _OWW_FRAME_SAMPLES:
      frame = pcm[:_OWW_FRAME_SAMPLES]
      pcm = pcm[_OWW_FRAME_SAMPLES:]

      preds = model.predict(
        frame,
        patience={patience_mdl: _OWW_PATIENCE_FRAMES},
        threshold={patience_mdl: threshold},
      )
      score = max(float(v) for v in preds.values()) if preds else 0.0
      now = time.monotonic()
      if score < threshold or now - last_fire < _COOLDOWN_S:
        continue

      params.put_bool("BodyVoiceAssistantActive", True)
      cloudlog.event("body_wake_word_trigger", score=score, model=patience_mdl)
      last_fire = now
      model.reset()


if __name__ == "__main__":
  main()
