#!/usr/bin/env python3
"""
Stream comma body (or any device publishing rawAudioData) through OpenAI Whisper and print transcripts.

Requires micd running so rawAudioData is on the bus (on comma body, micd runs whenever CP.notCar).

Usage on device:
  export OPENAI_API_KEY=sk-...
  python tools/body/whisper_from_mic.py

Optional:
  --chunk-seconds 5   # send this much audio per API call (default 5)
  --language en       # ISO-639-1 code; omit for auto-detect
  --min-rms 500       # skip API call if chunk RMS below this (int16 scale; ~0 = always send)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import wave

import numpy as np
import requests

from cereal import messaging

# Must match system/micd.py
SAMPLE_RATE = 16_000

WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"


def pcm16_to_wav_bytes(pcm: np.ndarray) -> bytes:
  buf = io.BytesIO()
  with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(pcm.astype(np.int16, copy=False).tobytes())
  return buf.getvalue()


def transcribe_chunk(
  wav_bytes: bytes,
  api_key: str,
  language: str | None,
  timeout_s: float,
) -> str:
  headers = {"Authorization": f"Bearer {api_key}"}
  files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
  data: dict[str, str] = {"model": "whisper-1"}
  if language:
    data["language"] = language
  r = requests.post(WHISPER_URL, headers=headers, files=files, data=data, timeout=timeout_s)
  if not r.ok:
    return f"[error {r.status_code}] {r.text}"
  try:
    return r.json().get("text", "").strip()
  except Exception:
    return f"[bad json] {r.text[:200]}"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--chunk-seconds", type=float, default=5.0, help="Audio per transcription request")
  parser.add_argument("--language", type=str, default=None, help="e.g. en (optional)")
  parser.add_argument("--min-rms", type=float, default=0.0, help="Skip chunk if RMS below this (int16 samples)")
  parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
  parser.add_argument("--once", action="store_true", help="Transcribe one chunk then exit")
  args = parser.parse_args()

  api_key = os.environ.get("OPENAI_API_KEY", "").strip()
  if not api_key:
    print("Set OPENAI_API_KEY in the environment.", file=sys.stderr)
    sys.exit(1)

  samples_per_chunk = max(1, int(args.chunk_seconds * SAMPLE_RATE))
  sm = messaging.SubMaster(["rawAudioData"])
  pcm = np.empty(0, dtype=np.int16)
  warned_sr = False
  warned_no_mic = False

  print(f"Listening on rawAudioData; sending ~{args.chunk_seconds:.1f}s WAV chunks to Whisper.", flush=True)

  while True:
    sm.update(1000)
    if not sm.updated["rawAudioData"]:
      if sm.frame > 50 and sm.recv_frame["rawAudioData"] == 0 and not warned_no_mic:
        print("No rawAudioData yet — is micd running?", flush=True)
        warned_no_mic = True
      continue

    msg = sm["rawAudioData"]
    if msg.sampleRate != SAMPLE_RATE:
      if not warned_sr:
        print(f"Warning: sampleRate {msg.sampleRate} != expected {SAMPLE_RATE}", flush=True)
        warned_sr = True

    chunk = np.frombuffer(msg.data, dtype=np.int16)
    if chunk.size:
      pcm = np.concatenate((pcm, chunk))

    while pcm.size >= samples_per_chunk:
      block = pcm[:samples_per_chunk]
      pcm = pcm[samples_per_chunk:]

      if args.min_rms > 0:
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        if rms < args.min_rms:
          print(f"[skip quiet chunk rms={rms:.0f}]", flush=True)
          if args.once:
            return
          continue

      wav_bytes = pcm16_to_wav_bytes(block)
      text = transcribe_chunk(wav_bytes, api_key, args.language, args.timeout)
      print(text, flush=True)

      if args.once:
        return


if __name__ == "__main__":
  main()
