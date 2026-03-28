#!/usr/bin/env python3
"""
OpenAI Realtime API over WebRTC using the same device audio path as webrtcd:

  - Uplink:  ``BodyMicAudioTrack`` → ``rawAudioData`` (micd must be running)
  - Downlink: remote audio track → ``BodySpeaker`` → ``webrtcAudioData`` (soundd path)

This does **not** change the webrtcd HTTP server; it is a separate client that talks to
OpenAI’s Realtime WebRTC endpoint (``POST https://api.openai.com/v1/realtime/calls``),
matching the “unified interface” flow from OpenAI’s docs (browser would POST SDP to your
backend; here the script POSTs SDP directly with your API key).

Requires:
  - ``OPENAI_API_KEY``
  - micd publishing ``rawAudioData``
  - Network access to api.openai.com

Usage:
  export OPENAI_API_KEY=sk-...
  python tools/body/openai_realtime_webrtc.py
  python tools/body/openai_realtime_webrtc.py --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any

import requests
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from openpilot.system.webrtc.device.audio import BodyMicAudioTrack, BodySpeaker

REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"
ICE_GATHER_TIMEOUT_S = 30.0


async def _ice_gathering_complete(pc: RTCPeerConnection) -> None:
  loop = asyncio.get_running_loop()
  fut: asyncio.Future[None] = loop.create_future()

  @pc.on("icegatheringstatechange")
  def _on_gather() -> None:
    if pc.iceGatheringState == "complete" and not fut.done():
      fut.set_result(None)

  if pc.iceGatheringState == "complete":
    return
  await asyncio.wait_for(fut, timeout=ICE_GATHER_TIMEOUT_S)


def _post_realtime_calls(api_key: str, offer_sdp: str, session: dict[str, Any]) -> str:
  r = requests.post(
    REALTIME_CALLS_URL,
    headers={"Authorization": f"Bearer {api_key}"},
    files={
      "sdp": (None, offer_sdp, "application/sdp"),
      "session": (None, json.dumps(session), "application/json"),
    },
    timeout=120,
  )
  if not r.ok:
    raise RuntimeError(f"OpenAI {REALTIME_CALLS_URL} -> {r.status_code}: {r.text[:4000]}")
  return r.text


async def run_session(
  *,
  api_key: str,
  model: str,
  voice: str,
  instructions: str | None,
  verbose: bool,
) -> None:
  session: dict[str, Any] = {
    "type": "realtime",
    "model": model,
    "audio": {"output": {"voice": voice}},
  }
  if instructions:
    session["instructions"] = instructions

  pc = RTCPeerConnection()
  dc = pc.createDataChannel("oai-events")

  def on_dc_message(message: str | bytes) -> None:
    if isinstance(message, bytes):
      message = message.decode("utf-8", errors="replace")
    try:
      ev = json.loads(message)
    except json.JSONDecodeError:
      return
    typ = ev.get("type", "")
    if typ in ("response.output_audio_transcript.delta", "response.output_text.delta"):
      d = ev.get("delta", "")
      if d:
        print(d, end="", flush=True)
    elif typ == "error":
      print(f"\n[data channel error] {ev}", file=sys.stderr)
    elif verbose:
      print(f"\n[event] {typ}", flush=True)

  @dc.on("message")
  def _on_dc_message(message: str | bytes) -> None:
    on_dc_message(message)

  speaker = BodySpeaker()
  audio_to_speaker_started = False

  @pc.on("track")
  def on_track(track: MediaStreamTrack) -> None:
    nonlocal audio_to_speaker_started
    if track.kind != "audio" or audio_to_speaker_started:
      return
    audio_to_speaker_started = True
    speaker.start_track(track)

  mic = BodyMicAudioTrack()
  pc.addTrack(mic)

  offer = await pc.createOffer()
  await pc.setLocalDescription(offer)
  await _ice_gathering_complete(pc)

  local = pc.localDescription
  if local is None or not local.sdp:
    raise RuntimeError("Missing local SDP after ICE gathering")

  answer_sdp = await asyncio.to_thread(_post_realtime_calls, api_key, local.sdp, session)
  await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

  print(
    "\nConnected (WebRTC). Speaking into the comma mic; assistant plays like a webrtcd caller. Ctrl+C to stop.\n",
    flush=True,
  )

  stop = asyncio.Event()

  def request_stop() -> None:
    stop.set()

  loop = asyncio.get_running_loop()
  try:
    loop.add_signal_handler(signal.SIGINT, request_stop)
    loop.add_signal_handler(signal.SIGTERM, request_stop)
  except NotImplementedError:
    pass

  await stop.wait()

  mic.stop()
  await pc.close()
  await speaker.stop()


def main() -> None:
  logging.basicConfig(level=logging.WARNING)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="gpt-realtime", help="Realtime model name")
  parser.add_argument("--voice", default="marin", help="Output voice (e.g. marin, alloy)")
  parser.add_argument(
    "--instructions",
    default="You are talking to someone on a comma body. Keep replies short and clear.",
    help="Session instructions",
  )
  parser.add_argument("--verbose", action="store_true", help="Log Realtime data-channel event types")
  args = parser.parse_args()

  api_key = os.environ.get("OPENAI_API_KEY", "").strip()
  if not api_key:
    print("Set OPENAI_API_KEY.", file=sys.stderr)
    sys.exit(1)

  try:
    asyncio.run(
      run_session(
        api_key=api_key,
        model=args.model,
        voice=args.voice,
        instructions=args.instructions,
        verbose=args.verbose,
      )
    )
  except KeyboardInterrupt:
    pass
  except TimeoutError as e:
    print(f"Timed out: {e}", file=sys.stderr)
    sys.exit(1)
  except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
