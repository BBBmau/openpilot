#!/usr/bin/env python3
"""
OpenAI Realtime API over WebRTC using the same device audio path as webrtcd:

  - Uplink:  ``BodyMicAudioTrack`` → ``rawAudioData`` (micd must be running)
  - Downlink: remote audio track → ``BodySpeaker`` → ``bodyRealtimeAudioData`` (``soundd`` must be
    running — on comma body, manager starts ``soundd`` offroad via ``soundd_should_run``; otherwise
    use onroad or enable driver view / ``IsDriverViewEnabled``)

This does **not** change the webrtcd HTTP server; it is a separate client that talks to
OpenAI’s Realtime WebRTC endpoint (``POST https://api.openai.com/v1/realtime/calls``),
matching the “unified interface” flow from OpenAI’s docs (browser would POST SDP to your
backend; here the script POSTs SDP directly with your API key).

Requires:
  - ``OPENAI_API_KEY``
  - micd publishing ``rawAudioData``
  - soundd consuming ``bodyRealtimeAudioData`` (no audio if soundd is not running)
  - Network access to api.openai.com

Long reply delay is often **server VAD**: ambient noise keeps the model thinking you are still
talking until silence is detected. Tune ``--vad-threshold`` (higher → less sensitive) and
``--vad-silence-ms`` (lower → faster end-of-turn). Try ``--vad-mode semantic_vad`` in very noisy
environments.

Usage:
  export OPENAI_API_KEY=sk-...
  python tools/body/openai_realtime_webrtc.py
  python tools/body/openai_realtime_webrtc.py --verbose
  python tools/body/openai_realtime_webrtc.py --debug
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Any

import requests
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from openpilot.system.webrtc.device.audio import (
  BODY_REALTIME_PCM_SERVICE,
  BodyMicAudioTrack,
  BodySpeaker,
)

REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"
ICE_GATHER_TIMEOUT_S = 30.0
LOG = logging.getLogger("openai_realtime_webrtc")
MIC_LOG_INTERVAL = 50


class _DebugMicTrack(BodyMicAudioTrack):
  """Logs periodic uplink audio frames (proves mic → encoder → RTP path is alive)."""

  def __init__(self) -> None:
    super().__init__()
    self._mic_frames = 0
    self._t0 = time.monotonic()

  async def recv(self):
    frame = await super().recv()
    self._mic_frames += 1
    if self._mic_frames == 1 or self._mic_frames % MIC_LOG_INTERVAL == 0:
      dt = time.monotonic() - self._t0
      LOG.debug(
        "uplink mic: frame #%d (~%.1fs elapsed) samples=%d sample_rate=%d format=%s",
        self._mic_frames,
        dt,
        frame.samples,
        frame.sample_rate,
        frame.format.name if frame.format else "?",
      )
    return frame


async def _ice_gathering_complete(pc: RTCPeerConnection) -> None:
  loop = asyncio.get_running_loop()
  fut: asyncio.Future[None] = loop.create_future()

  @pc.on("icegatheringstatechange")
  def _on_gather() -> None:
    LOG.debug("ICE gathering state: %s", pc.iceGatheringState)
    if pc.iceGatheringState == "complete" and not fut.done():
      fut.set_result(None)

  if pc.iceGatheringState == "complete":
    return
  await asyncio.wait_for(fut, timeout=ICE_GATHER_TIMEOUT_S)


def _post_realtime_calls(api_key: str, offer_sdp: str, session: dict[str, Any]) -> str:
  t0 = time.monotonic()
  r = requests.post(
    REALTIME_CALLS_URL,
    headers={"Authorization": f"Bearer {api_key}"},
    files={
      "sdp": (None, offer_sdp, "application/sdp"),
      "session": (None, json.dumps(session), "application/json"),
    },
    timeout=120,
  )
  elapsed = time.monotonic() - t0
  LOG.info(
    "POST %s -> %s in %.2fs (offer SDP %d bytes, answer body %d bytes)",
    REALTIME_CALLS_URL,
    r.status_code,
    elapsed,
    len(offer_sdp),
    len(r.content),
  )
  if not r.ok:
    raise RuntimeError(f"OpenAI {REALTIME_CALLS_URL} -> {r.status_code}: {r.text[:4000]}")
  return r.text


def _turn_detection_payload(
  *,
  vad_mode: str,
  vad_threshold: float,
  vad_silence_ms: int,
  vad_prefix_ms: int,
  semantic_eagerness: str,
) -> dict[str, Any]:
  base_conv = {"create_response": True, "interrupt_response": True}
  if vad_mode == "semantic_vad":
    return {"type": "semantic_vad", "eagerness": semantic_eagerness, **base_conv}
  thr = max(0.0, min(1.0, float(vad_threshold)))
  return {
    "type": "server_vad",
    "threshold": thr,
    "prefix_padding_ms": vad_prefix_ms,
    "silence_duration_ms": vad_silence_ms,
    **base_conv,
  }


async def run_session(
  *,
  api_key: str,
  model: str,
  voice: str,
  instructions: str | None,
  verbose: bool,
  debug: bool,
  vad_mode: str,
  vad_threshold: float,
  vad_silence_ms: int,
  vad_prefix_ms: int,
  semantic_eagerness: str,
) -> None:
  session: dict[str, Any] = {
    "type": "realtime",
    "model": model,
    "audio": {"output": {"voice": voice}},
    "turn_detection": _turn_detection_payload(
      vad_mode=vad_mode,
      vad_threshold=vad_threshold,
      vad_silence_ms=vad_silence_ms,
      vad_prefix_ms=vad_prefix_ms,
      semantic_eagerness=semantic_eagerness,
    ),
  }
  if instructions:
    session["instructions"] = instructions

  LOG.info("session: model=%s voice=%s", model, voice)
  LOG.debug("session JSON: %s", json.dumps(session, indent=2) if debug else session)

  pc = RTCPeerConnection()

  @pc.on("connectionstatechange")
  def _on_conn_state() -> None:
    LOG.info("peer connection state: %s", pc.connectionState)

  @pc.on("iceconnectionstatechange")
  def _on_ice_conn() -> None:
    LOG.info("ICE connection state: %s", pc.iceConnectionState)

  @pc.on("icegatheringstatechange")
  def _on_ice_gather_outer() -> None:
    LOG.debug("ICE gathering (outer): %s", pc.iceGatheringState)

  dc = pc.createDataChannel("oai-events")

  @dc.on("open")
  def _on_dc_open() -> None:
    LOG.info('Realtime data channel "oai-events" open (readyState=%s)', dc.readyState)

  @dc.on("close")
  def _on_dc_close() -> None:
    LOG.info("Realtime data channel closed")

  dc_messages = 0

  def on_dc_message(message: str | bytes) -> None:
    nonlocal dc_messages
    if isinstance(message, bytes):
      message = message.decode("utf-8", errors="replace")
    try:
      ev = json.loads(message)
    except json.JSONDecodeError:
      LOG.warning("data channel non-JSON message (%d bytes)", len(message))
      return
    dc_messages += 1
    typ = ev.get("type", "")
    if typ in ("response.output_audio_transcript.delta", "response.output_text.delta"):
      d = ev.get("delta", "")
      if d:
        print(d, end="", flush=True)
    elif typ == "error":
      LOG.error("Realtime error event: %s", ev)
      print(f"\n[data channel error] {ev}", file=sys.stderr)
    elif debug:
      LOG.debug("data channel event #%d type=%s", dc_messages, typ)
    elif verbose:
      LOG.info("data channel event #%d type=%s", dc_messages, typ)

  @dc.on("message")
  def _on_dc_message(message: str | bytes) -> None:
    on_dc_message(message)

  speaker = BodySpeaker(pcm_service=BODY_REALTIME_PCM_SERVICE)
  audio_to_speaker_started = False

  @pc.on("track")
  def on_track(track: MediaStreamTrack) -> None:
    nonlocal audio_to_speaker_started
    LOG.info("remote track received: kind=%s id=%s", track.kind, getattr(track, "id", "?"))
    if track.kind != "audio" or audio_to_speaker_started:
      return
    audio_to_speaker_started = True
    LOG.info("starting BodySpeaker for remote audio downlink")
    speaker.start_track(track)

  mic: BodyMicAudioTrack = _DebugMicTrack() if debug else BodyMicAudioTrack()
  pc.addTrack(mic)

  stop = asyncio.Event()

  def request_stop() -> None:
    stop.set()

  loop = asyncio.get_running_loop()
  try:
    try:
      loop.add_signal_handler(signal.SIGINT, request_stop)
      loop.add_signal_handler(signal.SIGTERM, request_stop)
    except NotImplementedError:
      pass

    LOG.info("creating WebRTC offer and gathering ICE candidates...")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await _ice_gathering_complete(pc)

    local = pc.localDescription
    if local is None or not local.sdp:
      raise RuntimeError("Missing local SDP after ICE gathering")

    LOG.info("sending SDP to OpenAI realtime/calls ...")
    answer_sdp = await asyncio.to_thread(_post_realtime_calls, api_key, local.sdp, session)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
    LOG.info(
      "setRemoteDescription(answer) done; ICE=%s connection=%s dataChannel=%s",
      pc.iceConnectionState,
      pc.connectionState,
      dc.readyState,
    )

    print(
      "\nConnected (WebRTC). Speaking into the comma mic; assistant plays like a webrtcd caller. Ctrl+C to stop.\n",
      flush=True,
    )

    await stop.wait()
  finally:
    for sig in (signal.SIGINT, signal.SIGTERM):
      try:
        loop.remove_signal_handler(sig)
      except (NotImplementedError, ValueError, OSError):
        pass
    # Match webrtcd: stop downlink consumer, tear down PC (stops RTP sender reading mic), then stop cereal thread.
    try:
      await speaker.stop()
    except Exception:
      LOG.exception("cleanup: speaker.stop")
    try:
      await pc.close()
    except Exception:
      LOG.exception("cleanup: pc.close")
    await asyncio.sleep(0.1)
    mic.stop()
    mic.join_poll_thread(timeout=2.0)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="gpt-realtime", help="Realtime model name")
  parser.add_argument("--voice", default="marin", help="Output voice (e.g. marin, alloy)")
  parser.add_argument(
    "--instructions",
    default="You are talking to someone on a comma body. Keep replies short and clear.",
    help="Session instructions",
  )
  parser.add_argument(
    "--verbose",
    action="store_true",
    help="Log connection, signaling, and each Realtime data-channel event type (INFO)",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Log everything from --verbose plus ICE details, session JSON, mic uplink frames, and aiortc INFO",
  )
  parser.add_argument(
    "--vad-mode",
    choices=("server_vad", "semantic_vad"),
    default="server_vad",
    help="Realtime turn detection: server_vad uses silence (tune threshold/silence); semantic_vad uses utterance semantics",
  )
  parser.add_argument(
    "--vad-threshold",
    type=float,
    default=0.65,
    metavar="0-1",
    help="server_vad only: higher = louder required to count as speech (better in noisy rooms; default 0.65)",
  )
  parser.add_argument(
    "--vad-silence-ms",
    type=int,
    default=400,
    metavar="MS",
    help="server_vad only: silence length to end a turn; lower = faster replies (default 400)",
  )
  parser.add_argument(
    "--vad-prefix-ms",
    type=int,
    default=300,
    metavar="MS",
    help="server_vad only: audio kept before detected speech start (default 300)",
  )
  parser.add_argument(
    "--semantic-eagerness",
    choices=("low", "medium", "high", "auto"),
    default="high",
    help="semantic_vad only: higher chunks sooner (default high for lower latency)",
  )
  args = parser.parse_args()

  if args.debug:
    log_level = logging.DEBUG
  elif args.verbose:
    log_level = logging.INFO
  else:
    log_level = logging.WARNING
  logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")
  LOG.setLevel(log_level)
  if args.debug:
    logging.getLogger("aiortc").setLevel(logging.INFO)

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
        verbose=args.verbose or args.debug,
        debug=args.debug,
        vad_mode=args.vad_mode,
        vad_threshold=args.vad_threshold,
        vad_silence_ms=args.vad_silence_ms,
        vad_prefix_ms=args.vad_prefix_ms,
        semantic_eagerness=args.semantic_eagerness,
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
