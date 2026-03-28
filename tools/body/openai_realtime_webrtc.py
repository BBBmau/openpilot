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

Use ``--playback-gain`` if assistant audio is too quiet (soundd plays this path at fixed level,
unlike alert chimes). Try ``--voice cedar`` or ``--voice marin`` for higher-quality voices.
``--output-speed`` slightly below 1.0 can sound clearer on small speakers.

Long reply delay is often **server VAD**: ambient noise keeps the model thinking you are still
talking until silence is detected. Tune ``--vad-threshold`` (higher → less sensitive) and
``--vad-silence-ms`` (lower → faster end-of-turn). Try ``--vad-mode semantic_vad`` in very noisy
environments.

Replies getting **cut off mid-sentence** are usually **interrupt_response**: with the default
(off), user VAD does not cancel assistant audio. Enable ``--interrupt-response`` only if you want
barge-in in a quiet room (speaker bleed into the mic otherwise looks like “user talking”).

False “user” turns from echo/noise can still start **new** responses while audio plays. Defaults
use **far_field** noise shaping and a **higher VAD threshold**; raise ``--vad-threshold`` further
(up to ~0.95) if it still self-interrupts, or try ``--vad-mode semantic_vad`` with
``--semantic-eagerness low``.

**Half-duplex uplink** (on by default): while assistant PCM is playing (and briefly after), the
script sends **silence** on the WebRTC mic so OpenAI’s VAD does not hear speaker bleed. Use
``--no-half-duplex`` only with a headset or if you need true full-duplex.

With ``--server-auto-response`` off (the default), the client sends ``response.create`` only after
each ``input_audio_buffer.committed`` and **queues** another if a response is still in progress,
avoiding ``conversation_already_has_active_response`` when VAD double-fires during playback.

**User speech logging** (``--log-user-speech``, on by default): prints Realtime VAD events to stderr
(``speech_started`` / ``speech_stopped`` / ``committed``, etc.). Add ``--input-transcription`` to
enable ASR on committed user audio and log ``[user speech transcript]`` lines (separate billing).

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
from collections.abc import Callable
from typing import Any

import numpy as np
import requests
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from openpilot.system.webrtc.device.audio import (
  BODY_REALTIME_PCM_SERVICE,
  BodyMicAudioTrack,
  BodySpeaker,
  SPEAKER_SAMPLE_RATE,
)

REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"
ICE_GATHER_TIMEOUT_S = 30.0
LOG = logging.getLogger("openai_realtime_webrtc")
MIC_LOG_INTERVAL = 50

_USER_SPEECH_LOG_TYPES = frozenset({
  "input_audio_buffer.speech_started",
  "input_audio_buffer.speech_stopped",
  "input_audio_buffer.committed",
  "input_audio_buffer.timeout_triggered",
})
_USER_SPEECH_LOG_KEYS = ("item_id", "audio_start_ms", "audio_end_ms", "previous_item_id", "event_id")


def _format_user_speech_log_line(ev: dict[str, Any]) -> str:
  typ = ev.get("type", "?")
  parts: list[str] = [str(typ)]
  for key in _USER_SPEECH_LOG_KEYS:
    if key in ev and ev[key] is not None:
      parts.append(f"{key}={ev[key]}")
  return " ".join(parts)


# Matches ``selfdrive/ui/soundd.py`` SAMPLE_RATE / ``feed_webrtc_pcm`` (non-48kHz PCM is dropped).
_soundd_sr_mismatch_logged = False


class _DownlinkActivityGate:
  """Mute uplink while downlink PCM is loud, plus hangover (echo path + DAC tail)."""

  def __init__(self, *, rms_threshold: float, hangover_s: float, after_response_s: float) -> None:
    self._thr = rms_threshold
    self._hang = hangover_s
    self._after_resp = after_response_s
    self._suppress_until = 0.0

  def feed_pcm_int16(self, pcm: np.ndarray) -> None:
    if pcm.size == 0:
      return
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms >= self._thr:
      now = time.monotonic()
      self._suppress_until = max(self._suppress_until, now + self._hang)

  def suppress_uplink(self) -> bool:
    return time.monotonic() < self._suppress_until

  def on_assistant_response_done(self) -> None:
    now = time.monotonic()
    self._suppress_until = max(self._suppress_until, now + self._after_resp)


class _HalfDuplexMicTrack(BodyMicAudioTrack):
  """Sends silence to the peer while ``suppress_uplink()`` is true (same frame timing as mic)."""

  def __init__(self, suppress_uplink: Callable[[], bool]) -> None:
    super().__init__()
    self._suppress_uplink = suppress_uplink

  async def recv(self):
    frame = await super().recv()
    if self._suppress_uplink():
      frame.planes[0].update(np.zeros(frame.samples, dtype=np.int16).tobytes())
    return frame


def _soundd_body_realtime_sample_rate_check(published_hz: int) -> None:
  global _soundd_sr_mismatch_logged
  if published_hz == SPEAKER_SAMPLE_RATE or _soundd_sr_mismatch_logged:
    return
  _soundd_sr_mismatch_logged = True
  LOG.warning(
    "bodyRealtimeAudioData sampleRate %d != soundd %d; feed_webrtc_pcm ignores non-48kHz PCM (no downlink audio)",
    published_hz,
    SPEAKER_SAMPLE_RATE,
  )


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


class _DebugHalfDuplexMicTrack(_HalfDuplexMicTrack):
  """Half-duplex mic with periodic uplink logs."""

  def __init__(self, suppress_uplink: Callable[[], bool]) -> None:
    super().__init__(suppress_uplink)
    self._mic_frames = 0
    self._t0 = time.monotonic()

  async def recv(self):
    frame = await super().recv()
    self._mic_frames += 1
    if self._mic_frames == 1 or self._mic_frames % MIC_LOG_INTERVAL == 0:
      dt = time.monotonic() - self._t0
      LOG.debug(
        "uplink mic: frame #%d (~%.1fs elapsed) samples=%d sample_rate=%d gated=%s",
        self._mic_frames,
        dt,
        frame.samples,
        frame.sample_rate,
        self._suppress_uplink(),
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
  interrupt_response: bool,
  server_auto_response: bool,
) -> dict[str, Any]:
  base_conv = {"create_response": server_auto_response, "interrupt_response": interrupt_response}
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
  interrupt_response: bool,
  server_auto_response: bool,
  noise_reduction: str,
  playback_gain: float,
  output_speed: float,
  half_duplex: bool,
  half_duplex_rms: float,
  half_duplex_hangover_s: float,
  half_duplex_after_response_s: float,
  log_user_speech: bool,
  input_transcription: bool,
  input_transcription_model: str,
) -> None:
  gate: _DownlinkActivityGate | None = None
  if half_duplex:
    gate = _DownlinkActivityGate(
      rms_threshold=half_duplex_rms,
      hangover_s=half_duplex_hangover_s,
      after_response_s=half_duplex_after_response_s,
    )

  audio_input: dict[str, Any] = {
    "turn_detection": _turn_detection_payload(
      vad_mode=vad_mode,
      vad_threshold=vad_threshold,
      vad_silence_ms=vad_silence_ms,
      vad_prefix_ms=vad_prefix_ms,
      semantic_eagerness=semantic_eagerness,
      interrupt_response=interrupt_response,
      server_auto_response=server_auto_response,
    ),
  }
  if noise_reduction == "off":
    audio_input["noise_reduction"] = None
  else:
    audio_input["noise_reduction"] = {"type": noise_reduction}
  if input_transcription:
    audio_input["transcription"] = {"model": input_transcription_model}

  session: dict[str, Any] = {
    "type": "realtime",
    "model": model,
    "audio": {
      "output": {
        "voice": voice,
        "format": {"type": "audio/pcm", "rate": 24000},
        "speed": output_speed,
      },
      "input": audio_input,
    },
  }
  if instructions:
    session["instructions"] = instructions

  LOG.info("session: model=%s voice=%s", model, voice)
  if gate is not None:
    LOG.info(
      "half-duplex uplink: mute while downlink RMS≥%.0f (hangover %.2fs, +%.2fs after response.done)",
      half_duplex_rms,
      half_duplex_hangover_s,
      half_duplex_after_response_s,
    )
  if input_transcription:
    LOG.info("input transcription enabled (model=%s)", input_transcription_model)
    print(
      f"[user speech] input transcription on ({input_transcription_model}); "
      "expect [user speech transcript] lines after each turn",
      file=sys.stderr,
      flush=True,
    )
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
  # When create_response is false, we send response.create on each committed user turn. If VAD
  # commits again while a response is still streaming (echo/noise), the server rejects a second
  # create; queue one for after response.done instead.
  response_busy = False
  pending_response_create = False

  def _send_response_create() -> None:
    nonlocal response_busy, pending_response_create
    if dc.readyState != "open":
      LOG.warning("data channel not open; skipping response.create")
      return
    try:
      dc.send(json.dumps({"type": "response.create"}))
    except Exception:
      LOG.exception("response.create send failed")
      return
    response_busy = True

  def _on_input_committed() -> None:
    nonlocal response_busy, pending_response_create
    if server_auto_response:
      return
    if response_busy:
      pending_response_create = True
      LOG.debug("input_audio_buffer.committed while response active; queued response.create")
      return
    _send_response_create()

  def _on_response_done() -> None:
    nonlocal response_busy, pending_response_create
    if server_auto_response:
      return
    response_busy = False
    if pending_response_create:
      pending_response_create = False
      _send_response_create()

  def on_dc_message(message: str | bytes) -> None:
    nonlocal dc_messages, response_busy, pending_response_create
    if isinstance(message, bytes):
      message = message.decode("utf-8", errors="replace")
    try:
      ev = json.loads(message)
    except json.JSONDecodeError:
      LOG.warning("data channel non-JSON message (%d bytes)", len(message))
      return
    dc_messages += 1
    typ = ev.get("type", "")
    if log_user_speech and typ in _USER_SPEECH_LOG_TYPES:
      print(f"[user speech] {_format_user_speech_log_line(ev)}", file=sys.stderr, flush=True)
    if input_transcription:
      if typ == "conversation.item.input_audio_transcription.completed":
        tr = ev.get("transcript", "")
        print(
          f"[user speech transcript] item_id={ev.get('item_id')} transcript={json.dumps(tr)}",
          file=sys.stderr,
          flush=True,
        )
      elif typ == "conversation.item.input_audio_transcription.failed":
        print(f"[user speech transcript] FAILED {ev}", file=sys.stderr, flush=True)
    if typ == "input_audio_buffer.committed" and not server_auto_response:
      _on_input_committed()
    elif typ == "response.done":
      if gate is not None:
        gate.on_assistant_response_done()
      if not server_auto_response:
        _on_response_done()
    if typ in ("response.output_audio_transcript.delta", "response.output_text.delta"):
      d = ev.get("delta", "")
      if d:
        print(d, end="", flush=True)
    elif typ == "error":
      err = ev.get("error") or {}
      code = err.get("code")
      if not server_auto_response and code == "conversation_already_has_active_response":
        pending_response_create = True
        response_busy = True
        LOG.warning(
          "Realtime: active response in progress; queued deferred response.create (%s)",
          ev.get("event_id", ""),
        )
      else:
        LOG.error("Realtime error event: %s", ev)
        print(f"\n[data channel error] {ev}", file=sys.stderr)
    elif debug:
      LOG.debug("data channel event #%d type=%s", dc_messages, typ)
    elif verbose:
      LOG.info("data channel event #%d type=%s", dc_messages, typ)

  @dc.on("message")
  def _on_dc_message(message: str | bytes) -> None:
    on_dc_message(message)

  if half_duplex and gate is not None:
    mic: BodyMicAudioTrack = (
      _DebugHalfDuplexMicTrack(gate.suppress_uplink) if debug else _HalfDuplexMicTrack(gate.suppress_uplink)
    )
  elif debug:
    mic = _DebugMicTrack()
  else:
    mic = BodyMicAudioTrack()

  speaker = BodySpeaker(
    pcm_service=BODY_REALTIME_PCM_SERVICE,
    pcm_gain=playback_gain,
    on_publish_sample_rate=_soundd_body_realtime_sample_rate_check,
    on_downlink_pcm=gate.feed_pcm_int16 if gate is not None else None,
  )
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
    default=0.82,
    metavar="0-1",
    help="server_vad only: higher = louder required to count as speech (default 0.82 for body/speaker bleed; try 0.6–0.7 in a quiet room)",
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
    default="low",
    help="semantic_vad only: low waits longer before end-of-turn (default low, fewer false chunks from echo)",
  )
  parser.add_argument(
    "--interrupt-response",
    action="store_true",
    help="Let detected user speech cancel assistant playback (barge-in). Default off: avoids cutoff when the mic picks up the speaker.",
  )
  parser.add_argument(
    "--noise-reduction",
    choices=("far_field", "near_field", "off"),
    default="far_field",
    help="Realtime input noise reduction before VAD (default far_field for room mic; near_field for close-talk; off to disable)",
  )
  parser.add_argument(
    "--server-auto-response",
    action="store_true",
    help="Let the server call response.create on each committed turn (can error with conversation_already_has_active_response if VAD fires during playback)",
  )
  parser.add_argument(
    "--half-duplex/--no-half-duplex",
    dest="half_duplex",
    default=True,
    help="Mute WebRTC uplink while assistant audio is playing (default on; stops echo from triggering VAD)",
  )
  parser.add_argument(
    "--half-duplex-rms",
    type=float,
    default=380.0,
    metavar="INT16_RMS",
    help="Downlink RMS threshold to extend uplink mute (default 380; lower=more aggressive mute)",
  )
  parser.add_argument(
    "--half-duplex-hangover",
    type=float,
    default=0.5,
    metavar="SEC",
    help="Keep uplink muted this long after each loud downlink chunk (default 0.5)",
  )
  parser.add_argument(
    "--half-duplex-after-response",
    type=float,
    default=0.45,
    metavar="SEC",
    help="Extra uplink mute after response.done for speaker/DAC tail (default 0.45)",
  )
  parser.add_argument(
    "--log-user-speech/--no-log-user-speech",
    dest="log_user_speech",
    default=True,
    help="Log Realtime input VAD events to stderr (default on)",
  )
  parser.add_argument(
    "--input-transcription",
    action="store_true",
    help="Enable input audio transcription; log ASR text on stderr (extra API usage)",
  )
  parser.add_argument(
    "--input-transcription-model",
    default="gpt-4o-mini-transcribe",
    help="Transcription model when --input-transcription is set",
  )
  parser.add_argument(
    "--playback-gain",
    type=float,
    default=1.75,
    metavar="X",
    help="Digital gain on downlink PCM before soundd (1.0 = as decoded; default 1.75; clip at ±32767)",
  )
  parser.add_argument(
    "--output-speed",
    type=float,
    default=1.0,
    metavar="X",
    help="Realtime audio.output.speed (0.25–1.5; default 1.0; slightly lower can sound clearer)",
  )
  args = parser.parse_args()
  if args.playback_gain <= 0:
    print("--playback-gain must be > 0", file=sys.stderr)
    sys.exit(1)
  if not 0.25 <= args.output_speed <= 1.5:
    print("--output-speed must be between 0.25 and 1.5", file=sys.stderr)
    sys.exit(1)
  if not 0.0 <= args.vad_threshold <= 1.0:
    print("--vad-threshold must be between 0 and 1", file=sys.stderr)
    sys.exit(1)
  if args.half_duplex_rms < 0:
    print("--half-duplex-rms must be >= 0", file=sys.stderr)
    sys.exit(1)
  if args.half_duplex_hangover < 0 or args.half_duplex_after_response < 0:
    print("--half-duplex-hangover and --half-duplex-after-response must be >= 0", file=sys.stderr)
    sys.exit(1)

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
        interrupt_response=args.interrupt_response,
        server_auto_response=args.server_auto_response,
        noise_reduction=args.noise_reduction,
        playback_gain=args.playback_gain,
        output_speed=args.output_speed,
        half_duplex=args.half_duplex,
        half_duplex_rms=args.half_duplex_rms,
        half_duplex_hangover_s=args.half_duplex_hangover,
        half_duplex_after_response_s=args.half_duplex_after_response,
        log_user_speech=args.log_user_speech,
        input_transcription=args.input_transcription,
        input_transcription_model=args.input_transcription_model,
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
