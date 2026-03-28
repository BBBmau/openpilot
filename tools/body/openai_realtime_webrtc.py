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
  - For ``--screen-vision``: ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``
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

**Half-duplex uplink** (on by default): WebRTC uplink and (with ``--micd-suppress-during-assistant``)
``rawAudioData`` are **fully muted** for the whole assistant ``response`` (from
``response.create`` / ``response.created`` / ``output_audio_buffer.started`` through ``response.done``).
After ``response.done``, the client keeps the uplink off until an **API timing floor** (minimum hold
after ``response.done``, ``response.output_audio_transcript.done``, and ``response.output_audio.done``)
and, when ``soundd`` publishes ``sounddWebrtcQueueState`` (default on), until **soundd’s WebRTC/body
PCM queue** has stayed at/below ``--soundd-queue-nonempty-epsilon-samples`` for
``--soundd-stable-empty-ms`` plus ``--soundd-post-drain-hangover`` (room / OS buffer tail). If that
telemetry is missing, the API floor alone applies. Defaults favor **lower latency**; if the mic
picks up tail audio and retriggers, raise ``--half-duplex-min-after-response-done`` /
``--half-duplex-after-transcript`` or ``--commit-grace-after-unmute``. Use ``--no-soundd-speaker-gate``
for API-only timing, or ``--no-half-duplex`` for full-duplex (e.g. headset).

**micd** (device ``rawAudioData``): with ``--micd-suppress-during-assistant`` (default on, requires
half-duplex), the script sets param ``MicdSuppressRawAudio`` in lockstep with that gate so **micd**
publishes **silence** on ``rawAudioData`` while the assistant is speaking—reducing feedback paths
beyond WebRTC uplink (e.g. wake word, logging). Use ``--no-micd-suppress-during-assistant`` if you
need live ``rawAudioData`` during playback.

With ``--server-auto-response`` off (the default), the client sends ``response.create`` after each
``input_audio_buffer.committed`` while idle. Commits that arrive during an active response are
**ignored** (not queued)—queuing echo commits caused assistant→mic→assistant loops. A post-response
cooldown also blocks ``response.create`` briefly after ``response.done``.

**User speech logging** (``--log-user-speech``, on by default): prints Realtime VAD events to stderr
(``speech_started`` / ``speech_stopped`` / ``committed``, etc.). Add ``--input-transcription`` to
enable ASR on committed user audio and log ``[user speech transcript]`` lines (separate billing).

**Body camera + Gemini vision** (``--screen-vision``): registers Realtime function
``describe_visible_screen``. When speech triggers a tool call, the client waits for
``response.done``, grabs one decoded frame from the body **livestream** camera (same H.264 feed as
webrtcd: ``livestreamDriverEncodeData`` / ``livestreamWideRoadEncodeData`` from encoderd), encodes
PNG, POSTs to the Gemini multimodal API (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``), sends
``conversation.item.create`` with ``function_call_output``, then ``response.create`` for the spoken
reply. Optional ``--screenshot-path`` / ``--screenshot-cmd`` override the camera for desktop testing.

Usage:
  export OPENAI_API_KEY=sk-...
  python tools/body/openai_realtime_webrtc.py
  python tools/body/openai_realtime_webrtc.py --verbose
  python tools/body/openai_realtime_webrtc.py --debug
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import requests
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame

from cereal import messaging
from openpilot.common.params import Params
from openpilot.system.webrtc.device.audio import (
  BODY_REALTIME_PCM_SERVICE,
  BodyMicAudioTrack,
  BodySpeaker,
  SPEAKER_SAMPLE_RATE,
)
from openpilot.system.webrtc.device.video import grab_livestream_png_bytes

REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"
ICE_GATHER_TIMEOUT_S = 30.0
LOG = logging.getLogger("openai_realtime_webrtc")
MIC_LOG_INTERVAL = 50

# Realtime tool: capture screen → Gemini vision → function_call_output → response.create (audio).
SCREENSHOT_TOOL_NAME = "describe_visible_screen"

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


class _SounddQueueWatcher:
  """Background SubMaster for ``sounddWebrtcQueueState`` (PCM queued before soundd's output callback drains it)."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._queued_samples = 0
    self._seen = False
    self._alive = False
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    def _loop() -> None:
      sm = messaging.SubMaster(["sounddWebrtcQueueState"], poll="sounddWebrtcQueueState")
      while not self._stop.is_set():
        sm.update(50)
        with self._lock:
          self._queued_samples = int(sm["sounddWebrtcQueueState"].queuedSamples)
          self._seen = sm.seen["sounddWebrtcQueueState"]
          self._alive = sm.all_alive(["sounddWebrtcQueueState"])

    self._thread = threading.Thread(target=_loop, name="sounddWebrtcQueueState", daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=1.5)
      self._thread = None

  def telemetry_ok(self) -> bool:
    with self._lock:
      return self._seen and self._alive

  def queued_samples(self) -> int:
    with self._lock:
      return self._queued_samples


class _ResponseHalfDuplexGate:
  """Event-driven uplink mute: entire assistant response, then API timing floor + optional soundd queue drain.

  Soundd queue gating is armed only in ``on_response_done`` so idle listening is not muted by
  unrelated PCM in soundd’s webrtc deque or by stable-empty/hangover on startup.

  The deque can sit at a small steady backlog (RTP vs. soundd callback rate), so “drained” uses a
  sample epsilon below which we treat the queue as empty, plus an optional max wait fallback.
  """

  def __init__(
    self,
    *,
    after_transcript_s: float,
    after_audio_s: float,
    after_response_done_min_s: float,
    soundd_queue: _SounddQueueWatcher | None,
    soundd_stable_empty_s: float,
    soundd_post_drain_hangover_s: float,
    soundd_queue_nonempty_epsilon_samples: int,
    soundd_max_drain_wait_s: float,
    commit_grace_after_unmute_s: float,
  ) -> None:
    self._after_transcript_s = after_transcript_s
    self._after_audio_s = after_audio_s
    self._after_response_done_min_s = after_response_done_min_s
    self._soundd_queue = soundd_queue
    self._soundd_stable_empty_s = soundd_stable_empty_s
    self._soundd_post_drain_hangover_s = soundd_post_drain_hangover_s
    self._soundd_queue_nonempty_epsilon_samples = max(0, int(soundd_queue_nonempty_epsilon_samples))
    self._soundd_max_drain_wait_s = float(soundd_max_drain_wait_s)
    self._commit_grace_after_unmute_s = commit_grace_after_unmute_s
    self._in_response = False
    self._transcript_done_at: float | None = None
    self._audio_done_at: float | None = None
    self._api_release_at = 0.0
    self._empty_since: float | None = None
    self._hangover_until: float | None = None
    self._was_suppressed = False
    self._commit_grace_until = 0.0
    # Only run soundd queue / stable-empty / hangover after response.done. Otherwise idle would
    # mute whenever queuedSamples>0 (other webrtc/body PCM) or force an extra hangover on startup.
    self._armed_post_response_soundd = False
    self._soundd_drain_deadline: float | None = None
    self._soundd_drain_timeout_logged = False

  def _reset_post_done_drain(self) -> None:
    self._empty_since = None
    self._hangover_until = None

  def _soundd_queue_nonempty(self, q: int) -> bool:
    e = self._soundd_queue_nonempty_epsilon_samples
    return q > e if e > 0 else q > 0

  def suppress_uplink(self) -> bool:
    now = time.monotonic()
    if self._in_response:
      result = True
    elif now < self._api_release_at:
      result = True
    elif (
      self._armed_post_response_soundd
      and self._soundd_queue is not None
      and self._soundd_queue.telemetry_ok()
    ):
      q = self._soundd_queue.queued_samples()
      if (
        self._soundd_max_drain_wait_s > 0.0
        and self._soundd_drain_deadline is not None
        and now >= self._soundd_drain_deadline
      ):
        if not self._soundd_drain_timeout_logged:
          LOG.warning(
            "soundd speaker gate: drain wait exceeded %.1fs (queuedSamples=%d); forcing uplink unmute",
            self._soundd_max_drain_wait_s,
            q,
          )
          self._soundd_drain_timeout_logged = True
        result = False
      elif self._soundd_queue_nonempty(q):
        self._reset_post_done_drain()
        result = True
      else:
        if self._empty_since is None:
          self._empty_since = now
        if now - self._empty_since < self._soundd_stable_empty_s:
          result = True
        else:
          if self._hangover_until is None:
            self._hangover_until = now + self._soundd_post_drain_hangover_s
          result = now < self._hangover_until
    else:
      result = False

    if (
      self._armed_post_response_soundd
      and not result
      and not self._in_response
      and now >= self._api_release_at
    ):
      self._armed_post_response_soundd = False

    if self._was_suppressed and not result:
      self._commit_grace_until = max(
        self._commit_grace_until,
        now + self._commit_grace_after_unmute_s,
      )
    self._was_suppressed = result
    return result

  def commit_blocked(self) -> bool:
    now = time.monotonic()
    if self.suppress_uplink():
      return True
    return now < self._commit_grace_until

  def on_response_created(self) -> None:
    self._in_response = True
    self._transcript_done_at = None
    self._audio_done_at = None
    self._reset_post_done_drain()

  def ensure_response_active(self) -> None:
    """If response.created was missed, still mute from first output audio frame."""
    if not self._in_response:
      self.on_response_created()

  def on_output_audio_transcript_done(self) -> None:
    self._transcript_done_at = time.monotonic()

  def on_output_audio_done(self) -> None:
    self._audio_done_at = time.monotonic()

  def on_response_done(self) -> None:
    """Set API timing floor; after that, optional soundd drain + stable-empty + hangover extends mute."""
    now = time.monotonic()
    release = now + self._after_response_done_min_s
    if self._transcript_done_at is not None:
      release = max(release, self._transcript_done_at + self._after_transcript_s)
    else:
      release = max(release, now + self._after_transcript_s)
    if self._audio_done_at is not None:
      release = max(release, self._audio_done_at + self._after_audio_s)
    else:
      release = max(release, now + self._after_audio_s)
    self._api_release_at = max(self._api_release_at, release)
    self._in_response = False
    self._transcript_done_at = None
    self._audio_done_at = None
    self._armed_post_response_soundd = True
    self._soundd_drain_timeout_logged = False
    self._soundd_drain_deadline = (
      now + self._soundd_max_drain_wait_s if self._soundd_max_drain_wait_s > 0.0 else None
    )
    self._reset_post_done_drain()


class _HalfDuplexMicTrack(BodyMicAudioTrack):
  """Sends silence to the peer while ``suppress_uplink()`` is true (same frame timing as mic)."""

  def __init__(self, suppress_uplink: Callable[[], bool]) -> None:
    super().__init__()
    self._suppress_uplink = suppress_uplink

  async def recv(self):
    frame = await super().recv()
    if not self._suppress_uplink():
      return frame
    # New frame: in-place ``planes[0].update`` is not reliably encoded by PyAV/aiortc.
    n = int(frame.samples)
    silent = AudioFrame(format="s16", layout="mono", samples=n)
    silent.planes[0].update(np.zeros(n, dtype=np.int16).tobytes())
    silent.pts = frame.pts
    silent.sample_rate = frame.sample_rate
    silent.time_base = frame.time_base
    return silent


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


def _vision_tool_definitions() -> list[dict[str, Any]]:
  return [
    {
      "type": "function",
      "name": SCREENSHOT_TOOL_NAME,
      "description": (
        "Capture a still from the body livestream camera (road or wide road view), analyze it with "
        "a vision model, and return a text description. Call when the user asks what you see, what is "
        "ahead, or to describe the scene."
      ),
      "parameters": {
        "type": "object",
        "properties": {
          "focus": {
            "type": "string",
            "description": "Optional: what to emphasize (e.g. error text, map, speed).",
          },
        },
        "additionalProperties": False,
      },
    }
  ]


def _load_vision_png(
  *,
  path: str | None,
  cmd: str | None,
  body_camera: str,
  camera_timeout_s: float,
) -> bytes:
  if path:
    with open(path, "rb") as f:
      return f.read()
  if cmd:
    argv = shlex.split(cmd, posix=os.name != "nt")
    r = subprocess.run(argv, capture_output=True, timeout=45, check=False)
    if r.returncode != 0:
      err = (r.stderr or b"").decode("utf-8", errors="replace")[:800]
      raise RuntimeError(f"vision command failed (exit {r.returncode}): {err}")
    if not r.stdout:
      raise RuntimeError("vision command produced no stdout")
    return r.stdout
  return grab_livestream_png_bytes(body_camera, timeout_s=camera_timeout_s)


def _gemini_describe_png(api_key: str, model: str, png_bytes: bytes, instruction: str) -> str:
  url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
  body: dict[str, Any] = {
    "contents": [
      {
        "parts": [
          {
            "inline_data": {
              "mime_type": "image/png",
              "data": base64.standard_b64encode(png_bytes).decode("ascii"),
            },
          },
          {"text": instruction},
        ],
      },
    ],
  }
  r = requests.post(url, params={"key": api_key}, json=body, timeout=120)
  if not r.ok:
    raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:2000]}")
  data = r.json()
  cands = data.get("candidates") or []
  if not cands:
    raise RuntimeError(f"Gemini: no candidates in {data!r}")
  parts = (cands[0].get("content") or {}).get("parts") or []
  texts: list[str] = []
  for p in parts:
    if isinstance(p, dict) and p.get("text"):
      texts.append(str(p["text"]))
  if not texts:
    raise RuntimeError(f"Gemini: no text in response {data!r}")
  return "\n".join(texts).strip()


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
  half_duplex_after_transcript_s: float,
  half_duplex_after_audio_s: float,
  half_duplex_after_response_done_min_s: float,
  post_response_commit_cooldown_s: float,
  micd_suppress_during_assistant: bool,
  soundd_speaker_gate: bool,
  soundd_stable_empty_s: float,
  soundd_post_drain_hangover_s: float,
  soundd_queue_nonempty_epsilon_samples: int,
  soundd_max_drain_wait_s: float,
  commit_grace_after_unmute_s: float,
  log_user_speech: bool,
  input_transcription: bool,
  input_transcription_model: str,
  screen_vision: bool,
  gemini_api_key: str | None,
  gemini_model: str,
  screenshot_path: str | None,
  screenshot_cmd: str | None,
  vision_body_camera: str,
  vision_camera_timeout_s: float,
) -> None:
  loop = asyncio.get_running_loop()
  soundd_watcher: _SounddQueueWatcher | None = None
  if half_duplex and soundd_speaker_gate:
    soundd_watcher = _SounddQueueWatcher()
    soundd_watcher.start()

  gate: _ResponseHalfDuplexGate | None = None
  if half_duplex:
    gate = _ResponseHalfDuplexGate(
      after_transcript_s=half_duplex_after_transcript_s,
      after_audio_s=half_duplex_after_audio_s,
      after_response_done_min_s=half_duplex_after_response_done_min_s,
      soundd_queue=soundd_watcher,
      soundd_stable_empty_s=soundd_stable_empty_s,
      soundd_post_drain_hangover_s=soundd_post_drain_hangover_s,
      soundd_queue_nonempty_epsilon_samples=soundd_queue_nonempty_epsilon_samples,
      soundd_max_drain_wait_s=soundd_max_drain_wait_s,
      commit_grace_after_unmute_s=commit_grace_after_unmute_s,
    )

  Params().put_bool_nonblocking("MicdSuppressRawAudio", False)

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
  if screen_vision and gemini_api_key:
    session["tools"] = _vision_tool_definitions()
    vis_hint = (
      " When the user asks what you see, what is ahead, or to describe the scene, call "
      f"{SCREENSHOT_TOOL_NAME}; then summarize the returned description briefly in speech."
    )
    session["instructions"] = (session.get("instructions") or "") + vis_hint
    LOG.info(
      "screen vision: tool=%s gemini_model=%s camera=%s timeout=%.1fs path=%s cmd=%s",
      SCREENSHOT_TOOL_NAME,
      gemini_model,
      vision_body_camera,
      vision_camera_timeout_s,
      screenshot_path or "(livestream)",
      screenshot_cmd or "(livestream)",
    )

  LOG.info("session: model=%s voice=%s", model, voice)
  if gate is not None:
    if soundd_watcher is not None:
      LOG.info(
        "half-duplex uplink: mute through each response; after API floor (max response.done+%.2fs, "
        "transcript.done+%.2fs, output_audio.done+%.2fs), also wait for soundd queue drain "
        "(queuedSamples<=%d treated empty, stable %.0fms + %.2fs hangover; max wait %.1fs from response.done)",
        half_duplex_after_response_done_min_s,
        half_duplex_after_transcript_s,
        half_duplex_after_audio_s,
        soundd_queue_nonempty_epsilon_samples,
        soundd_stable_empty_s * 1000.0,
        soundd_post_drain_hangover_s,
        soundd_max_drain_wait_s,
      )
    else:
      LOG.info(
        "half-duplex uplink: mute through each response; unmute after max(response.done+%.2fs, "
        "transcript.done+%.2fs, output_audio.done+%.2fs)",
        half_duplex_after_response_done_min_s,
        half_duplex_after_transcript_s,
        half_duplex_after_audio_s,
      )
    if micd_suppress_during_assistant:
      LOG.info("micd: MicdSuppressRawAudio follows half-duplex (rawAudioData silence while gated)")
  elif micd_suppress_during_assistant:
    LOG.warning("--micd-suppress-during-assistant ignored without half-duplex")
  if post_response_commit_cooldown_s > 0:
    LOG.info(
      "post-response: ignore response.create from commits for %.2fs after each response.done",
      post_response_commit_cooldown_s,
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
  # Commits during an active response are dropped (not queued)—echo was chaining response.create.
  ignore_response_create_until = [0.0]
  response_busy = False
  pending_response_create = False

  responses_done_ids: set[str] = set()
  response_done_waiters: dict[str, asyncio.Event] = {}
  vision_suppress_commits = [0]

  def _mark_response_done_for_vision(rid: str | None) -> None:
    if not screen_vision or not rid:
      return
    responses_done_ids.add(rid)
    e = response_done_waiters.pop(rid, None)
    if e is not None:
      e.set()

  async def _wait_response_done_for_vision(rid: str) -> None:
    if not rid or rid in responses_done_ids:
      return
    e = asyncio.Event()
    response_done_waiters[rid] = e
    try:
      await asyncio.wait_for(e.wait(), 120.0)
    finally:
      response_done_waiters.pop(rid, None)

  async def _execute_screen_vision_tool(call_id: str, response_id: str, arguments_str: str) -> None:
    vision_suppress_commits[0] += 1
    try:
      await _wait_response_done_for_vision(response_id)
      focus = ""
      try:
        args = json.loads(arguments_str) if arguments_str else {}
        if isinstance(args, dict):
          focus = str(args.get("focus") or "").strip()
      except json.JSONDecodeError:
        pass
      instruction = (
        "Describe this image clearly and concisely for a voice assistant to read aloud to the user."
      )
      if focus:
        instruction += f" Emphasize: {focus}."
      try:
        png = await asyncio.to_thread(
          _load_vision_png,
          path=screenshot_path,
          cmd=screenshot_cmd,
          body_camera=vision_body_camera,
          camera_timeout_s=vision_camera_timeout_s,
        )
        assert gemini_api_key is not None
        desc = await asyncio.to_thread(
          _gemini_describe_png, gemini_api_key, gemini_model, png, instruction
        )
        out = json.dumps({"ok": True, "description": desc})
      except Exception as exc:
        LOG.exception("screen vision tool failed")
        out = json.dumps({"ok": False, "description": "", "error": str(exc)})
      if dc.readyState != "open":
        LOG.warning("data channel closed before function_call_output")
        return
      try:
        dc.send(
          json.dumps(
            {
              "type": "conversation.item.create",
              "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": out,
              },
            }
          )
        )
      except Exception:
        LOG.exception("conversation.item.create (function_call_output) send failed")
        return
      _send_response_create()
    finally:
      vision_suppress_commits[0] -= 1

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
    if gate is not None:
      gate.on_response_created()

  def _on_input_committed() -> None:
    nonlocal response_busy, pending_response_create
    if server_auto_response:
      return
    if vision_suppress_commits[0] > 0:
      LOG.debug("input_audio_buffer.committed during screen vision tool; skip response.create")
      return
    if gate is not None and gate.commit_blocked():
      LOG.debug("half-duplex gate or post-unmute grace; not scheduling response.create")
      return
    if response_busy:
      LOG.debug(
        "input_audio_buffer.committed while response active; not scheduling response.create "
        "(echo-loop guard)",
      )
      return
    now = time.monotonic()
    if now < ignore_response_create_until[0]:
      LOG.debug(
        "input_audio_buffer.committed during post-response cooldown; not scheduling response.create "
        "(%.2fs left)",
        ignore_response_create_until[0] - now,
      )
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
    if (
      screen_vision
      and gemini_api_key
      and typ == "response.function_call_arguments.done"
      and ev.get("name") == SCREENSHOT_TOOL_NAME
    ):
      call_id = ev.get("call_id")
      response_id = ev.get("response_id") or ""
      if call_id:
        loop.create_task(
          _execute_screen_vision_tool(call_id, response_id, ev.get("arguments") or "")
        )
        LOG.info(
          "screen vision: scheduled tool call_id=%s response_id=%s",
          call_id,
          response_id or "(none)",
        )
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
      resp_obj = ev.get("response")
      if isinstance(resp_obj, dict):
        _mark_response_done_for_vision(resp_obj.get("id"))
      now_sync = time.monotonic()
      if gate is not None:
        gate.on_response_done()
      ignore_response_create_until[0] = max(
        ignore_response_create_until[0],
        now_sync + post_response_commit_cooldown_s,
      )
      if dc.readyState == "open":
        try:
          dc.send(json.dumps({"type": "input_audio_buffer.clear"}))
        except Exception:
          LOG.debug("input_audio_buffer.clear after response.done failed", exc_info=True)
      if not server_auto_response:
        _on_response_done()
    if gate is not None:
      if typ == "response.created":
        gate.on_response_created()
      elif typ in ("response.output_audio.delta", "output_audio_buffer.started"):
        # WebRTC: model audio is mostly on RTP; ``response.output_audio.delta`` is often absent.
        # ``output_audio_buffer.started`` still fires on the data channel (see Realtime server events).
        gate.ensure_response_active()
      elif typ == "response.output_audio_transcript.done":
        gate.on_output_audio_transcript_done()
      elif typ == "response.output_audio.done":
        gate.on_output_audio_done()
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
        if gate is not None:
          gate.ensure_response_active()
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
  micd_suppress_task: asyncio.Task[None] | None = None
  if micd_suppress_during_assistant and gate is not None:

    async def _micd_suppress_loop() -> None:
      p = Params()
      while not stop.is_set():
        p.put_bool_nonblocking("MicdSuppressRawAudio", gate.suppress_uplink())
        await asyncio.sleep(0.02)

    micd_suppress_task = asyncio.create_task(_micd_suppress_loop())

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
    if micd_suppress_task is not None:
      micd_suppress_task.cancel()
      try:
        await micd_suppress_task
      except asyncio.CancelledError:
        pass
    Params().put_bool_nonblocking("MicdSuppressRawAudio", False)
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
    if soundd_watcher is not None:
      soundd_watcher.stop()


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
    help="Mute WebRTC uplink for each assistant response (response.created→done) plus tail timing below",
  )
  parser.add_argument(
    "--half-duplex-after-transcript",
    type=float,
    default=0.5,
    metavar="SEC",
    help="Keep uplink muted at least this long after response.output_audio_transcript.done (default 0.5; raise if echo)",
  )
  parser.add_argument(
    "--half-duplex-after-audio",
    type=float,
    default=0.25,
    metavar="SEC",
    help="Keep uplink muted at least this long after response.output_audio.done (default 0.25)",
  )
  parser.add_argument(
    "--half-duplex-min-after-response-done",
    type=float,
    default=0.3,
    metavar="SEC",
    help="Minimum uplink mute after response.done (before transcript/audio math); default 0.3 (was 2.0)",
  )
  parser.add_argument(
    "--post-response-commit-cooldown",
    type=float,
    default=1.15,
    metavar="SEC",
    help="Do not send response.create on input_audio_buffer.committed until this long after response.done "
    "(default 1.15; increase if commits race before uplink unmutes)",
  )
  parser.add_argument(
    "--micd-suppress-during-assistant/--no-micd-suppress-during-assistant",
    dest="micd_suppress_during_assistant",
    default=True,
    help="With half-duplex, set MicdSuppressRawAudio so micd publishes silence on rawAudioData while gated (default on)",
  )
  parser.add_argument(
    "--soundd-speaker-gate/--no-soundd-speaker-gate",
    dest="soundd_speaker_gate",
    default=True,
    help="With half-duplex, keep uplink muted until soundd’s webrtc PCM queue is drained (sounddWebrtcQueueState); "
    "falls back to API-only timing if telemetry is absent (default on)",
  )
  parser.add_argument(
    "--soundd-stable-empty-ms",
    type=float,
    default=50.0,
    metavar="MS",
    help="With --soundd-speaker-gate, require queue at/below epsilon for this long before hangover (default 50)",
  )
  parser.add_argument(
    "--soundd-post-drain-hangover",
    type=float,
    default=0.15,
    metavar="SEC",
    help="With --soundd-speaker-gate, extra mute after stable-empty queue (default 0.15)",
  )
  parser.add_argument(
    "--soundd-queue-nonempty-epsilon-samples",
    type=int,
    default=7200,
    metavar="N",
    help="With --soundd-speaker-gate, treat queuedSamples<=N as empty (~7200≈150ms at 48kHz; 0=strict q==0)",
  )
  parser.add_argument(
    "--soundd-max-drain-wait",
    type=float,
    default=15.0,
    metavar="SEC",
    help="With --soundd-speaker-gate, force uplink unmute after this long waiting on the queue (0=disable)",
  )
  parser.add_argument(
    "--commit-grace-after-unmute",
    type=float,
    default=0.2,
    metavar="SEC",
    help="After uplink unmutes, block response.create this long (default 0.2; echo buffer)",
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
    "--screen-vision",
    action="store_true",
    help="Register describe_visible_screen: body camera frame → Gemini → function_call_output → response.create",
  )
  parser.add_argument(
    "--vision-camera",
    choices=("driver", "wideRoad"),
    default=None,
    help="Livestream source for vision (default: Params LivestreamCamera, else driver)",
  )
  parser.add_argument(
    "--vision-timeout",
    type=float,
    default=15.0,
    metavar="SEC",
    help="Max seconds to wait for a decodable livestream frame when using the body camera",
  )
  parser.add_argument(
    "--gemini-model",
    default="gemini-2.0-flash",
    metavar="MODEL",
    help="Gemini model id for screen vision (e.g. gemini-2.0-flash, gemini-2.0-flash-exp)",
  )
  parser.add_argument(
    "--screenshot-path",
    default=None,
    metavar="PATH",
    help="Read PNG from this file instead of the livestream camera (desktop testing)",
  )
  parser.add_argument(
    "--screenshot-cmd",
    default=None,
    metavar="CMD",
    help="Shell command that writes PNG bytes to stdout; overrides livestream (parsed with shlex)",
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
  if args.half_duplex_after_transcript < 0 or args.half_duplex_after_audio < 0:
    print("--half-duplex-after-transcript and --half-duplex-after-audio must be >= 0", file=sys.stderr)
    sys.exit(1)
  if args.half_duplex_min_after_response_done < 0:
    print("--half-duplex-min-after-response-done must be >= 0", file=sys.stderr)
    sys.exit(1)
  if args.post_response_commit_cooldown < 0:
    print("--post-response-commit-cooldown must be >= 0", file=sys.stderr)
    sys.exit(1)
  if args.soundd_stable_empty_ms < 0:
    print("--soundd-stable-empty-ms must be >= 0", file=sys.stderr)
    sys.exit(1)
  if args.soundd_post_drain_hangover < 0:
    print("--soundd-post-drain-hangover must be >= 0", file=sys.stderr)
    sys.exit(1)
  if args.soundd_queue_nonempty_epsilon_samples < 0:
    print("--soundd-queue-nonempty-epsilon-samples must be >= 0", file=sys.stderr)
    sys.exit(1)
  if args.soundd_max_drain_wait < 0:
    print("--soundd-max-drain-wait must be >= 0", file=sys.stderr)
    sys.exit(1)
  if args.commit_grace_after_unmute < 0:
    print("--commit-grace-after-unmute must be >= 0", file=sys.stderr)
    sys.exit(1)
  if args.vision_timeout <= 0:
    print("--vision-timeout must be > 0", file=sys.stderr)
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

  gemini_key: str | None = None
  if args.screen_vision:
    gemini_key = (
      os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not gemini_key:
      print("Set GEMINI_API_KEY or GOOGLE_API_KEY for --screen-vision.", file=sys.stderr)
      sys.exit(1)

  vision_cam = args.vision_camera
  if vision_cam is None:
    raw_lc = Params().get("LivestreamCamera")
    if isinstance(raw_lc, bytes):
      raw_lc = raw_lc.decode("utf-8", errors="replace")
    vision_cam = raw_lc if raw_lc in ("driver", "wideRoad") else "driver"

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
        half_duplex_after_transcript_s=args.half_duplex_after_transcript,
        half_duplex_after_audio_s=args.half_duplex_after_audio,
        half_duplex_after_response_done_min_s=args.half_duplex_min_after_response_done,
        post_response_commit_cooldown_s=args.post_response_commit_cooldown,
        micd_suppress_during_assistant=args.micd_suppress_during_assistant,
        soundd_speaker_gate=args.soundd_speaker_gate,
        soundd_stable_empty_s=args.soundd_stable_empty_ms / 1000.0,
        soundd_post_drain_hangover_s=args.soundd_post_drain_hangover,
        soundd_queue_nonempty_epsilon_samples=args.soundd_queue_nonempty_epsilon_samples,
        soundd_max_drain_wait_s=args.soundd_max_drain_wait,
        commit_grace_after_unmute_s=args.commit_grace_after_unmute,
        log_user_speech=args.log_user_speech,
        input_transcription=args.input_transcription,
        input_transcription_model=args.input_transcription_model,
        screen_vision=args.screen_vision,
        gemini_api_key=gemini_key,
        gemini_model=args.gemini_model,
        screenshot_path=args.screenshot_path,
        screenshot_cmd=args.screenshot_cmd,
        vision_body_camera=vision_cam,
        vision_camera_timeout_s=args.vision_timeout,
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
