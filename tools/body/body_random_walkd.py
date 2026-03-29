#!/usr/bin/env python3
"""
Random walk on the comma body via bodyjim + localhost webrtcd.

Started by manager when comma body is active (``comma_body_stack_should_run`` in
``system/manager/process_config.py``).  Requires ``pip install bodyjim`` on device.

Params:
  ``BodyRandomWalkHumanRender`` (default on) — pygame window; off → headless.

Env vars:
  ``BODY_RANDOM_WALK_VERBOSE=1``           — full tracebacks on failures.
  ``BODY_RANDOM_WALK_CAMERAS=driver``      — override camera (default: ``LivestreamCamera`` param).
  ``BODY_RANDOM_WALK_RECEIVE_TIMEOUT=15``  — bodyjim first-frame timeout (seconds).
"""
from __future__ import annotations

import os
import signal
import sys
import time

import numpy as np

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

_VERBOSE = os.environ.get("BODY_RANDOM_WALK_VERBOSE", "").strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# aiortc RTX bug workaround
# ---------------------------------------------------------------------------

def _patch_aiortc_rtx_bug() -> None:
  """
  aiortc <=1.10.1: after unwrapping an RTX retransmission packet, ``_handle_rtp_packet`` still
  passes the ``video/rtx`` codec to the decoder thread which crashes with
  ``ValueError: No decoder found for MIME type 'video/rtx'``.
  Fixed upstream in https://github.com/aiortc/aiortc/pull/1260 (one-line fix).

  The device filesystem (``/usr/local/venv``) is read-only, so we monkeypatch ``decoder_worker``
  in memory to silently skip RTX codecs instead of crashing.
  """
  try:
    import aiortc.rtcrtpreceiver as mod
    _original_decoder_worker = mod.decoder_worker

    def _patched_decoder_worker(loop, input_q, output_q):
      """Same as original but skips RTX codec frames that slip through the unfixed receiver."""
      import asyncio as _asyncio
      from aiortc.codecs import get_decoder as _get_decoder, is_rtx as _is_rtx

      codec_name = None
      decoder = None
      while True:
        task = input_q.get()
        if task is None:
          _asyncio.run_coroutine_threadsafe(output_q.put(None), loop)
          break
        codec, encoded_frame = task
        if _is_rtx(codec):
          continue
        if codec.name != codec_name:
          decoder = _get_decoder(codec)
          codec_name = codec.name
        for frame in decoder.decode(encoded_frame):
          _asyncio.run_coroutine_threadsafe(output_q.put(frame), loop)
      if decoder is not None:
        del decoder

    import inspect

    try:
      src = inspect.getsource(mod.RTCRtpReceiver._handle_rtp_packet)
    except (OSError, TypeError):
      src = ""
    if "codec = self.__codecs[apt]" not in src and "codec = self._RTCRtpReceiver__codecs[apt]" not in src:
      mod.decoder_worker = _patched_decoder_worker
      cloudlog.info("body_random_walkd: monkeypatched aiortc decoder_worker to skip RTX codecs")
    else:
      cloudlog.info("body_random_walkd: aiortc RTX fix already present upstream")
  except Exception as e:
    cloudlog.warning("body_random_walkd: aiortc RTX monkeypatch failed: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bodyjim_cameras(params: Params) -> list[str]:
  """Camera list for bodyjim — must match webrtcd's outgoing track (``LivestreamCamera``)."""
  raw = os.environ.get("BODY_RANDOM_WALK_CAMERAS", "").strip()
  if raw:
    cam = raw.split(",")[0].strip()
    if cam in ("driver", "wideRoad"):
      return [cam]
    cloudlog.warning("body_random_walkd: invalid BODY_RANDOM_WALK_CAMERAS=%r; falling back to param", raw)
  active = params.get("LivestreamCamera") or "driver"
  if active not in ("driver", "wideRoad"):
    active = "driver"
  return [active]


def _patch_bodyjim_timeouts() -> None:
  """bodyjim defaults (2 s receive / 3 s connect) are too short for first keyframe on device."""
  import bodyjim.data_stream as ds

  raw = os.environ.get("BODY_RANDOM_WALK_RECEIVE_TIMEOUT", "15").strip()
  try:
    v = float(raw)
    if v > 0:
      ds.RECEIVE_TIMEOUT_SECONDS = v
  except ValueError:
    cloudlog.warning("body_random_walkd: bad BODY_RANDOM_WALK_RECEIVE_TIMEOUT=%r", raw)
  cloudlog.info("body_random_walkd: bodyjim receive_timeout=%.1fs", ds.RECEIVE_TIMEOUT_SECONDS)


def _safe_close(env: object | None) -> None:
  if env is not None:
    try:
      env.close()  # type: ignore[union-attr]
    except Exception:
      pass


def _log_failure(stage: str, err: BaseException, *, elapsed_s: float | None = None) -> None:
  elapsed = f" after {elapsed_s:.2f}s" if elapsed_s is not None else ""
  detail = str(err) or "(empty — typical of Future.result(timeout) waiting for decoded video)"
  cloudlog.warning("body_random_walkd: %s failed%s: %s: %s", stage, elapsed, type(err).__name__, detail)
  if _VERBOSE:
    cloudlog.exception("body_random_walkd: %s traceback", stage)


def _log_pipeline_probe(cameras: list[str]) -> None:
  """Quick probe of the livestream encode topic after a reset timeout."""
  import cereal.messaging as messaging

  cam = cameras[0] if cameras else "driver"
  try:
    from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack, livestream_encode_data_diag

    service = LiveStreamVideoStreamTrack.camera_to_sock_mapping[cam]
  except Exception:
    return

  sock = messaging.sub_sock(service, conflate=False)
  deadline = time.monotonic() + 2.0
  while time.monotonic() < deadline:
    msg = messaging.recv_one_or_none(sock)
    if msg is not None:
      d = livestream_encode_data_diag(msg)
      cloudlog.warning(
        "body_random_walkd: probe %s flags=0x%x header=%dB data=%dB sync_ok=%s nal_types=%s",
        service, d["flags"], d["header_bytes"], d["data_bytes"], d["sync_ok"], d.get("nal_types_sample", ""),
      )
      return
    time.sleep(0.05)
  cloudlog.warning("body_random_walkd: probe %s — no messages (camerad / stream_encoderd not running?)", service)


def _prime_datachannel(env: object, should_stop: callable, *, timeout_s: float = 20.0) -> bool:
  """Poll ``env.step`` until the WebRTC data channel opens. Returns False if stopped."""
  t0 = time.monotonic()
  zero = np.zeros(2, dtype=np.float32)
  while time.monotonic() - t0 < timeout_s:
    if should_stop():
      return False
    try:
      env.step(zero)  # type: ignore[union-attr]
      return True
    except AssertionError as e:
      if "Session not started" not in (str(e.args[0]) if e.args else ""):
        raise
      time.sleep(0.05)
  raise TimeoutError("data channel not ready — is webrtcd running?")


# ---------------------------------------------------------------------------
# Connection attempt (one try)
# ---------------------------------------------------------------------------

def _try_connect(BodyEnv, body_ip: str, cameras: list[str], render_mode: str | None,
                 should_stop: callable) -> object | None:
  """Attempt init → reset → data-channel prime. Returns env on success, None on failure."""
  env = None
  try:
    env = BodyEnv(body_ip, cameras, [], render_mode=render_mode)
  except Exception as e:
    _log_failure("BodyEnv.__init__", e)
    return None

  try:
    t0 = time.monotonic()
    env.reset()
    cloudlog.info("body_random_walkd: reset ok in %.2fs", time.monotonic() - t0)
  except Exception as e:
    _log_failure("BodyEnv.reset", e, elapsed_s=time.monotonic() - t0)
    if isinstance(e, TimeoutError):
      _log_pipeline_probe(cameras)
    _safe_close(env)
    return None

  try:
    if not _prime_datachannel(env, should_stop):
      _safe_close(env)
      return None  # stopped
  except TimeoutError as e:
    _log_failure("datachannel_prime", e)
    _safe_close(env)
    return None

  return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  stop = False

  def _stop(*_args: object) -> None:
    nonlocal stop
    stop = True

  def _stop_check() -> bool:
    return stop

  signal.signal(signal.SIGTERM, _stop)
  signal.signal(signal.SIGINT, _stop)

  params = Params()
  human = params.get_bool("BodyRandomWalkHumanRender")
  if not human:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

  _patch_aiortc_rtx_bug()

  BodyEnv = None
  while BodyEnv is None and not stop:
    try:
      _patch_bodyjim_timeouts()
      from bodyjim import BodyEnv as _BodyEnv

      BodyEnv = _BodyEnv
    except ImportError:
      cloudlog.event("body_random_walkd: bodyjim not installed; pip install bodyjim", error=True)
      for _ in range(60):
        if stop:
          return
        time.sleep(1.0)

  if BodyEnv is None:
    return

  pygame = None
  if human:
    try:
      import pygame as _pygame

      pygame = _pygame
    except ImportError:
      cloudlog.event("body_random_walkd: pygame missing", error=True)
      while not stop:
        time.sleep(5.0)
      return

  render_mode = "human" if human else None
  cameras = _bodyjim_cameras(params)
  cloudlog.info("body_random_walkd: cameras=%r render=%s", cameras, render_mode)

  while not stop:
    env = None
    while env is None and not stop:
      env = _try_connect(BodyEnv, "127.0.0.1", cameras, render_mode, _stop_check)
      if env is None and not stop:
        time.sleep(1.0)

    if env is None:
      break

    if human and pygame is not None:
      pygame.init()

    try:
      while not stop:
        env.step(env.action_space.sample())
        if stop:
          break
        if human and pygame is not None:
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
              stop = True
              break
          if not stop:
            env.render()
    except Exception:
      cloudlog.exception("body_random_walkd: session error; reconnecting")
    finally:
      _safe_close(env)


if __name__ == "__main__":
  try:
    main()
  except Exception:
    cloudlog.exception("body_random_walkd: fatal")
    sys.exit(1)
