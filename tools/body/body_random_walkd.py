#!/usr/bin/env python3
"""
Random walk on the comma body via bodyjim + localhost webrtcd.

Started by manager when comma body is active (``comma_body_stack_should_run`` in
``system/manager/process_config.py``).  Requires ``pip install bodyjim`` on device.

Env vars:
  ``BODY_RANDOM_WALK_VERBOSE=1``           — full tracebacks on failures.
  ``BODY_RANDOM_WALK_CAMERAS=driver``      — override camera (default: ``LivestreamCamera`` param).
  ``BODY_RANDOM_WALK_RECEIVE_TIMEOUT=15``  — bodyjim first-frame timeout (seconds).
  ``BODY_RANDOM_WALK_FRAME_SCALE=4``       — downsample factor for received video (1 = full res).
"""
from __future__ import annotations

import os
import signal
import sys
import time
import types

import av.logging
import numpy as np

av.logging.set_level(av.logging.PANIC)

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

_VERBOSE = os.environ.get("BODY_RANDOM_WALK_VERBOSE", "").strip().lower() in ("1", "true", "yes")
_FRAME_SCALE = max(1, int(os.environ.get("BODY_RANDOM_WALK_FRAME_SCALE", "4")))


# ---------------------------------------------------------------------------
# aiortc / bodyjim patches (applied once before import)
# ---------------------------------------------------------------------------

def _patch_aiortc() -> None:
  """Fix two aiortc issues that prevent video from working on the device.

  1. **RTX crash** (aiortc <=1.10.1): ``decoder_worker`` receives ``video/rtx``
     codec after retransmission unwrapping and crashes.  We replace it with a
     version that silently skips RTX frames.  The replacement also downscales
     decoded frames in-thread (controlled by ``BODY_RANDOM_WALK_FRAME_SCALE``)
     to keep the output queue small and the event loop unblocked.

  2. **VP8 negotiation**: VP8 is first in aiortc's default codec list.
     webrtcd sends pre-encoded H264 — if VP8 is negotiated the receiver gets
     garbage.  We remove VP8 from the codec registry so H264 is always used.
  """
  try:
    import inspect

    import aiortc.rtcrtpreceiver as mod
    from aiortc.codecs import CODECS

    # --- RTX + downscale decoder worker -----------------------------------
    def _patched_decoder_worker(loop, input_q, output_q):
      import asyncio as _asyncio
      from aiortc.codecs import get_decoder as _get_decoder, is_rtx as _is_rtx

      scale = _FRAME_SCALE
      codec_name: str | None = None
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
        try:
          for frame in decoder.decode(encoded_frame):
            if scale > 1:
              frame = frame.reformat(width=frame.width // scale, height=frame.height // scale, format="rgb24")
            _asyncio.run_coroutine_threadsafe(output_q.put(frame), loop)
        except Exception:
          pass
      if decoder is not None:
        del decoder

    try:
      src = inspect.getsource(mod.RTCRtpReceiver._handle_rtp_packet)
    except (OSError, TypeError):
      src = ""
    if "codec = self.__codecs[apt]" not in src and "codec = self._RTCRtpReceiver__codecs[apt]" not in src:
      mod.decoder_worker = _patched_decoder_worker

    # --- Force H264 -------------------------------------------------------
    CODECS["video"] = [c for c in CODECS["video"] if c.mimeType != "video/VP8"]
  except Exception as e:
    cloudlog.warning("body_random_walkd: aiortc patch failed: %s", e)


def _mock_pygame() -> None:
  """Stub ``pygame`` so bodyjim imports without the ~25 MB C library."""
  pg = types.ModuleType("pygame")
  for sub in ("display", "font", "event", "surfarray", "transform", "pkgdata"):
    m = types.ModuleType(f"pygame.{sub}")
    setattr(pg, sub, m)
    sys.modules[f"pygame.{sub}"] = m
  pg.Surface = type("Surface", (), {"__init__": lambda *a, **kw: None})
  pg.QUIT = 256
  pg.init = pg.quit = lambda: None
  pg.display.init = pg.display.quit = lambda: None
  sys.modules["pygame"] = pg


def _patch_bodyjim() -> None:
  """Adjust bodyjim defaults for running on the device.

  * Increase the receive timeout (default 2 s is too short for H264 keyframe).
  * Replace ``_receive_async`` so it calls ``to_ndarray()`` on the already-
    downscaled RGB frames produced by the patched decoder worker.
  """
  import asyncio as _aio

  import bodyjim.data_stream as ds

  raw = os.environ.get("BODY_RANDOM_WALK_RECEIVE_TIMEOUT", "15").strip()
  try:
    v = float(raw)
    if v > 0:
      ds.RECEIVE_TIMEOUT_SECONDS = v
  except ValueError:
    pass

  if _FRAME_SCALE <= 1:
    return

  async def _receive_downscaled(self: ds.DataStreamSession):
    camera_coroutines = [self._camera_tracks[cam].recv() for cam in self._cameras]
    frames = await _aio.gather(*camera_coroutines)
    self._last_recv_time = __import__("time").time()
    return (
      {cam: frame.to_ndarray() for cam, frame in zip(self._cameras, frames, strict=True)},
      {svc: self._message_storage[svc] for svc in self._requested_services},
      {svc: self._message_validity[svc] for svc in self._requested_services},
      {svc: self._message_log_mono_times[svc] for svc in self._requested_services},
    )

  ds.DataStreamSession._receive_async = _receive_downscaled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bodyjim_cameras(params: Params) -> list[str]:
  """Camera list — must match webrtcd's outgoing track (``LivestreamCamera``)."""
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


def _safe_close(env: object | None) -> None:
  if env is not None:
    try:
      env.close()  # type: ignore[union-attr]
    except Exception:
      pass


def _log_failure(stage: str, err: BaseException, *, elapsed_s: float | None = None) -> None:
  elapsed = f" after {elapsed_s:.2f}s" if elapsed_s is not None else ""
  detail = str(err) or "(timeout waiting for decoded video)"
  cloudlog.warning("body_random_walkd: %s failed%s: %s: %s", stage, elapsed, type(err).__name__, detail)
  if _VERBOSE:
    cloudlog.exception("body_random_walkd: %s traceback", stage)


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

def _try_connect(BodyEnv, body_ip: str, cameras: list[str],
                 should_stop: callable) -> object | None:
  """Attempt init → reset → data-channel prime. Returns env on success, None on failure."""
  env = None
  try:
    env = BodyEnv(body_ip, cameras, [], render_mode=None)
  except Exception as e:
    _log_failure("BodyEnv.__init__", e)
    return None

  try:
    t0 = time.monotonic()
    env.reset()
    cloudlog.info("body_random_walkd: reset ok in %.2fs", time.monotonic() - t0)
  except Exception as e:
    _log_failure("BodyEnv.reset", e, elapsed_s=time.monotonic() - t0)
    _safe_close(env)
    return None

  try:
    if not _prime_datachannel(env, should_stop):
      _safe_close(env)
      return None
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

  os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
  _patch_aiortc()
  _mock_pygame()

  params = Params()

  BodyEnv = None
  while BodyEnv is None and not stop:
    try:
      _patch_bodyjim()
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

  cameras = _bodyjim_cameras(params)
  cloudlog.info("body_random_walkd: cameras=%r", cameras)

  while not stop:
    env = None
    while env is None and not stop:
      env = _try_connect(BodyEnv, "127.0.0.1", cameras, _stop_check)
      if env is None and not stop:
        time.sleep(1.0)

    if env is None:
      break

    try:
      action = env.action_space.sample()
      hold_steps = 0
      while not stop:
        if hold_steps <= 0:
          action = env.action_space.sample()
          hold_steps = np.random.randint(20, 80)
        env.step(action)
        hold_steps -= 1
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
