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

import av.logging
import numpy as np

av.logging.set_level(av.logging.PANIC)

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

    def _patched_decoder_worker(loop, input_q, output_q):
      """Same as original but skips RTX codec frames and downscales in-thread."""
      import asyncio as _asyncio
      import os as _os
      import sys as _sys
      from aiortc.codecs import get_decoder as _get_decoder, is_rtx as _is_rtx

      _scale = max(1, int(_os.environ.get("BODY_RANDOM_WALK_FRAME_SCALE", "4")))
      codec_name = None
      decoder = None
      n_in = n_rtx = n_out = n_err = 0
      _log_at = {1, 2, 3, 5, 10, 25, 50, 100, 500, 1000}
      while True:
        task = input_q.get()
        if task is None:
          _asyncio.run_coroutine_threadsafe(output_q.put(None), loop)
          break
        codec, encoded_frame = task
        n_in += 1
        if _is_rtx(codec):
          n_rtx += 1
          if n_rtx <= 3:
            print(f"body_random_walkd: decoder: skip RTX #{n_rtx}", file=_sys.stderr, flush=True)
          continue
        if codec.name != codec_name:
          print(f"body_random_walkd: decoder: codec {codec_name!r}->{codec.name!r}", file=_sys.stderr, flush=True)
          decoder = _get_decoder(codec)
          codec_name = codec.name
        try:
          for frame in decoder.decode(encoded_frame):
            if _scale > 1:
              frame = frame.reformat(width=frame.width // _scale, height=frame.height // _scale, format="rgb24")
            n_out += 1
            _asyncio.run_coroutine_threadsafe(output_q.put(frame), loop)
            if n_out <= 3:
              print(f"body_random_walkd: decoder: output #{n_out} {frame.width}x{frame.height}", file=_sys.stderr, flush=True)
        except Exception as exc:
          n_err += 1
          if n_err <= 5:
            print(f"body_random_walkd: decoder: error #{n_err}: {exc}", file=_sys.stderr, flush=True)
        if n_in in _log_at:
          rss = _rss_mb()
          print(f"body_random_walkd: decoder: in={n_in} out={n_out} rtx={n_rtx} err={n_err} rss={rss:.0f}MB", file=_sys.stderr, flush=True)
      print(f"body_random_walkd: decoder: exit in={n_in} out={n_out} rtx={n_rtx} err={n_err} rss={_rss_mb():.0f}MB", file=_sys.stderr, flush=True)
      if decoder is not None:
        del decoder

    import inspect

    try:
      src = inspect.getsource(mod.RTCRtpReceiver._handle_rtp_packet)
    except (OSError, TypeError):
      src = ""
    if "codec = self.__codecs[apt]" not in src and "codec = self._RTCRtpReceiver__codecs[apt]" not in src:
      mod.decoder_worker = _patched_decoder_worker
      print("body_random_walkd: aiortc RTX monkeypatch applied", file=sys.stderr, flush=True)
    else:
      print("body_random_walkd: aiortc RTX fix already upstream — no patch needed", file=sys.stderr, flush=True)
  except Exception as e:
    print(f"body_random_walkd: aiortc RTX monkeypatch FAILED: {e}", file=sys.stderr, flush=True)
    cloudlog.warning("body_random_walkd: aiortc RTX monkeypatch failed: %s", e)


def _force_h264_codec() -> None:
  """Remove VP8 from aiortc's CODECS registry so SDP always negotiates H264.

  webrtcd's LiveStreamVideoStreamTrack sends pre-encoded H264 packets. If VP8 is
  negotiated instead (VP8 is listed first in aiortc's default codec list), the
  receiver gets garbage and decodes nothing. aiortc's SDP builder reads the
  CODECS["video"] list directly — patching getCapabilities is insufficient.
  """
  try:
    from aiortc.codecs import CODECS

    before = [c.mimeType for c in CODECS["video"]]
    CODECS["video"] = [c for c in CODECS["video"] if c.mimeType != "video/VP8"]
    after = [c.mimeType for c in CODECS["video"]]
    print(f"body_random_walkd: video codecs {before} -> {after}", file=sys.stderr, flush=True)
  except Exception as e:
    print(f"body_random_walkd: _force_h264_codec FAILED: {e}", file=sys.stderr, flush=True)


def _mock_pygame() -> None:
  """Install a lightweight pygame stub so bodyjim imports without loading the C library.

  bodyjim's env.py does ``import pygame`` at module scope.  The actual pygame
  functions are only called inside ``render()`` and ``close()`` — both gated
  behind ``self._window is not None`` which stays None when render_mode is None.
  A stub module satisfies the import for ~0 MB instead of ~25 MB.
  """
  import types

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
  print("body_random_walkd: pygame stubbed (headless)", file=sys.stderr, flush=True)


def _rss_mb() -> float:
  """Current RSS in MB (Linux /proc)."""
  try:
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) / 1024.0
  except Exception:
    pass
  return -1.0


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


_FRAME_SCALE = max(1, int(os.environ.get("BODY_RANDOM_WALK_FRAME_SCALE", "4")))


def _patch_bodyjim_frame_downscale() -> None:
  """Replace bodyjim's ``_receive_async`` with a version that skips the full-size
  ``to_ndarray(format='rgb24')`` allocation.

  The decoder worker already outputs downscaled YUV frames (controlled by
  ``BODY_RANDOM_WALK_FRAME_SCALE``).  This patch just does the lightweight
  YUV→RGB conversion on the already-small frame, keeping the event loop
  unblocked and avoiding ~7 MB temporaries.
  """
  if _FRAME_SCALE <= 1:
    return

  import asyncio as _aio
  import bodyjim.data_stream as ds

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
  print(f"body_random_walkd: frame downscale {_FRAME_SCALE}x (in decoder thread)", file=sys.stderr, flush=True)


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


def _log_receiver_state(env: object | None) -> None:
  """Best-effort check of bodyjim's WebRTC receiver after a failed reset."""
  if env is None:
    return
  try:
    import threading
    ds = getattr(env, "_data_stream", None)
    if ds is None:
      cloudlog.warning("body_random_walkd: receiver: env._data_stream is None")
      return
    stream = getattr(ds, "_stream", None)
    pc = getattr(stream, "peer_connection", None) if stream else None
    pc_state = pc.connectionState if pc else "no_pc"
    ice_state = pc.iceConnectionState if pc else "no_pc"

    decoder_threads = [t for t in threading.enumerate() if "decoder" in t.name.lower()]
    alive_decoders = [t.name for t in decoder_threads if t.is_alive()]
    dead_decoders = [t.name for t in decoder_threads if not t.is_alive()]

    cloudlog.warning(
      "body_random_walkd: receiver: pc=%s ice=%s decoder_threads_alive=%s dead=%s",
      pc_state, ice_state, alive_decoders or "none", dead_decoders or "none",
    )
  except Exception as e:
    cloudlog.warning("body_random_walkd: receiver probe failed: %s", e)


def _log_pipeline_probe(cameras: list[str], env: object | None = None) -> None:
  """Quick probe of the livestream encode topic + receiver state after a reset timeout."""
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
      break
    time.sleep(0.05)
  else:
    cloudlog.warning("body_random_walkd: probe %s — no messages (camerad / stream_encoderd not running?)", service)

  _log_receiver_state(env)


def _prime_datachannel(env: object, should_stop: callable, *, timeout_s: float = 20.0) -> bool:
  """Poll ``env.step`` until the WebRTC data channel opens. Returns False if stopped."""
  t0 = time.monotonic()
  zero = np.zeros(2, dtype=np.float32)
  while time.monotonic() - t0 < timeout_s:
    if should_stop():
      return False
    try:
      env.step(zero)  # type: ignore[union-attr]
      _log_datachannel_state(env)
      return True
    except AssertionError as e:
      if "Session not started" not in (str(e.args[0]) if e.args else ""):
        raise
      time.sleep(0.05)
  raise TimeoutError("data channel not ready — is webrtcd running?")


def _log_datachannel_state(env: object) -> None:
  """Check data channel + cereal to verify the full control path works."""
  import cereal.messaging as cmsg

  try:
    ds = getattr(env, "_data_stream", None)
    ch = getattr(ds, "_channel", None) if ds else None
    state = ch.readyState if ch else "no_channel"
    print(f"body_random_walkd: data channel: {state}", file=sys.stderr, flush=True)
  except Exception as e:
    print(f"body_random_walkd: data channel check failed: {e}", file=sys.stderr, flush=True)

  try:
    sock = cmsg.sub_sock("testJoystick", conflate=True)
    time.sleep(0.1)
    env.step(np.array([0.5, 0.3], dtype=np.float32))  # type: ignore[union-attr]
    time.sleep(0.5)
    msg = cmsg.recv_one_or_none(sock)
    if msg is not None:
      axes = list(msg.testJoystick.axes)
      print(f"body_random_walkd: cereal testJoystick OK: axes={axes}", file=sys.stderr, flush=True)
    else:
      print("body_random_walkd: cereal testJoystick: NO messages (webrtcd bridge not forwarding?)", file=sys.stderr, flush=True)
  except Exception as e:
    print(f"body_random_walkd: cereal testJoystick check failed: {e}", file=sys.stderr, flush=True)

  try:
    sm = cmsg.SubMaster(["selfdriveState", "carControl", "controlsState"])
    sm.update(1000)
    enabled = sm["selfdriveState"].enabled
    active = sm["selfdriveState"].active
    long_active = sm["carControl"].carControl.longActive
    lat_active = sm["carControl"].carControl.latActive
    is_joystick = sm["controlsState"].controlsState.lateralControlState.which() == "debugState"
    ctrl = f"enabled={enabled} active={active} longActive={long_active} latActive={lat_active} joystick={is_joystick}"
    print(f"body_random_walkd: control state: {ctrl}", file=sys.stderr, flush=True)
  except Exception as e:
    print(f"body_random_walkd: control state check failed: {e}", file=sys.stderr, flush=True)


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
      _log_pipeline_probe(cameras, env)
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
  print(f"body_random_walkd: startup rss={_rss_mb():.0f}MB", file=sys.stderr, flush=True)

  _patch_aiortc_rtx_bug()
  _force_h264_codec()
  _mock_pygame()

  BodyEnv = None
  while BodyEnv is None and not stop:
    try:
      _patch_bodyjim_timeouts()
      _patch_bodyjim_frame_downscale()
      from bodyjim import BodyEnv as _BodyEnv

      BodyEnv = _BodyEnv
      print(f"body_random_walkd: bodyjim loaded rss={_rss_mb():.0f}MB", file=sys.stderr, flush=True)
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
      env = _try_connect(BodyEnv, "127.0.0.1", cameras, None, _stop_check)
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
