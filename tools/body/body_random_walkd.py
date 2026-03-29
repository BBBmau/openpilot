#!/usr/bin/env python3
"""
Local random walk on the comma device: ``BodyEnv("127.0.0.1", ...)`` → localhost ``webrtcd``
(same idea as https://github.com/commaai/bodyjim/blob/master/examples/random_walk.py ).

Manager starts this when comma body is active with **ignition** or full onroad — same gate as
``webrtcd`` / ``bridge``: ``comma_body_stack_should_run`` in ``system/manager/process_config.py``
(``LiveIgnition`` or ``deviceState.started``, plus ``CP.notCar``).

Params: ``BodyRandomWalkHumanRender`` (default on) = pygame window; off = headless
(``SDL_VIDEODRIVER=dummy``). Requires ``pip install bodyjim`` on device.

**Manual run alongside manager:** the manager may also start ``bodyrandomwalkd``. Stop the daemon first, e.g.
``pkill -f body_random_walkd``, or use ``BLOCK=bodyrandomwalkd`` for a longer manual test
(see ``system/manager/manager.py``). ``webrtcd`` allows **multiple** ``/stream`` clients; if ``reset`` still fails,
check logs and that ``webrtcd`` is running.

After ``reset()``, video can arrive before the WebRTC data channel used for ``testJoystick``; we poll
``step`` until ``send`` works so you do not hit ``Session not started``.

**Diagnosing ``connect/reset failed``** (logs name the stage):

1. **BodyEnv.__init__** — HTTP ``GET http://127.0.0.1:5001/schema?...`` (bodyjim fetches observation schema before WebRTC). If this fails: ``webrtcd`` not running, wrong port, or firewall.
2. **BodyEnv.reset** — POST ``/stream`` for WebRTC. Other HTTP errors usually mean bad SDP/codec, ``webrtcd`` down, or a proxy/firewall issue; **restart webrtcd** if the server is wedged.
3. **webrtc_datachannel_prime** — data channel never opened (timeout). Mismatching **teleoprtc** / ``webrtcd``.

If stderr shows ``Error parsing message 'logMonoTime'``, **bodyjim** expects every data-channel JSON
object to include ``logMonoTime``, ``valid``, and ``data`` (same envelope as bridged cereal).
Control messages from older ``webrtcd`` (e.g. ``activeCamera``, ``clockSync`` pong) omitted those
fields and triggered a ``KeyError``. Use a current ``webrtcd`` that sends the full envelope.

On device: ``curl -sS 'http://127.0.0.1:5001/schema?services=' | head`` and confirm **manager** shows ``webrtcd`` green.

Set env **BODY_RANDOM_WALK_VERBOSE=1** for full tracebacks on each failed attempt (noisy).

**``BodyEnv.reset`` → ``TimeoutError``** (~``RECEIVE_TIMEOUT``): bodyjim’s first ``receive()``
blocks until **decoded frames exist for every requested camera**. ``webrtcd`` (see
``system/webrtc/webrtcd.py``) publishes **one** outgoing video track named from the
``LivestreamCamera`` param (``driver`` or ``wideRoad``), not “whatever the client asked for”.
If this script used ``["driver"]`` while the param was ``wideRoad``, bodyjim would wait on the
wrong track and hit ``Future.result(timeout)`` with an empty message. The camera list is therefore
aligned with ``LivestreamCamera`` by default; override with ``BODY_RANDOM_WALK_CAMERAS=wideRoad``
(or ``driver``) if needed.

Also raise ``BODY_RANDOM_WALK_RECEIVE_TIMEOUT`` (default **15** s) if the encoder is slow to
produce a keyframe. Optional ``BODY_RANDOM_WALK_CONNECT_TIMEOUT`` overrides the 3s WebRTC connect wait.

Set **``BODY_RANDOM_WALK_DEBUG=1``** for extra logs: configured timeouts, monotonic elapsed time
around ``reset()``, a snapshot of bodyjim’s ``DataStreamSession`` after a failed ``reset``, and a
short **burst scan** of ``livestream*EncodeData`` (how many messages look like WebRTC sync points vs P-frames).
Set **``BODY_RANDOM_WALK_PIPELINE_BURST=1``** alone to log that burst scan without full debug noise.

On **TimeoutError** during ``reset``, the script probes the matching ``livestream*EncodeData`` socket
for a few seconds (override with ``BODY_RANDOM_WALK_PIPELINE_PROBE_S``). No packets usually means
``stream_encoderd`` or ``camerad`` is not running — historically ``stream_encoderd`` only ran when
``deviceState.started`` (onroad) while ``webrtcd`` could run on ``LiveIgnition`` alone; openpilot
aligns those gates in ``system/manager/process_config.py`` (restart manager after updating).
"""
from __future__ import annotations

import os
import signal
import sys
import time
from collections.abc import Callable

import numpy as np

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

_VERBOSE = os.environ.get("BODY_RANDOM_WALK_VERBOSE", "").strip().lower() in ("1", "true", "yes")
_DEBUG = os.environ.get("BODY_RANDOM_WALK_DEBUG", "").strip().lower() in ("1", "true", "yes")
_PIPELINE_BURST = os.environ.get("BODY_RANDOM_WALK_PIPELINE_BURST", "").strip().lower() in ("1", "true", "yes")


def _bodyjim_cameras(params: Params) -> list[str]:
  """Single camera name bodyjim should receive — must match webrtcd's outgoing track (``LivestreamCamera``)."""
  raw = os.environ.get("BODY_RANDOM_WALK_CAMERAS", "").strip()
  if raw:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if parts:
      if len(parts) > 1:
        cloudlog.warning(
          "body_random_walkd: webrtcd publishes one video track; using first camera from BODY_RANDOM_WALK_CAMERAS=%r",
          raw,
        )
      cam = parts[0]
      if cam in ("driver", "wideRoad"):
        return [cam]
      cloudlog.warning(
        "body_random_walkd: invalid BODY_RANDOM_WALK_CAMERAS=%r (want driver or wideRoad); using LivestreamCamera param",
        cam,
      )
  active = params.get("LivestreamCamera") or "driver"
  if active not in ("driver", "wideRoad"):
    active = "driver"
  return [active]


def _patch_bodyjim_timeouts() -> None:
  """bodyjim uses 2s receive / 3s connect; first video frame on comma body often needs more."""
  import bodyjim.data_stream as ds  # noqa: PLC0415

  recv = os.environ.get("BODY_RANDOM_WALK_RECEIVE_TIMEOUT", "15").strip()
  if recv:
    try:
      v = float(recv)
      if v > 0:
        ds.RECEIVE_TIMEOUT_SECONDS = v
    except ValueError:
      cloudlog.warning("body_random_walkd: invalid BODY_RANDOM_WALK_RECEIVE_TIMEOUT=%r", recv)
  conn = os.environ.get("BODY_RANDOM_WALK_CONNECT_TIMEOUT", "").strip()
  if conn:
    try:
      v = float(conn)
      if v > 0:
        ds.CONNECT_TIMEOUT_SECONDS = v
    except ValueError:
      cloudlog.warning("body_random_walkd: invalid BODY_RANDOM_WALK_CONNECT_TIMEOUT=%r", conn)
  cloudlog.info(
    "body_random_walkd: bodyjim timeouts receive=%.1fs connect=%.1fs",
    ds.RECEIVE_TIMEOUT_SECONDS,
    ds.CONNECT_TIMEOUT_SECONDS,
  )


def _log_connect_failure(stage: str, err: BaseException, *, elapsed_s: float | None = None) -> None:
  elapsed = f" after {elapsed_s:.2f}s" if elapsed_s is not None else ""
  detail = str(err) if str(err) else (
    "(empty message — typical of Future.result(timeout) while waiting for decoded video from webrtcd)"
  )
  cloudlog.warning(
    "body_random_walkd: %s failed%s: %s: %s",
    stage,
    elapsed,
    type(err).__name__,
    detail,
  )
  if _VERBOSE:
    cloudlog.exception("body_random_walkd: %s traceback", stage)


def _log_bodyjim_stream_debug(env: object, label: str) -> None:
  """Best-effort snapshot of bodyjim DataStreamSession (private attrs; API may differ by version)."""
  if not _DEBUG:
    return
  try:
    import bodyjim.data_stream as bj_ds  # noqa: PLC0415

    cloudlog.info(
      "body_random_walkd: %s bodyjim RECEIVE_TIMEOUT_SECONDS=%.3f CONNECT_TIMEOUT_SECONDS=%.3f",
      label,
      bj_ds.RECEIVE_TIMEOUT_SECONDS,
      bj_ds.CONNECT_TIMEOUT_SECONDS,
    )
  except Exception as e:
    cloudlog.warning("body_random_walkd: %s could not read bodyjim.data_stream timeouts: %s", label, e)

  ds_sess = getattr(env, "_data_stream", None)
  cloudlog.info("body_random_walkd: %s env._data_stream is None=%s", label, ds_sess is None)
  if ds_sess is None:
    return

  th = getattr(ds_sess, "_runner_thread", None)
  th_alive = th.is_alive() if th is not None else None
  ch = getattr(ds_sess, "_channel", None)
  tracks = getattr(ds_sess, "_camera_tracks", None)
  track_keys = list(tracks.keys()) if isinstance(tracks, dict) else None
  cams = getattr(ds_sess, "_cameras", None)
  cloudlog.info(
    "body_random_walkd: %s stream runner_alive=%s data_channel_open=%s cameras=%r track_keys=%s",
    label,
    th_alive,
    ch is not None,
    cams,
    track_keys,
  )


def _log_livestream_sync_burst_scan(sock: object, service: str, *, max_msgs: int = 40, budget_s: float = 1.0) -> None:
  """Read a few more encoded frames on the same subscriber — sync_ok counts match LiveStreamVideoStreamTrack logic."""
  import cereal.messaging as messaging  # noqa: PLC0415

  from openpilot.system.webrtc.device.video import livestream_encode_data_diag  # noqa: PLC0415

  deadline = time.monotonic() + budget_s
  n_ok = n_tot = 0
  while time.monotonic() < deadline and n_tot < max_msgs:
    msg = messaging.recv_one_or_none(sock)
    if msg is None:
      time.sleep(0.005)
      continue
    n_tot += 1
    if livestream_encode_data_diag(msg)["sync_ok"]:
      n_ok += 1
  cloudlog.info(
    "body_random_walkd: pipeline: burst on %s: read %d msgs in ≤%.2fs (%d track_sync_ok) — "
    "if sync_ok>0 but reset fails, video likely dies after msgq (WebRTC / bodyjim decode).",
    service,
    n_tot,
    budget_s,
    n_ok,
  )


def _log_livestream_pipeline_diagnosis(cameras: list[str], params: Params) -> None:
  """If msgq has no livestream encode data, WebRTC cannot send video; log gates and probe the topic."""
  import cereal.messaging as messaging  # noqa: PLC0415

  cam = cameras[0] if cameras else "driver"
  try:
    from openpilot.system.webrtc.device.video import (  # noqa: PLC0415
      LiveStreamVideoStreamTrack,
      livestream_encode_data_diag,
    )

    service = LiveStreamVideoStreamTrack.camera_to_sock_mapping[cam]
  except Exception as e:
    cloudlog.warning("body_random_walkd: pipeline diagnosis: camera %r: %s", cam, e)
    return

  probe_s = 2.0
  raw = os.environ.get("BODY_RANDOM_WALK_PIPELINE_PROBE_S", "").strip()
  if raw:
    try:
      probe_s = max(0.2, min(30.0, float(raw)))
    except ValueError:
      cloudlog.warning("body_random_walkd: invalid BODY_RANDOM_WALK_PIPELINE_PROBE_S=%r", raw)

  sm = messaging.SubMaster(["deviceState", "carParams"])
  for _ in range(100):
    sm.update(100)
    if sm.seen["deviceState"] and sm.seen["carParams"]:
      break

  started = bool(sm["deviceState"].started) if sm.seen["deviceState"] else None
  not_car_s = (
    str(bool(sm["carParams"].notCar)) if sm.seen["carParams"] else "(no carParams message yet)"
  )
  cloudlog.warning(
    "body_random_walkd: pipeline snapshot: deviceState.started=%s carParams.notCar=%s "
    "LiveIgnition=%s IsDriverViewEnabled=%s LivestreamCamera=%r bodyjim_cameras=%r",
    started,
    not_car_s,
    params.get_bool("LiveIgnition"),
    params.get_bool("IsDriverViewEnabled"),
    params.get("LivestreamCamera") or "driver",
    cameras,
  )

  sock = messaging.sub_sock(service, conflate=False)
  deadline = time.monotonic() + probe_s
  got = 0
  last_mono = None
  while time.monotonic() < deadline:
    msg = messaging.recv_one_or_none(sock)
    if msg is not None:
      got += 1
      last_mono = msg.logMonoTime
      break
    time.sleep(0.05)

  if got:
    diag = livestream_encode_data_diag(msg)
    cloudlog.warning(
      "body_random_walkd: pipeline: msgq ok — %s within %.1fs logMonoTime=%s",
      service,
      probe_s,
      last_mono,
    )
    cloudlog.warning(
      "body_random_walkd: pipeline: first frame which=%s encodeType=%s flags=0x%x keyframe_bit=%s "
      "header=%dB data=%dB slice_has_start_code=%s sync_ok=%s has_codec_header=%s nal_types=%s — "
      "sync_ok means IDR (NAL 5) or keyframe flag (what webrtcd emits first). P-frames + SPS/PPS only "
      "do not decode on a fresh peer. If header=0, fix stream_encoderd / repeat_codec_header.",
      diag["which"],
      diag["encode_type"],
      diag["flags"],
      diag["keyframe_bit"],
      diag["header_bytes"],
      diag["data_bytes"],
      diag["slice_has_start_code"],
      diag["sync_ok"],
      diag["has_codec_header"],
      diag.get("nal_types_sample", ""),
    )
    if _DEBUG or _PIPELINE_BURST:
      _log_livestream_sync_burst_scan(sock, service)
  else:
    cloudlog.warning(
      "body_random_walkd: pipeline: no messages on %s within %.1fs — need camerad + stream_encoderd. "
      "For comma body with ignition but not onroad, manager must run stream_encoderd and camerad "
      "under the same conditions as webrtcd (see comma_body_stack_should_run / driverview in process_config).",
      service,
      probe_s,
    )


def _prime_webrtc_datachannel(env, should_stop: Callable[[], bool], *, timeout_s: float = 20.0) -> bool:
  """``DataStreamSession.send`` needs ``RTCDataChannel``; video can work before the channel opens.

  Returns False if ``should_stop()`` before the channel is ready.
  """
  t0 = time.monotonic()
  zero = np.zeros(2, dtype=np.float32)
  while time.monotonic() - t0 < timeout_s:
    if should_stop():
      return False
    try:
      env.step(zero)
      return True
    except AssertionError as e:
      msg = str(e.args[0]) if e.args else ""
      if "Session not started" not in msg:
        raise
      time.sleep(0.05)
  raise TimeoutError(
    "bodyjim data channel not ready in time — check webrtcd is running and teleoprtc matches bodyjim"
  )


def main() -> None:
  stop = False

  def _stop(*_args: object) -> None:
    nonlocal stop
    stop = True

  signal.signal(signal.SIGTERM, _stop)
  signal.signal(signal.SIGINT, _stop)

  params = Params()
  human = params.get_bool("BodyRandomWalkHumanRender")
  if not human:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

  BodyEnv = None
  while BodyEnv is None and not stop:
    try:
      _patch_bodyjim_timeouts()
      from bodyjim import BodyEnv as _BodyEnv  # noqa: PLC0415

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
      import pygame as _pygame  # noqa: PLC0415

      pygame = _pygame
    except ImportError:
      cloudlog.event("body_random_walkd: pygame missing (required for human render)", error=True)
      while not stop:
        time.sleep(5.0)
      return

  render_mode = "human" if human else None
  body_ip = "127.0.0.1"
  cameras = _bodyjim_cameras(params)
  cloudlog.info(
    "body_random_walkd: bodyjim cameras=%r (must match webrtcd track from LivestreamCamera param unless BODY_RANDOM_WALK_CAMERAS is set)",
    cameras,
  )

  while not stop:
    env = None
    while env is None and not stop:
      env_try = None
      try:
        cloudlog.info("body_random_walkd: BodyEnv(%s) render_mode=%s", body_ip, render_mode)
        try:
          env_try = BodyEnv(body_ip, cameras, [], render_mode=render_mode)
        except Exception as e:
          _log_connect_failure(
            "BodyEnv.__init__ (GET http://%s:5001/schema — is webrtcd running?)" % body_ip,
            e,
          )
          time.sleep(1.0)
          continue

        t_reset: float | None = None
        try:
          import bodyjim.data_stream as bj_ds  # noqa: PLC0415

          cloudlog.info(
            "body_random_walkd: BodyEnv.reset() starting cameras=%r "
            "(bodyjim receive_timeout=%.1fs connect_timeout=%.1fs)",
            cameras,
            bj_ds.RECEIVE_TIMEOUT_SECONDS,
            bj_ds.CONNECT_TIMEOUT_SECONDS,
          )
          t_reset = time.monotonic()
          env_try.reset()
          cloudlog.info(
            "body_random_walkd: BodyEnv.reset() ok in %.2fs",
            time.monotonic() - t_reset,
          )
        except Exception as e:
          elapsed = (time.monotonic() - t_reset) if t_reset is not None else None
          _log_connect_failure(
            "BodyEnv.reset (WebRTC / first frames — wait_for_connection + first receive() per camera)",
            e,
            elapsed_s=elapsed,
          )
          if isinstance(e, TimeoutError):
            _log_livestream_pipeline_diagnosis(cameras, params)
          _log_bodyjim_stream_debug(env_try, "after failed reset")
          try:
            env_try.close()
          except Exception:
            pass
          env_try = None
          time.sleep(1.0)
          continue

        try:
          if not _prime_webrtc_datachannel(env_try, lambda: stop):
            try:
              env_try.close()
            except Exception:
              pass
            return
        except TimeoutError as e:
          _log_connect_failure("webrtc_datachannel_prime (RTCDataChannel for testJoystick)", e)
          try:
            env_try.close()
          except Exception:
            pass
          env_try = None
          time.sleep(1.0)
          continue

        try:
          if human and pygame is not None:
            pygame.init()
        except Exception as e:
          _log_connect_failure("pygame.init", e)
          try:
            env_try.close()
          except Exception:
            pass
          env_try = None
          time.sleep(1.0)
          continue

        env = env_try
        env_try = None
      except Exception as e:
        _log_connect_failure("connect (unexpected)", e)
        if env_try is not None:
          try:
            env_try.close()
          except Exception:
            pass
        time.sleep(1.0)

    if env is None:
      break

    try:
      while not stop:
        # bodyjim sets _last_observation in step(); render() asserts it is set (unlike raw Gym examples).
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
      try:
        env.close()
      except Exception:
        cloudlog.exception("body_random_walkd: env.close()")


if __name__ == "__main__":
  try:
    main()
  except Exception:
    cloudlog.exception("body_random_walkd: fatal")
    sys.exit(1)
