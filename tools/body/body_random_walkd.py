#!/usr/bin/env python3
"""
Local random walk on the comma device: ``BodyEnv("127.0.0.1", ...)`` → localhost ``webrtcd``
(same idea as https://github.com/commaai/bodyjim/blob/master/examples/random_walk.py ).

Manager starts this when **``BodyRandomWalkEnabled``** (default on) and comma body is active with
**ignition** or full onroad — see ``body_random_walk_should_run`` in
``system/manager/process_config.py`` (``LiveIgnition`` or ``deviceState.started``, plus ``CP.notCar``).
``webrtcd`` / ``bridge`` still use ``comma_body_stack_should_run`` only, so you can disable random
walk and keep teleop.

Params: ``BodyRandomWalkHumanRender`` (default on) = pygame window when display works; off = headless
(``SDL_VIDEODRIVER=dummy``). If pygame display init fails, we fall back to headless for this process.
Requires ``pip install bodyjim`` on device.

**Manual run alongside manager:** the manager may also start ``bodyrandomwalkd``. Stop the daemon first, e.g.
``pkill -f body_random_walkd``, or use ``BLOCK=bodyrandomwalkd`` for a longer manual test
(see ``system/manager/manager.py``). ``webrtcd`` allows **multiple** ``/stream`` clients; if ``reset`` still fails,
check logs and that ``webrtcd`` is running.

After ``reset()``, video can arrive before the WebRTC data channel used for ``testJoystick``; we poll
``step`` until ``send`` works so you do not hit ``Session not started``.

**Diagnosing ``connect/reset failed``** (logs name the stage):

1. **BodyEnv.__init__** — HTTP GET ``schema`` on port 5001 (bodyjim before WebRTC). If this fails:
   ``webrtcd`` not running, wrong port, or firewall.
2. **BodyEnv.reset** — POST ``/stream`` for WebRTC. Other HTTP errors: bad SDP/codec, ``webrtcd`` down,
   or proxy/firewall; **restart webrtcd** if the server is wedged.
3. **webrtc_datachannel_prime** — data channel never opened (timeout). Mismatching **teleoprtc** / ``webrtcd``.

On device: ``curl -sS 'http://127.0.0.1:5001/schema?services=' | head`` and confirm **manager** shows ``webrtcd`` green.

Set env **BODY_RANDOM_WALK_VERBOSE=1** for full tracebacks on each failed attempt (noisy).
"""
from __future__ import annotations

import os
import signal
import sys
import time
from collections.abc import Callable

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

_STEP_HZ = 20

_VERBOSE = os.environ.get("BODY_RANDOM_WALK_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def _log_connect_failure(stage: str, err: BaseException) -> None:
  cloudlog.warning(
    "body_random_walkd: %s failed: %s: %s",
    stage,
    type(err).__name__,
    err,
  )
  if _VERBOSE:
    cloudlog.exception("body_random_walkd: %s traceback", stage)


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

  def should_stop() -> bool:
    return stop

  params = Params()
  want_pygame_window = params.get_bool("BodyRandomWalkHumanRender")
  if not want_pygame_window:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

  pygame = None
  use_human_render = False
  if want_pygame_window:
    try:
      import pygame as _pygame

      pygame = _pygame
      pygame.init()
      if not pygame.display.get_init():
        pygame.display.init()
      use_human_render = True
    except ImportError:
      cloudlog.event("body_random_walkd: pygame missing (required for human render)", error=True)
      while not stop:
        time.sleep(5.0)
      return
    except pygame.error as e:
      cloudlog.warning(
        "body_random_walkd: pygame display failed (%s); headless BodyEnv (set DISPLAY or BodyRandomWalkHumanRender=0)",
        e,
      )
      try:
        pygame.quit()
      except Exception:
        pass
      pygame = None
      os.environ["SDL_VIDEODRIVER"] = "dummy"
      use_human_render = False

  BodyEnv = None
  while BodyEnv is None and not stop:
    try:
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

  render_mode = "human" if use_human_render else None
  body_ip = "127.0.0.1"
  cameras = ["driver"]

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
            f"BodyEnv.__init__ (GET http://{body_ip}:5001/schema — is webrtcd running?)",
            e,
          )
          time.sleep(1.0)
          continue

        try:
          env_try.reset()
        except Exception as e:
          _log_connect_failure("BodyEnv.reset (WebRTC / first frames)", e)
          try:
            env_try.close()
          except Exception:
            pass
          env_try = None
          time.sleep(1.0)
          continue

        try:
          if not _prime_webrtc_datachannel(env_try, should_stop):
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
      rk = Ratekeeper(_STEP_HZ, print_delay_threshold=None)
      while not stop:
        # bodyjim sets _last_observation in step(); render() asserts it is set (unlike raw Gym examples).
        env.step(env.action_space.sample())
        if stop:
          break
        if use_human_render and pygame is not None:
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
              stop = True
              break
          if not stop:
            env.render()
        rk.keep_time()
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
