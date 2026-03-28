#!/usr/bin/env python3
"""
On-device Gymnasium-style random walk for comma body using bodyjim → localhost webrtcd.

Runs for a notCar platform while manager sees panda **ignition** or full **onroad**
(``deviceState.started``). Ignition alone is enough even when startup conditions block ``started``.

Requires the ``bodyjim`` package on the device::

  pip install bodyjim

(or ``pip install -e '.[bodyjim]'`` from an openpilot tree that lists the optional extra).

Params (see ``common/params_keys.h``); both default **on** so ignition/onroad starts human mode with no SSH:

- ``BodyRandomWalkEnabled`` — set false to disable the daemon entirely.
- ``BodyRandomWalkHumanRender`` — if true (default), ``render_mode=\"human\"`` (pygame on device); if false,
  headless (``SDL_VIDEODRIVER=dummy``).

While this daemon is connected, it holds the single ``webrtcd`` WebRTC session; remote teleop /
laptop bodyjim cannot connect until you go offroad or disable the feature.
"""
from __future__ import annotations

import os
import signal
import sys
import time

from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog


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
      from bodyjim import BodyEnv as _BodyEnv  # noqa: PLC0415 — optional dependency

      BodyEnv = _BodyEnv
    except ImportError:
      cloudlog.event("body_random_walkd: bodyjim not installed; pip install bodyjim", error=True)
      for _ in range(60):
        if stop:
          return
        time.sleep(1.0)

  if BodyEnv is None:
    return

  if human:
    try:
      import pygame  # noqa: PLC0415
    except ImportError:
      cloudlog.event("body_random_walkd: pygame missing (required for human render)", error=True)
      while not stop:
        time.sleep(5.0)
      return

  render_mode = "human" if human else None
  rk = Ratekeeper(20, print_delay_threshold=None)

  while not stop:
    env = None
    while env is None and not stop:
      try:
        cloudlog.info("body_random_walkd: connecting BodyEnv(127.0.0.1) render_mode=%s", render_mode)
        env = BodyEnv("127.0.0.1", ["driver"], [], render_mode=render_mode)
        env.reset()
      except Exception:
        cloudlog.warning("body_random_walkd: connect/reset failed; retry in 1s")
        time.sleep(1.0)

    if env is None:
      break

    try:
      while not stop:
        if human:
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
              stop = True
              break
          env.render()
        if stop:
          break
        action = env.action_space.sample()
        env.step(action)
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
