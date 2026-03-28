#!/usr/bin/env python3
"""
Comma body → LiveKit (WebRTC): publish mic + camera, play Gemini agent audio via soundd.

Uses the same device paths as ``openai_realtime_webrtc.py``:

- Uplink: ``BodyMicAudioTrack`` → 16 kHz PCM into LiveKit
- Downlink: remote agent audio → ``BodySpeaker`` → ``bodyRealtimeAudioData`` (soundd)

Video: VisionIPC from ``camerad`` (NV12 → RGBA) at a low FPS for the worker to forward
to Gemini Live.

Requires optional deps: ``uv sync --extra gemini_live``

Environment:

- ``LIVEKIT_URL``, ``LIVEKIT_ROOM_NAME``
- Either ``LIVEKIT_TOKEN`` (JWT) **or** ``LIVEKIT_API_KEY`` + ``LIVEKIT_API_SECRET`` to mint a token

Run (after starting ``gemini_live_worker/bot.py`` in the same room)::

  export LIVEKIT_URL=wss://... LIVEKIT_ROOM_NAME=body-demo LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
  PYTHONPATH=. uv run --extra gemini_live python tools/body/gemini_livekit_body_client.py
"""
from __future__ import annotations

import argparse
import asyncio
import fractions
import logging
import os
import signal
import time
from typing import Any

import numpy as np
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from livekit import api, rtc

from msgq.visionipc import VisionIpcClient, VisionStreamType

from openpilot.system.camerad.snapshot import extract_image
from openpilot.system.webrtc.device.audio import (
  BODY_REALTIME_PCM_SERVICE,
  BodyMicAudioTrack,
  BodySpeaker,
  MIC_SAMPLE_RATE,
)

LOG = logging.getLogger("gemini_livekit_body")

_STREAMS: dict[str, VisionStreamType] = {
  "driver": VisionStreamType.VISION_STREAM_DRIVER,
  "wideRoad": VisionStreamType.VISION_STREAM_WIDE_ROAD,
}


def _mint_token(*, room: str, identity: str, api_key: str, api_secret: str) -> str:
  t = api.AccessToken(api_key, api_secret)
  t.with_identity(identity).with_name(identity).with_grants(
    api.VideoGrants(
      room_join=True,
      room=room,
    ),
  )
  return t.to_jwt()


class _LiveKitAudioPlayback:
  """Presents LiveKit ``AudioStream`` as an object with ``async recv()`` for ``BodySpeaker``."""

  def __init__(self, stream: rtc.AudioStream) -> None:
    self._stream = stream
    self._pts = 0

  async def recv(self) -> AudioFrame:
    try:
      ev = await self._stream.__anext__()
    except StopAsyncIteration as e:
      raise MediaStreamError from e
    lf = ev.frame
    spc = lf.samples_per_channel
    raw = np.frombuffer(lf.data, dtype=np.int16)
    if lf.num_channels == 1:
      mono = raw.reshape(-1)
    else:
      mono = raw.reshape(lf.num_channels, spc).mean(axis=0).astype(np.int16)

    frame = AudioFrame(format="s16", layout="mono", samples=spc)
    frame.planes[0].update(mono.tobytes())
    frame.sample_rate = lf.sample_rate
    frame.pts = self._pts
    frame.time_base = fractions.Fraction(1, lf.sample_rate)
    self._pts += spc
    return frame


async def _mic_to_livekit(mic: BodyMicAudioTrack, source: rtc.AudioSource) -> None:
  while True:
    frame = await mic.recv()
    n = int(frame.samples)
    lk = rtc.AudioFrame.create(MIC_SAMPLE_RATE, 1, n)
    pcm = frame.to_ndarray()
    if pcm.ndim > 1:
      pcm = pcm.reshape(-1)
    np.copyto(np.frombuffer(lk.data, dtype=np.int16), pcm.astype(np.int16))
    await source.capture_frame(lk)


async def _vision_to_livekit(
  client: VisionIpcClient,
  vsource: rtc.VideoSource,
  fps: float,
) -> None:
  period = 1.0 / max(0.25, min(fps, 30.0))
  while True:
    t0 = time.monotonic()
    try:
      buf = await asyncio.to_thread(client.recv)
      rgb = await asyncio.to_thread(extract_image, buf)
    except Exception:
      LOG.exception("vision: frame grab failed")
      await asyncio.sleep(period)
      continue
    h, w = rgb.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    vf = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGBA, rgba.tobytes())
    vsource.capture_frame(vf)
    dt = time.monotonic() - t0
    await asyncio.sleep(max(0.0, period - dt))


async def run(args: argparse.Namespace, stop: asyncio.Event) -> None:
  url = args.url or os.getenv("LIVEKIT_URL")
  room = args.room or os.getenv("LIVEKIT_ROOM_NAME")
  if not url or not room:
    raise SystemExit("LIVEKIT_URL and LIVEKIT_ROOM_NAME (or --url / --room) are required.")

  token = args.token or os.getenv("LIVEKIT_TOKEN")
  if not token:
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    if not key or not secret:
      raise SystemExit("Set LIVEKIT_TOKEN or LIVEKIT_API_KEY + LIVEKIT_API_SECRET.")
    token = _mint_token(room=room, identity=args.identity, api_key=key, api_secret=secret)

  loop = asyncio.get_running_loop()
  room_obj = rtc.Room(loop=loop)

  mic = BodyMicAudioTrack()
  speaker = BodySpeaker(
    pcm_service=BODY_REALTIME_PCM_SERVICE,
    pcm_gain=float(args.playback_gain),
  )
  agent_audio_started = False

  def _on_track_subscribed(
    track: rtc.Track,
    publication: rtc.RemoteTrackPublication,
    _participant: rtc.RemoteParticipant,
  ) -> None:
    nonlocal agent_audio_started
    if track.kind != rtc.TrackKind.KIND_AUDIO:
      return
    if agent_audio_started:
      LOG.warning("ignoring extra remote audio track %s", publication.sid)
      return
    agent_audio_started = True

    async def _attach() -> None:
      stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
      speaker.start_track(_LiveKitAudioPlayback(stream))

    asyncio.create_task(_attach())

  room_obj.on("track_subscribed", _on_track_subscribed)

  LOG.info("connecting to LiveKit room=%s", room)
  await room_obj.connect(
    url,
    token,
    options=rtc.RoomOptions(auto_subscribe=True),
  )
  LOG.info("connected as %s", args.identity)

  audio_source = rtc.AudioSource(MIC_SAMPLE_RATE, 1)
  audio_track = rtc.LocalAudioTrack.create_audio_track("microphone", audio_source)
  audio_opts = rtc.TrackPublishOptions()
  audio_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
  await room_obj.local_participant.publish_track(audio_track, audio_opts)

  tasks: list[asyncio.Task[Any]] = [
    asyncio.create_task(_mic_to_livekit(mic, audio_source), name="mic-uplink"),
  ]

  if not args.no_video:
    st = _STREAMS.get(args.camera)
    if st is None:
      raise SystemExit(f"unknown camera {args.camera!r}")
    vipc = VisionIpcClient("camerad", st, True)
    try:
      vipc.connect(True)
    except Exception as e:
      LOG.warning("VisionIPC connect failed (%s); continuing audio-only", e)
    else:
      try:
        buf0 = vipc.recv()
        rgb0 = extract_image(buf0)
        h0, w0 = rgb0.shape[:2]
      except Exception as e:
        LOG.warning("first vision frame failed (%s); audio-only", e)
      else:
        vsource = rtc.VideoSource(w0, h0)
        vtrack = rtc.LocalVideoTrack.create_video_track("camera", vsource)
        vopts = rtc.TrackPublishOptions(
          source=rtc.TrackSource.SOURCE_CAMERA,
          video_encoding=rtc.VideoEncoding(
            max_framerate=min(30, max(1, int(args.video_fps + 0.5))),
            max_bitrate=2_500_000,
          ),
        )
        await room_obj.local_participant.publish_track(vtrack, vopts)
        vsource.capture_frame(
          rtc.VideoFrame(
            w0,
            h0,
            rtc.VideoBufferType.RGBA,
            _rgb_to_rgba_bytes(rgb0),
          ),
        )
        tasks.append(
          asyncio.create_task(
            _vision_to_livekit(vipc, vsource, float(args.video_fps)),
            name="video-uplink",
          ),
        )

  await stop.wait()

  for t in tasks:
    t.cancel()
    try:
      await t
    except asyncio.CancelledError:
      pass

  await speaker.stop()
  await room_obj.disconnect()
  mic.stop()
  mic.join_poll_thread(timeout=2.0)


def _rgb_to_rgba_bytes(rgb: np.ndarray) -> bytes:
  h, w = rgb.shape[:2]
  rgba = np.empty((h, w, 4), dtype=np.uint8)
  rgba[:, :, :3] = rgb
  rgba[:, :, 3] = 255
  return rgba.tobytes()


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--url", default=None, help="LiveKit URL (or LIVEKIT_URL)")
  p.add_argument("--room", default=None, help="Room name (or LIVEKIT_ROOM_NAME)")
  p.add_argument("--token", default=None, help="JWT (or LIVEKIT_TOKEN); else mint from API key/secret")
  p.add_argument("--identity", default=os.getenv("LIVEKIT_IDENTITY", "comma-body"), help="Participant identity")
  p.add_argument("--no-video", action="store_true", help="Microphone only")
  p.add_argument("--camera", choices=list(_STREAMS), default="driver")
  p.add_argument("--video-fps", type=float, default=2.0, help="VisionIPC publish rate (Gemini-friendly)")
  p.add_argument(
    "--playback-gain",
    type=float,
    default=1.0,
    help="Gain for downlink PCM (same idea as openai_realtime_webrtc)",
  )
  args = p.parse_args()

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  stop = asyncio.Event()

  def request_stop() -> None:
    stop.set()

  try:
    loop.add_signal_handler(signal.SIGINT, request_stop)
    loop.add_signal_handler(signal.SIGTERM, request_stop)
  except NotImplementedError:
    pass

  try:
    loop.run_until_complete(run(args, stop))
  finally:
    for sig in (signal.SIGINT, signal.SIGTERM):
      try:
        loop.remove_signal_handler(sig)
      except (NotImplementedError, ValueError, OSError):
        pass
    loop.close()


if __name__ == "__main__":
  main()
