#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import os
import ssl
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

# aiortc and its dependencies have lots of internal warnings :(
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # TODO: remove this when google-crc32c publish a python3.12 wheel

import capnp
from aiohttp import web
if TYPE_CHECKING:
  from aiortc.rtcdatachannel import RTCDataChannel

from openpilot.system.webrtc.schema import generate_field
from cereal import messaging, log
from openpilot.common.params import Params


class CerealOutgoingMessageProxy:
  def __init__(self, sm: messaging.SubMaster):
    self.sm = sm
    self.channels: list[RTCDataChannel] = []

  def add_channel(self, channel: 'RTCDataChannel'):
    self.channels.append(channel)

  def to_json(self, msg_content: Any):
    if isinstance(msg_content, capnp._DynamicStructReader):
      msg_dict = msg_content.to_dict()
    elif isinstance(msg_content, capnp._DynamicListReader):
      msg_dict = [self.to_json(msg) for msg in msg_content]
    elif isinstance(msg_content, bytes):
      msg_dict = msg_content.decode()
    else:
      msg_dict = msg_content

    return msg_dict

  def update(self):
    # this is blocking in async context...
    self.sm.update(0)
    for service, updated in self.sm.updated.items():
      if not updated:
        continue
      msg_dict = self.to_json(self.sm[service])
      mono_time, valid = self.sm.logMonoTime[service], self.sm.valid[service]
      outgoing_msg = {"type": service, "logMonoTime": mono_time, "valid": valid, "data": msg_dict}
      encoded_msg = json.dumps(outgoing_msg).encode()
      for channel in self.channels:
        channel.send(encoded_msg)


class CerealIncomingMessageProxy:
  def __init__(self, pm: messaging.PubMaster):
    self.pm = pm

  def send(self, message: bytes):
    msg_json = json.loads(message)
    msg_type, msg_data = msg_json["type"], msg_json["data"]
    size = None
    if not isinstance(msg_data, dict):
      size = len(msg_data)

    msg = messaging.new_message(msg_type, size=size)
    setattr(msg, msg_type, msg_data)
    self.pm.send(msg_type, msg)


class CerealProxyRunner:
  def __init__(self, proxy: CerealOutgoingMessageProxy):
    self.proxy = proxy
    self.is_running = False
    self.task = None
    self.logger = logging.getLogger("webrtcd")

  def start(self):
    assert self.task is None
    self.task = asyncio.create_task(self.run())

  def stop(self):
    if self.task is None or self.task.done():
      return
    self.task.cancel()
    self.task = None

  async def run(self):
    from aiortc.exceptions import InvalidStateError

    while True:
      try:
        self.proxy.update()
      except InvalidStateError:
        self.logger.warning("Cereal outgoing proxy invalid state (connection closed)")
        break
      except Exception:
        self.logger.exception("Cereal outgoing proxy failure")
      await asyncio.sleep(0.01)


class DynamicPubMaster(messaging.PubMaster):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.lock = asyncio.Lock()

  async def add_services_if_needed(self, services):
    async with self.lock:
      for service in services:
        if service not in self.sock:
          self.sock[service] = messaging.pub_sock(service)


def validate_body_sound_name(sound_name: Any) -> str:
  from openpilot.system.webrtc.device.audio import BODY_SOUND_NAMES

  if sound_name not in BODY_SOUND_NAMES:
    raise ValueError(f"unsupported body sound: {sound_name}")
  return sound_name


class StreamSession:
  shared_pub_master = DynamicPubMaster([])

  def __init__(self, sdp: str, cameras: list[str], incoming_services: list[str], outgoing_services: list[str],
               audio_output=None, debug_mode: bool = False, app: dict | None = None):
    from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack
    from openpilot.system.webrtc.device.audio import BodyMicAudioTrack, BodySpeaker
    from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
    from teleoprtc import WebRTCAnswerBuilder
    from teleoprtc.info import parse_info_from_offer

    self.logger = logging.getLogger("webrtcd")
    config = parse_info_from_offer(sdp)
    builder = WebRTCAnswerBuilder(sdp)

    # Use the camera the encoder is currently active on, so reconnects don't get a blank stream
    active_camera = Params().get("LivestreamCamera") or "driver"
    if active_camera not in ("driver", "wideRoad"):
      active_camera = "driver"
    self.video_track = LiveStreamVideoStreamTrack(active_camera) if not debug_mode else VideoStreamTrack()
    builder.add_video_stream(active_camera, self.video_track)

    self.outgoing_audio_track: BodyMicAudioTrack | None = None
    if config.expected_audio_track:
      try:
        if debug_mode:
          builder.add_audio_stream(AudioStreamTrack())
        else:
          self.outgoing_audio_track = BodyMicAudioTrack()
          builder.add_audio_stream(self.outgoing_audio_track)
      except Exception:
        self.logger.exception("Failed to initialize body microphone track")

    self.audio_output: BodySpeaker | None = audio_output if (config.incoming_audio_track or config.incoming_datachannel) else None
    if self.audio_output is None and (config.incoming_audio_track or config.incoming_datachannel):
      try:
        self.audio_output = BodySpeaker()
      except Exception:
        self.logger.exception("Failed to initialize body speaker output")
    if config.incoming_audio_track:
      builder.offer_to_receive_audio_stream()

    self.stream = builder.stream()
    self.identifier = str(uuid.uuid4())

    self.incoming_bridge: CerealIncomingMessageProxy | None = None
    self.incoming_bridge_services = incoming_services
    self.outgoing_bridge: CerealOutgoingMessageProxy | None = None
    self.outgoing_bridge_runner: CerealProxyRunner | None = None
    if len(incoming_services) > 0:
      self.incoming_bridge = CerealIncomingMessageProxy(self.shared_pub_master)
    if len(outgoing_services) > 0:
      self.outgoing_bridge = CerealOutgoingMessageProxy(messaging.SubMaster(outgoing_services))
      self.outgoing_bridge_runner = CerealProxyRunner(self.outgoing_bridge)

    self.run_task: asyncio.Task | None = None
    self.cleaned_up = False
    self._app = app
    self.logger.info(
      "New stream session (%s), cameras %s, incoming services %s, outgoing services %s, send audio %s, receive audio %s",
      self.identifier, cameras, incoming_services, outgoing_services, config.expected_audio_track, config.incoming_audio_track,
    )

  def start(self):
    self.run_task = asyncio.create_task(self.run())

  def stop(self):
    if self.run_task is None or self.run_task.done():
      return
    self.run_task.cancel()
    self.run_task = None
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      asyncio.run(self.post_run_cleanup())
    else:
      loop.create_task(self.post_run_cleanup())

  async def get_answer(self):
    return await self.stream.start()

  def _handle_clock_sync(self, payload: dict):
    data = payload.get("data", {})
    if data.get("action") != "ping":
      return
    pong = json.dumps({
      "type": "clockSync",
      "logMonoTime": time.monotonic_ns(),
      "valid": True,
      "data": {
        "action": "pong",
        "browserSendTime": data.get("browserSendTime"),
        "deviceTime": time.time() * 1000,  # noqa: TID251
      },
    })
    try:
      if self.stream.has_messaging_channel():
        self.stream.get_messaging_channel().send(pong)
    except Exception:
      self.logger.warning("Failed to send clockSync pong")

  def _handle_switch_camera(self, payload: dict):
    data = payload.get("data")
    if not isinstance(data, dict):
      return
    camera = data.get("camera")
    if camera in ("driver", "wideRoad"):
      try:
        Params().put("LivestreamCamera", camera)
      except Exception:
        self.logger.warning("Failed to write LivestreamCamera param")
      if hasattr(self, 'video_track') and hasattr(self.video_track, 'switch_camera'):
        self.video_track.switch_camera(camera)
      self.logger.info("Switched livestream camera to %s", camera)

  async def message_handler(self, message: bytes):
    try:
      payload = json.loads(message) if isinstance(message, (bytes, str)) else None
      if isinstance(payload, dict) and payload.get("type") == "switchCamera":
        self._handle_switch_camera(payload)
        return
      if isinstance(payload, dict) and payload.get("type") == "clockSync":
        self._handle_clock_sync(payload)
        return
      if isinstance(payload, dict) and payload.get("type") == "enableTimingSei":
        enabled = bool(payload.get("data", {}).get("enabled"))
        if hasattr(self.video_track, 'timing_sei_enabled'):
          self.video_track.timing_sei_enabled = enabled
          self.logger.info("Timing SEI %s", "enabled" if enabled else "disabled")
        return
      if self.incoming_bridge is not None:
        self.incoming_bridge.send(message)
    except Exception:
      self.logger.exception("Cereal incoming proxy failure")

  async def run(self):
    try:
      await self.stream.wait_for_connection()
      if self.audio_output is not None and self.stream.has_incoming_audio_track():
        self.audio_output.start_track(self.stream.get_incoming_audio_track())
      if self.stream.has_messaging_channel():
        if self.incoming_bridge is not None:
          await self.shared_pub_master.add_services_if_needed(self.incoming_bridge_services)
        self.stream.set_message_handler(self.message_handler)
        if self.outgoing_bridge_runner is not None:
          channel = self.stream.get_messaging_channel()
          self.outgoing_bridge_runner.proxy.add_channel(channel)
          self.outgoing_bridge_runner.start()
      # Tell the client which camera is currently active
      if self.stream.has_messaging_channel():
        try:
          active = getattr(self.video_track, '_camera_type', 'driver')
          self.stream.get_messaging_channel().send(
            json.dumps(
              {
                "type": "activeCamera",
                "logMonoTime": time.monotonic_ns(),
                "valid": True,
                "data": {"camera": active},
              }
            )
          )
        except Exception:
          pass
      self.logger.info("Stream session (%s) connected", self.identifier)

      await self.stream.wait_for_disconnection()

      self.logger.info("Stream session (%s) ended", self.identifier)
    except Exception:
      self.logger.exception("Stream session failure")
    finally:
      await self.post_run_cleanup()

  async def post_run_cleanup(self):
    if self.cleaned_up:
      return
    self.cleaned_up = True

    streams: dict = self._app["streams"] if self._app is not None else {}
    streams.pop(self.identifier, None)

    await self.stream.stop()
    if self.outgoing_bridge is not None:
      self.outgoing_bridge_runner.stop()
    if self.outgoing_audio_track is not None:
      self.outgoing_audio_track.stop()
    if self.audio_output is not None:
      shared = (
        self._app is not None
        and self.audio_output is self._app.get("body_audio_output")
      )
      if shared:
        if not _get_active_streams(streams):
          await self.audio_output.stop()
      else:
        await self.audio_output.stop()
    if self._app is not None:
      if not _get_active_streams(streams):
        Params().put_bool("JoystickDebugMode", False)
    else:
      Params().put_bool("JoystickDebugMode", False)


@dataclass
class StreamRequestBody:
  sdp: str
  cameras: list[str]
  bridge_services_in: list[str] = field(default_factory=list)
  bridge_services_out: list[str] = field(default_factory=list)


@dataclass
class SoundRequestBody:
  sound: str

def _add_cors_headers(request: 'web.Request', response: 'web.Response'):
  response.headers["Access-Control-Allow-Origin"] = "*"
  response.headers["Access-Control-Allow-Headers"] = "Content-Type"
  response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
  response.headers["Access-Control-Allow-Private-Network"] = "true"


@web.middleware
async def cors_middleware(request: 'web.Request', handler):
  try:
    response = await handler(request)
  except web.HTTPException as ex:
    _add_cors_headers(request, ex)
    raise
  _add_cors_headers(request, response)
  return response


async def stream_options(request: 'web.Request'):
  response = web.Response()
  _add_cors_headers(request, response)
  return response


REQUIRED_VIDEO_CODEC = "H264"

def _validate_sdp_video_codecs(sdp: str):
  import aiortc.sdp
  desc = aiortc.sdp.SessionDescription.parse(sdp)
  required_mime = f"video/{REQUIRED_VIDEO_CODEC}"
  for m in desc.media:
    if m.kind != "video":
      continue
    offered_mimes = {c.mimeType for c in m.rtp.codecs}
    if required_mime not in offered_mimes:
      raise web.HTTPBadRequest(
        text=json.dumps({"error": "unsupported_codec", "message": f"Frontend must offer {REQUIRED_VIDEO_CODEC} via setCodecPreferences()"}),
        content_type="application/json",
      )


def _cleanup_stale_streams(stream_dict: dict):
  stale = [sid for sid, s in stream_dict.items() if s.run_task is None or s.run_task.done()]
  for sid in stale:
    del stream_dict[sid]


def _get_active_streams(stream_dict: dict) -> list[str]:
  return [sid for sid, s in stream_dict.items() if s.run_task is not None and not s.run_task.done()]


async def get_stream(request: 'web.Request'):
  logger = logging.getLogger("webrtcd")
  try:
    stream_dict, debug_mode = request.app['streams'], request.app['debug']

    _cleanup_stale_streams(stream_dict)

    raw_body = await request.json()
    body = StreamRequestBody(**raw_body)
    _validate_sdp_video_codecs(body.sdp)

    session = StreamSession(body.sdp, body.cameras, body.bridge_services_in, body.bridge_services_out,
                            request.app['body_audio_output'], debug_mode, request.app)
    answer = await session.get_answer()
    session.start()
    Params().put_bool("JoystickDebugMode", True)

    stream_dict[session.identifier] = session

    active_camera = getattr(session.video_track, '_camera_type', 'driver')
    # Only sdp + type in JSON: clients (e.g. bodyjim) often do RTCSessionDescription(**response.json()).
    # activeCamera is also sent on the data channel after connect; expose here for HTTP-only callers via header.
    response = web.json_response({"sdp": answer.sdp, "type": answer.type})
    response.headers["X-Openpilot-Active-Camera"] = active_camera
    response.headers["Access-Control-Expose-Headers"] = "X-Openpilot-Active-Camera"
    _add_cors_headers(request, response)
    return response
  except web.HTTPException:
    raise
  except Exception:
    logger.exception("Error in /stream handler")
    raise


async def post_sound(request: 'web.Request'):
  try:
    raw_body = await request.json()
    body = SoundRequestBody(**raw_body)
    sound_name = validate_body_sound_name(body.sound)
  except web.HTTPException:
    raise
  except (TypeError, ValueError, json.JSONDecodeError) as err:
    raise web.HTTPBadRequest(
      text=json.dumps({"error": "invalid_sound", "message": str(err)}),
      content_type="application/json",
    ) from err

  audio_output = request.app['body_audio_output']
  if audio_output is None:
    raise web.HTTPServiceUnavailable(
      text=json.dumps({"error": "audio_unavailable", "message": "Body audio output is unavailable"}),
      content_type="application/json",
    )

  audio_output.play_sound(sound_name)
  return web.Response(status=200, text="OK")


async def get_schema(request: 'web.Request'):
  services = request.query["services"].split(",")
  services = [s for s in services if s]
  assert all(s in log.Event.schema.fields and not s.endswith("DEPRECATED") for s in services), "Invalid service name"
  schema_dict = {s: generate_field(log.Event.schema.fields[s]) for s in services}
  return web.json_response(schema_dict)

TRUST_HTML = """<!DOCTYPE html>
<html><head><title>comma body</title>
<style>
  body { background: #111; color: #fff; font-family: -apple-system, system-ui, sans-serif;
         display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
  .card { text-align: center; max-width: 400px; padding: 40px; }
  h1 { font-size: 24px; margin-bottom: 8px; }
  p { color: #aaa; font-size: 14px; }
  .check { font-size: 64px; margin-bottom: 16px; }
</style></head>
<body><div class="card">
  <h1>SSL Certificate Accepted</h1>
  <p>You can close this tab and return to the connect app.</p>
  <script>
    if (window.opener) {
      window.opener.postMessage({ type: 'ssl_cert_accepted' }, '*');
    }
    setTimeout(() => window.close(), 100);
  </script>
</div></body></html>"""


async def get_trust(request: 'web.Request'):
  return web.Response(content_type="text/html", text=TRUST_HTML)


async def post_notify(request: 'web.Request'):
  try:
    payload = await request.json()
  except Exception as e:
    raise web.HTTPBadRequest(text="Invalid JSON") from e

  for session in list(request.app.get('streams', {}).values()):
    try:
      ch = session.stream.get_messaging_channel()
      ch.send(json.dumps(payload))
    except Exception:
      continue

  return web.Response(status=200, text="OK")

async def on_shutdown(app: 'web.Application'):
  for session in app['streams'].values():
    session.stop()
  if app.get('body_audio_output') is not None:
    await app['body_audio_output'].stop()
  del app['streams']


CERT_PATH = "/data/webrtc_cert.pem"
KEY_PATH = "/data/webrtc_key.pem"


def create_ssl_cert():
  logger = logging.getLogger("webrtcd")
  try:
    proc = subprocess.run(
      f'openssl req -x509 -newkey rsa:4096 -nodes -out {CERT_PATH} -keyout {KEY_PATH} '  # noqa: ISC002
      f'-days 365 -subj "/C=US/ST=California/O=commaai/OU=comma body"',
      capture_output=True, shell=True,
    )
    proc.check_returncode()
  except subprocess.CalledProcessError as ex:
    raise ValueError(f"Error creating SSL certificate:\n[stdout]\n{proc.stdout.decode()}\n[stderr]\n{proc.stderr.decode()}") from ex
  logger.info("SSL certificate created")


def create_ssl_context():
  logger = logging.getLogger("webrtcd")
  if not os.path.exists(CERT_PATH) or not os.path.exists(KEY_PATH):
    logger.info("Creating SSL certificate...")
    create_ssl_cert()
  else:
    logger.info("SSL certificate exists")
  ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
  ssl_ctx.load_cert_chain(CERT_PATH, KEY_PATH)
  return ssl_ctx


def webrtcd_thread(host: str, port: int, debug: bool):
  from openpilot.system.webrtc.device.audio import BodySpeaker

  logging.basicConfig(level=logging.CRITICAL, handlers=[logging.StreamHandler()])
  logging_level = logging.DEBUG if debug else logging.INFO
  logging.getLogger("WebRTCStream").setLevel(logging_level)
  logging.getLogger("webrtcd").setLevel(logging_level)

  logger = logging.getLogger("webrtcd")

  app = web.Application(middlewares=[cors_middleware])
  app['streams'] = dict()
  app['debug'] = debug
  try:
    app['body_audio_output'] = BodySpeaker()
  except Exception:
    logger.exception("Failed to initialize shared body audio output")
    app['body_audio_output'] = None
  app.on_shutdown.append(on_shutdown)
  app.router.add_route("OPTIONS", "/stream", stream_options)
  app.router.add_route("OPTIONS", "/sound", stream_options)
  app.router.add_post("/stream", get_stream)
  app.router.add_post("/sound", post_sound)
  app.router.add_post("/notify", post_notify)
  app.router.add_get("/schema", get_schema)
  app.router.add_get("/trust", get_trust)

  https_port = port + 1

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)

  runner = web.AppRunner(app)

  async def start():
    await runner.setup()

    http_site = web.TCPSite(runner, host, port)
    await http_site.start()
    logger.info("HTTP server running on %s:%d", host, port)

    https_site = web.TCPSite(runner, host, https_port, ssl_context=create_ssl_context())
    await https_site.start()
    logger.info("HTTPS server running on %s:%d", host, https_port)

  loop.run_until_complete(start())
  loop.run_forever()


def main():
  parser = argparse.ArgumentParser(description="WebRTC daemon")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
  parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")
  args = parser.parse_args()

  webrtcd_thread(args.host, args.port, args.debug)


if __name__=="__main__":
  main()
