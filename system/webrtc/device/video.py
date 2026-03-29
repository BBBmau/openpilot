import asyncio
import struct
import time

import av
from teleoprtc.tracks import TiciVideoStreamTrack

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON

# 16-byte UUID identifying openpilot frame-timing SEI messages
TIMING_SEI_UUID = bytes([
  0xa5, 0xe0, 0xc4, 0xa4, 0x5b, 0x6e, 0x4e, 0x1e,
  0x9c, 0x7e, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
])


def _escape_rbsp(data: bytes) -> bytearray:
  """Insert H.264 emulation-prevention bytes (0x03) where required."""
  out = bytearray()
  zeros = 0
  for b in data:
    if zeros >= 2 and b <= 3:
      out.append(3)
      zeros = 0
    zeros = zeros + 1 if b == 0 else 0
    out.append(b)
  return out


def create_timing_sei(capture_ms: float, encode_ms: float, send_delay_ms: float, send_wall_ms: float) -> bytes:
  """Build an H.264 SEI NAL (user_data_unregistered) carrying frame timing."""
  ts_data = struct.pack('>4d', capture_ms, encode_ms, send_delay_ms, send_wall_ms)
  sei_payload = TIMING_SEI_UUID + ts_data  # 16 + 32 = 48 bytes

  # payload_type=5, payload_size=48, then RBSP stop bit
  rbsp = bytes([5, len(sei_payload)]) + sei_payload + bytes([0x80])
  escaped = _escape_rbsp(rbsp)

  # start-code (4 bytes) + NAL header (forbidden=0, ref_idc=0, type=6 SEI)
  return b'\x00\x00\x00\x01\x06' + bytes(escaped)


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  camera_to_sock_mapping = {
    "driver": "livestreamDriverEncodeData",
    "wideRoad": "livestreamWideRoadEncodeData",
  }

  def __init__(self, camera_type: str):
    dt = DT_DMON if camera_type == "driver" else DT_MDL
    super().__init__(camera_type, dt)

    self._camera_type = camera_type
    # Do not conflate: latest-only often lands on a P-frame with empty ``header`` (SPS/PPS only on
    # keyframes in encoder.cc). Remote WebRTC decoders then never emit a frame until a keyframe.
    self._sock = messaging.sub_sock(self.camera_to_sock_mapping[camera_type], conflate=False)
    self._pts = 0
    self._t0_ns = time.monotonic_ns()
    self.timing_sei_enabled = False

  def switch_camera(self, camera_type: str):
    if camera_type not in self.camera_to_sock_mapping or camera_type == self._camera_type:
      return
    self._camera_type = camera_type
    self._sock = messaging.sub_sock(self.camera_to_sock_mapping[camera_type], conflate=False)

  async def recv(self):
    while True:
      msg = messaging.recv_one_or_none(self._sock)
      if msg is not None:
        break
      await asyncio.sleep(0.005)

    evta = getattr(msg, msg.which())

    frame_data = evta.header + evta.data
    if self.timing_sei_enabled:
      capture_ms = (evta.idx.timestampEof - evta.idx.timestampSof) / 1e6
      encode_ms = (msg.logMonoTime - evta.idx.timestampEof) / 1e6
      send_delay_ms = (time.monotonic_ns() - msg.logMonoTime) / 1e6
      send_wall_ms = time.time() * 1000  # noqa: TID251
      sei_nal = create_timing_sei(capture_ms, encode_ms, send_delay_ms, send_wall_ms)
      frame_data = evta.header + sei_nal + evta.data

    packet = av.Packet(frame_data)
    packet.time_base = self._time_base

    self._pts = ((time.monotonic_ns() - self._t0_ns) * self._clock_rate) // 1_000_000_000
    packet.pts = self._pts
    self.log_debug("track sending frame %d", self._pts)

    return packet

  def codec_preference(self) -> str | None:
    return "H264"


def _video_frame_to_png_bytes(frame: av.VideoFrame) -> bytes:
  rgb = frame.reformat(format="rgb24")
  enc = av.CodecContext.create("png", "w")
  chunks: list[bytes] = []
  for pkt in enc.encode(rgb):
    chunks.append(bytes(pkt))
  for pkt in enc.encode(None):
    chunks.append(bytes(pkt))
  return b"".join(chunks)


def grab_livestream_png_bytes(camera_type: str, *, timeout_s: float = 15.0) -> bytes:
  """
  Pull H.264 from ``livestreamDriverEncodeData`` / ``livestreamWideRoadEncodeData`` (encoderd),
  decode one frame, return PNG bytes — same source as ``LiveStreamVideoStreamTrack`` / webrtcd video.
  """
  if camera_type not in LiveStreamVideoStreamTrack.camera_to_sock_mapping:
    raise ValueError(f"camera_type must be 'driver' or 'wideRoad', not {camera_type!r}")
  service = LiveStreamVideoStreamTrack.camera_to_sock_mapping[camera_type]
  sock = messaging.sub_sock(service, conflate=False)
  codec = av.CodecContext.create("h264", "r")
  deadline = time.monotonic() + timeout_s
  last_av_err: av.FFmpegError | None = None
  while time.monotonic() < deadline:
    batch = messaging.drain_sock(sock, wait_for_one=True)
    for msg in batch:
      evta = getattr(msg, msg.which())
      raw = bytes(evta.header) + bytes(evta.data)
      if not raw:
        continue
      try:
        decoded = codec.decode(av.Packet(raw))
      except av.FFmpegError as e:
        last_av_err = e
        continue
      for frame in decoded:
        return _video_frame_to_png_bytes(frame)
  detail = f" (last decode error: {last_av_err})" if last_av_err is not None else ""
  raise RuntimeError(
    f"No decodable H.264 frame from {service} within {timeout_s:.1f}s.{detail} "
    "Is encoderd running and livestream publishing?"
  )
