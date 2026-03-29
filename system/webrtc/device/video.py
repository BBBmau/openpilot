import asyncio
import struct
import time
from typing import Any

import av
from teleoprtc.tracks import TiciVideoStreamTrack

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON

# 16-byte UUID identifying openpilot frame-timing SEI messages
TIMING_SEI_UUID = bytes([
  0xa5, 0xe0, 0xc4, 0xa4, 0x5b, 0x6e, 0x4e, 0x1e,
  0x9c, 0x7e, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
])

# EncodeIndex.flags — same as V4L2_BUF_FLAG_KEYFRAME in system/loggerd/encoder/encoder.h
LIVESTREAM_KEYFRAME_FLAG = 8


def _h264_annex_b_has_sps_pps_or_idr(data: bytes) -> bool:
  """True if Annex-B H.264 contains SPS (7), PPS (8), or IDR slice (5)."""
  n = len(data)
  i = 0
  while i < n:
    if i + 3 <= n and data[i : i + 3] == b"\x00\x00\x01":
      sc = 3
    elif i + 4 <= n and data[i : i + 4] == b"\x00\x00\x00\x01":
      sc = 4
    else:
      i += 1
      continue
    pos = i + sc
    if pos < n and (data[pos] & 0x1F) in (5, 7, 8):
      return True
    i = pos

  return False


def _livestream_frame_is_decoder_sync_point(idx_flags: int, header: bytes, frame_data: bytes) -> bool:
  """First RTP must carry something a decoder can sync from (see encoder.cc header + V4L separate mode)."""
  if idx_flags & LIVESTREAM_KEYFRAME_FLAG:
    return True
  if header:
    return True
  return _h264_annex_b_has_sps_pps_or_idr(frame_data)


def livestream_encode_data_diag(msg: Any) -> dict[str, bool | int | str]:
  """Compact fields for logging / tools (e.g. body_random_walkd when WebRTC reset times out)."""
  which = msg.which()
  evta = getattr(msg, which)
  idx = evta.idx
  hdr = bytes(evta.header)
  dat = bytes(evta.data)
  frame = hdr + dat
  return {
    "which": which,
    "encode_type": str(idx.type),
    "flags": int(idx.flags),
    "header_bytes": len(hdr),
    "data_bytes": len(dat),
    "keyframe_bit": bool(idx.flags & LIVESTREAM_KEYFRAME_FLAG),
    "sync_ok": _livestream_frame_is_decoder_sync_point(idx.flags, hdr, frame),
  }


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
    # WebRTC peer decoders need SPS/PPS (EncodeData.header is only set on keyframes in encoder.cc).
    # If we start on a P-frame, nothing decodes until the next IDR — skip until the first keyframe.
    self._need_keyframe = True

  def switch_camera(self, camera_type: str):
    if camera_type not in self.camera_to_sock_mapping or camera_type == self._camera_type:
      return
    self._camera_type = camera_type
    self._sock = messaging.sub_sock(self.camera_to_sock_mapping[camera_type], conflate=False)
    self._need_keyframe = True

  async def recv(self):
    while True:
      while True:
        msg = messaging.recv_one_or_none(self._sock)
        if msg is not None:
          break
        await asyncio.sleep(0.005)

      evta = getattr(msg, msg.which())
      idx = evta.idx
      hdr = bytes(evta.header)
      frame_data = hdr + bytes(evta.data)
      if not frame_data:
        continue

      if self._need_keyframe:
        if not _livestream_frame_is_decoder_sync_point(idx.flags, hdr, frame_data):
          continue
        self._need_keyframe = False

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
