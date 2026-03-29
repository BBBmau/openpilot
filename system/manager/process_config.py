import os
import operator
import platform

from cereal import car
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE, PC, TICI
from openpilot.system.manager.process import PythonProcess, NativeProcess, DaemonProcess

WEBCAM = os.getenv("USE_WEBCAM") is not None

_cached_is_body: bool | None = None
def _is_body(CP: car.CarParams, params: Params) -> bool:
  """Check notCar from live CarParams, falling back to CarParamsPersistent.

  card only publishes carParams when onroad. For body-specific processes that
  must run offroad (micd, bodywaked), we need to know it's a body before card
  has ever published in this manager session.
  """
  global _cached_is_body
  if CP.notCar:
    _cached_is_body = True
    return True
  if _cached_is_body is not None:
    return _cached_is_body
  cp_bytes = params.get("CarParamsPersistent")
  if cp_bytes:
    try:
      _cached_is_body = car.CarParams.from_bytes(cp_bytes).notCar
      return _cached_is_body
    except Exception:
      pass
  return False

def driverview(started: bool, params: Params, CP: car.CarParams) -> bool:
  """Vision for UI / DM; on comma body also when ``LiveIgnition`` so ``stream_encoderd`` can attach."""
  if started or params.get_bool("IsDriverViewEnabled"):
    return True
  return CP.notCar and params.get_bool("LiveIgnition")

def notcar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and CP.notCar

def iscar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not CP.notCar

def logging(started: bool, params: Params, CP: car.CarParams) -> bool:
  run = (not CP.notCar) or not params.get_bool("DisableLogging")
  return started and run

def ublox_available() -> bool:
  return os.path.exists('/dev/ttyHS0') and not os.path.exists('/persist/comma/use-quectel-gps')

def ublox(started: bool, params: Params, CP: car.CarParams) -> bool:
  use_ublox = ublox_available()
  if use_ublox != params.get_bool("UbloxAvailable"):
    params.put_bool("UbloxAvailable", use_ublox)
  return started and use_ublox

def joystick(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("JoystickDebugMode")

def not_joystick(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not params.get_bool("JoystickDebugMode")

def long_maneuver(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and params.get_bool("LongitudinalManeuverMode")

def not_long_maneuver(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not params.get_bool("LongitudinalManeuverMode")

def qcomgps(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not ublox_available()

def always_run(started: bool, params: Params, CP: car.CarParams) -> bool:
  return True

def only_onroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started

def micd_onroad_or_body(started: bool, params: Params, CP: car.CarParams) -> bool:
  """Comma body needs the mic while nominally offroad so wake-word can bring the UI 'onroad'."""
  return started or _is_body(CP, params) or params.get_bool("BodyWakeWordEnabled")


def soundd_should_run(started: bool, params: Params, CP: car.CarParams) -> bool:
  """Like driverview for bench/cars, plus body offroad so webrtcAudioData/bodyRealtimeAudioData reach speakers."""
  if driverview(started, params, CP):
    return True
  if CP.notCar:
    return True
  if not PC and HARDWARE.get_device_type() == "tizi":
    return True
  return False


def bodywaked_should_run(started: bool, params: Params, CP: car.CarParams) -> bool:
  return params.get_bool("BodyWakeWordEnabled")


def comma_body_stack_should_run(started: bool, params: Params, CP: car.CarParams) -> bool:
  """Comma body: WebRTC/bridge/random-walk run when ignition is on.

  Uses _is_body fallback so the stack starts even before card publishes
  carParams (e.g. wake-word boot).  BodyWakeWordEnabled acts as a secondary
  body-device indicator when CarParamsPersistent is unavailable.
  """
  if not (_is_body(CP, params) or params.get_bool("BodyWakeWordEnabled")):
    return False
  return started or params.get_bool("LiveIgnition") or params.get_bool("BodyWakeIgnition")

def only_offroad(started: bool, params: Params, CP: car.CarParams) -> bool:
  return not started

def or_(*fns):
  return lambda *args: operator.or_(*(fn(*args) for fn in fns))

def and_(*fns):
  return lambda *args: operator.and_(*(fn(*args) for fn in fns))

procs = [
  DaemonProcess("manage_athenad", "system.athena.manage_athenad", "AthenadPid"),

  NativeProcess("loggerd", "system/loggerd", ["./loggerd"], logging),
  NativeProcess("encoderd", "system/loggerd", ["./encoderd"], only_onroad),
  # Same gate as webrtcd: livestream H.264 must flow whenever WebRTC stack runs (not only full onroad).
  NativeProcess("stream_encoderd", "system/loggerd", ["./encoderd", "--stream"], comma_body_stack_should_run),
  PythonProcess("logmessaged", "system.logmessaged", always_run),

  NativeProcess("camerad", "system/camerad", ["./camerad"], driverview, enabled=not WEBCAM),
  PythonProcess("webcamerad", "tools.webcam.camerad", driverview, enabled=WEBCAM),
  PythonProcess("proclogd", "system.proclogd", only_onroad, enabled=platform.system() != "Darwin"),
  PythonProcess("journald", "system.journald", only_onroad, platform.system() != "Darwin"),
  PythonProcess("micd", "system.micd", micd_onroad_or_body),
  PythonProcess("bodywaked", "selfdrive.ui.body.bodywaked", bodywaked_should_run),
  PythonProcess("timed", "system.timed", always_run, enabled=not PC),

  PythonProcess("modeld", "selfdrive.modeld.modeld", only_onroad),
  PythonProcess("dmonitoringmodeld", "selfdrive.modeld.dmonitoringmodeld", and_(driverview, iscar), enabled=(WEBCAM or not PC)),

  PythonProcess("sensord", "system.sensord.sensord", only_onroad, enabled=not PC),
  PythonProcess("ui", "selfdrive.ui.ui", always_run, restart_if_crash=True),
  PythonProcess("soundd", "selfdrive.ui.soundd", soundd_should_run),
  PythonProcess("locationd", "selfdrive.locationd.locationd", only_onroad),
  NativeProcess("_pandad", "selfdrive/pandad", ["./pandad"], always_run, enabled=False),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd", only_onroad),
  PythonProcess("torqued", "selfdrive.locationd.torqued", only_onroad),
  PythonProcess("controlsd", "selfdrive.controls.controlsd", and_(not_joystick, iscar)),
  PythonProcess("joystickd", "tools.joystick.joystickd", or_(joystick, notcar)),
  PythonProcess("selfdrived", "selfdrive.selfdrived.selfdrived", only_onroad),
  PythonProcess("card", "selfdrive.car.card", only_onroad),
  PythonProcess("deleter", "system.loggerd.deleter", always_run),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", and_(driverview, iscar), enabled=(WEBCAM or not PC)),
  PythonProcess("qcomgpsd", "system.qcomgpsd.qcomgpsd", qcomgps, enabled=TICI),
  PythonProcess("pandad", "selfdrive.pandad.pandad", always_run),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd", only_onroad),
  PythonProcess("lagd", "selfdrive.locationd.lagd", only_onroad),
  PythonProcess("ubloxd", "system.ubloxd.ubloxd", ublox, enabled=TICI),
  PythonProcess("pigeond", "system.ubloxd.pigeond", ublox, enabled=TICI),
  PythonProcess("plannerd", "selfdrive.controls.plannerd", not_long_maneuver),
  PythonProcess("maneuversd", "tools.longitudinal_maneuvers.maneuversd", long_maneuver),
  PythonProcess("radard", "selfdrive.controls.radard", only_onroad),
  PythonProcess("hardwared", "system.hardware.hardwared", always_run),
  PythonProcess("tombstoned", "system.tombstoned", always_run, enabled=not PC),
  PythonProcess("updated", "system.updated.updated", only_offroad, enabled=not PC),
  PythonProcess("uploader", "system.loggerd.uploader", always_run),
  PythonProcess("statsd", "system.statsd", always_run),
  PythonProcess("feedbackd", "selfdrive.ui.feedback.feedbackd", only_onroad),

  # debug procs
  NativeProcess("bridge", "cereal/messaging", ["./bridge"], comma_body_stack_should_run),
  PythonProcess("webrtcd", "system.webrtc.webrtcd", comma_body_stack_should_run),
  PythonProcess("bodyrandomwalkd", "tools.body.body_random_walkd", comma_body_stack_should_run),
  PythonProcess("openai_realtime_webrtcd", "tools.body.openai_realtime_webrtc", comma_body_stack_should_run),
  PythonProcess("joystick", "tools.joystick.joystick_control", and_(joystick, iscar)),
]

managed_processes = {p.name: p for p in procs}
