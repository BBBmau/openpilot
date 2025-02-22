#!/usr/bin/env python3
import time

from openpilot.system.remoted.commands import command_map
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert

# this should contain all the commands that can be run remotely
def runCommand(command: str, args:dict):
  print(f"Running command: {command} with args: {args}")

  if command in command_map:
    command_map[command](args)
  else:
    pass

def remoteCommand(command: str, args:dict) -> str:
  params = Params()

  if (not params.get_bool("IsOffroad")) or (params.get_bool("IsRunningCommand") and command != "flashLights"):
    print("Device is Busy")
    return "Device is Busy"

  params.put_bool("IsRunningCommand", True)
  set_offroad_alert("Offroad_IsRunningCommand", True, extra_text=f"{command} {args}")
  runCommand(command, args)
  time.sleep(8.0)  # Give hardwared time to read the param

  params.put_bool("IsRunningCommand", False)
  params.remove("Offroad_IsRunningCommand")

  return "Command Executed!"
