#!/usr/bin/env python3
import time
import threading
from openpilot.system.remoted.commands import command_map
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert

lock = threading.Lock()

# this should contain all the commands that can be run remotely
def runCommand(command: str, args:dict):
  if lock.locked() and args['isActive'] == "False":
    lock.release()
  with lock:
    print(f"Running command: {command} with args: {args}")

    if command in command_map:
      command_map[command](args)
    else:
      pass

def remoteCommand(command: str, args:dict) -> str:
  params = Params()

  if (params.get_bool("IsRunningCommand") and command != "flashLights"):
    print("Device is Busy")
    return "Device is Busy"

  params.put_bool("IsRunningCommand", True)
  set_offroad_alert("Offroad_IsRunningCommand", True, extra_text=f"{command} {args}")
  runCommand(command, args)
  time.sleep(8.0)  # Give hardwared time to read the param

  params.put_bool("IsRunningCommand", False)
  params.remove("Offroad_IsRunningCommand")

  return "Command Executed!"
