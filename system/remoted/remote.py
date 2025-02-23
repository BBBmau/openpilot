#!/usr/bin/env python3
import time
import threading
from openpilot.system.remoted.commands import command_map
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert

lock = threading.Lock()

# this should contain all the commands that can be run remotely
def runCommand(command: str, args: dict):
  # Check if the command is already running and should be canceled
  if lock.locked() and args['isActive'] == "False":
    print(f"Cancelling command: {command} as isActive is False")
    lock.release()
    return

  # Acquire the lock to run the command
  lock.acquire()
  try:
    print(f"Running command: {command} with args: {args}")

    if command in command_map:
      command_map[command](args)
    else:
      print(f"Command {command} not found in command_map.")
  finally:
    # Always release the lock in the finally block
    lock.release()

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
