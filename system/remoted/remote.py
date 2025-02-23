#!/usr/bin/env python3
import time
import threading
from openpilot.system.remoted.commands import command_map
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
import argparse

lock = threading.Lock()

# this should contain all the commands that can be run remotely
def runCommand(command: str, args: dict):
  # Check if the command is already running and should be canceled
  if lock.locked() and args['isActive'] == "False":
    print(f"Cancelling command: {command} as isActive is False")
    lock.release()
    return "Lights off"

  # Attempt to acquire the lock to run the command
  acquired = lock.acquire(blocking=False)
  if not acquired:
    print(f"Could not acquire lock for command: {command}")
    return "Could not acquire lock"

  try:
    print(f"Running command: {command} with args: {args}")

    if command in command_map:
      command_map[command](args)
    else:
      print(f"Command {command} not found in command_map.")
  finally:
    # Only release the lock if it was acquired
    if acquired:
      lock.release()

def remoteCommand(command: str, args:dict) -> str:
  params = Params()

  if (params.get_bool("IsRunningCommand") and command != "flashLights"):
    print("Device is Busy")
    return "Device is Busy"

  params.put_bool("IsRunningCommand", True)
  set_offroad_alert("Offroad_IsRunningCommand", True, extra_text=f"{command} {args}")
  res =runCommand(command, args)
  if res is not None:
    return res
  time.sleep(8.0)  # Give hardwared time to read the param

  params.put_bool("IsRunningCommand", False)
  params.remove("Offroad_IsRunningCommand")

  return "Command Executed!"


def main():
  parser = argparse.ArgumentParser(description='Remote command execution.')
  parser.add_argument('command', choices=command_map.keys(), help='Command to execute')
  parser.add_argument('--isActive', type=str, default="True", help='Active status for flashLights command')
  # Add more arguments here as needed for other commands

  args = parser.parse_args()

  remoteCommand(args.command, {"isActive": args.isActive})

if __name__ == "__main__":
  main()