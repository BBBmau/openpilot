#!/usr/bin/env python3
import time
import threading
from openpilot.system.remoted.commands import command_map
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
import argparse

# Dictionary to keep track of running command threads
running_threads = {}

# this should contain all the commands that can be run remotely
def runCommand(command: str, args: dict):
  def command_thread():
    try:
      print(f"Running command: {command} with args: {args}")

      if command in command_map:
        command_map[command](args)
      else:
        print(f"Command {command} not found in command_map.")
    finally:
      # Ensure the IsRunningCommand flag is reset after command execution
      params.put_bool("IsRunningCommand", False)
      params.remove("Offroad_IsRunningCommand")
      # Remove the thread from the running_threads dictionary
      running_threads.pop(command, None)

  params = Params()

  # Check if the command is already running and should be canceled
  if args['isActive'] == "False":
    if command in running_threads:
      print(f"Cancelling command: {command} as isActive is False")
      # Terminate the thread (not directly possible, so we signal it to stop)
      running_threads[command].do_run = False
      return "Command cancelled"
    else:
      return "No running command to cancel"

  # Start a new thread for the command
  thread = threading.Thread(target=command_thread)
  thread.do_run = True  # Custom attribute to control the thread
  running_threads[command] = thread
  print("Flash Lights execution started")
  thread.start()
  return "runCommand end"

def remoteCommand(command: str, args:dict) -> str:
  params = Params()

  if (params.get_bool("IsRunningCommand") and command != "flashLights"):
    print("Device is Busy")
    return "Device is Busy"

  params.put_bool("IsRunningCommand", True)
  set_offroad_alert("Offroad_IsRunningCommand", True, extra_text=f"{command} {args}")
  res = runCommand(command, args)
  if res is not None:
    return res

  return "Command Executed!"

def main():
  parser = argparse.ArgumentParser(description='Remote command execution.')
  parser.add_argument('command', choices=command_map.keys(), help='Command to execute')
  parser.add_argument('--isActive', type=str, default="True", help='Active status for flashLights command')
  # Add more arguments here as needed for other commands

  args = parser.parse_args()

  print(remoteCommand(args.command, {"isActive": args.isActive}))

if __name__ == "__main__":
  main()