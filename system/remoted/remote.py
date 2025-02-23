#!/usr/bin/env python3
import time
import threading
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
import argparse
import json
import os

# File to store the running threads state
RUNNING_THREADS_FILE = "running_threads.json"

def load_running_threads():
    if os.path.exists(RUNNING_THREADS_FILE):
        with open(RUNNING_THREADS_FILE, 'r') as f:
            try:
                # Ensure the loaded data is a dictionary
                data = json.load(f)
                if isinstance(data, list):
                    # Convert list to dictionary with None as default value
                    return {command: None for command in data}
                return data
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {RUNNING_THREADS_FILE}. Returning empty dictionary.")
                return {}
    return {}

def save_running_threads():
    # Save only the command names, not the Thread objects
    with open(RUNNING_THREADS_FILE, 'w') as f:
        json.dump(list(running_threads.keys()), f)

# Load the running threads state at the start
running_threads = load_running_threads()

# this should contain all the commands that can be run remotely
def runCommand(command: str, args: dict):
  def command_thread():
    from openpilot.system.remoted.commands import command_map
    try:
      print(f"Running command: {command} with args: {args}")

      if command in command_map:
        # Example of a loop that checks the do_run flag
        command_map[command](args)
        # Add a sleep or wait mechanism to prevent busy-waiting
      else:
        print(f"Command {command} not found in command_map.")
    finally:
      # Ensure the IsRunningCommand flag is reset after command execution
      params.put_bool("IsRunningCommand", False)
      params.remove("Offroad_IsRunningCommand")
      # Remove the thread from the running_threads dictionary
      running_threads.pop(command, None)
      save_running_threads()  # Save state after removing a command

  params = Params()

  # Check if the command is already running and should be canceled
  if args['isActive'] == "False":
    if command in running_threads:
      print(f"Cancelling command: {command} as isActive is False")
      # Check if the thread object is not None before accessing do_run
      if running_threads[command] is not None:
        running_threads[command].do_run = False
      running_threads.pop(command, None)
      save_running_threads()  # Save state after cancelling a command
      return "Command cancelled"
    else:
      return "No running command to cancel"

  # Start a new thread for the command
  thread = threading.Thread(target=command_thread)
  thread.do_run = True  # Custom attribute to control the thread
  running_threads[command] = thread
  save_running_threads()  # Save state after adding a new command
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
  parser.add_argument('command', choices="flashLights", help='Command to execute')
  parser.add_argument('--isActive', type=str, default="True", help='Active status for flashLights command')
  # Add more arguments here as needed for other commands

  args = parser.parse_args()

  print(remoteCommand(args.command, {"isActive": args.isActive}))

if __name__ == "__main__":
  main()