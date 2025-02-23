import time
import numpy as np
from cereal import messaging
import argparse
from cereal import car, log
import threading
import os
import json

sm = messaging.SubMaster([ 'selfdriveState',
                          'carState', 'carOutput',
                          'driverMonitoringState', 'onroadEvents'], poll='selfdriveState')
pm = messaging.PubMaster(['carControl', 'controlsState'])

CC = car.CarControl.new_message()
CS = sm['carState']

RUNNING_THREADS_FILE = "running_threads.json"

def is_command_active(command_name):
    if os.path.exists(RUNNING_THREADS_FILE):
        with open(RUNNING_THREADS_FILE, 'r') as f:
            try:
                data = json.load(f)
                return command_name in data
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {RUNNING_THREADS_FILE}.")
    return False

def flashLights(args: dict):
  print("Flash Lights")
  lightFlip = False
  while is_command_active('flashLights'):
    print("Running...")
    # Debug: Check the current state of the command
    print(f"Command active: {is_command_active('flashLights')}")
    # Check if the thread should stop
    if not getattr(threading.current_thread(), "do_run", True):
      print("Flash Lights command deactivated.")
      break  # Exit the loop if deactivated

    start_time = time.time()
    while time.time() - start_time < 1:
      CC.hudControl.leftLaneVisible = lightFlip
      CC.hudControl.rightLaneVisible = lightFlip
      CC.leftBlinker = lightFlip
      CC.rightBlinker = lightFlip
      CC.enabled = lightFlip
      flash_send = messaging.new_message('carControl')
      flash_send.valid = CS.canValid
      flash_send.carControl = CC
      pm.send('carControl', flash_send)
      print(start_time)
      lightFlip = not lightFlip
      time.sleep(0.1)  # Add a small delay to prevent rapid looping

  # Ensure any necessary cleanup here
  print("Flash Lights command completed or canceled.")

hvac_args = {
  "hvac_auto": bool,
  "hvac_on": bool,
  "mode": int,
  "fan_speed": int,
}

def hvac(args: hvac_args):
  print(args)

command_map = {
  "flashLights": flashLights,
  "hvac": hvac,
  # Add more commands here as needed
}

def main():
  parser = argparse.ArgumentParser(description='Remote command execution.')
  parser.add_argument('command', choices=command_map.keys(), help='Command to execute')
  parser.add_argument('--isActive', type=str, default="True", help='Active status for flashLights command')
  # Add more arguments here as needed for other commands

  args = parser.parse_args()

  if args.command == "flashLights":
    flashLights({"isActive": args.isActive})
  elif args.command == "hvac":
    # Example: Add handling for hvac command if needed
    hvac({})  # Pass appropriate arguments

if __name__ == "__main__":
  main()
