import time
import numpy as np
from cereal import messaging
import argparse
from cereal import car, log

sm = messaging.SubMaster([ 'selfdriveState',
                          'carState', 'carOutput',
                          'driverMonitoringState', 'onroadEvents'], poll='selfdriveState')
pm = messaging.PubMaster(['carControl', 'controlsState'])

CC = car.CarControl.new_message()
CS = sm['carState']

def flashLights(args: dict):
  print("Flash Lights")
  lightFlip = False
  while args['isActive'] == "True":
    CC.hudControl.leftLaneVisible = lightFlip
    CC.hudControl.rightLaneVisible = not lightFlip
    CC.leftBlinker = True
    CC.rightBlinker = True
    flash_send = messaging.new_message('carControl')
    flash_send.valid = CS.canValid
    flash_send.carControl = CC
    pm.send('carControl', flash_send)

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
