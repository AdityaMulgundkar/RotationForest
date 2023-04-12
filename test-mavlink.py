import time
# from pymavlink import mavutil
import struct
import numpy as np
import re
import os
import sys

cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/../RotationForest/dist/")
# sys.path.insert(0, cur_path+"~/PX4-Autopilot/src/modules/mavlink/mavlink/..")

from pymavlink import mavutil

mavutil.set_dialect("common")

master = mavutil.mavlink_connection('udpin:0.0.0.0:14552')
master.wait_heartbeat()


def request_message_interval(message_id: int, frequency_hz: float):
    """
    Request MAVLink message in a desired frequency,
    documentation for SET_MESSAGE_INTERVAL:
        https://mavlink.io/en/messages/common.html#MAV_CMD_SET_MESSAGE_INTERVAL
    Args:
        message_id (int): MAVLink message ID
        frequency_hz (float): Desired frequency in Hz
    """
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        message_id, # The MAVLink message ID
        1e6 / frequency_hz, # The interval between two messages in microseconds. Set to -1 to disable and 0 to request default rate.
        0, 0, 0, 0, # Unused parameters
        0, # Target address of message stream (if message has target address fields). 0: Flight-stack default (recommended), 1: address of requestor, 2: broadcast.
    )

request_message_interval(12921, 50)


while True:
    try:
        msg = master.recv_match(blocking=False, timeout=1)
        if not msg:
            continue
        type = msg.get_type()
        
        # print(f"msg: {msg}")
        if type == 'UNKNOWN_12921' or type == 'DESIRED_VELOCITY_RATES':
            print(f"{msg}")
        
    except:
        pass

    time.sleep(0.01)
    # print("main loop")