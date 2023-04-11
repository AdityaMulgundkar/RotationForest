import pickle
from RotationForest import *
import time
import math
# from dronekit import connect, VehicleMode
import asyncio
from mavsdk import System
# from pymavlink import mavutil

import os
import sys
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/../RotationForest/dist/")
# sys.path.insert(0, cur_path+"~/PX4-Autopilot/src/modules/mavlink/mavlink/..")

from pymavlink import mavutil

import numpy as np
import csv

import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings("ignore")

timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"dist/real-logs/log_{timestamp}.csv"

master = 0

pymavlink_string = "udpin:0.0.0.0:14553"
dronekit_string = "udpin:0.0.0.0:14554"
mavsdk_string = "udp://:14554"

    
def set_param(name: str, value: float, type: int=0,
                  timeout: float=1, retries: int=10):
    name = name.encode('utf8')
    master.mav.param_set_send(
            master.target_system,
            master.target_component,
            name, value, type
    )
    # logging.info(f'set_param({name=}, {value=}, {type=})')

    while not (msg := master.recv_match(type='PARAM_VALUE', blocking=True,
                                          timeout=timeout)) and retries > 0:
        retries -= 1
        # logging.debug(f'param set timed out after {timeout}s, retrying...')
        master.mav.param_set_send(
                master.target_system,
                master.target_component,
                name, value, type
            )
    return msg

async def run():
    def log_this(time, value):
        log_writer.writerow([time, value])
    last_time = 0
    R, P, P_1, P_d, Y = 0, 0, 0, 0, 0
    controlFlag = True
    # Connect to the drone
    drone = System()
    await drone.connect(system_address=mavsdk_string)
    with open(filename, "a", newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        Rotate = pickle.load(open("models/rfc-M35", 'rb'))

        mavutil.set_dialect("common")
        master = mavutil.mavlink_connection(pymavlink_string)
        master.wait_heartbeat()

        while True:
            try:
                msg = master.recv_match(blocking=False, timeout=1)
                if not msg:
                    continue
                type = msg.get_type()
                if type == 'UNKNOWN_12921' or type == 'DESIRED_VELOCITY_RATES':
                    # print(f"{msg}")
                    P_d = math.degrees(msg.pdes)
                
                if type == 'ATTITUDE_TARGET':
                    # print(f"{msg}")
                    P_1 = P
                    R, P, Y = math.degrees(msg.body_roll_rate), math.degrees(msg.body_pitch_rate), math.degrees(Y)
                    last_time = msg.time_boot_ms
            except:
                pass
            
            print(f"R, P, P_1, P_d, Y: {R, P, P_1, P_d, Y}")
            
            xte = np.asarray([R, P, P_1, P_d, Y]).reshape(1, 3)
            preds_rotate = Rotate.predict(xte)
            log_this(last_time, preds_rotate[0])
            print(f"Pred: {preds_rotate[0]}")

            if(preds_rotate[0]!=0 and controlFlag):
                await drone.param.set_param_int('FAULTY_M2', 1)
                print("SET CONTROL FLAG FALSE")
                controlFlag = False

            time.sleep(0.0001)

# Run the asyncio loop
asyncio.run(run())