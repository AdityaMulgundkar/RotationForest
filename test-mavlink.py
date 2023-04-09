import time
from pymavlink import mavutil
import struct
import numpy as np
import re

mavutil.set_dialect("common")

master = mavutil.mavlink_connection('udpin:0.0.0.0:14554')
master.wait_heartbeat()

while True:
    try:
        msg = master.recv_match(blocking=False, timeout=1)
        if not msg:
            continue
        type = msg.get_type()
        
        if type == 'UNKNOWN_12921' or type == 'DESIRED_VELOCITY_RATES':
            print(f"{msg}")
        
    except:
        pass

    time.sleep(0.01)
    # print("main loop")