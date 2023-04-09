import pickle
from RotationForest import *
import time
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

log_writer = 0
last_time = 0
R, R_1, R_d = 0, 0, 0

def log_this(time, value):
    log_writer.writerow([time, value])
    
with open(filename, "a", newline='') as csvfile:
    log_writer = csv.writer(csvfile)
    Rotate = pickle.load(open("models/rfc-M12", 'rb'))

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
                # print(f"{msg}")
                R_d = msg.rdes
            
            if type == 'ATTITUDE_TARGET':
                # print(f"{msg}")
                R, R_1 = msg.body_roll_rate, R
                last_time = msg.time_boot_ms
        except:
            pass
        
        # print(f"time: {last_time}")
        # print(f"R, R_1, R_d: {R, R_1, R_d}")
        
        xte = np.asarray([R, R_1, R_d]).reshape(1, 3)
        preds_rotate = Rotate.predict(xte)
        log_this(last_time, preds_rotate[0])
        print(preds_rotate[0])

        time.sleep(0.01)