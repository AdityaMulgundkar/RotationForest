import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix
from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import os

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_num = 6

# folder path
dir_path = f'dist/real-logs-2/m{motor_num}/'

# list file and directories
fileNames = os.listdir(dir_path)
print(fileNames)

j = 0
for fname in fileNames:
    if(fname.endswith('.csv')):
        last = 0
        j = j + 1

        aX_te, aY_te = [], []
        Xdata_te, Ydata_te = np.asarray([]), np.asarray([])
        data = pd.read_csv(f"dist/real-logs-2/m{motor_num}/{fname}")
        for i in range(1, len(data.loc[:, "R"])):
            if i < 1:
                ax = (
                    data.loc[:, "R"][i],
                    data.loc[:, "R"][i],
                    data.loc[:, "RDes"][i],
                    data.loc[:, "P"][i],
                    data.loc[:, "P"][i],
                    data.loc[:, "PDes"][i],
                    data.loc[:, "Y"][i],
                    data.loc[:, "Y"][i],
                    data.loc[:, "YDes"][i],
                )
            else:
                ax = (
                    data.loc[:, "R"][i],
                    data.loc[:, "R"][i-1],
                    data.loc[:, "RDes"][i],
                    data.loc[:, "P"][i],
                    data.loc[:, "P"][i-1],
                    data.loc[:, "PDes"][i],
                    data.loc[:, "Y"][i],
                    data.loc[:, "Y"][i-1],
                    data.loc[:, "YDes"][i],
                )
            ay = data.loc[:, "FaultIn"][i]
            if ay != 0:
                last = i
            
            aX_te.append(ax)
            aY_te.append(ay)

        print(f"Real fault is at {last}")

        xte, yte = np.asarray(aX_te).reshape(len(aX_te), len(
            ax)), np.asarray(aY_te).reshape(len(aY_te), 1)


        Rotate = pickle.load(open("models/rfc-Mall", 'rb'))
        preds_rotate = Rotate.predict(xte)
        # print(preds_rotate)

        c5 = confusion_matrix(yte, preds_rotate, normalize='true', labels=[0, motor_num])
        print(f"conf: {c5}")

        preds_rotate = np.insert(preds_rotate, 0,5)
        data.insert(7, "RRPrediction", preds_rotate)

        # print(data)

        # plt.plot(xte)
        plt.figure(j)
        plt.plot(yte, label=f"M{motor_num} Actual Fault")
        plt.plot(preds_rotate, linestyle='dotted', linewidth='2', label=f"RRF Classifier")
        plt.legend(loc="upper left")
        plt.xlabel("Sampling")
        plt.ylabel("Fault Classification")
        plt.title(f"RRF Classifier for Motor {motor_num} in Real Flight")
        # plt.show()
        plt.savefig(f"dist/real-logs-2/m{motor_num}/{fname[:-4]}.png", dpi=300)
        # plt.savefig(f"dist/real-logs-2/m{motor_num}/{fname[:-4]}.pdf")