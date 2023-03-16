import matplotlib.pyplot as plt
import pickle
from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_num = 1

# TODO:
# Get current roll pitch yaw value from onboard sensor
# Attach timestamp to each sensor input packet (above)
# Use Rotate.predict on the single dataset (last fetched)
## Pass 0 values for des rates in the predictor. Pass rates if possible for R, P, Y and so on.
## In case you cant fetch rates, jut pass zeros. Doesnt matter what output classifier gives. (This is worst case scenario)
# Attach a timestamp after prediction result
# Maintain an open log file (csv?)
# Write each entry into same file
# Pref name the log file with current timestamp, to avoid overrriding

aX_te, aY_te = [], []
Xdata_te, Ydata_te = np.asarray([]), np.asarray([])
# data = pd.read_csv(f"dist/hexa-x/real-cases/m{motor_num}.csv")
data = pd.read_csv(f"dist/hexa-x/err20/m{motor_num}/test10.csv")
# data = pd.read_csv(f"dist/hexa-x/err10/m{motor_num}/test10.csv")
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
    aX_te.append(ax)
    aY_te.append(ay)

xte, yte = np.asarray(aX_te).reshape(len(aX_te), len(
    ax)), np.asarray(aY_te).reshape(len(aY_te), 1)


Rotate = pickle.load(open("models/rfc-Mall", 'rb'))
preds_rotate = Rotate.predict(xte)
# print(preds_rotate)
preds_rotate = np.insert(preds_rotate, 0,5)
data.insert(7, "RRPrediction", preds_rotate)

print(data)

# plt.plot(xte)
plt.plot(yte, label=f"M{motor_num} Actual Fault")
plt.plot(preds_rotate, linestyle='dotted', linewidth='2', label=f"RRF Classifier")
plt.legend(loc="upper left")
plt.xlabel("Sampling")
plt.ylabel("Fault Classification")
# plt.axvline(x = 552, color = 'r', label = 'axvline - full height', linestyle='dotted')
plt.show()