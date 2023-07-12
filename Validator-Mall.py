import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix
from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_num = 2
# fname = "log_416_2023-2-28-11-46-20"
# fname = "log_417_2023-2-28-11-47-54"
fname = "log_420_2023-2-28-12-00-24"
# motor_num = 5
# fname = "log_439_2023-2-28-13-08-12"
# fname = "log_436_2023-2-28-12-57-14"


last = 0

aX_te, aY_te = [], []
Xdata_te, Ydata_te = np.asarray([]), np.asarray([])
# data = pd.read_csv(f"dist/hexa-x/real-cases/m{motor_num}.csv")
data = pd.read_csv(f"dist/munjaal/m{motor_num}/{fname}.csv")
# data = pd.read_csv(f"dist/hexa-x/graphs/V-L1-paper/m{motor_num}.csv")
# data = pd.read_csv(f"dist/hexa-x/err20/m{motor_num}/test10.csv")
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
plt.plot(yte, label=f"M{motor_num} Actual Fault")
plt.plot(preds_rotate, linestyle='dotted', linewidth='2', label=f"RRF Classifier")
plt.legend(loc="upper left")
plt.xlabel("Sampling")
plt.ylabel("Fault Classification")
# plt.axvline(x = 552, color = 'r', label = 'axvline - full height', linestyle='dotted')
plt.show()
