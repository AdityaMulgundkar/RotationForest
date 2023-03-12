import matplotlib.pyplot as plt
import pickle
from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_num = 5

aX_te, aY_te = [], []
Xdata_te, Ydata_te = np.asarray([]), np.asarray([])
# data = pd.read_csv(f"dist/hexa-x/real-cases/m{motor_num}.csv")
data = pd.read_csv(f"dist/hexa-x/err20/m{motor_num}/test10.csv")
# data = pd.read_csv(f"dist/hexa-x/err10/m{motor_num}/test10.csv")
for i in range(1, len(data.loc[:, "R"])):
    if i < 1:
        ax = (
            data.loc[:, "R"][i],
            data.loc[:, "P"][i],
            data.loc[:, "P"][i],
            data.loc[:, "PDes"][i],
            data.loc[:, "Y"][i],
        )
    else:
        ax = (
            data.loc[:, "R"][i],
            data.loc[:, "P"][i],
            data.loc[:, "P"][i-1],
            data.loc[:, "PDes"][i],
            data.loc[:, "Y"][i],
        )
    ay = data.loc[:, "FaultIn"][i]
    aX_te.append(ax)
    aY_te.append(ay)

xte, yte = np.asarray(aX_te).reshape(len(aX_te), len(
    ax)), np.asarray(aY_te).reshape(len(aY_te), 1)


Rotate = pickle.load(open("models/rfc-M35", 'rb'))
preds_rotate = Rotate.predict(xte)
print(preds_rotate)


# plt.plot(xte)
plt.plot(yte, label=f"M{motor_num} Actual Fault")
plt.plot(preds_rotate, linestyle='dotted', linewidth='2', label=f"RRF Classifier")
plt.legend(loc="upper left")
plt.xlabel("Sampling")
plt.ylabel("Fault Classification")
# plt.axvline(x = 552, color = 'r', label = 'axvline - full height', linestyle='dotted')
plt.show()
