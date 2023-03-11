from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_num = 1

import pandas as pd

aX_te, aY_te = [], []
Xdata_te, Ydata_te = np.asarray([]), np.asarray([])
data = pd.read_csv(f"dist/hexa-x/err80/m{motor_num}/real-test.csv")
# data = pd.read_csv(f"dist/hexa-x/err80/m{motor_num}/test10.csv")
# data = pd.read_csv(f"dist/hexa-x/err90/m{motor_num}/test10.csv")
for i in range(1,len(data.loc[:, "R"])):
    # ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
    # ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i]
    # ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
    # if(i < 3 or i > len(data.loc[:, "R"])):
    #     ax = data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
    # else:
    #     ax = data.loc[:,"R"][i], data.loc[:,"R"][i-1], data.loc[:,"R"][i-2], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"P"][i-1],  data.loc[:,"P"][i-2], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i-1], data.loc[:,"Y"][i-2], data.loc[:,"YDes"][i]

    if i < 10 or i > len(data.loc[:, "R"]):
            ax = (
                data.loc[:, "R"][i],
                *([data.loc[:, "R"][i]] * 3),
                data.loc[:, "RDes"][i],
                data.loc[:, "P"][i],
                *([data.loc[:, "P"][i]] * 3),
                data.loc[:, "PDes"][i],
                data.loc[:, "Y"][i],
                *([data.loc[:, "Y"][i]] * 3),
                data.loc[:, "YDes"][i]
            )
    else:
            ax = (
                data.loc[:, "R"][i],
                data.loc[:, "R"][i-1],
                data.loc[:, "R"][i-2],
                data.loc[:, "R"][i-3],
                data.loc[:, "RDes"][i],
                data.loc[:, "P"][i],
                data.loc[:, "P"][i-1],
                data.loc[:, "P"][i-2],
                data.loc[:, "P"][i-3],
                data.loc[:, "PDes"][i],
                data.loc[:, "Y"][i],
                data.loc[:, "Y"][i-1],
                data.loc[:, "Y"][i-2],
                data.loc[:, "Y"][i-3],
                data.loc[:, "YDes"][i]
                )    
    ay = data.loc[:,"FaultIn"][i]
    aX_te.append(ax)
    aY_te.append(ay)

xte, yte = np.asarray(aX_te).reshape(len(aX_te),len(ax)), np.asarray(aY_te).reshape(len(aY_te),1)

import pickle

Rotate = pickle.load(open("models/rfc-12-err90", 'rb'))
preds_rotate = Rotate.predict(xte)
print(preds_rotate)

import matplotlib.pyplot as plt

# plt.plot(xte)
plt.plot(yte)
plt.plot(preds_rotate, linestyle='dotted')
# plt.axvline(x = 552, color = 'r', label = 'axvline - full height', linestyle='dotted')
plt.show()