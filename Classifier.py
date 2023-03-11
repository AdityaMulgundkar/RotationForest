from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_num = 1

import pandas as pd

# motor_choice = [1,2,3,4,5,6]
motor_choice = [1,2]

for motor in motor_choice:
    for j in range(1,10):
        data = pd.read_csv(f"dist/hexa-x/err90/m{motor}/test{j}.csv")
        # data = pd.read_csv(f"dist/hexa-x/err90/m{motor}/test{j}.csv")
        for i in range(1,len(data.loc[:, "R"])):
            # ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i]
            # ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
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
            aX.append(ax)
            aY.append(ay) 

    
Xdata, Ydata = np.asarray(aX).reshape(len(aX),len(ax)), np.asarray(aY).reshape(len(aY),1)

xtr = Xdata
ytr = Ydata

Rotate = RotationForest(n_trees=1000, n_features=3)
Rotate.fit(xtr, ytr)

import pickle
pickle.dump(Rotate, open("models/rfc-12-err90", 'wb'))