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
# motor_choice = [1,2]
motor_choice = [3,5]

for motor in motor_choice:
    for j in range(1,10):
        # data = pd.read_csv(f"dist/hexa-x/err80/m{motor}/test{j}.csv")
        data = pd.read_csv(f"dist/hexa-x/err90/m{motor}/test{j}.csv")
        for i in range(1,len(data.loc[:, "R"])):
            # M35 forw - R, P, P-1, PDes, Y
            if i < 1:
                ax = (
                    data.loc[:,"R"][i],
                    data.loc[:,"P"][i],
                    data.loc[:,"P"][i],
                    data.loc[:,"PDes"][i],
                    data.loc[:,"Y"][i],
                    )
            else:
                ax = (
                    data.loc[:,"R"][i],
                    data.loc[:,"P"][i],
                    data.loc[:,"P"][i-1],
                    data.loc[:,"PDes"][i],
                    data.loc[:,"Y"][i],
                    )
            ay = data.loc[:,"FaultIn"][i]
            aX.append(ax)
            aY.append(ay) 

    
Xdata, Ydata = np.asarray(aX).reshape(len(aX),len(ax)), np.asarray(aY).reshape(len(aY),1)

xtr = Xdata
ytr = Ydata

Rotate = RotationForest(n_trees=200, n_features=3)
Rotate.fit(xtr, ytr)

import pickle
pickle.dump(Rotate, open("models/rfc-M35", 'wb'))