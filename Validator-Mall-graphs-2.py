import matplotlib.pyplot as plt
import pickle
from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_num = 3

motors = [1, 3, 6]
# motors = [3]
last = 0

for motor_num in motors:
    aX_te, aY_te = [], []
    
    Xdata_te, Ydata_te = np.asarray([]), np.asarray([])
    data = pd.read_csv(f"dist/hexa-x/real-cases/m{motor_num}.csv")
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

        if ay == 0:
            last = i

        if i - last < 10:
            aX_te.append(ax)
            aY_te.append(ay)

    xte, yte = np.asarray(aX_te).reshape(len(aX_te), len(
        ax)), np.asarray(aY_te).reshape(len(aY_te), 1)
    print(f"last {last}")
    print(xte.shape, yte.shape)

    Rotate = pickle.load(open("models/rfc-Mall", 'rb'))
    preds_rotate = Rotate.predict(xte)
    # print(preds_rotate.shape)
    data.to_csv(f"dist/hexa-x/graphs/V-L1-paper/m{motor_num}.csv", sep=',', index=False)
    # data.to_csv(f"dist/hexa-x/graphs/V-L1-paper/m{motor_num}.csv", sep='\t')