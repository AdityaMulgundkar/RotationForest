import matplotlib.pyplot as plt
import pickle
from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import timeit


aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_num = 1

# TODO:
# Get current roll pitch yaw value from onboard sensor
# Attach timestamp to each sensor input packet (above)
# Use Rotate_All.predict on the single dataset (last fetched)
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
# for i in range(1, len(data.loc[:, "R"])):

for i in range(1, 2):
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

xte,

models = ["models/rfc-Mall","models/rfc-M12","models/rfc-M35","models/rfc-M46"]
results = []

for model in models:
    rint = random.randint(1,300)
    for i in range(rint, rint+1):
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

        xte,
    Rotate = pickle.load(open(model,'rb'))
    preds_Rotate = None
    results.append(timeit.timeit(stmt='preds_Rotate=Rotate.predict(xte)',number=100,globals=globals()))


# Rotate_All = pickle.load(open("models/rfc-Mall", 'rb'))
# #Mall,M12,35,46
# preds_Rotate_All=None
# def predTimeAll():
#     global preds_Rotate_All 
#     preds_Rotate_All = Rotate_All.predict(xte)

# Rotate_12 = pickle.load(open("models/rfc-M12", 'rb'))
# #Mall,M12,35,46
# preds_Rotate_12=None
# def predTime12():
#     global preds_Rotate_12
#     preds_Rotate_12 = Rotate_12.predict(xte)

# Rotate_35 = pickle.load(open("models/rfc-M35", 'rb'))
# #Mall,M12,35,46
# preds_Rotate_35=None
# def predTime35():
#     global preds_Rotate_35 
#     preds_Rotate_35 = Rotate_35.predict(xte)

# Rotate_46 = pickle.load(open("models/rfc-M46", 'rb'))
# #Mall,M12,35,46
# preds_Rotate_46=None
# def predTime46():
#     global preds_Rotate_46 
#     preds_Rotate_46 = Rotate_46.predict(xte)




# result = timeit.timeit(stmt='predTimeAll()', globals=globals(), number=5)

# print(preds_Rotate_All)
# preds_Rotate_All = np.insert(preds_Rotate_All, 0,5)

# data.insert(7, "RRPrediction", preds_Rotate_All)

# print(data)

# # plt.plot(xte)
# plt.plot(yte, label=f"M{motor_num} Actual Fault")
# plt.plot(preds_Rotate_All, linestyle='dotted', linewidth='2', label=f"RRF Classifier")
# plt.legend(loc="upper left")
# plt.xlabel("Sampling")
# plt.ylabel("Fault Classification")
# # plt.axvline(x = 552, color = 'r', label = 'axvline - full height', linestyle='dotted')
# plt.show()
