from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_num = 3

import pandas as pd
# for k in range(1,6):
#     for j in range(1,10):
#         data = pd.read_csv(f"dist/hexa-x/m{k}/test{j}.csv")
#         for i in range(1,len(data.loc[:, "R"])):
#             # if(i < 3 or i > len(data.loc[:, "R"])-3):
#             #     ax = data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
#             # else:
#             #     ax = data.loc[:,"R"][i-2], data.loc[:,"R"][i-1], data.loc[:,"R"][i], data.loc[:,"R"][i+1], data.loc[:,"R"][i+2], data.loc[:,"RDes"][i], data.loc[:,"P"][i-2], data.loc[:,"P"][i-1], data.loc[:,"P"][i], data.loc[:,"P"][i+1], data.loc[:,"P"][i+2], data.loc[:,"PDes"][i], data.loc[:,"Y"][i-2], data.loc[:,"Y"][i-1], data.loc[:,"Y"][i], data.loc[:,"Y"][i+1], data.loc[:,"Y"][i+2], data.loc[:,"YDes"][i]
#             # ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i]
#             ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
#             ay = data.loc[:,"FaultIn"][i]
#             aX.append(ax)
#             aY.append(ay)

for j in range(1,10):
    data = pd.read_csv(f"dist/hexa-x/m{motor_num}/test{j}.csv")
    for i in range(1,len(data.loc[:, "R"])):
        # if(i < 3 or i > len(data.loc[:, "R"])-3):
        #     ax = data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
        # else:
        #     ax = data.loc[:,"R"][i-2], data.loc[:,"R"][i-1], data.loc[:,"R"][i], data.loc[:,"R"][i+1], data.loc[:,"R"][i+2], data.loc[:,"RDes"][i], data.loc[:,"P"][i-2], data.loc[:,"P"][i-1], data.loc[:,"P"][i], data.loc[:,"P"][i+1], data.loc[:,"P"][i+2], data.loc[:,"PDes"][i], data.loc[:,"Y"][i-2], data.loc[:,"Y"][i-1], data.loc[:,"Y"][i], data.loc[:,"Y"][i+1], data.loc[:,"Y"][i+2], data.loc[:,"YDes"][i]
        # ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i]
        ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
        ay = data.loc[:,"FaultIn"][i]
        aX.append(ax)
        aY.append(ay)    
        
Xdata, Ydata = np.asarray(aX).reshape(len(aX),len(ax)), np.asarray(aY).reshape(len(aY),1)

# Xdata, Ydata = generate_data(3000)
print(f"Load x: {Xdata.shape}")
print(f"Load y: {Ydata.shape}")
# xtr, xte, ytr, yte = train_test_split(Xdata, Ydata, test_size=0.3)
xtr = Xdata
ytr = Ydata

aX_te, aY_te = [], []
Xdata_te, Ydata_te = np.asarray([]), np.asarray([])
data = pd.read_csv(f"dist/hexa-x/m{motor_num}/real-test.csv")
for i in range(1,len(data.loc[:, "R"])):
    # ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
    # ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i]
    ax = data.loc[:,"R"][i], data.loc[:,"RDes"][i], data.loc[:,"P"][i], data.loc[:,"PDes"][i], data.loc[:,"Y"][i], data.loc[:,"YDes"][i]
    ay = data.loc[:,"FaultIn"][i]
    aX_te.append(ax)
    aY_te.append(ay)

xte, yte = np.asarray(aX_te).reshape(len(aX_te),len(ax)), np.asarray(aY_te).reshape(len(aY_te),1)

# print(f"xte: {(xte)}")
# print(f"yte: {(yte)}")

Rotate = RotationForest(n_trees=200, n_features=3)
Random = RandomForestClassifier(n_estimators=200)
Linear = LogisticRegression()

Rotate.fit(xtr, ytr)
Random.fit(xtr, ytr)
Linear.fit(xtr, ytr)

preds_rotate = Rotate.predict(xte)
preds_random = Random.predict(xte)
preds_linear = Linear.predict(xte)
obs = yte

print("Validating...")

df = pd.DataFrame(data=[preds_rotate, preds_random, obs, preds_linear]).T
df['rotate'] = df[2].eq(df[0]).astype(int)
df['random'] = df[2].eq(df[1]).astype(int)
df['linear'] = df[2].eq(df[3]).astype(int)

print(f"Accuracy rotate: {df['rotate'].sum() / len(df) * 100}%")
print(f"Accuracy random: {df['random'].sum() / len(df) * 100}%")
print(f"Accuracy linear: {df['linear'].sum() / len(df) * 100}%")

import matplotlib.pyplot as plt

# plt.plot(xte)
plt.plot(yte)
plt.plot(preds_rotate, linestyle='dotted')
plt.axvline(x = 552, color = 'r', label = 'axvline - full height', linestyle='dotted')
plt.show()