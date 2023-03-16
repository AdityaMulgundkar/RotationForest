import matplotlib.pyplot as plt
import pickle
from RotationForest import *
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

motor_choice = [1,2,3,4,5,6]
# motor_choice = [1,2]
# motor_choice = [3,5]
# motor_choice = [1,3,4]

for motor in motor_choice:
    for j in range(1,10):
        # data = pd.read_csv(f"dist/hexa-x/err20/m{motor}/test{j}.csv")
        data = pd.read_csv(f"dist/hexa-x/err10/m{motor}/test{j}.csv")
        for i in range(1,len(data.loc[:, "R"])):
            # Mall - R, R-1, RDes, P, P-1, PDes, Y
            if i < 1:
                ax = (
                    data.loc[:,"R"][i],
                    data.loc[:,"RDes"][i],
                    )
            else:
                ax = (
                    data.loc[:,"R"][i],
                    data.loc[:,"RDes"][i],
                    )
            if data.loc[:,"FaultIn"][i] == 2:
                ay = 1
            elif data.loc[:,"FaultIn"][i] == 5:
                ay = 3
            elif data.loc[:,"FaultIn"][i] == 6:
                ay = 4
            else:
                ay = data.loc[:,"FaultIn"][i]
            aX.append(ax)
            aY.append(ay) 

    
Xdata, Ydata = np.asarray(aX).reshape(len(aX),len(ax)), np.asarray(aY).reshape(len(aY),1)

xtr = Xdata
ytr = Ydata

Linear = LogisticRegression()
Linear.fit(xtr, ytr)

Ada = AdaBoostClassifier(n_estimators=100, random_state=0)
Ada.fit(xtr, ytr)

Random = RandomForestClassifier(n_estimators=200)
Random.fit(xtr, ytr)

Rotate = RotationForest(n_trees=200, n_features=3)
Rotate.fit(xtr, ytr)

plot_colors = "ryb"

# Plot the decision boundary
ax = plt.subplot(2, 3, 1)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
DecisionBoundaryDisplay.from_estimator(
        Linear,
        xtr,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel="x",
        ylabel="y",
    )
# Plot the training points
for i, color in zip(range(1), plot_colors):
    idx = np.where(ytr == i)
    plt.scatter(
            xtr[idx, 0],
            xtr[idx, 1],
            label=ytr[i],
            s=15,
)

ax = plt.subplot(2, 3, 2)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
DecisionBoundaryDisplay.from_estimator(
        Ada,
        xtr,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel="x",
        ylabel="y",
    )
# Plot the training points
for i, color in zip(range(1), plot_colors):
    idx = np.where(ytr == i)
    plt.scatter(
            xtr[idx, 0],
            xtr[idx, 1],
            label=ytr[i],
            s=15,
)

ax = plt.subplot(2, 3, 3)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
DecisionBoundaryDisplay.from_estimator(
        Random,
        xtr,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel="x",
        ylabel="y",
    )
# Plot the training points
for i, color in zip(range(1), plot_colors):
    idx = np.where(ytr == i)
    plt.scatter(
            xtr[idx, 0],
            xtr[idx, 1],
            label=ytr[i],
            s=15,
)

plt.show()