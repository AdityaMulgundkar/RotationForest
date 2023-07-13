import matplotlib.pyplot as plt
import pickle

from sklearn.naive_bayes import GaussianNB
from RotationForest import *
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

aX, aY = [], []
Xdata, Ydata = np.asarray([]), np.asarray([])

# motor_choice = [1, 2, 3, 4, 5, 6]
# motor_choice = [1,6]
# motor_choice = [1,2,3,5]
motor_choice = [1,2]
# motor_choice = [2]
# motor_choice = [3,5]
# motor_choice = [1,3,4]

for motor in motor_choice:
    for j in range(1, 10):
        # data = pd.read_csv(f"dist/hexa-x/err20/m{motor}/test{j}.csv")
        data = pd.read_csv(f"dist/hexa-x/err10/m{motor}/test{j}.csv")
        for i in range(1, len(data.loc[:, "R"])):
            # Mall - R, R-1, RDes, P, P-1, PDes, Y
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
            if data.loc[:, "FaultIn"][i] == 2:
                ay = 1
            elif data.loc[:, "FaultIn"][i] == 5:
                ay = 3
            elif data.loc[:, "FaultIn"][i] == 6:
                ay = 4
            else:
                ay = data.loc[:, "FaultIn"][i]
            aX.append(ax)
            aY.append(ay)


Xdata, Ydata = np.asarray(aX).reshape(
    len(aX), len(ax)), np.asarray(aY).reshape(len(aY), 1)

xtr = Xdata
ytr = Ydata

# norm
mean = np.mean(xtr)
xmean = np.subtract(xtr, mean)
std = np.std(xmean)
xtr = np.divide(xmean, std)

yt = np.array([])
for i in range(0, len(xtr)):
    if (ytr[i] == [0]):
        yt = np.append(yt, int(0))
    else:
        yt = np.append(yt, int(1))
print(f"ytr: {ytr}")

pca = PCA(n_components=2)
xtr = pca.fit_transform(xtr)

Logistic = LogisticRegression()
Logistic.fit(xtr, ytr)

Gaussian = GaussianNB()
Gaussian.fit(xtr, ytr)

Ada = AdaBoostClassifier(n_estimators=100, random_state=0)
Ada.fit(xtr, ytr)

Random = RandomForestClassifier(n_estimators=200)
Random.fit(xtr, ytr)

Rotate = RotationForest(n_trees=2000, n_features=3)
Rotate.fit(xtr, ytr)

# Plot the decision boundary
ax = plt.subplot(2, 3, 1)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plot_decision_regions(xtr, yt.astype(np.int_), clf=Logistic, legend=2, filler_feature_values=[1], feature_index=[0, 1])
plt.title('Logistic Regression')

ax = plt.subplot(2, 3, 2)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plot_decision_regions(xtr, yt.astype(np.int_), clf=Gaussian, legend=2)
plt.title('Gaussian Naive Bayes')

ax = plt.subplot(2, 3, 3)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plot_decision_regions(xtr, yt.astype(np.int_), clf=Ada, legend=2)
plt.title('AdaBoost Classifier')

ax = plt.subplot(2, 3, 4)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plot_decision_regions(xtr, yt.astype(np.int_), clf=Random, legend=2)
plt.title('Random Forest Classifier')

ax = plt.subplot(2, 3, 5)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plot_decision_regions(xtr, yt.astype(np.int_), clf=Rotate, legend=2)
plt.title('Rotation Forest Classifier')

# Adding axes annotations
plt.show()
