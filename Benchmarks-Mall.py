import warnings
from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
import pandas as pd

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings("ignore")

aX, aY = [], []
xtr, ytr = np.asarray([]), np.asarray([])
xte, yte = np.asarray([]), np.asarray([])

# motor_choice = [1]
motor_choice = [1, 2, 3, 4, 5, 6]

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
            ay = data.loc[:, "FaultIn"][i]
            aX.append(ax)
            aY.append(ay)
    
xtr, ytr = np.asarray(aX).reshape(
    len(aX), len(ax)), np.asarray(aY).reshape(len(aY), 1)
results_df = []

# motor_choice = [2]
motor_choice = [1, 2, 3, 4, 5, 6]

Logistic = LogisticRegression()
Gaussian = GaussianNB()
Ada = AdaBoostClassifier()
Random = RandomForestClassifier(n_estimators=200)
Rotate = RotationForest(n_trees=200, n_features=3)

Logistic.fit(xtr, ytr)
Gaussian.fit(xtr, ytr)
Ada.fit(xtr, ytr)
Random.fit(xtr, ytr)
Rotate.fit(xtr, ytr)

for motor in motor_choice:
    xte, yte = np.asarray([]), np.asarray([])
    aX, aY = [], []
    for j in range(1, 10):
        # data = pd.read_csv(f"dist/hexa-x/real-cases/m{motor}.csv")
        data = pd.read_csv(f"dist/hexa-x/err20/m{motor}/test{j}.csv")
        # data = pd.read_csv(f"dist/hexa-x/err10/m{motor}/test{j}.csv")
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
            
            ay = data.loc[:, "FaultIn"][i]
            # if ay != 0:
            #     aX.append(ax)
            #     aY.append(ay)
            aX.append(ax)
            aY.append(ay)

    xte, yte = np.asarray(aX).reshape(
        len(aX), len(ax)), np.asarray(aY).reshape(len(aY), 1)

    preds_logistic = Logistic.predict(xte)
    preds_gauss = Gaussian.predict(xte)
    preds_ada = Ada.predict(xte)
    preds_random = Random.predict(xte)
    preds_rotate = Rotate.predict(xte)
    obs = yte

    df = pd.DataFrame(data=[preds_rotate, preds_random,
                      obs, preds_logistic, preds_gauss, preds_ada]).T

    c1 = confusion_matrix(yte, preds_logistic, normalize='true', labels=[0, motor])
    c2 = confusion_matrix(yte, preds_gauss, normalize='true', labels=[0, motor])
    c3 = confusion_matrix(yte, preds_ada, normalize='true', labels=[0, motor])
    c4 = confusion_matrix(yte, preds_random, normalize='true', labels=[0, motor])
    c5 = confusion_matrix(yte, preds_rotate, normalize='true', labels=[0, motor])

    # print("\n")
    print([f"M{motor} T +ve", c1[1, 1], c2[1, 1], c3[1, 1], c4[1, 1], c5[1, 1]])
    # print([f"M{motor} F -ve", c1[1, 0], c2[1, 0], c3[1, 0], c4[1, 0], c5[1, 0]])
    # print([f"M{motor} F +ve", c1[0, 1], c2[0, 1], c3[0, 1], c4[0, 1], c5[0, 1]])