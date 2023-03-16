from RotationForest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

aX, aY = [], []
xtr, ytr = np.asarray([]), np.asarray([])
xte, yte = np.asarray([]), np.asarray([])

motor_num = 1

import pandas as pd

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
                    data.loc[:,"R"][i],
                    data.loc[:,"RDes"][i],
                    data.loc[:,"P"][i],
                    data.loc[:,"P"][i],
                    data.loc[:,"PDes"][i],
                    data.loc[:,"Y"][i],
                    data.loc[:,"Y"][i],
                    data.loc[:,"YDes"][i],
                    )
            else:
                ax = (
                    data.loc[:,"R"][i],
                    data.loc[:,"R"][i-1],
                    data.loc[:,"RDes"][i],
                    data.loc[:,"P"][i],
                    data.loc[:,"P"][i-1],
                    data.loc[:,"PDes"][i],
                    data.loc[:,"Y"][i],
                    data.loc[:,"Y"][i-1],
                    data.loc[:,"YDes"][i],
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
    
xtr, ytr = np.asarray(aX).reshape(len(aX),len(ax)), np.asarray(aY).reshape(len(aY),1)
results_df = []

for motor in motor_choice:
    xte, yte = np.asarray([]), np.asarray([])
    for j in range(1,10):
        data = pd.read_csv(f"dist/hexa-x/err20/m{motor}/test{j}.csv")
        # data = pd.read_csv(f"dist/hexa-x/err10/m{motor}/test{j}.csv")
        for i in range(1,len(data.loc[:, "R"])):
            # Mall - R, R-1, RDes, P, P-1, PDes, Y
            if i < 1:
                ax = (
                    data.loc[:,"R"][i],
                    data.loc[:,"R"][i],
                    data.loc[:,"RDes"][i],
                    data.loc[:,"P"][i],
                    data.loc[:,"P"][i],
                    data.loc[:,"PDes"][i],
                    data.loc[:,"Y"][i],
                    data.loc[:,"Y"][i],
                    data.loc[:,"YDes"][i],
                    )
            else:
                ax = (
                    data.loc[:,"R"][i],
                    data.loc[:,"R"][i-1],
                    data.loc[:,"RDes"][i],
                    data.loc[:,"P"][i],
                    data.loc[:,"P"][i-1],
                    data.loc[:,"PDes"][i],
                    data.loc[:,"Y"][i],
                    data.loc[:,"Y"][i-1],
                    data.loc[:,"YDes"][i],
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
    
    xte, yte = np.asarray(aX).reshape(len(aX),len(ax)), np.asarray(aY).reshape(len(aY),1)

    Linear = LogisticRegression()
    Gaussian = GaussianNB()
    Ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    Random = RandomForestClassifier(n_estimators=200)
    Rotate = RotationForest(n_trees=200, n_features=3)

    Linear.fit(xtr, ytr)
    Gaussian.fit(xtr, ytr)
    Ada.fit(xtr, ytr)
    Random.fit(xtr, ytr)
    Rotate.fit(xtr, ytr)

    preds_linear = Linear.predict(xte)
    preds_gauss = Gaussian.predict(xte)
    preds_ada = Ada.predict(xte)
    preds_random = Random.predict(xte)
    preds_rotate = Rotate.predict(xte)
    obs = yte

    df = pd.DataFrame(data=[preds_rotate, preds_random, obs, preds_linear, preds_gauss, preds_ada]).T
    # df = pd.DataFrame(data=[obs, preds_linear, preds_random, preds_rotate]).T
    a1 = df[2].eq(df[3]).astype(int).sum() / len(df)
    a2 = df[2].eq(df[4]).astype(int).sum() / len(df)
    a3 = df[2].eq(df[5]).astype(int).sum() / len(df)
    a4 = df[2].eq(df[1]).astype(int).sum() / len(df)
    a5 = df[2].eq(df[0]).astype(int).sum() / len(df)

    # results_df.append([df['linear'], df['gauss'], df['ada'], df['random'], df['rotate']])
    results_df.append([f"M{motor}",a1, a2, a3, a4, a5])

    # print(f"Linear: {df['linear'].sum() / len(df)}")
    # print(f"Gauss: {df['gauss'].sum() / len(df)}")
    # print(f"AdaBoost: {df['gauss'].sum() / len(df)}")
    # print(f"Random: {df['random'].sum() / len(df)}")
    # print(f"Rotate: {df['rotate'].sum() / len(df)}")
print(f"Results: {results_df}")