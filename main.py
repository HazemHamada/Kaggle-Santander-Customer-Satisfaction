import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import lightgbm as lgb
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score

import gc
import warnings


warnings.filterwarnings("ignore")
warnings.filterwarnings("error", message=".*check_inverse*.", category=UserWarning, append=False)

data = pd.read_csv('train.csv')

data = data.fillna(data.mean())
data = data.drop_duplicates()

label = data.TARGET
data = data.drop('TARGET', axis=1)
"""
clf = Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(penalty='l1', loss='squared_hinge', dual=False))),
      ('classification', RandomForestClassifier())
    ])
clf.fit(data, label)
"""
#sfm = SelectFromModel(ExtraTreesClassifier(n_estimators=50))
sfm = SelectFromModel(LinearSVC(penalty='l1', loss='squared_hinge', dual=False))
data = sfm.fit_transform(data, label)
data = preprocessing.scale(data)
transformer = FunctionTransformer(np.log1p, validate=True)
transformer.transform(data)
data = preprocessing.normalize(data, norm='l2')

TrainX, TestX, TrainY, TestY = train_test_split(data, label, test_size=0.2, random_state=1)


def train_lgb(Xtrain, Ytrain, Xvalid, Yvalid):
    dtrain = lgb.Dataset(Xtrain, label=Ytrain)
    dvalid = lgb.Dataset(Xvalid, label=Yvalid)
    param = {'num_leaves': 250, 'objective': 'binary',
             'metric': 'auc'}
    print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                    early_stopping_rounds=10, verbose_eval=False)
    valid_pred = bst.predict(Xvalid)
    valid_score = metrics.roc_auc_score(Yvalid, valid_pred)
    print(f"Validation AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(Yvalid, valid_pred.round())
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return bst


bstm = train_lgb(TrainX,TrainY, TestX,  TestY)

# Validation AUC score: 0.8189
# Accuracy: 96.03%


def trainSVM(VTrainX,VTrainY, VTestX,  VTestY):
    clf = svm.SVC()
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    print(f"Validation2 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy2: %.2f%%" % (accuracy * 100.0))
    return clf


svmm = trainSVM(TrainX,TrainY, TestX,  TestY)

# Validation2 AUC score: 0.5000
# Accuracy2: 96.03%


def trainMLPClassifier(VTrainX,VTrainY, VTestX,  VTestY):
    clf = MLPClassifier()
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    print(f"Validation3 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy3: %.2f%%" % (accuracy * 100.0))
    return clf


mlpm = trainMLPClassifier(TrainX,TrainY, TestX,  TestY)

# Validation3 AUC score: 0.5036
# Accuracy3: 95.96%


def trainRFC(VTrainX,VTrainY, VTestX,  VTestY):
    clf = RandomForestClassifier()
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    print(f"Validation4 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy4: %.2f%%" % (accuracy * 100.0))
    return clf


RFC = trainRFC(TrainX,TrainY, TestX,  TestY)

# Validation4 AUC score: 0.5087
# Accuracy4: 95.71%

def trainXGBClassifier(VTrainX,VTrainY, VTestX,  VTestY):
    clf = XGBClassifier()
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    print(f"Validation5 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy5: %.2f%%" % (accuracy * 100.0))
    return clf


XGB = trainXGBClassifier(TrainX,TrainY, TestX,  TestY)

# Validation5 AUC score: 0.5012
# Accuracy5: 95.97%


def trainKNeighborsClassifier(VTrainX,VTrainY, VTestX,  VTestY):
    clf = KNeighborsClassifier()
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    print(f"Validation6 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy6: %.2f%%" % (accuracy * 100.0))
    return clf


KNC = trainKNeighborsClassifier(TrainX,TrainY, TestX,  TestY)

# Validation6 AUC score: 0.5122
# Accuracy6: 95.79%
