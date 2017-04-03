__author__ = 'YXIE1'

import pandas as pd
import numpy as np
import time
import math

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import log_loss as LOSS
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.linear_model import ElasticNet

import xgboost as xgb

start_time=time.time()
test_run=True
total_file='./data/total_r.csv'
train_file="./data/train_r.csv"
test_file="./data/test_r.csv"
submission_file="./data/sample_submission.csv"
total_data=pd.read_csv(total_file)

train=pd.read_csv(train_file)
test=pd.read_csv(test_file)
submission=pd.read_csv(submission_file).sort(['id'], ascending=[1])

y=pd.DataFrame(train['fault_severity'])
y['id']=y.index
y=y[['id','fault_severity']]


X_train=train.drop(['id', 'fault_severity'], axis=1)
X_test=test.drop(['id'], axis=1)


scaler=StandardScaler().fit(X_train)

X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

bnb=BernoulliNB(alpha=1.0)
logistic = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                          fit_intercept=True, intercept_scaling=2.0,
                         class_weight='balanced', random_state=None, solver='lbfgs')

rtree=RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=8,
                             criterion="entropy",  min_samples_split=3, random_state=1234)

calibrated_model=CalibratedClassifierCV(logistic, method='isotonic', cv=5)


if not test_run:
    # logistic.fit(X_train, y)
    calibrated_model.fit(X_train, y['fault_severity'])
    pred1=calibrated_model.predict_proba(X_test)
    submission['predict_0']=pred1[:,0]
    submission['predict_1']=pred1[:,1]
    submission['predict_2']=pred1[:,2]
    submission.to_csv('./submission/latest_rf_submission.csv', index=False)
    # result1 = logistic.predict_proba(X_test)

else:
    print("performing test run")
    rnd_state=np.random.RandomState(1234)
    loss=[]
    for run in range(1,10):
        # train_i, test_i = train_test_split(np.arange(X_train.shape[0]), train_size = 0.8, random_state = rnd_state, stratify=y['fault_severity'].as_matrix() )
        train_i, test_i = train_test_split(np.arange(X_train.shape[0]), train_size = 0.8, random_state = rnd_state, stratify=y['fault_severity'].as_matrix() )
        train_features=X_train[train_i]
        test_features=X_train[test_i]
        y_train=y.ix[train_i]
        y_test=y.ix[test_i]
        # score=train_and_eval_auc(calibrated_bnb, train_features, y_train, test_features, y_test)
        calibrated_model.fit(train_features, y_train['fault_severity'])
        # enet.fit(train_features, y_train['fault_severity'])

        p=calibrated_model.predict_proba(test_features)
        # p=enet.predict_proba(test_features)
        ohe=OneHotEncoder()
        mlabel=ohe.fit_transform(y_test['fault_severity'].as_matrix().reshape(-1,1)).toarray()
        #
        score=LOSS(mlabel, p[:,])
        loss.append(score)
        print("LOSS score for test run %i is %.6f" %(run, score))

    print("Mean LOSS is %.6f:" %np.mean(loss))



end_time=time.time()
duration=end_time-start_time
print("it takes %.3f seconds to run the code"  %(duration))

