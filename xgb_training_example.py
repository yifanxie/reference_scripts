__author__ = 'xie'

#!/usr/bin/python

'''
Based on https://www.kaggle.com/justdoit/rossmann-store-sales/xgboost-in-python-with-rmspe/code
Score : 0.101423
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV,  RandomizedSearchCV
import xgboost as xgb

# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = (y != 0)
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.expm1(y)
    yhat = np.expm1(yhat)
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

def toBinary(featureCol, df):
    values = set(df[featureCol].unique())
    newCol = [featureCol + '_' + val for val in values]
    for val in values:
        df[featureCol + '_' + val] = df[featureCol].map(lambda x: 1 if x == val else 0)
    return newCol

# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    #
    #features.append('StateHoliday')
    #data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    #data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    #data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    #data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)

    # features.append('StoreType')
    for x in ['a', 'b', 'c', 'd']:
        features.append('StoreType_' + x)
        data['StoreType_' + x] = data['StoreType'].map(lambda y: 1 if y == x else 0)

    newCol = toBinary('Assortment', data)
    features += newCol

## Start of main script

print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
store = pd.read_csv("../input/store.csv")

print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)

print('training data processed')

params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }
num_boost_round = 300

print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.01)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)
dtest = xgb.DMatrix(test[features])

watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100, \
  feval=rmspe_xg, verbose_eval=True)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
indices = yhat < 0
yhat[indices] = 0
error = rmspe(np.expm1(yhat), X_valid.Sales.values)
print('RMSPE: {:.6f}'.format(error))



print("Make predictions on the test set")
test_probs = gbm.predict(dtest)
indices = test_probs < 0
test_probs[indices] = 0
# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales':  np.expm1(test_probs)})
result.to_csv("xgboost_8_submission.csv", index=False)