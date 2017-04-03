__author__ = 'xie'
import pandas as pd
import numpy as np
import time
import xgboost as xgb
import random
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

#'convert reflectivity (dbz) to mm/hr
def marshall_palmer(dbz):
    mp=((10**(dbz/10))/200)**0.625
    return mp


def mae(preds, dm):
    # print ("hello")
    labels=dm.get_label()
    # err=mean_absolute_error(preds, labels)

    elab=np.expm1(labels)
    epreds=np.expm1(preds)
    err=mean_absolute_error(elab, epreds)
    # err=np.mean(np.abs(epreds-elab))

    # err=float(err)
    # print(type(err))
    # print(elab, epreds)
    return'error', err



if __name__ == '__main__':

    train_file='./data/qiqi_traindf.csv'
    test_file='./data/qiqi_testdf.csv'

    train_df=pd.read_csv(train_file)
    test_file='./data/qiqi_testdf.csv'
    test_df=pd.read_csv(test_file)

    submission_sample_file='./data/sample_solution.csv'
    submission_sample=pd.read_csv(submission_sample_file)

    start_time=time.time()
    expected_cutoff=70

    # train=train_df.drop(['Unnamed: 0', 'Id', 'Expected'], axis=1).head(100000)
    # train=train_df.drop(['Unnamed: 0', 'Id'], axis=1).head(10000)

    train=train_df.drop(['Unnamed: 0', 'Id'], axis=1)

    train=train[train['Expected']<expected_cutoff]

    train=train[train['Ref_mean']!=0]



    train['mp_min']=marshall_palmer(pd.Series(train['Ref_min']))
    train['mp_mean']=marshall_palmer(pd.Series(train['Ref_mean']))
    train['mp_max']=marshall_palmer(pd.Series(train['Ref_max']))
    train['mp_sd']=marshall_palmer(pd.Series(train['Ref_sd']))

    label_train=train['Expected']

    label_tr_log = np.log1p(label_train)

    dtrain_train = xgb.DMatrix(train, label=label_tr_log)

    #xgb setting starts here
    params = {}
    params["objective"] = "reg:linear"
    params["booster"]="gbtree"
    params["eta"] = 0.2
    params["max_depth"] = 16
    params["subsample"] = 0.5
    params["min_child_weight"] = 1
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    params["nthread"]=4
    plst = list(params.items())

    watchlist=[(dtrain_train, 'train')]
    # watchlist=[(dtrain_train, 'train')]

    # model = xgb.train(plst, dtrain,num_boost_round=100,
    #                   feval=mae)

    model = xgb.train(plst, dtrain_train,num_boost_round=1000, early_stopping_rounds=100,
                      evals=watchlist,
                      feval=mae)


    test=test_df.drop(['Unnamed: 0','Id'], axis=1)
    test['mp_min']=marshall_palmer(pd.Series(test['Ref_min']))
    test['mp_mean']=marshall_palmer(pd.Series(test['Ref_mean']))
    test['mp_max']=marshall_palmer(pd.Series(test['Ref_max']))
    test['mp_sd']=marshall_palmer(pd.Series(test['Ref_sd']))

    # test=test[test['Ref_mean']>0].head(100000)
    # label=np.log1p(train_df['Expected'])
    # test_eval=test[test['Ref_mean']!=0]
    dtest = xgb.DMatrix(test)

    preds1 = model.predict(dtest)
    preds = np.expm1(preds1)
    end_time=time.time()
    duration=end_time-start_time
    print("it takes %.3f seconds"  %(duration))



    submission=submission_sample.copy()
    # submission['Expected']=0
    # submission.ix[test_eval.index]['Expected']=10

    submission['Expected']=preds

    # test[test_eval.index]=preds
    print("finished")

    submission.to_csv('submission_yx_full.csv', index=False)

