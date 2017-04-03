__author__ = 'YXIE1'

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler





def xgb_learning_set_train_size(labels, train, test, num_rounds=1000, train_size=100):
    train_y = labels[0:train_size]
    test_y=labels[train_size: train.shape[0]]

    test=test[train_size:train.shape[0],:]
    train=train[0:train_size,:]


    preds=xgb_learning2(train_y, train, test)


    rmse_score = (np.sum((np.log1p(preds)-np.log1p(test_y))**2)/len(test_y))**0.5

    print "rmse score is %.6f" %rmse_score

    return  preds



def xgb_learning(labels, train, test, num_rounds=1000):
    label_log = np.log1p(labels)

    # fit a random forest model
    params = {}
    params["objective"] = "reg:linear"
    # params["eta"] = 0.05
    params["eta"] = 0.1

    params["min_child_weight"] = 1
    params["subsample"] = 1
    # params["subsample"] = 0.5

    params["gamma"]=0.1 #0.28
    params["colsample_bytree"] = 1
    params["silent"] = 1
    params["max_depth"] = 10
    # params["seed"]=3


    plst = list(params.items())

    xgtrain = xgb.DMatrix(train, label=label_log)
    xgtest = xgb.DMatrix(test)

    # num_rounds = 1000
    # model = xgb.train(plst, xgtrain, num_rounds)
    # preds = model.predict(xgtest)

    model = xgb.train(plst, xgtrain, num_rounds)
    preds1 = model.predict(xgtest)
    preds = np.expm1(preds1)



    # rmse_score = (np.sum((np.log1p(preds)-np.log1p(y))**2)/len(npdata))**0.5
    # print "rmse score is %.6f" %rmse_score

    return  preds


def xgb_learning2(labels, train, test, num_rounds=1000):
    label_log = np.log1p(labels)

    # fit a random forest model
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.1
    params["min_child_weight"] = 1
    params["subsample"] = 1

    params["gamma"]=0.1
    params["colsample_bytree"] = 1
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 10



    plst = list(params.items())

    xgtrain = xgb.DMatrix(train, label=label_log)
    xgtest = xgb.DMatrix(test)

    # num_rounds = 1000
    # model = xgb.train(plst, xgtrain, num_rounds)
    # preds = model.predict(xgtest)

    model = xgb.train(plst, xgtrain, num_rounds)
    preds1 = model.predict(xgtest)
    #
    # num_rounds = 2000
    # model = xgb.train(plst, xgtrain, num_rounds)
    # preds2 = model.predict(xgtest)


    # print ('model 2')
    num_rounds = 3000
    model = xgb.train(plst, xgtrain, num_rounds)
    preds3 = model.predict(xgtest)

    # preds = np.expm1(preds1)
    # preds = (np.expm1(preds1)+ np.expm1(preds2))/2
    preds = np.expm1((preds1+preds3)/2)

    return  preds



# the following model is not very successful, more testing required
def xgb_learning3(labels, train, test):
    label_log = np.log1p(labels)

    #benchmark param xgbooost2
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.02
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.6
    params["scale_pos_weight"] = 0.8
    params["silent"] = 1
    params["max_depth"] = 8
    params["max_delta_step"]=2
    params["seed"]=3

    plst = list(params.items())

    xgtrain = xgb.DMatrix(train, label=label_log)
    xgtest = xgb.DMatrix(test)

    #print ('model 1')
    num_rounds = 2000
    model = xgb.train(plst, xgtrain, num_rounds)
    preds1 = model.predict(xgtest)

    # print ('model 2')
    num_rounds = 3000
    model = xgb.train(plst, xgtrain, num_rounds)
    preds2 = model.predict(xgtest)

    # print ('model 4')
    num_rounds = 4000
    model = xgb.train(plst, xgtrain, num_rounds)
    preds4 = model.predict(xgtest)

    # print('model 3: power 1/16 4000')
    label_log = np.power(labels,1./16)

    xgtrain = xgb.DMatrix(train, label=label_log)
    xgtest = xgb.DMatrix(test)
    num_rounds = 4000


    model = xgb.train(plst, xgtrain, num_rounds)
    preds3 = model.predict(xgtest)

    preds = (np.expm1( (preds1+preds2+preds4)/3)+np.power(preds3,16))/2
    return  preds

def linear_learning(labels, train, test):
    # label_log=np.log1p(labels)
    linear=LinearRegression()
    model=linear.fit(train, labels)
    preds=model.predict(test)
    # preds=np.expm1(preds1)
    return  preds, model


if __name__ == '__main__':

    file_name='./product_data_model_training.xlsx'
    xl_file = pd.ExcelFile(file_name)
    data=xl_file.parse('training_data_P199')
    train_size=177

    # data=xl_file.parse('training_data_P169')
    # train_size=152

    # data=xl_file.parse('training_data_20150815')
    # train_size=100

    # data=xl_file.parse('training_data_fac1')
    # train_size=140


    test_run=True
    test_round=50



    # data['Threading']=data['Threading']/data['Surfacing Process Fee']
    # data['Sandblasting Fee']=data['Sandblasting Fee']/data['Surfacing Process Fee']
    # data['Plating Fee']=data['Plating Fee']/data['Surfacing Process Fee']
    # data['Polishing Fee']=data['Polishing Fee']/data['Surfacing Process Fee']

    # data=data.drop(data.index[[0,40]])

    #Drop statistical outliner
    # data=data[data["PID"]!='P022']
    data=data[data["PID"]!='P088']
    data=data[data["PID"]!='P140']
    data=data[data["PID"]!='P124']
    data=data[data["PID"]!='P166']
    data=data[data["PID"]!='P167']
    data=data[data["PID"]!='P168']
    data=data[data["PID"]!='P169']
    #


    # data['Area']=np.log1p(data['Area'])
    # data['Volume']=np.log1p(data['Volume'])


    data['VARatio']=data['Area']/data['Volume']
    data['VASum']=data['Area']+data['Volume']
    data['VAProd']=data['Area']*data['Volume']

    data['V2A2']=(data['Area']**2)*(data['Volume']**2)

    # data['MP']=data['Material']+data['Process']



    data.fillna(0,inplace=True)

    cleandata=data.drop(['PID', #'Material',
                         #'Process',
                          'Threading',
                        'Sandblasting Fee',
                        'Plating Fee',
                         'Polishing Fee',
                         'Surfacing Process Fee',
                         #'Total Handling Fee'
                         # 'Area',
                         # 'Volume',
                         'Wax Model Fee',
                         'Raw Material Fee' ,
                         'Molding Fee',
                          'Finetuning Fee',
                          'Pu Fee',
                          'Unit Price'
                         ], axis=1)







    lbl_M=preprocessing.LabelEncoder()
    lbl_M.fit(list(cleandata['Material']))
    cleandata['Material']=lbl_M.transform(cleandata['Material'])

    lbl_P=preprocessing.LabelEncoder()
    lbl_P.fit(list(cleandata['Process']))
    cleandata['Process']=lbl_P.transform(cleandata['Process'])

    # lbl_MP=preprocessing.LabelEncoder()
    # lbl_MP.fit(list(cleandata['MP']))
    # cleandata['MP']=lbl_MP.transform(cleandata['MP'])
    #

    cleandata['PARatio']=cleandata['Process']/cleandata['Area']
    cleandata['PAProd']=cleandata['Process']*cleandata['Area']


    # cleandata['PARatio']=cleandata['MP']/cleandata['Area']
    # cleandata['PAProd']=cleandata['MP']*cleandata['Area']

    clean_cols=['Material',
                'Process',
                # 'MP',
                'Volume',
                'Area',
                'VARatio',
                'VASum',
                'VAProd',
                'V2A2',
                'PAProd',
                'PARatio',
                'Total Handling Fee']

    cleandata=cleandata[clean_cols]
    npdata=np.array(cleandata)

    # y=np.array(cleandata["Unite Price"])
    y=np.array(cleandata["Total Handling Fee"])

    # y=np.float32(npdata[:,npdata.shape[1]-1])
    train=npdata[:,0:(npdata.shape[1]-1)]
    #testing learning with linear regression
    # preds, model=linear_learning(y, train, train)

    # ch2=SelectKBest(chi2, k=7)
    # train=ch2.fit_transform(train,y)



    if test_run:
        print ("perform cross validation")
        rmse=[]
        rnd_state=np.random.RandomState(1234)
        for run in range(0, test_round):
            # scaler=StandardScaler()
            # scaler=MinMaxScaler()
            train_i, test_i = train_test_split(np.arange(train.shape[0]), train_size = 0.8, random_state = rnd_state )
            tr_train=train[train_i]
            tr_test=train[test_i]
            tr_train_y=y[train_i]
            tr_test_y=y[test_i]

            # tr_train=scaler.fit_transform(tr_train)
            # tr_test=scaler.transform(tr_test)

            tr_preds=xgb_learning(tr_train_y, tr_train, tr_test)
            # tr_preds=xgb_learning3(tr_train_y, tr_train, tr_test)


            rmse_score = (np.sum((np.log1p(tr_preds)-np.log1p(tr_test_y))**2)/len(test_i))**0.5
            # rmse_score = (np.sum(tr_preds-tr_test_y)**2)/len(test_i)**0.5
            rmse.append(rmse_score)
            print ("logistic regression score for test run %i is %.6f" %(run, rmse_score))
        print ("Mean logistic regression RMSE is %.6f:" %np.mean(rmse))
        print ("standard deviation of RMSE in test run is %.5f:" %np.array(rmse).std())
    else:



        preds=xgb_learning_set_train_size(y, train, train, train_size=train_size)

        delta=(preds-y[train_size:len(y)])/y[train_size:len(y)]
        abs_delta=abs((preds-y[train_size:len(y)])/y[train_size:len(y)])

        preds=np.concatenate((np.zeros(train_size),preds))
        delta=np.concatenate((np.zeros(train_size),delta))
        abs_delta=np.concatenate((np.zeros(train_size),abs_delta))






        data['Predict Price']=pd.Series(preds, index=data.index)
        data['Delta']=pd.Series(delta, index=data.index)
        data['Abs_Delta']=pd.Series(abs_delta, index=data.index)

        data=data.drop(['Threading',
                        'Sandblasting Fee',
                        'Plating Fee',
                         'Polishing Fee',
                         'Surfacing Process Fee',
                         #'Total Handling Fee'
                         # 'Area',
                         # 'Volume',
                         'Wax Model Fee',
                         'Raw Material Fee' ,
                         'Molding Fee',
                          'Finetuning Fee',
                          'Pu Fee',
                          'Unit Price'
                         ], axis=1)



        # data.to_csv('predict_test.csv', index=False)
        # rmse_score = (np.sum((np.log1p(preds)-np.log1p(y))**2)/len(npdata))**0.5
        #
        # print "rmse score is %.6f" %rmse_score

        writer = pd.ExcelWriter('predict_test.xlsx')
        data.to_excel(writer, 'predict_test')
        writer.save()