import os
# from KaggleWord2VecUtility import KaggleWord2VecUtility
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.stem.porter as Porter
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition, pipeline, metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import cPickle
from sklearn.metrics import roc_auc_score as AUC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss




def load_data(pickle_file):
    load_file=open(pickle_file,'rb')
    data=cPickle.load(load_file)
    return  data



############################################################################################################################################
def train_and_eval_auc( model, train_x, train_y, test_x, test_y ):
    model.fit( train_x, train_y )
    p = model.predict_proba( test_x )
    auc = AUC( test_y, p[:,1] )
    return auc

def train_and_eval_auc2( model1, model2, train_x, train_y, test_x, test_y ):
    model1.fit( train_x, train_y )
    model2.fit( train_x, train_y )

    p1 = model1.predict_proba( test_x )*0.85
    p2=model2.predict_proba( test_x )*0.15
    p=p1+p2
    auc = AUC( test_y, p[:,1] )
    return auc

def train_and_eval_auc3( model1, model2, model3, train_x, train_y, test_x, test_y ):
    model1.fit( train_x, train_y )
    model2.fit( train_x, train_y )
    model3.fit( train_x, train_y )

    w1=0.8075795
    w2=0.19129704
    w3=0.00112346

    p1 = model1.predict_proba( test_x )*w1
    p2=model2.predict_proba( test_x )*w2
    p3=model3.predict_proba( test_x )*w3


    p=p1+p2+p3

    auc = AUC( test_y, p[:,1])
    return auc


def SelectK_model(X, X_test, y):



    #print "creating SelectK model"
    logistic = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                              fit_intercept=True, intercept_scaling=1.0,
                             class_weight=None, random_state=None)

    # knn=KNeighborsClassifier(n_neighbors=2)
    bnb=BernoulliNB(alpha=1.0)
    # tree=DecisionTreeClassifier(max_depth=5)
    # rtree=RandomForestClassifier(n_estimators=50)
    # calibrated_rtree=
    # calibrated_logistic=CalibratedClassifierCV(logistic, method='isotonic', cv=10)
    #
    # calibrated_rtree=CalibratedClassifierCV(rtree, method='isotonic', cv= 5)
    calibrated_bnb=CalibratedClassifierCV(bnb, method='isotonic', cv= 5)
    # calibrated_knn=CalibratedClassifierCV(knn, method='isotonic', cv= 5)

    selectK_value=250000
    logistic.C=50
    ch2=SelectKBest(chi2, k=selectK_value)
    X=ch2.fit_transform(X,y)
    X_test=ch2.transform(X_test)
    # model = GridSearchCV(estimator=model, param_grid=param_grid, verbose=0,
    #                     refit=True, n_jobs=4, scoring='roc_auc', cv=10)
    logistic.fit(X,y)
    calibrated_bnb.fit(X,y)

    result1 = logistic.predict_proba(X_test)[:,1]
    result2=calibrated_bnb.predict_proba(X_test)[:,1]
    w1=0.9
    w2=0.1
    result=result1*w1+result2*w2
    return result


def SVD_model(X, X_test, y):
   #print "creating SVD model"
    logistic = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                              fit_intercept=True, intercept_scaling=1.0,
                             class_weight=None, random_state=None)
    logistic.fit(X,y)
    result = logistic.predict_proba(X_test)[:,1]
    return result


if __name__ == '__main__':

    start_time=time.time()

    print 'loading svd data'
    svd_datapath='./pickledata/tut_ng3_mindf3_mf800k_svd400.dat'
    svd_X, svd_y, svd_X_test, svd_testid=load_data(svd_datapath)

    print 'load selectK data'
    selectK_datapath='./pickledata/train_ultrain_test_ng3_mindf3_mf800k.dat'

    selectK_X_all, y, lentrain, lentest, testid=load_data(selectK_datapath)
    selectK_X = selectK_X_all[:lentrain]
    selectK_X_test = selectK_X_all[lentrain:(lentrain+lentest)]

    lm_datapath='./pickledata/tut_ng3_mindf3_mf800k_lemma.dat'
    lm_X_all, lm_y, lm_lentrain, lm_lentest, lm_testid=load_data(lm_datapath)
    lm_X = lm_X_all[:lm_lentrain]
    lm_X_test = lm_X_all[lm_lentrain:(lm_lentrain+lm_lentest)]



    test_run=True
    if not test_run:
        result1=SelectK_model(selectK_X, selectK_X_test, y)
        # result2=SVD_model(svd_X, svd_X_test,y)
        result2=SelectK_model(lm_X, lm_X_test, lm_y )
        result=result1*0.7+result2*0.3
        output = pd.DataFrame( data={"id":testid, "sentiment":result} )


        # Use pandas to write the comma-separated output file
        output.to_csv( "SelectK_LM.csv", index=False, quoting=3 )

        print "Wrote results to output document"

    else:

        print "performing test run"
        run=0
        auc=[]
        # result=np.array([])
        rnd_state=np.random.RandomState(1234)
        for run in xrange(1, 11):
            train_i, test_i = train_test_split(np.arange(svd_X.shape[0]), train_size = 0.8, random_state = rnd_state )
            selectK_train_features=selectK_X[train_i]
            selectK_test_features=selectK_X[test_i]

            lm_train_features=lm_X[train_i]
            lm_test_features=lm_X[test_i]

            svd_train_features=svd_X[train_i]
            svd_test_features=svd_X[test_i]
            y_train=y.ix[train_i]
            y_test=y.ix[test_i]

            result1=SelectK_model(selectK_train_features, selectK_test_features, y_train)
            result2=SelectK_model(lm_train_features, lm_test_features, y_train)

            result3=SVD_model(svd_train_features, svd_test_features, y_train)

            result=result1*0.7+result2*0.20+result3*0.1
        #
            score = AUC(y_test, result)
        #
            auc.append(score)
            print "AUC score for test run %i is %.6f" %(run, score)


        print "Mean logistic regression AUC is %.6f:" %np.mean(auc)

    end_time=time.time()
    duration=end_time-start_time
    print "it takes %.3f seconds"  %(duration)
1