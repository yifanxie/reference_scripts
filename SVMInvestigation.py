# import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# import PixelbyPixelClassifier as PP
import time
import pickle
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from random import randrange
from sklearn.learning_curve import learning_curve


## consider adding scikit-learn for feature scaling


case_sensitive_setting=True
keeprest=False
##modelfile="SVM_ucn.xml"


##fs=None
##fs="JC5A"
##fs="JC5A"
fs="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
chrmatch=False

############################################################################################################################################
def TrainPbyPModel(samples, responses, paramsin):
##  SVM method
    model = cv2.SVM()
    model.train(samples, responses, params=paramsin)
    return model



############################################################################################################################################
def SaveModel(model, method, datapath):
    svm_modelfile=datapath+modelfile

    if method==1:
        print "KNN model saving is not implemented"
    elif method==2:
        print "saving SVM model"
        model.save(svm_modelfile)



############################################################################################################################################
def LoadSVMModel(datapath):
    if case_sensitive_setting:
        modelfile="svm_cs.xml"
    else:
        modelfile="svm_ncs.xml"

    if fs!=None:
        modelfile="svm_fs.xml"

    svm_modelfile=datapath+modelfile
    model=cv2.SVM()
    model.load(svm_modelfile)
    return model


############################################################################################################################################
def CreateDataSets(data):
    np.random.shuffle(data)
    total_size=len(data)
    rate=0.7
    size_TR=int(total_size*rate)
    size_TE=int(total_size*(1.0-rate))

    training_set=data[:size_TR]
    testing_set=data[size_TR:total_size]

    return training_set, testing_set





############################################################################################################################################
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



############################################################################################################################################
if __name__ == '__main__':

##    datapath="./74KFntTrain/"
    datapath="./74KTrain/"
    datafile=datapath+"PbyP_Training.data"


##    segpath="./SegmentationTrain/"
##    datafile=segpath+"PbyP_Segmentation.data"

##    data=PP.LoadSegIndicators(datafile)
    data=PP.LoadIndicators(datafile, fs, keeprest, case_sensitive_setting, chrmatch)
## Best practice set of parameter for SVM
    params1= dict( kernel_type = cv2.SVM_POLY,
            svm_type = cv2.SVM_C_SVC,
            C = 100, gamma=5, degree=3, coef0=10)

## Experimental set of parameter for SVM
    params2= dict( kernel_type = cv2.SVM_POLY,
            svm_type = cv2.SVM_C_SVC,
            C = 5, gamma=0.5, degree=3, coef0=10)


##    clf = svm.SVC(C=5.0, kernel='poly', degree=3, gamma=0.5, coef0=10.0, shrinking=True,
##    probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
##    max_iter=-1, random_state=None)


##    params2 = dict( kernel_type = cv2.SVM_LINEAR,
##                    svm_type = cv2.SVM_C_SVC,
##                    C = 100)

    start_time=time.time()


    print "perform SVM training"





    labels=data[:,0]
    features=data[:,1:]

    seed =randrange(100)


    trainfeatures, testfeatures, trainlabels, testlabels=train_test_split(features, labels,
                                                                          test_size=0.2,
                                                                          random_state=seed)


##    orgtestdata=PP.LoadIndicators(testdatafile, fs, keeprest, case_sensitive_setting)
##
##    testdata1, testdata2=CreateDataSets(orgtestdata)


##    trainlabels=data[:,0]
##    trainfeatures=data[:,1:]
##    testlabels=testdata[:,0]
##    testfeatures=testdata[:,1:]
##



## ******************* Feature Scaling *******************
    print "performing feature scaling"
    min_max_scaler=preprocessing.MinMaxScaler()
    trainfeatures_fs=min_max_scaler.fit_transform(trainfeatures)
    testfeatures_fs=min_max_scaler.transform(testfeatures)


## ******************* OpenCV SVM *******************
    print "training CV SVM model"

    model0=TrainPbyPModel(trainfeatures_fs,trainlabels, params1)

####        print "performing predictions"
    results=model0.predict_all(testfeatures_fs)
    results=results.ravel()
    testerror=float(len(testlabels)-np.sum(testlabels==results))/float(len(testlabels))
    print"error rate with SVM 1  is %.4f" %testerror

##    SaveModel(model0,2, statmodelpath)



## ******************* Scikit-learning SVM *******************
##    clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=10.0, degree=3,
##    gamma=5, kernel='poly', max_iter=-1, probability=False, random_state=None,
##    shrinking=True, tol=0.001, verbose=False)

##    clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True,
##    probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
##    max_iter=-1, random_state=None)

##    clf=svm.LinearSVC(C=5.0, class_weight=None, dual=True, fit_intercept=True,
##    intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
##    random_state=None, tol=0.0001, verbose=0)



    print "training SVM model"

    clf = svm.SVC(C=5.0, kernel='poly', degree=3, gamma=0.5, coef0=10.0, shrinking=True,
    probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    max_iter=-1, random_state=None)


    clf.fit(trainfeatures_fs, trainlabels)
    results2=clf.predict(testfeatures_fs)

    results2=results2.ravel()
    testerror2=float(len(testlabels)-np.sum(testlabels==results2))/float(len(testlabels))
    print"error rate with SVM 2 is %.4f" %testerror2







## ******************* Learning Curve Ploting*******************
print "performing feature scaling for learning curve"
min_max_scaler=preprocessing.MinMaxScaler()
features_fs=min_max_scaler.fit_transform(features)
cv = cross_validation.ShuffleSplit(labels.shape[0], n_iter=10,
                                   test_size=0.2, random_state=0)



title = "Learning Curves (SVM scikit-learn)"
clf_ls = svm.SVC(C=5.0, kernel='poly', degree=3, gamma=0.5, coef0=10.0, shrinking=True,
probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
max_iter=-1, random_state=None)

plot_learning_curve(clf_ls, title, features_fs, labels, (0.7, 1.01), cv=cv, n_jobs=4)
plt.show()

## ******************* Scikit-learning RBM + SVM *******************
##    print "train RBM+SVM model"
##
####    trainfeatures = (trainfeatures - np.min(trainfeatures, 0)) / (np.max(trainfeatures, 0) + 0.0001)  # 0-1 scaling
##
##
##
##    rbm = BernoulliRBM(random_state=0, verbose=True)
##    classifier = Pipeline(steps=[('rbm', rbm), ('svm', clf)])
##    rbm.learning_rate = 0.06
##    rbm.n_iter = 20
##
##    # More components tend to give better prediction performance, but larger
##    # fitting time
##    rbm.n_components = 400
##
##    classifier.fit(trainfeatures_fs, trainlabels)
##    results3=classifier.predict(testfeatures_fs)
##
##    results3=results3.ravel()
##    testerror3=float(len(testlabels)-np.sum(testlabels==results3))/float(len(testlabels))
##    print"error rate with SVM 3 is %.4f" %testerror3
##
##
##
##    end_time=time.time()
##    duration=end_time-start_time
##    print "it takes %.3f seconds"  %(duration)
