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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# from Word2vec_Model_Creation import
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)

    #assert function triggers an error if the folloing condition is false
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]



    # 5. Return a list of words
    return(words)


###################################################
## convert raw review text into words this time apply porter stemmer
def review_to_wordlist2(raw_review):
    # set up Porter Stemmer
    stemmer=Porter.PorterStemmer()

    # Clean up tag and markup using BeautifulSoup
    review_text=BeautifulSoup(raw_review).get_text()

    # Remove all non letters and replace with space
    letters_only=re.sub("[^a-zA-Z]", " ", review_text)

    # convert all to lower case
    lower_case=letters_only.lower()

    # tokenization
    words=lower_case.split()

    # construct set of english stopwords
    stops=set(stopwords.words("english"))

    # remove english stopwords using nltk's stopwords list
    meaningful_words=[w for w in words if not w in stops]

    # stem word tokens
    stemmed_words=[]
    for word in meaningful_words:
        stemmed_words.append(stemmer.stem(word))

    return (stemmed_words)


def load_data(pickle_file):
    load_file=open(pickle_file,'rb')
    data=cPickle.load(load_file)
    return  data


def create_data():
    train = pd.read_csv("./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test=pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3 )
    unlabeled_train=pd.read_csv("./data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
    y = train["sentiment"]

    print "Cleaning and parsing movie reviews...\n"

    print "cleaning training data"
    traindata = []
    for i in xrange( 0, len(train["review"])):
        traindata.append(" ".join(review_to_wordlist(train["review"][i], False)))

    print "cleaning unlabelled training data"
    ul_traindata=[]
    for i in xrange( 0, len(unlabeled_train["review"])):
        ul_traindata.append(" ".join(review_to_wordlist(unlabeled_train["review"][i], False)))

    print "cleaning test data"
    testdata=[]
    for i in xrange(0,len(test["review"])):
        testdata.append(" ".join(review_to_wordlist(test["review"][i], False)))

    print 'vectorizing... ',

    tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1,3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

    X_all = traindata + testdata + ul_traindata

    # X_all = traindata + testdata

    lentrain = len(traindata)
    lentest=len(testdata)

    print "fitting pipeline... ",
    tfv.fit(X_all)
    X_all = tfv.transform(X_all)

    print X_all.shape

    return X_all, lentrain, lentest, y

############################################################################################################################################
def train_and_eval_auc( model, train_x, train_y, test_x, test_y ):
    model.fit( train_x, train_y )
    p = model.predict_proba( test_x )
    auc = AUC( test_y, p[:,1] )
    return auc

def train_and_eval_auc2( model1, model2, train_x, train_y, test_x, test_y ):
    model1.fit( train_x, train_y )
    model2.fit( train_x, train_y )

    p1 = model1.predict_proba( test_x )*0.09
    p2=model2.predict_proba( test_x )*0.01
    p=p1+p2
    auc = AUC( test_y, p[:,1] )
    return auc

def train_and_eval_auc3( model1, model2, model3, train_x, train_y, test_x, test_y ):
    model1.fit( train_x, train_y )
    model2.fit( train_x, train_y )

    p1 = model1.predict_proba( test_x )*0.7
    p2=model2.predict_proba( test_x )*0.2
    p3=model3.predict_proba( test_x )*0.1

    p=p1+p2+p3
    auc = AUC( test_y, p[:,1] )
    return auc


if __name__ == '__main__':
    # Kappa Scorer
    # kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)


    # model = GridSearchCV(estimator=model, param_grid=param_grid, verbose=5,
    #                     refit=True, n_jobs=4, scoring=kappa_scorer)


    start_time=time.time()
    datapath='./pickledata/train_ultrain_test_ng3_mindf3_mf800k.dat'

    X_all, y, lentrain, lentest, testid=load_data(datapath)

    X = X_all[:lentrain]
    X_test = X_all[lentrain:(lentrain+lentest)]

    print "creating model"
    print "select K"


    # param_grid={'C':[1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75],}
    # param_grid={'C':[100, 115, 125, 135, 150],}

    # param_grid={'C':[25],}
    logistic = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                              fit_intercept=True, intercept_scaling=1.0,
                             class_weight=None, random_state=None)
    # bagging=BaggingClassifier(base_estimator=logistic, n_jobs=4, verbose=5, n_estimators=200)
    ada=AdaBoostClassifier(base_estimator=logistic, n_estimators=10)

    bnb=BernoulliNB(alpha=1.0)


    test_run=True
    selectK_value=255000
    SelectP_value=20
    logistic.C=50
    if not test_run:
        ch2=SelectKBest(chi2, k=selectK_value)
        X=ch2.fit_transform(X,y)
        X_test=ch2.transform(X_test)
        # model = GridSearchCV(estimator=model, param_grid=param_grid, verbose=0,
        #                     refit=True, n_jobs=4, scoring='roc_auc', cv=10)
        logistic.fit(X,y)
        bnb.fit(X,y)
        # print("Best score: %0.6f" % model.best_score_)
        # print("Best parameters set:")
        # best_parameters = model.best_estimator_.get_params()
        # for param_name in sorted(param_grid.keys()):
        #     print("\t%s: %r" % (param_name, best_parameters[param_name]))

        result1 = logistic.predict_proba(X_test)[:,1]
        result2=bnb.predict_proba(X_test)[:,1]
        result=result1*0.9+result2*0.1
        output = pd.DataFrame( data={"id":testid, "sentiment":result} )


        # Use pandas to write the comma-separated output file
        output.to_csv( "Logistic_Bnb.csv", index=False, quoting=3 )

        print "Wrote results to output document"

    else:
        print "performing test run"

        auc=[]
        rnd_state=np.random.RandomState(1234)
        for run in xrange(1, 11):
            train_i, test_i = train_test_split(np.arange(X.shape[0]), train_size = 0.8, random_state = rnd_state )
            train_features=X[train_i]
            test_features=X[test_i]
            y_train=y.ix[train_i]
            y_test=y.ix[test_i]

            ch2=SelectKBest(chi2, k=selectK_value)
            # ch2=SelectPercentile(chi2, 10)
            train_features=ch2.fit_transform(train_features,y_train)
            test_features=ch2.transform(test_features)
            score=train_and_eval_auc2(logistic, bnb, train_features, y_train, test_features, y_test)
            # score=train_and_eval_auc(ada, train_features, y_train, test_features, y_test)
            auc.append(score)
            print "AUC score for test run %i is %.6f" %(run, score)

        print "Mean logistic regression AUC is %.6f:" %np.mean(auc)

    end_time=time.time()
    duration=end_time-start_time
    print "it takes %.3f seconds"  %(duration)
