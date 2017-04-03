import os
# from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.stem.porter as Porter
# from Word2vec_Model_Creation import

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

    return stemmed_words


# train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
#                 delimiter="\t", quoting=3)
# test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
#                quoting=3 )
train = pd.read_csv("./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test=pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3 )

y = train["sentiment"]


print "Cleaning and parsing movie reviews...\n"      
traindata = []
for i in xrange( 0, len(train["review"])):
    traindata.append(" ".join(review_to_wordlist2(train["review"][i])))
    # traindata.append(" ".join(review_to_wordlist(train["review"][i], False)))
testdata = []
for i in xrange(0,len(test["review"])):
    # testdata.append(" ".join(review_to_wordlist(test["review"][i], False)))
    traindata.append(" ".join(review_to_wordlist2(test["review"][i])))

print 'vectorizing... ',

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
X_all = traindata + testdata

lentrain = len(traindata)

print "fitting pipeline... ",
tfv.fit(X_all)
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)

print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X, y, cv=20, scoring='roc_auc'))

print "Retrain on all training data, predicting test labels...\n"
model.fit(X,y)
result = model.predict_proba(X_test)[:,1]
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
# output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)


# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

print "Wrote results to Bag_of_Words_model.csv"