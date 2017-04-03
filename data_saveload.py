# from KaggleWord2VecUtility import KaggleWord2VecUtility
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.stem.porter as Porter
from nltk.stem import WordNetLemmatizer
import pickle



def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "html.parser").get_text()
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
## convert raw review text into words this time apply lemmatizer
def review_to_wordlist3(raw_review, remove_stopwords=False):
    # set up Porter Stemmer    stemmer=Porter.PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    # Clean up tag and markup using BeautifulSoup
    review_text=BeautifulSoup(raw_review, "html.parser").get_text()

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
    lemmatized_words=[]
    for word in meaningful_words:
        lemmatized_words.append(lemmatizer.lemmatize(word))

    return (lemmatized_words)

###################################################
## convert raw review text into words this time apply porter stemmer
def review_to_wordlist2(raw_review, remove_stopwords=False):
    # set up Porter Stemmer
    stemmer=Porter.PorterStemmer()

    # Clean up tag and markup using BeautifulSoup
    review_text=BeautifulSoup(raw_review, "html.parser").get_text()

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
    data=pickle.load(load_file)
    return  data


def pickle_data(path, data):
    file=path
    save_file=open(file, 'wb')
    pickle.dump(data, save_file, -1)
    save_file.close()

############################

############################################################################################################################################
if __name__ == '__main__':
    start_time=time.time()

    train = pd.read_csv("./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test=pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3 )
    unlabeled_train=pd.read_csv("./data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

    y = train["sentiment"]

    print("Cleaning and parsing movie reviews...\n")

    print("cleaning training data")
    traindata = []
    for i in range( 0, len(train["review"])):
        traindata.append(" ".join(review_to_wordlist(train["review"][i], False)))

    print("cleaning unlabelled training data")
    ul_traindata=[]
    for i in range( 0, len(unlabeled_train["review"])):
        ul_traindata.append(" ".join(review_to_wordlist2(unlabeled_train["review"][i], False)))

    print("cleaning test data")
    testdata=[]
    for i in range(0,len(test["review"])):
        testdata.append(" ".join(review_to_wordlist2(test["review"][i], False)))

    print('vectorizing... ',)

    tfv = TfidfVectorizer(min_df=3, max_features=800000, analyzer='word',
                          ngram_range=(1,3), sublinear_tf=True)

    X_all = traindata + testdata + ul_traindata


    # X_all = traindata + testdata

    lentrain = len(traindata)
    lentest=len(testdata)

    print("fitting pipeline... ",)
    tfv.fit(X_all)
    X_all = tfv.transform(X_all)


    print(X_all.shape)




    end_time=time.time()
    duration=end_time-start_time
    print("it takes %.3f seconds"  %(duration))

    testid=test['id']

    datapath='./pickledata/tut_ng3_mindf3_mf800k_ps.dat'
    pickle_data(datapath,(X_all, y, lentrain, lentest, testid))


