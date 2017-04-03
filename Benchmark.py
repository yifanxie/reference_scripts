__author__ = 'xie'

import nltk
import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("./data/train.csv").fillna("")
test  = pd.read_csv("./data/test.csv").fillna("")

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_name, extractor in self.features:
            extractor.fit(X[column_name], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.transform(X[column_name])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else:
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.fit_transform(X[column_name], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else:
            return extracted[0]

def identity(x):
    return x

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

#                          Feature Set Name            Data Frame Column              Transformer
features = FeatureMapper([('QueryBagOfWords',          'query',                       CountVectorizer(max_features=200)),
                          ('TitleBagOfWords',          'product_title',               CountVectorizer(max_features=200)),
                          ('DescriptionBagOfWords',    'product_description',         CountVectorizer(max_features=200)),
                          ('QueryTokensInTitle',       'query_tokens_in_title',       SimpleTransform()),
                          ('QueryTokensInDescription', 'query_tokens_in_description', SimpleTransform())])

def extract_features(data):
    # the r" string prefix below denote that this is a raw string notation.
    # (?) is the UNICODE FLAG
    # \b matches empty string
    # \w matches any alphanumeric character and the underscore (without LOCALE and UNICODE flags)

    # regular expression object that specify pattern of all alphanumeric strings with length greater than 1
    token_pattern = re.compile(r"(?u)\b\w\w+\b")

    #the following statements create new column in the dataframe object, and assign initial value
    data["query_tokens_in_title"] = 0.0
    data["query_tokens_in_description"] = 0.0
    for i, row in data.iterrows():
        #i is the index
        # row contain values of the row of index i for each column

        #the following statements extract all sub-string with lenght >2 in the respective column of the current row, and save them in a set
        query = set(x.lower() for x in token_pattern.findall(row["query"]))
        title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))

        #calculate and assign value to the two new feature that is created, by calculating the intersection between query and title sets, and then query and description sets
        if len(title) > 0:
            #pandas.set_value method: set value of the given column
            data.set_value(i, "query_tokens_in_title", len(query.intersection(title))/len(title))
        if len(description) > 0:
            data.set_value(i, "query_tokens_in_description", len(query.intersection(description))/len(description))



#the following statement create the "query_tokens_in_title" and "query_tokens_in_description" features in the train and test dataframe objects,
# and then calculate and assign the value of these features to each row of these two objects
extract_features(train)
extract_features(test)

pipeline = Pipeline([("extract_features", features),
                     ("classify", RandomForestClassifier(n_estimators=200,
                                                         n_jobs=1,
                                                         min_samples_split=2,
                                                         random_state=1))])

pipeline.fit(train, train["median_relevance"])

predictions = pipeline.predict(test)

submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
submission.to_csv("python_benchmark.csv", index=False)