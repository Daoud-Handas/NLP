import nltk
from nltk import word_tokenize, PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


from model.dumb_model import LogisticRegressionModel, RandomForestModel, DumbModel

from sklearn.base import BaseEstimator

from model.dumb_model import DumbModel, LogisticRegressionModel



def make_model(config=None):
    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        ("RandomForest", RandomForestClassifier())
    ])

