import nltk
from nltk import word_tokenize, PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from model.dumb_model import DumbModel, LogisticRegressionModel

from model.dumb_model import RandomForestModel


def make_model(config=None):
    return Pipeline([
        ("dumbModel", RandomForestModel())
    ])