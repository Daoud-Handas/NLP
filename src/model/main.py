import nltk
from nltk import word_tokenize, PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline



def make_model(config=None):
    return Pipeline([
        # ("count_vectorizer", CountVectorizer()), # only when X contains text
        ("RandomForestClassifier", RandomForestClassifier())
    ])
