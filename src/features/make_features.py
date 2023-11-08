import string
from ast import literal_eval

import nltk
from nltk import word_tokenize, ngrams, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def make_features(df, task, config=None):
    X = df["video_name"]
    y = get_output(df, task)
    y = y.apply(literal_eval)
    if config:
        if config["use_lowercase"]:
            X = X.str.lower()
        if config["use_stopwords"]:
            X = X.apply(remove_stopwords)
        if config["use_stemming"]:
            X = X.apply(stemming)
        if config["use_tokenization"]:
            X = X.apply(tokenize)
        if config["use_ngram"]:
            X = X.apply(make_ngrams)
        if config["use_ngram_range"]:
            X = X.apply(make_mgrams_range)
        if config["use_extract_word_features"]:
            X = X.apply(extract_word_features)
            X = X.apply(lambda x: [d['sum'] for d in x])

    return X, y


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords.words('french')])


def stemming(text):
    stemmer = PorterStemmer()
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text


def tokenize(text):
    return " ".join([word for word in word_tokenize(text)])


def make_ngrams(text, n=3):
    words = word_tokenize(text)
    n_grams = list(ngrams(words, n))
    return ' '.join([' '.join(grams) for grams in n_grams])


def make_mgrams_range(text, min_n=1, max_n=4):
    words = word_tokenize(text)
    n_grams = []
    for n in range(min_n, max_n + 1):
        n_grams += list(ngrams(words, n))
    return ' '.join([' '.join(grams) for grams in n_grams])


def is_starting_word(word, sentence):
    index = sentence.index(word)
    if index == 0 or index == 1:
        return 1
    else:
        return 0


def is_final_word(word, sentence):
    index = sentence.index(word)
    if index + len(word) == len(sentence):
        return 3
    else:
        return 0

def is_capitalized(word):
    if word.isupper():
        return 5
    else:
        return 0


def is_punctuation(word):
    if word in string.punctuation or word == "``" or word == "''":
        return 7
    else:
        return 0

def extract_word_features(sentence):
    words = word_tokenize(sentence)
    features = []

    for i, word in enumerate(words):
        if word in string.punctuation or word == "``" or word == "''":
            info = {
                "is_starting_word": 0,
                "is_final_word": 0,
                "is_capitalized": 0,
                "is_punctuation": is_punctuation(word),
            }
        else:
            info = {
                "is_starting_word": is_starting_word(word, sentence),
                "is_final_word": is_final_word(word, sentence),
                "is_capitalized": is_capitalized(word),
                "is_punctuation": 0,
            }
        info["sum"] = sum(info.values())
        features.append(info)
    return features