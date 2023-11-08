from ast import literal_eval

import nltk
import numpy as np
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re


# nltk.download('stopwords') -> uncomment if you don't have stopwords
# nltk.download('punkt') -> uncomment if you don't have punkt


def make_features(df, task, config=None):
    X = df["video_name"]
    y = get_output(df, task)
    if config:
        if config["use_lowercase"]:
            X = X.str.lower()
        if config["use_stopwords"]:
            X = X.apply(remove_stopwords)
        if config["use_stemming"]:
            X = X.apply(stemming)
        if config["use_tokenization"]:
            X = X.apply(tokenize)
        if config["make_ngrams"]:
            X = X.apply(make_ngrams)
        if config["make_mgrams_range"]:
            X = X.apply(make_mgrams_range)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)
    return X, y

# Fonction pour convertir une chaîne de caractères en liste de mots
def convert_string_to_list(input_string):
    # Supprimez les caractères non alphabétiques et non numériques
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    # Séparez la chaîne en mots en utilisant l'espace comme délimiteur
    word_list = cleaned_string.split()
    return word_list


def make_features_is_name(df, task):
    X = df["tokens"]
    y = get_output(df, task)
    y = y.apply(literal_eval)

    # Parcourez les chaînes de caractères mal formatées et convertissez-les en listes de mots
    for i, input_string in enumerate(X):
        X[i] = convert_string_to_list(input_string)

    if task == "is_name":
        X = X.apply(extract_word_info)
        sentences = []
        for title in X:
            features = []
            for word in title:
                features.append(
                    [word["is_start_word"], word["is_end_word"], word["is_capitalized"],
                     word["is_punctuation"]])
            sentences.append([item for sublist in features for item in sublist])

        # get biggest sentence
        max_len_x = max([len(sentence) for sentence in sentences])

        # pad sentences
        for i, sentence in enumerate(sentences):
            if len(sentence) < max_len_x:
                sentences[i] = np.pad(sentence, (0, max_len_x - len(sentence)), 'constant', constant_values=(-1))

        # get biggest label
        # pad y
        max_len_y = max([len(label) for label in y])

        for i, label in enumerate(y):
            if len(label) < max_len_y:
                y[i] = np.pad(label, (0, max_len_y - len(label)), 'constant', constant_values=(-1))

        return sentences, y


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


def is_start_word(i):
    return int(i == 0)


def is_end_word(i, len_text):
    return int(i == len_text - 1)


def is_capitalized(word):
    return int(word[0].isupper())


def is_punctuation(word):
    return int(word in [".", ",", "!", "?"])


def extract_word_info(words):
    word_info = []

    for i, word in enumerate(words):
        info = {
            "word": word,
            "is_start_word": is_start_word(i),
            "is_end_word": is_end_word(i, len(words)),
            "is_capitalized": is_capitalized(word),
            "is_punctuation": is_punctuation(word)
        }
        word_info.append(info)

    return word_info
