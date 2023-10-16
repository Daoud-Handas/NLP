import nltk
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


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
        if config["use_ngram"]:
            X = X.apply(make_ngrams)
        if config["use_ngram_range"]:
            X = X.apply(make_mgrams_range)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)
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
