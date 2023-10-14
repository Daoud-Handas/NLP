import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer


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
