import pickle

import click
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer

from data.make_dataset import make_dataset, make_train_test_split
from features.make_features import make_features
from model.main import make_model
from model.dumb_model import LogisticRegressionModel, RandomForestModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--features", help="Can be tokenize, stopwords, stemming or lowercase")
def train(task, input_filename, model_dump_filename, features):
    df = make_dataset(input_filename)
    arr_features = features.split(",")
    config = {
        "use_stemming": True if "stemming" in arr_features else False,
        "use_lowercase": True if "lowercase" in arr_features else False,
        "use_stopwords": True if "stopwords" in arr_features else False,
        "use_tokenization": True if "tokenize" in arr_features else False,
        "use_ngram": True if "ngram" in arr_features else False,
        "use_ngram_range": True if "ngram_range" in arr_features else False,
        "use_extract_word_features": True if "extract_word_features" in arr_features else False,
    }

    if task == "is_name":
        X, y = make_features(df, task, config)
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)
        rf_classifier = RandomForestModel()
        rf_classifier.fit(X, y)
        # Dump the model to a file
        with open(model_dump_filename, "wb") as model_file:
            pickle.dump(rf_classifier, model_file)
        return rf_classifier.dump(model_dump_filename)
    else:
        X, y = make_features(df, task, config)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)
        model = LogisticRegressionModel()
        model.fit(X, y)
        return model.dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(task, input_filename, model_dump_filename, output_filename):
    print(f"Predicting for task {task}")
    df = make_dataset(input_filename)
    X, y = make_features(df, task)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    model = LogisticRegressionModel()
    model.load(model_dump_filename)

    predict = model.predict(X)
    print(predict)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--features", help="Can be tokenize, stopwords, stemming or lowercase")
def evaluate(task, input_filename, features):
    print(f"Evaluating feature {features}")
    arr_features = features.split(",")

    config = {
        "use_stemming": True if "stemming" in arr_features else False,
        "use_lowercase": True if "lowercase" in arr_features else False,
        "use_stopwords": True if "stopwords" in arr_features else False,
        "use_tokenization": True if "tokenize" in arr_features else False,
        "use_ngram": True if "ngram" in arr_features else False,
        "use_ngram_range": True if "ngram_range" in arr_features else False,
        "use_extract_word_features": True if "extract_word_features" in arr_features else False,
    }

    df = make_dataset(input_filename)
    X, y = make_features(df, task,config)
    print(X)
    print(y)
    model = make_model(config)
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    print()
    scores = cross_val_score(model, X, y, scoring="accuracy")
    print(scores)

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
