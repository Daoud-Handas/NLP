import click
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

from NLP.data.make_dataset import make_dataset
from features.make_features import make_features, make_features_is_name
from model.main import make_model


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--feature", help="Can be tokenize, stopwords, stemming or lowercase")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/model.json", help="File to dump model")
def train(task, feature, input_filename, model_dump_filename):
    print(f"Evaluating feature {feature}")

    config = {
        "use_stemming": True if feature == "stemming" else False,
        "use_lowercase": True if feature == "lowercase" else False,
        "use_stopwords": True if feature == "stopwords" else False,
        "use_tokenization": True if feature == "tokenize" else False,
        "is_start_word": True if feature == "is_start_word" else False,
        "is_end_word": True if feature == "is_end_word" else False,
        "is_capitalized": True if feature == "is_capitalized" else False,
        "is_punctuation": True if feature == "is_punctuation" else False,
        "use_ngram": True if feature == "ngram" else False,
        "use_ngram_range": True if feature == "ngram_range" else False,
    }

    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    model = make_model(config)

    model.fit(X, y)
    return model.dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--feature", help="Can be tokenize, stopwords, stemming or lowercase")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model()
    model.load(model_dump_filename)

    y_pred = model.predict(X)

    df["prediction"] = y_pred

    df.to_csv(output_filename)


def predict(task, input_filename, model_dump_filename, output_filename):
    print(f"Predicting for task {task}")
    df = make_dataset(input_filename)
    X, y = make_features(df, task)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    model = make_model()
    model.load(model_dump_filename)

    predict = model.predict(X)
    print(predict)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--feature", help="Can be tokenize, stopwords, stemming or lowercase")
def evaluate(task, input_filename, feature):
    print(f"Evaluating feature {feature}")

    # Object with .fit, .predict methods
    df = make_dataset(input_filename)

    if task == "is_name":
        X, y = make_features_is_name(df, task)
    else:
        X, y = make_features(df, task)

    X = np.array(X)
    y = y.values

    model = make_model()

    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Divisez vos données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # Entraînez le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Faites des prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Calculez l'accuracy sur les données de test
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
