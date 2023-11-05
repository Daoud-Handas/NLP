import click
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main import make_model

from model.dumb_model import DumbModel, LogisticRegressionModel, RandomForestModel, LinearModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model()
    model.fit(X, y)

    return model.dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
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


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--feature", help="Can be tokenize, stopwords, stemming or lowercase")
def evaluate(task, input_filename, feature):
    print(f"Evaluating feature {feature}")

    config = {
        "use_stemming": True if feature == "stemming" else False,
        "use_lowercase": True if feature == "lowercase" else False,
        "use_stopwords": True if feature == "stopwords" else False,
        "use_tokenization": True if feature == "tokenize" else False,
        "is_start_word": True if feature == "is_start_word" else False,
        "is_end_word": True if feature == "is_end_word" else False,
        "is_capitalized": True if feature == "is_capitalized" else False,
        "is_punctuation": True if feature == "is_punctuation" else False
    }
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task, config)

    print(X[0])

    # Object with .fit, .predict methods
    model = make_model(config)

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
