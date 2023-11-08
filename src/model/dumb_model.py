import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
import json

class DumbModel:
    """Dumb model always predict 0"""

    def fit(self, X, y):
        pass

    def predict(self, X):
        return [0] * len(X)

    def dump(self, filename_output):
        pass


# make a class for linear model
class LogisticRegressionModel:
    """Linear model"""

    def __init__(self):
        self.model = LogisticRegression()
        self.trained = None

    def fit(self, X, y):
        try:
            print("fitting")
            print(X)
            print(y)
            self.trained = self.model.fit(X, y)
        except Exception as e:
            print("An error occurred during training:")
            print(e)

    def predict(self, X):
        if self.trained is None:
            raise ValueError("Model not trained")
        return self.trained.predict(X)

    def dump(self, filename_output):
        print("dumping")
        print(self.trained)
        # if self.trained is None:
        #     raise ValueError("Model not trained")
        # else:
        #     model_dict = {
        #         "coef": self.trained.coef_.tolist(),
        #         "intercept": self.trained.intercept_.tolist(),
        #         "classes": self.trained.classes_.tolist(),
        #         "solver": self.trained.solver,
        #     }
        #     with open(filename_output, 'w') as json_file:
        #         json.dump(model_dict, json_file)

    def load(self, filename_input):
        with open(filename_input) as json_file:
            model_dict = json.load(json_file)
            self.trained = LogisticRegression()
            self.trained.coef_ = model_dict["coef"]
            self.trained.intercept_ = model_dict["intercept"]
            self.trained.classes_ = model_dict["classes"]
            self.trained.solver = model_dict["solver"]



# make a class for random forest model
class RandomForestModel:
    """Random forest model"""

    def __init__(self):
        self.model = RandomForestClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def dump(self, filename_output):
        with open(filename_output, "w") as f:
            f.write("RandomForestModel")


class LinearModel:
    """Linear model"""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def dump(self, filename_output):
        with open(filename_output, "w") as f:
            f.write("LinearModel")
