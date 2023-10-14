from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression


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

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def dump(self, filename_output):
        with open(filename_output, "w") as f:
            f.write("LinearModel")


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
