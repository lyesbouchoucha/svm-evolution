import numpy as np


class NotSeparableError(Exception):
    """Raised when a hard-margin model cannot separate the data."""


class BaseClassifier:

    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        raise NotImplementedError("fit() must be implemented by the subclass.")

    def decision_function(self, X):
        raise NotImplementedError(
            "decision_function() must be implemented by the subclass."
        )

    def predict(self, X):
        # np.sign(0) == 0 would leave boundary points unclassified
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
