"""Common base class for the four classifiers."""

import numpy as np


class NotSeparableError(Exception):
    """Raised when a hard margin model cannot separate the data."""


class BaseClassifier:
    """Shared interface. Labels are expected in {-1, +1}.

    project(X) returns the decision function f(x); predict(X) returns its
    sign. The plots need f(x) itself, since the margins are its level sets.
    """

    def fit(self, X, y):
        raise NotImplementedError

    def project(self, X):
        raise NotImplementedError

    def predict(self, X):
        # np.sign would return 0 on the boundary, which is not a label.
        return np.where(self.project(X) >= 0, 1.0, -1.0)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
