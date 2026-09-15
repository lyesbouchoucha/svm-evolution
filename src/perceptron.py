"""Rosenblatt's Perceptron (1958): error-driven learning.

Update rule: whenever a point is misclassified, push the hyperplane
towards it. Nothing in the algorithm refers to a margin, which is exactly
the gap the SVM fills.
"""

import numpy as np

from src.base import BaseClassifier


class Perceptron(BaseClassifier):
    """Classic perceptron.

    Two attributes are exposed for the project's demonstrations:
        errors_per_epoch_ : updates per epoch. Reaches 0 on separable data,
                            oscillates forever otherwise.
        converged_        : True if an epoch went through with no update.

    random_state also matters here: the visiting order changes the
    hyperplane found, since the algorithm stops at the first separator it
    reaches rather than at an optimal one.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, shuffle=True,
                 random_state=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.shuffle = shuffle
        self.random_state = random_state
        self.errors_per_epoch_ = []
        self.converged_ = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_per_epoch_ = []
        self.converged_ = False

        for _ in range(self.n_iterations):
            order = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
            errors = 0

            for i in order:
              
                if y[i] * (X[i] @ self.weights + self.bias) <= 0:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    errors += 1

            self.errors_per_epoch_.append(errors)
            if errors == 0:
                self.converged_ = True
                break

        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias
