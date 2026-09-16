"""Perceptron (Rosenblatt, 1958)."""

import numpy as np

from src.base import BaseClassifier


class Perceptron(BaseClassifier):

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.errors_per_epoch = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.errors_per_epoch = []

        for epoch in range(self.n_iterations):
            errors = 0
            for i in range(n_samples):
                # "<= 0" rather than "< 0", so that w = 0 is not a fixed point.
                if y[i] * (np.dot(X[i], self.w) + self.b) <= 0:
                    self.w = self.w + self.learning_rate * y[i] * X[i]
                    self.b = self.b + self.learning_rate * y[i]
                    errors = errors + 1
            # Zero on separable data, positive forever otherwise.
            self.errors_per_epoch.append(errors)

        return self

    def project(self, X):
        return np.dot(X, self.w) + self.b
