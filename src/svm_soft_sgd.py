"""Soft Margin SVM: hinge loss and L2 penalty, by stochastic subgradient descent.

    minimise  lambda ||w||^2 + (1/n) sum_i max(0, 1 - y_i f(x_i))
"""

import numpy as np
from numpy import linalg

from src.base import BaseClassifier


class SoftSVM(BaseClassifier):

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=500):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_iterations):
            for i in range(n_samples):
                margin = y[i] * (np.dot(X[i], self.w) + self.b)

                if margin >= 1:
                    # Hinge loss is zero: only the penalty has a gradient.
                    gradient_w = 2 * self.lambda_param * self.w
                    gradient_b = 0.0
                else:
                    # Subgradient of the hinge loss adds -y_i x_i and -y_i.
                    gradient_w = 2 * self.lambda_param * self.w - y[i] * X[i]
                    gradient_b = -y[i]

                self.w = self.w - self.learning_rate * gradient_w
                self.b = self.b - self.learning_rate * gradient_b

        self.margin = 1.0 / linalg.norm(self.w)

        return self

    def project(self, X):
        return np.dot(X, self.w) + self.b
