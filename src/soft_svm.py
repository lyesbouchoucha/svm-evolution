"""Soft Margin SVM: hinge loss + L2 regularisation, optimised by SGD.

The constrained problem with slack variables

    min (1/2)||w||^2 + C sum_i xi_i
    s.t. y_i f(x_i) >= 1 - xi_i,  xi_i >= 0

is exactly equivalent to the unconstrained problem

    min  lambda ||w||^2 + (1/n) sum_i max(0, 1 - y_i f(x_i))

which is what this class minimises by stochastic subgradient descent. In
other words the hinge loss IS the soft margin formulation, which is why
gradient descent on the hinge loss can never act as a hard margin SVM.

Note on lambda: a LARGE lambda means strong regularisation, small ||w||,
so a WIDE margin and more tolerated violations. The model gets softer,
not harder.
"""

import numpy as np

from src.base import BaseClassifier


class SoftSVM(BaseClassifier):

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=500):
        super().__init__()
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        for _ in range(self.n_iterations):
            for i in range(X.shape[0]):
                if y[i] * (X[i] @ self.weights + self.bias) >= 1:
                    # Margin satisfied: only the L2 term has a gradient
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                    )
                else:
                    # Margin violated: L2 term + hinge subgradient
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - y[i] * X[i]
                    )
                    self.bias += self.learning_rate * y[i]

        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias

    @property
    def margin_(self):
        return 1.0 / np.linalg.norm(self.weights)
