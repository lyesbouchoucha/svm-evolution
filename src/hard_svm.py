import numpy as np
from src.base import BaseClassifier

class HardSVM(BaseClassifier):
    
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            for i, x_i in enumerate(X):
                if y[i] * (np.dot(x_i, self.weights) + self.bias) < 1:
                    self.weights = self.weights - self.learning_rate * (self.weights - y[i] * x_i)
                    self.bias += self.learning_rate * y[i]
                else:
                    self.weights = self.weights - self.learning_rate * self.weights

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)
