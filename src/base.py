import numpy as np

class BaseClassifier:

    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        raise NotImplementedError("La méthode fit() doit être implémentée par la classe enfant.")

    def predict(self, X):
        raise NotImplementedError("La méthode predict() doit être implémentée par la classe enfant.")
