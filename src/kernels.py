import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3, coef0=1.0):
    return (np.dot(x1, x2) + coef0) ** degree

def gaussian_kernel(x1, x2, gamma=0.1):
    distance_sq = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-gamma * distance_sq)
