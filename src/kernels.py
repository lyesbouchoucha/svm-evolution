"""Kernel functions. Each takes two points and returns K(x1, x2)."""

import numpy as np
from numpy import linalg


def linear_kernel(x1, x2):
    """K(x1, x2) = x1 . x2"""
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, degree=3, coef0=1.0):
    """K(x1, x2) = (x1 . x2 + c)^d"""
    return (np.dot(x1, x2) + coef0) ** degree


def gaussian_kernel(x1, x2, gamma=1.0):
    """K(x1, x2) = exp(-gamma * ||x1 - x2||^2)"""
    return np.exp(-gamma * linalg.norm(x1 - x2) ** 2)
