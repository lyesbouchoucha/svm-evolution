"""Kernel functions, vectorised.

Each kernel takes two matrices (n1, d) and (n2, d) and returns the Gram
matrix (n1, n2). Working with matrices rather than pairs of points is what
makes SMO and boundary plotting fast enough to be usable.

Use functools.partial to fix a hyperparameter:

    kernel = partial(rbf_kernel, gamma=2.0)
"""

import numpy as np


def linear_kernel(X1, X2):
    return np.asarray(X1) @ np.asarray(X2).T


def polynomial_kernel(X1, X2, degree=3, coef0=1.0):
    return (np.asarray(X1) @ np.asarray(X2).T + coef0) ** degree


def rbf_kernel(X1, X2, gamma=1.0):
    """Gaussian kernel exp(-gamma * ||x1 - x2||^2).

    ||x1 - x2||^2 is expanded as ||x1||^2 + ||x2||^2 - 2 x1.x2 so the whole
    distance matrix comes from one matrix product.

    Mind gamma: on moons or circles, squared distances are of order 1 to 9,
    so gamma=0.1 gives an almost flat kernel and therefore an almost linear
    boundary. A useful range is [1, 10].
    """
    X1, X2 = np.asarray(X1), np.asarray(X2)
    sq = (np.sum(X1**2, axis=1)[:, None]
          + np.sum(X2**2, axis=1)[None, :]
          - 2.0 * (X1 @ X2.T))
    return np.exp(-gamma * np.maximum(sq, 0.0))
