"""Hard Margin SVM, solved in the primal as a quadratic program.

    minimise    (1/2) ||w||^2
    subject to  y_i (w . x_i + b) >= 1   for every i
"""

import cvxopt
import cvxopt.solvers
import numpy as np
from numpy import linalg

from src.base import BaseClassifier, NotSeparableError

cvxopt.solvers.options['show_progress'] = False


class HardSVM(BaseClassifier):

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # cvxopt minimises (1/2) z' P z + q' z under G z <= h. Here
        # z = [w, b], and the last diagonal entry of P is 1e-9 instead of 0
        # so that P is positive definite, as the solver requires.
        P = np.identity(n_features + 1)
        P[n_features, n_features] = 1e-9
        q = np.zeros(n_features + 1)

        # One row per constraint, rewritten as -y_i [x_i, 1] . z <= -1.
        X_augmented = np.hstack((X, np.ones((n_samples, 1))))
        G = -X_augmented * y[:, None]
        h = -np.ones(n_samples)

        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)

        # Same constraints, zero objective: this only asks whether an
        # admissible point exists. Needed because cvxopt.qp fails with a
        # numerical error instead of reporting an infeasible problem.
        feasibility = cvxopt.solvers.lp(q, G, h)
        if feasibility['status'] != 'optimal':
            raise NotSeparableError(
                "The data is not linearly separable: no (w, b) satisfies "
                "y_i (w . x_i + b) >= 1 for every i."
            )

        solution = cvxopt.solvers.qp(P, q, G, h)
        z = np.ravel(solution['x'])
        self.w = z[:n_features]
        self.b = z[n_features]
        self.margin = 1.0 / linalg.norm(self.w)

        # solution['z'] holds the Lagrange multipliers. They are non-zero
        # only on the margin, but an interior-point solver returns small
        # values rather than exact zeros, hence the threshold.
        alphas = np.ravel(solution['z'])
        self.support = np.arange(n_samples)[alphas > 1e-5]

        return self

    def project(self, X):
        return np.dot(X, self.w) + self.b
