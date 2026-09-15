"""Hard Margin SVM, solved as a constrained quadratic program.

    min_{w, b}  (1/2) ||w||^2
    s.t.        y_i (w . x_i + b) >= 1   for every i

There is no tolerance parameter here. If the data is not linearly
separable, the feasible set is empty and the problem has no solution at
all. That is the limitation the soft margin was invented to lift.
"""

import numpy as np
from cvxopt import matrix, solvers

from src.base import BaseClassifier, NotSeparableError

solvers.options["show_progress"] = False


class HardSVM(BaseClassifier):

    def __init__(self, sv_tol=1e-5):
        super().__init__()
        self.sv_tol = sv_tol
        self.support_ = None
        self.margin_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        # Optimisation variable: z = [w_1, ..., w_d, b]
        # Objective (1/2) z^T P z with P = diag(1, ..., 1, 0): the bias is
        # not penalised. The 1e-9 keeps P positive definite for the solver.
        P = np.eye(n_features + 1)
        P[-1, -1] = 1e-9
        q = np.zeros(n_features + 1)

        # Constraints as G z <= h:
        #   y_i (w . x_i + b) >= 1   <=>   -y_i [x_i, 1] . z <= -1
        G = -y[:, None] * np.hstack([X, np.ones((n_samples, 1))])
        h = -np.ones(n_samples)

        # Feasibility check first. On an infeasible problem cvxopt.qp does
        # not report 'primal infeasible': it crashes on a negative square
        # root inside the KKT system. An LP with a zero objective has the
        # same constraints and diagnoses it properly.
        feasibility = solvers.lp(matrix(q), matrix(G), matrix(h))
        if feasibility["status"] != "optimal":
            raise NotSeparableError(
                "The data is not linearly separable: no (w, b) satisfies "
                "y_i (w.x_i + b) >= 1 for every i."
            )

        solution = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        z = np.array(solution["x"]).ravel()
        self.weights = z[:-1]
        self.bias = float(z[-1])
        self.margin_ = 1.0 / np.linalg.norm(self.weights)

        # Lagrange multipliers: alpha_i > 0 only for points sitting exactly
        # on the margin (KKT complementarity). Interior-point methods never
        # return exact zeros, hence the threshold.
        alphas = np.array(solution["z"]).ravel()
        self.support_ = np.where(alphas > self.sv_tol)[0]

        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias
