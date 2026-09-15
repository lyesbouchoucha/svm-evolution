"""Kernel SVM: the dual problem solved by Sequential Minimal Optimization.

    max_alpha  sum_i alpha_i - (1/2) sum_ij a_i a_j y_i y_j K(x_i, x_j)
    s.t.       0 <= alpha_i <= C   and   sum_i alpha_i y_i = 0

Why the dual: the data appears only through inner products, which can be
replaced by K(x_i, x_j) without ever computing phi(x). That is what makes
the RBF kernel usable, its feature space being infinite-dimensional. The
soft margin alone would not justify the dual -- plain SGD on the hinge
loss handles that in the linear case.

Why two alphas at a time: the equality constraint sum_i alpha_i y_i = 0
means a single alpha cannot move on its own. Two is the minimum, and for
two the optimum has a closed form -- hence "minimal optimization".
"""

import numpy as np

from src.base import BaseClassifier
from src.kernels import linear_kernel


class KernelSVM(BaseClassifier):
    """Dual SVM optimised by SMO (simplified version of Platt's algorithm).

    C          : upper bound on the alphas (the soft margin parameter).
    max_passes : stop after this many consecutive sweeps with no change.
    max_iter   : hard cap, since max_passes is reset whenever an alpha
                 moves and therefore does not bound the total work.
    """

    def __init__(self, C=1.0, kernel=linear_kernel, tol=1e-3, max_passes=5,
                 max_iter=100_000, random_state=None):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]

        self.X, self.y = X, y
        self.alphas = np.zeros(n)
        self.bias = 0.0

        K = self.kernel(X, X)          # Gram matrix, computed once
        rng = np.random.default_rng(self.random_state)
        passes = 0
        self.n_iter_ = 0

        while passes < self.max_passes and self.n_iter_ < self.max_iter:
            changed = 0

            for i in range(n):
                self.n_iter_ += 1
                E_i = (self.alphas * y) @ K[:, i] + self.bias - y[i]

                # Does point i violate the KKT conditions?
                if not ((y[i] * E_i < -self.tol and self.alphas[i] < self.C)
                        or (y[i] * E_i > self.tol and self.alphas[i] > 0)):
                    continue

                j = rng.integers(n - 1)
                if j >= i:
                    j += 1             # uniform draw over [0, n) without i
                E_j = (self.alphas * y) @ K[:, j] + self.bias - y[j]

                a_i_old, a_j_old = self.alphas[i], self.alphas[j]

                # Bounds on alpha_j imposed by 0 <= alpha <= C together
                # with sum_i alpha_i y_i = 0
                if y[i] != y[j]:
                    L = max(0.0, a_j_old - a_i_old)
                    H = min(self.C, self.C + a_j_old - a_i_old)
                else:
                    L = max(0.0, a_i_old + a_j_old - self.C)
                    H = min(self.C, a_i_old + a_j_old)
                if L >= H:
                    continue

                # Second derivative of the objective along the update
                # direction; a non-negative value means no interior minimum
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                self.alphas[j] = np.clip(a_j_old - y[j] * (E_i - E_j) / eta, L, H)
                if abs(self.alphas[j] - a_j_old) < 1e-5:
                    continue

                # Keep sum_i alpha_i y_i unchanged
                self.alphas[i] = a_i_old + y[i] * y[j] * (a_j_old - self.alphas[j])

                # Recentre the bias so that KKT holds for i and j
                d_i = y[i] * (self.alphas[i] - a_i_old)
                d_j = y[j] * (self.alphas[j] - a_j_old)
                b_i = self.bias - E_i - d_i * K[i, i] - d_j * K[i, j]
                b_j = self.bias - E_j - d_i * K[i, j] - d_j * K[j, j]

                if 0 < self.alphas[i] < self.C:
                    self.bias = b_i
                elif 0 < self.alphas[j] < self.C:
                    self.bias = b_j
                else:
                    self.bias = 0.5 * (b_i + b_j)

                changed += 1

            passes = passes + 1 if changed == 0 else 0

        self.converged_ = self.n_iter_ < self.max_iter
        self.support_ = np.where(self.alphas > 1e-8)[0]

        return self

    def decision_function(self, X):
        K = self.kernel(self.X, np.asarray(X, dtype=float))   # (n_train, n_test)
        return (self.alphas * self.y) @ K + self.bias
