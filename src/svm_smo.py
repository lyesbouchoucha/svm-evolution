"""Kernel SVM: the dual problem solved by Sequential Minimal Optimization.

    maximise    sum_i a_i - (1/2) sum_ij a_i a_j y_i y_j K(x_i, x_j)
    subject to  0 <= a_i <= C   and   sum_i a_i y_i = 0

    f(x) = sum_i a_i y_i K(x_i, x) + b

SMO repeatedly picks two multipliers violating the KKT conditions and sets
them to their exact optimum, until none violates them. See the README for
the derivation.
"""

import numpy as np

from src.base import BaseClassifier
from src.kernels import linear_kernel, polynomial_kernel, gaussian_kernel


class KernelSVM(BaseClassifier):
    """Dual SVM optimised by SMO (the simplified version of Platt's method).

    kernel      : 'linear', 'polynomial' or 'gaussian'
    C           : upper bound on the multipliers (soft margin parameter)
    tol         : tolerance on the KKT conditions
    max_passes  : consecutive sweeps without any update before stopping
    max_iter    : overall iteration limit, since max_passes is reset to zero
                  whenever a multiplier moves
    """

    def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree=3, coef0=1.0,
                 tol=1e-3, max_passes=5, max_iter=300000, random_state=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.random_state = random_state

    def compute_kernel(self, x1, x2):
        if self.kernel == 'linear':
            return linear_kernel(x1, x2)
        if self.kernel == 'polynomial':
            return polynomial_kernel(x1, x2, self.degree, self.coef0)
        if self.kernel == 'gaussian':
            return gaussian_kernel(x1, x2, self.gamma)
        raise ValueError("kernel must be 'linear', 'polynomial' or 'gaussian'")

    def fit(self, X, y):
        n_samples = X.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.X = X
        self.y = y
        self.alphas = np.zeros(n_samples)
        self.b = 0.0

        # Gram matrix, computed once: the loop below reuses it constantly.
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.compute_kernel(X[i], X[j])
        self.K = K

        passes = 0
        self.n_iter = 0

        while passes < self.max_passes and self.n_iter < self.max_iter:
            changed = 0

            for i in range(n_samples):
                self.n_iter = self.n_iter + 1
                error_i = np.sum(self.alphas * y * K[:, i]) + self.b - y[i]

                # y_i * error_i equals y_i f(x_i) - 1, so these two tests
                # detect a multiplier that violates the KKT conditions.
                below_cap = self.alphas[i] < self.C and y[i] * error_i < -self.tol
                above_zero = self.alphas[i] > 0 and y[i] * error_i > self.tol

                if below_cap or above_zero:
                    # Second point of the pair. Platt uses a heuristic; a
                    # uniform draw is enough at this scale.
                    j = np.random.randint(n_samples)
                    while j == i:
                        j = np.random.randint(n_samples)

                    error_j = np.sum(self.alphas * y * K[:, j]) + self.b - y[j]
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    # Range left for a_j once the equality constraint and
                    # the box 0 <= a <= C are both imposed.
                    if y[i] != y[j]:
                        low = max(0.0, alpha_j_old - alpha_i_old)
                        high = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        low = max(0.0, alpha_i_old + alpha_j_old - self.C)
                        high = min(self.C, alpha_i_old + alpha_j_old)

                    # Second derivative of the objective in a_j. It must be
                    # strictly negative for an interior maximum to exist.
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]

                    if low < high and eta < 0:
                        alpha_j_new = alpha_j_old - y[j] * (error_i - error_j) / eta
                        alpha_j_new = min(max(alpha_j_new, low), high)

                        if abs(alpha_j_new - alpha_j_old) > 1e-5:
                            self.alphas[j] = alpha_j_new
                            # a_i absorbs the change, keeping sum a_i y_i = 0.
                            self.alphas[i] = alpha_i_old + y[i] * y[j] * (
                                alpha_j_old - alpha_j_new)

                            # Bias that restores y f(x) = 1 for whichever
                            # point has 0 < a < C; the midpoint if neither
                            # does.
                            delta_i = y[i] * (self.alphas[i] - alpha_i_old)
                            delta_j = y[j] * (self.alphas[j] - alpha_j_old)
                            b_i = (self.b - error_i - delta_i * K[i, i]
                                   - delta_j * K[i, j])
                            b_j = (self.b - error_j - delta_i * K[i, j]
                                   - delta_j * K[j, j])

                            if 0 < self.alphas[i] < self.C:
                                self.b = b_i
                            elif 0 < self.alphas[j] < self.C:
                                self.b = b_j
                            else:
                                self.b = (b_i + b_j) / 2

                            changed = changed + 1

            if changed == 0:
                passes = passes + 1
            else:
                passes = 0

        self.converged = self.n_iter < self.max_iter

        # Points with a non-zero multiplier: the only ones f(x) depends on.
        self.support = np.arange(n_samples)[self.alphas > 1e-8]

        return self

    def project(self, X):
        # Only the support vectors contribute to the sum.
        values = np.zeros(len(X))
        for k in range(len(X)):
            total = 0.0
            for i in self.support:
                total = total + self.alphas[i] * self.y[i] * self.compute_kernel(
                    self.X[i], X[k])
            values[k] = total
        return values + self.b
