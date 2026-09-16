"""Generate every figure used in the README.

Run from the project root:
    python generate_visuals.py
"""

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

from src.base import NotSeparableError
from src.perceptron import Perceptron
from src.plotting import plot_margin, plot_kernel_trick, plot_lifted_points
from src.svm_hard import HardSVM
from src.svm_smo import KernelSVM
from src.svm_soft_sgd import SoftSVM


# Features standardised so one learning rate suits all three datasets.
# Labels converted to {-1, +1}: scikit-learn returns {0, 1}, and a label of
# 0 would cancel every update of the form y_i * x_i.
def generate_blobs():
    X, y = make_blobs(n_samples=100, centers=2, random_state=6)
    return StandardScaler().fit_transform(X), np.where(y == 0, -1.0, 1.0)


def generate_moons():
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    return StandardScaler().fit_transform(X), np.where(y == 0, -1.0, 1.0)


def generate_circles():
    X, y = make_circles(n_samples=200, noise=0.08, factor=0.45, random_state=42)
    return StandardScaler().fit_transform(X), np.where(y == 0, -1.0, 1.0)


# --- Scenario 0: why maximise the margin? ----------------------------------
# Both reach 100% accuracy. The perceptron stops at the first separator it
# finds, so two permutations of the same data give two different answers.
print('[0] Why maximise the margin?')
X, y = generate_blobs()

for name, seed in (('a', 23), ('b', 20)):
    np.random.seed(seed)
    order = np.random.permutation(len(y))
    perceptron = Perceptron(n_iterations=100).fit(X[order], y[order])
    angle = np.degrees(np.arctan2(perceptron.w[1], perceptron.w[0]))
    plot_margin(perceptron, X, y,
                'Perceptron - data order #%d - hyperplane at %.0f deg, '
                'accuracy %.0f%%' % (seed, angle, 100 * perceptron.score(X, y)),
                '00_perceptron_%s.png' % name, show_margins=False)
    print('    order #%d: angle = %.1f deg, accuracy = %.2f'
          % (seed, angle, perceptron.score(X, y)))

hard_svm = HardSVM().fit(X, y)
plot_margin(hard_svm, X, y,
            'Hard Margin SVM - margin %.2f, %d support vectors, deterministic'
            % (hard_svm.margin, len(hard_svm.support)),
            '00_hard_svm.png')
print('    hard SVM: margin = %.3f, %d support vectors'
      % (hard_svm.margin, len(hard_svm.support)))


# --- Scenario 1: the hard margin trap --------------------------------------
# One point inside the opposite class leaves the constraints with no
# solution, while the soft margin absorbs it as a violation.
print('\n[1] The hard margin trap')
X_outlier = np.vstack((X, [-0.5, -0.7]))
y_outlier = np.append(y, -1.0)

try:
    HardSVM().fit(X_outlier, y_outlier)
    print('    hard SVM converged, which was not expected')
except NotSeparableError as error:
    print('    hard SVM -> NotSeparableError: %s' % error)

soft_svm = SoftSVM(n_iterations=400).fit(X_outlier, y_outlier)
plot_margin(soft_svm, X_outlier, y_outlier,
            'Soft Margin SVM - outlier absorbed, accuracy %.0f%%, margin %.2f'
            % (100 * soft_svm.score(X_outlier, y_outlier), soft_svm.margin),
            '01_soft_svm_outlier.png')
print('    soft SVM: accuracy = %.3f, margin = %.3f'
      % (soft_svm.score(X_outlier, y_outlier), soft_svm.margin))


# --- Scenario 2: the linear limit ------------------------------------------
print('\n[2] The linear limit (moons)')
X_moons, y_moons = generate_moons()

soft_moons = SoftSVM(n_iterations=400).fit(X_moons, y_moons)
plot_margin(soft_moons, X_moons, y_moons,
            'Soft Margin SVM on moons - a line cannot do better than %.0f%%'
            % (100 * soft_moons.score(X_moons, y_moons)),
            '02_soft_svm_moons.png')
print('    linear soft SVM: accuracy = %.3f' % soft_moons.score(X_moons, y_moons))

# Squared distances are of order 1 to 9 here, hence gamma = 2. C = 10 is
# above the largest multiplier needed, so none is capped and every support
# vector satisfies y f(x) = 1.
kernel_moons = KernelSVM(kernel='gaussian', C=10.0, gamma=2.0,
                         random_state=0).fit(X_moons, y_moons)
plot_margin(kernel_moons, X_moons, y_moons,
            'SMO + gaussian kernel (gamma = 2, C = 10) on moons\n'
            'accuracy %.1f%%'
            % (100 * kernel_moons.score(X_moons, y_moons)),
            '02_kernel_svm_moons.png')
print('    gaussian kernel SVM: accuracy = %.3f, %d support vectors, '
      'iterations = %d, converged = %s'
      % (kernel_moons.score(X_moons, y_moons), len(kernel_moons.support),
         kernel_moons.n_iter, kernel_moons.converged))


# --- Scenario 3: the ultimate test -----------------------------------------
print('\n[3] Concentric circles')
X_circles, y_circles = generate_circles()

kernel_circles = KernelSVM(kernel='gaussian', C=10.0, gamma=2.0,
                           random_state=0).fit(X_circles, y_circles)
plot_margin(kernel_circles, X_circles, y_circles,
            'SMO + gaussian kernel (gamma = 2, C = 10) on circles\n'
            'accuracy %.0f%%'
            % (100 * kernel_circles.score(X_circles, y_circles)),
            '03_kernel_svm_circles.png')
print('    gaussian kernel SVM: accuracy = %.3f, %d support vectors'
      % (kernel_circles.score(X_circles, y_circles), len(kernel_circles.support)))


# --- Why a kernel works ----------------------------------------------------
# Figure 04 uses no kernel: a linear SVM on phi(x) = (x1, x2, x1^2 + x2^2).
# Figure 05 uses the gaussian model fitted above.
print('\n[4] The kernel trick, in three dimensions')
X_lifted = np.column_stack((X_circles,
                            X_circles[:, 0] ** 2 + X_circles[:, 1] ** 2))
linear_in_3d = HardSVM().fit(X_lifted, y_circles)
plot_kernel_trick(linear_in_3d, X_circles, y_circles, '04_kernel_trick.png')
print('    linear SVM in R^3: margin = %.3f, %d support vectors, accuracy = %.2f'
      % (linear_in_3d.margin, len(linear_in_3d.support),
         linear_in_3d.score(X_lifted, y_circles)))

print('\n[5] Every point at its own decision value')
plot_lifted_points(kernel_circles, X_circles, y_circles, '05_lifted_points.png')
margin_values = y_circles * kernel_circles.project(X_circles)
support = kernel_circles.support
print('    support vectors: y f(x) between %.4f and %.4f'
      % (margin_values[support].min(), margin_values[support].max()))
print('    nearest point that is not a support vector: y f(x) = %.3f'
      % np.min(np.delete(margin_values, support)))

print('\nAll figures written to figures/')
