"""Measure training and prediction cost for the four models.

Run from the project root:
    python benchmark.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

from src.perceptron import Perceptron
from src.svm_hard import HardSVM
from src.svm_smo import KernelSVM
from src.svm_soft_sgd import SoftSVM

SIZES = [50, 100, 200, 400, 800]


def blobs(n_samples):
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=6)
    return StandardScaler().fit_transform(X), np.where(y == 0, -1.0, 1.0)


def moons(n_samples):
    X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    return StandardScaler().fit_transform(X), np.where(y == 0, -1.0, 1.0)


def time_fit(model, X, y):
    start = time.time()
    model.fit(X, y)
    return time.time() - start


perceptron_times = []
hard_times = []
soft_times = []
smo_times = []
support_counts = []

print('%6s %12s %10s %10s %10s %6s'
      % ('n', 'perceptron', 'hard QP', 'soft SGD', 'SMO', 'SV'))

for n_samples in SIZES:
    X_blobs, y_blobs = blobs(n_samples)
    X_moons, y_moons = moons(n_samples)

    perceptron_times.append(
        time_fit(Perceptron(n_iterations=100), X_blobs, y_blobs))
    hard_times.append(time_fit(HardSVM(), X_blobs, y_blobs))
    soft_times.append(time_fit(SoftSVM(n_iterations=100), X_moons, y_moons))

    kernel_svm = KernelSVM(kernel='gaussian', C=10.0, gamma=2.0, random_state=0)
    smo_times.append(time_fit(kernel_svm, X_moons, y_moons))
    support_counts.append(len(kernel_svm.support))

    print('%6d %11.3fs %9.3fs %9.3fs %9.2fs %6d'
          % (n_samples, perceptron_times[-1], hard_times[-1],
             soft_times[-1], smo_times[-1], support_counts[-1]))

# Prediction cost: a linear model evaluates one dot product per point,
# a kernel model one kernel per support vector per point.
X_moons, y_moons = moons(400)
linear_model = SoftSVM(n_iterations=100).fit(X_moons, y_moons)
kernel_model = KernelSVM(kernel='gaussian', C=10.0, gamma=2.0,
                         random_state=0).fit(X_moons, y_moons)
test = np.random.randn(5000, 2)

start = time.time()
linear_model.project(test)
linear_prediction = time.time() - start

start = time.time()
kernel_model.project(test)
kernel_prediction = time.time() - start

print('\nPrediction on 5000 points, model trained on 400:')
print('  linear: %.4fs' % linear_prediction)
print('  kernel: %.4fs, with %d support vectors'
      % (kernel_prediction, len(kernel_model.support)))
print('  sparsity: %d support vectors out of 400 points (%.0f%%)'
      % (len(kernel_model.support), 100 * len(kernel_model.support) / 400))

figure, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(SIZES, perceptron_times, marker='o', label='Perceptron')
axes[0].plot(SIZES, hard_times, marker='o', label='Hard margin (primal QP)')
axes[0].plot(SIZES, soft_times, marker='o', label='Soft margin (SGD)')
axes[0].plot(SIZES, smo_times, marker='o', label='Kernel SVM (SMO)')
axes[0].set_xlabel('training set size $n$')
axes[0].set_ylabel('fitting time (s)')
axes[0].set_yscale('log')
axes[0].set_title('Training cost', fontsize=11)
axes[0].legend(fontsize=8)

axes[1].plot(SIZES, support_counts, marker='o', color='#00008B',
             label='support vectors')
axes[1].plot(SIZES, SIZES, linestyle='--', color='gray', label='$n$')
axes[1].set_xlabel('training set size $n$')
axes[1].set_ylabel('number of points')
axes[1].set_title('Support vectors kept by the kernel SVM', fontsize=11)
axes[1].legend(fontsize=8)

figure.tight_layout()
figure.savefig('figures/06_complexity.png', dpi=130, bbox_inches='tight')
plt.close(figure)
print('\n  -> figures/06_complexity.png')
