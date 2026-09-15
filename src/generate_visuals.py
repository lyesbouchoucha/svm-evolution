"""Generate every figure used in the README.

Run from the project root:
    python generate_visuals.py
"""

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

from src.base import NotSeparableError
from src.perceptron import Perceptron
from src.svm_hard import HardSVM
from src.svm_soft_sgd import SoftSVM
from src.svm_smo import KernelSVM
from src.kernels import rbf_kernel


def plot_decision_boundary(model, X, y, title, filename, show_margins=True):
    """Plot the decision boundary, the margins and the support vectors."""
    plt.figure(figsize=(7, 6))

    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # decision_function, not predict: we need the continuous value of
    # w.x + b to draw the -1 and +1 contours, i.e. the margins. predict()
    # only returns the sign, which gives a boundary and nothing else.
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], alpha=0.12,
                 colors=["#1f77b4", "#d62728"])
    plt.contour(xx, yy, Z, levels=[0], colors="black", linewidths=2)
    if show_margins:
        plt.contour(xx, yy, Z, levels=[-1, 1], colors="black",
                    linewidths=1, linestyles="dashed")

    plt.scatter(X[y == -1, 0], X[y == -1, 1], c="#1f77b4", edgecolors="k",
                s=45, label="class -1")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="#d62728", edgecolors="k",
                s=45, label="class +1")

    support = getattr(model, "support_", None)
    if support is not None and len(support) > 0:
        plt.scatter(X[support, 0], X[support, 1], s=220, facecolors="none",
                    edgecolors="black", linewidths=1.6,
                    label=f"support vectors ({len(support)})")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # Equal aspect ratio: without it a correct margin looks distorted and
    # the hyperplane appears badly oriented.
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="upper right", fontsize=8)
    plt.savefig(f"figures/{filename}", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> figures/{filename}")


# Datasets are standardised so that a single learning rate works everywhere,
# and labels are mapped to {-1, +1} right after generation: sklearn returns
# {0, 1}, where y=0 silently cancels every gradient update.
def blobs(random_state=6):
    X, y = make_blobs(n_samples=100, centers=2, random_state=random_state)
    return StandardScaler().fit_transform(X), np.where(y == 0, -1.0, 1.0)


def moons(random_state=42):
    X, y = make_moons(n_samples=200, noise=0.15, random_state=random_state)
    return StandardScaler().fit_transform(X), np.where(y == 0, -1.0, 1.0)


def circles(random_state=42):
    X, y = make_circles(n_samples=200, noise=0.08, factor=0.45,
                        random_state=random_state)
    return StandardScaler().fit_transform(X), np.where(y == 0, -1.0, 1.0)


# --- SCENARIO 0: Why maximise the margin? -----------------------------------
# Both models reach 100% accuracy, but the perceptron stops at the first
# separator it finds: reordering the data changes its answer. The hard margin
# returns one deterministic hyperplane, defined by its support vectors.
print("[0] Why maximise the margin?")
X, y = blobs()

for tag, seed in (("a", 23), ("b", 20)):
    perm = np.random.default_rng(seed).permutation(len(y))
    model = Perceptron(n_iterations=100).fit(X[perm], y[perm])
    angle = np.degrees(np.arctan2(model.weights[1], model.weights[0]))
    plot_decision_boundary(
        model, X, y,
        f"Perceptron, data order #{seed} (angle {angle:.0f} deg)",
        f"00_perceptron_{tag}.png", show_margins=False)
    print(f"    perceptron order #{seed}: angle={angle:.1f} deg, "
          f"accuracy={model.score(X, y):.2f}")

hard = HardSVM().fit(X, y)
plot_decision_boundary(
    hard, X, y,
    f"Hard Margin SVM (margin {hard.margin_:.2f}, deterministic)",
    "00_hard_svm.png")
print(f"    hard SVM: margin={hard.margin_:.3f}, "
      f"{len(hard.support_)} support vectors")


# --- SCENARIO 1: The hard margin trap ---------------------------------------
# One point on the wrong side empties the feasible set: no (w, b) satisfies
# every constraint, so the problem has no solution at all.
print("\n[1] The hard margin trap")
X_out = np.vstack([X, [-0.5, -0.7]])
y_out = np.append(y, -1.0)

try:
    HardSVM().fit(X_out, y_out)
    print("    hard SVM converged (unexpected)")
except NotSeparableError as exc:
    print(f"    hard SVM -> NotSeparableError: {exc}")

soft = SoftSVM(learning_rate=0.01, lambda_param=0.01, n_iterations=400)
soft.fit(X_out, y_out)
plot_decision_boundary(
    soft, X_out, y_out,
    f"Soft Margin SVM: outlier absorbed (accuracy {soft.score(X_out, y_out):.0%})",
    "01_soft_svm_outlier.png")
print(f"    soft SVM: accuracy={soft.score(X_out, y_out):.3f}, "
      f"margin={soft.margin_:.3f}")


# --- SCENARIO 2: The linear limit (moons) -----------------------------------
print("\n[2] The linear limit (moons)")
X_moons, y_moons = moons()

soft_moons = SoftSVM(learning_rate=0.01, lambda_param=0.01, n_iterations=400)
soft_moons.fit(X_moons, y_moons)
plot_decision_boundary(
    soft_moons, X_moons, y_moons,
    f"Soft Margin SVM: linear boundary fails "
    f"(accuracy {soft_moons.score(X_moons, y_moons):.0%})",
    "02_soft_svm_moons.png")
print(f"    linear soft SVM: accuracy={soft_moons.score(X_moons, y_moons):.3f}")

# gamma=2, not the 0.1 default of many tutorials: squared distances here are
# of order 1 to 9, so a small gamma flattens the kernel into a linear model.
kernel_moons = KernelSVM(C=1.0, kernel=partial(rbf_kernel, gamma=2.0),
                         random_state=0)
kernel_moons.fit(X_moons, y_moons)
plot_decision_boundary(
    kernel_moons, X_moons, y_moons,
    f"SMO + RBF kernel, gamma=2 "
    f"(accuracy {kernel_moons.score(X_moons, y_moons):.0%})",
    "02_kernel_svm_moons.png", show_margins=False)
print(f"    RBF kernel SVM : accuracy={kernel_moons.score(X_moons, y_moons):.3f}, "
      f"iters={kernel_moons.n_iter_}, converged={kernel_moons.converged_}")


# --- SCENARIO 3: The ultimate test (circles) --------------------------------
print("\n[3] Concentric circles")
X_circ, y_circ = circles()

kernel_circ = KernelSVM(C=1.0, kernel=partial(rbf_kernel, gamma=2.0),
                        random_state=0)
kernel_circ.fit(X_circ, y_circ)
plot_decision_boundary(
    kernel_circ, X_circ, y_circ,
    f"SMO + RBF kernel: one class enclosing the other "
    f"(accuracy {kernel_circ.score(X_circ, y_circ):.0%})",
    "03_kernel_svm_circles.png", show_margins=False)
print(f"    RBF kernel SVM: accuracy={kernel_circ.score(X_circ, y_circ):.3f}")

print("\nAll figures written to figures/")
