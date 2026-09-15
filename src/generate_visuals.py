import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles

# Import your custom models (ensure these files are accessible in your environment)
from src.soft_svm import SoftSVM
from src.svm_smo import KernelSVM
from src.kernels import gaussian_kernel

def plot_decision_boundary(model, X, y, title, filename):
    """Utility function to plot the decision boundary and save the image."""
    plt.figure(figsize=(8, 6))
    
    # Create a mesh grid to evaluate the model
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict on the entire grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and the data points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# --- SCENARIO 1: The Hard Margin (Outliers) ---
# Generate linearly separable data, then add one extreme outlier
X_blobs, y_blobs = make_blobs(n_samples=100, centers=2, random_state=6)
y_blobs = np.where(y_blobs == 0, -1, 1)

X_outlier = np.vstack([X_blobs, [6.5, -4.5]]) 
y_outlier = np.append(y_blobs, -1)

soft_model = SoftSVM(learning_rate=0.001, lambda_param=0.01)
soft_model.fit(X_outlier, y_outlier)
plot_decision_boundary(soft_model, X_outlier, y_outlier, 
                       "Soft SVM: Robustness to Outliers", 
                       "scenario1_outlier.png")

# --- SCENARIO 2: The Linear Limit (Moons Dataset) ---
X_moons, y_moons = make_moons(n_samples=100, noise=0.1, random_state=42)
y_moons = np.where(y_moons == 0, -1, 1)

# Soft SVM (Linear) attempting to solve non-linear data
soft_model_moons = SoftSVM(learning_rate=0.001, lambda_param=0.01)
soft_model_moons.fit(X_moons, y_moons)
plot_decision_boundary(soft_model_moons, X_moons, y_moons, 
                       "Soft SVM: Failing on Non-Linear Data (Moons)", 
                       "scenario2_linear_limit.png")

# Kernel SVM (Gaussian) solving the same data
kernel_model_moons = KernelSVM(C=1.0, kernel=gaussian_kernel)
kernel_model_moons.fit(X_moons, y_moons)
plot_decision_boundary(kernel_model_moons, X_moons, y_moons, 
                       "Kernel SVM (Gaussian): Solving Non-Linear Data", 
                       "scenario2_gaussian_moons.png")

# --- SCENARIO 3: The Ultimate Test (Circles Dataset) ---
X_circles, y_circles = make_circles(n_samples=100, noise=0.1, factor=0.2, random_state=42)
y_circles = np.where(y_circles == 0, -1, 1)

kernel_model_circles = KernelSVM(C=1.0, kernel=gaussian_kernel)
kernel_model_circles.fit(X_circles, y_circles)
plot_decision_boundary(kernel_model_circles, X_circles, y_circles, 
                       "Kernel SVM (Gaussian): Concentric Circles", 
                       "scenario3_gaussian_circles.png")
