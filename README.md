# svm-evolution-from-scratch

This repository provides a from-scratch implementation of linear and non-linear classifiers using Python and NumPy. The objective of this project is to demonstrate the mathematical and algorithmic progression that led to modern Support Vector Machines (SVM). 

This project builds the algorithms step-by-step, highlighting the limitations of early models and how subsequent mathematical innovations resolved them.

## Implemented Algorithms

The project is structured around four core implementations, each addressing a specific mathematical challenge:

1. **Perceptron (`src/perceptron.py`)**
   * Implements the classic error-driven learning algorithm.
   * **Limitation:** Finds any separating hyperplane without optimizing for the margin, leading to poor generalization. It fails to converge on non-linearly separable data.

2. **Hard Margin SVM (`src/svm_hard.py`)**
   * Introduces margin maximization. The objective is updated to ensure data points are not just correctly classified, but lie outside a defined margin.
   * **Limitation:** Strictly requires linearly separable data. The presence of outliers or overlapping classes prevents convergence.

3. **Soft Margin SVM (`src/svm_soft_sgd.py`)**
   * Introduces the Hinge Loss function and an L2 regularization term to allow for misclassifications (soft margin). Optimized using Stochastic Gradient Descent (SGD).
   * **Limitation:** While robust to noise and outliers, it is still restricted to linear decision boundaries.

4. **Kernel SVM with SMO Algorithm (`src/svm_smo.py` & `src/kernels.py`)**
   * Solves the dual optimization problem using John Platt's Sequential Minimal Optimization (SMO) algorithm.
   * Leverages the Kernel Trick (Linear, Polynomial, and Gaussian) to project data into higher dimensions, allowing the model to learn complex, non-linear decision boundaries.

## Repository Structure

```text
.
├── src/
│   ├── perceptron.py       # Standard Perceptron implementation
│   ├── svm_hard.py         # Hard Margin SVM 
│   ├── svm_soft_sgd.py     # Soft Margin SVM optimized via SGD
│   ├── svm_smo.py          # Dual SVM optimized via SMO algorithm
│   └── kernels.py          # Linear, Polynomial, and gaussian kernel functions
├── notebooks/
│   └── demonstrations.ipynb # Visualizations of decision boundaries and margins
├── tests/
│   └── test_models.py      # Unit tests verifying algorithm correctness
├── requirements.txt        # Project dependencies
└── README.md
