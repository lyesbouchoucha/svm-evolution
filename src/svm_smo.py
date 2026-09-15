import numpy as np
from src.base import BaseClassifier
from src.kernels import linear_kernel

class KernelSVM(BaseClassifier):
    
    def __init__(self, C=1.0, kernel=linear_kernel, tol=1e-3, max_passes=5):
        super().__init__()
        self.C = C                  
        self.kernel = kernel       
        self.tol = tol              
        self.max_passes = max_passes 
        self.alphas = None          
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, _ = X.shape
        self.alphas = np.zeros(n_samples)
        self.bias = 0.0

        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                E_i = self._decision_function_single(self.X[i]) - self.y[i]
                
                if (self.y[i] * E_i < -self.tol and self.alphas[i] < self.C) or \
                   (self.y[i] * E_i > self.tol and self.alphas[i] > 0):
                    
                    
                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                    
                    E_j = self._decision_function_single(self.X[j]) - self.y[j]
                    
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                        
                    if L == H:
                        continue
                        
                    
                    eta = 2.0 * self.kernel(self.X[i], self.X[j]) - \
                          self.kernel(self.X[i], self.X[i]) - \
                          self.kernel(self.X[j], self.X[j])
                          
                    if eta >= 0:
                        continue
                        
          
                    self.alphas[j] -= (self.y[j] * (E_i - E_j)) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                        
                   
                    self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])
                    
                    
                    b1 = self.bias - E_i - self.y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(self.X[i], self.X[i]) - \
                         self.y[j] * (self.alphas[j] - alpha_j_old) * self.kernel(self.X[i], self.X[j])
                    b2 = self.bias - E_j - self.y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(self.X[i], self.X[j]) - \
                         self.y[j] * (self.alphas[j] - alpha_j_old) * self.kernel(self.X[j], self.X[j])
                         
                    if 0 < self.alphas[i] < self.C:
                        self.bias = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.bias = b2
                    else:
                        self.bias = (b1 + b2) / 2.0
                        
                    num_changed_alphas += 1
                    
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    def _decision_function_single(self, x):

        result = self.bias
        for i in range(len(self.alphas)):
            if self.alphas[i] > 0: 
                result += self.alphas[i] * self.y[i] * self.kernel(self.X[i], x)
        return result

    def predict(self, X):
        
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = np.sign(self._decision_function_single(X[i]))
        return y_pred
