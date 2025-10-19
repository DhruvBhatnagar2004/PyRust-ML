"""
Python fallback implementations for when Rust extensions are not available.
This allows the project to run and demonstrate functionality without Rust compilation.
"""

import numpy as np
from typing import Union, List


class PythonLinearRegression:
    """Python implementation of Linear Regression using Ordinary Least Squares"""
    
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.fitted_ = False
    
    def fit(self, X: Union[np.ndarray, List[List[float]]], y: Union[np.ndarray, List[float]]):
        """Fit the linear regression model"""
        X = np.array(X)
        y = np.array(y)
        
        # Add bias column
        X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
        
        # Normal equation: theta = (X^T * X)^(-1) * X^T * y
        try:
            theta = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
            self.weights = theta[:-1]
            self.bias = theta[-1]
            self.fitted_ = True
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.weights = theta[:-1]
            self.bias = theta[-1]
            self.fitted_ = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Make predictions"""
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def score(self, X: Union[np.ndarray, List[List[float]]], y: Union[np.ndarray, List[float]]) -> float:
        """Calculate R-squared score"""
        y_pred = self.predict(X)
        y = np.array(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class PythonSVM:
    """Python implementation of Support Vector Machine with linear kernel"""
    
    def __init__(self, learning_rate=0.01, lambda_reg=0.01, max_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.max_iters = max_iters
        self.weights = None
        self.bias = 0.0
        self.fitted_ = False
    
    def fit(self, X: Union[np.ndarray, List[List[float]]], y: Union[np.ndarray, List[float]]):
        """Fit the SVM model"""
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1/1
        y = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Gradient descent
        for _ in range(self.max_iters):
            for i in range(n_samples):
                decision = np.dot(X[i], self.weights) + self.bias
                
                if y[i] * decision < 1:
                    # Misclassified or in margin
                    self.weights -= self.learning_rate * (
                        self.lambda_reg * self.weights - y[i] * X[i]
                    )
                    self.bias -= self.learning_rate * (-y[i])
                else:
                    # Correctly classified
                    self.weights -= self.learning_rate * self.lambda_reg * self.weights
        
        self.fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Make predictions"""
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        decisions = X @ self.weights + self.bias
        return np.where(decisions >= 0, 1, -1)
    
    def decision_function(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Get decision function values"""
        if not self.fitted_:
            raise ValueError("Model must be fitted before calculating decision function")
        
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def score(self, X: Union[np.ndarray, List[List[float]]], y: Union[np.ndarray, List[float]]) -> float:
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        y = np.array(y)
        y_binary = np.where(y <= 0, -1, 1)
        return np.mean(y_pred == y_binary)


class PythonKMeans:
    """Python implementation of K-Means clustering"""
    
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.fitted_ = False
    
    def fit(self, X: Union[np.ndarray, List[List[float]]]):
        """Fit the K-Means model"""
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        np.random.seed(42)
        self.cluster_centers_ = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centers = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tol):
                break
            
            self.cluster_centers_ = new_centers
        
        self.fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Predict cluster labels"""
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def fit_predict(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Fit the model and predict cluster labels"""
        self.fit(X)
        return self.labels_
    
    def inertia(self, X: Union[np.ndarray, List[List[float]]]) -> float:
        """Calculate within-cluster sum of squares"""
        if not self.fitted_:
            raise ValueError("Model must be fitted before calculating inertia")
        
        X = np.array(X)
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.cluster_centers_[k])**2)
        return inertia