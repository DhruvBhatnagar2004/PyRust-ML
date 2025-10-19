"""
Rust-accelerated Support Vector Machine implementation
"""

import numpy as np
from typing import Union, Optional
try:
    from ._rust import SVM as _RustSVM
    RUST_AVAILABLE = True
except ImportError:
    # Fallback for development without compiled Rust extension
    _RustSVM = None
    RUST_AVAILABLE = False

if not RUST_AVAILABLE:
    from .fallback import PythonSVM as _FallbackSVM


class RustSVM:
    """
    Support Vector Machine implementation using Rust for high performance.
    
    Uses linear kernel and perceptron-like optimization.
    
    Attributes:
        kernel (str): Kernel type ('linear' supported)
        C (float): Regularization parameter
        gamma (float): Kernel coefficient
        fitted_ (bool): Whether the model has been fitted
    
    Example:
        >>> from pyrustml import RustSVM
        >>> model = RustSVM(kernel="linear", C=1.0)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, kernel: str = "linear", C: float = 1.0, gamma: float = 1.0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        
        if RUST_AVAILABLE:
            self._model = _RustSVM(kernel, C, gamma)
            self._using_rust = True
        else:
            self._model = _FallbackSVM(kernel, C, gamma)
            self._using_rust = False
        self.fitted_ = False
    
    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'RustSVM':
        """
        Fit the SVM model.
        
        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data targets of shape (n_samples,) with values -1 or 1
            
        Returns:
            self: Returns the instance itself
        """
        # Convert inputs to proper format for Rust or use directly for Python
        if self._using_rust:
            if isinstance(X, np.ndarray):
                X = X.tolist()
            if isinstance(y, np.ndarray):
                y = y.tolist()
        
        # Validate input shapes
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        self._model.fit(X, y)
        self.fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels (-1 or 1) of shape (n_samples,)
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if self._using_rust:
            if isinstance(X, np.ndarray):
                X = X.tolist()
            predictions = self._model.predict(X)
            return np.array(predictions)
        else:
            return self._model.predict(X)
    
    def decision_function(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Get decision function values (distances from hyperplane).
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Decision function values of shape (n_samples,)
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before calculating decision function")
            
        if isinstance(X, np.ndarray):
            X = X.tolist()
            
        decisions = self._model.decision_function(X)
        return np.array(decisions)
    
    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        """
        Calculate the accuracy score.
        
        Args:
            X: Test data features of shape (n_samples, n_features)
            y: Test data targets of shape (n_samples,)
            
        Returns:
            Accuracy score
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before calculating score")
            
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()
            
        return self._model.score(X, y)
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.
        
        Args:
            deep: If True, will return the parameters for this estimator
                 and contained subobjects that are estimators.
                 
        Returns:
            Dictionary of parameter names mapped to their values
        """
        return {
            'learning_rate': self.learning_rate,
            'lambda_reg': self.lambda_reg,
            'max_iters': self.max_iters
        }
    
    def set_params(self, **params) -> 'RustSVM':
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Estimator parameters
            
        Returns:
            self: Returns the instance itself
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        
        # Recreate the Rust model with new parameters
        self._model = _RustSVM(self.learning_rate, self.lambda_reg, self.max_iters)
        self.fitted_ = False
        return self