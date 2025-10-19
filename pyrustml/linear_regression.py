"""
Rust-accelerated Linear Regression implementation
"""

import numpy as np
from typing import Union, Optional
try:
    from ._rust import LinearRegression as _RustLinearRegression
    RUST_AVAILABLE = True
except ImportError:
    # Fallback for development without compiled Rust extension
    _RustLinearRegression = None
    RUST_AVAILABLE = False

if not RUST_AVAILABLE:
    from .fallback import PythonLinearRegression as _FallbackLinearRegression


class RustLinearRegression:
    """
    Linear Regression implementation using Rust for high performance.
    
    Uses Ordinary Least Squares (OLS) method for fitting the linear model.
    
    Attributes:
        fitted_ (bool): Whether the model has been fitted
    
    Example:
        >>> from pyrustml import RustLinearRegression
        >>> model = RustLinearRegression()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self):
        if RUST_AVAILABLE:
            self._model = _RustLinearRegression()
            self._using_rust = True
        else:
            self._model = _FallbackLinearRegression()
            self._using_rust = False
        self.fitted_ = False
    
    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'RustLinearRegression':
        """
        Fit the linear regression model.
        
        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data targets of shape (n_samples,)
            
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
            Predicted values of shape (n_samples,)
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
    
    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        """
        Calculate the R-squared score.
        
        Args:
            X: Test data features of shape (n_samples, n_features)
            y: Test data targets of shape (n_samples,)
            
        Returns:
            R-squared score
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before calculating score")
        
        if self._using_rust:
            if isinstance(X, np.ndarray):
                X = X.tolist()
            if isinstance(y, np.ndarray):
                y = y.tolist()
            return self._model.score(X, y)
        else:
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
        return {}
    
    def set_params(self, **params) -> 'RustLinearRegression':
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Estimator parameters
            
        Returns:
            self: Returns the instance itself
        """
        # No parameters to set for basic linear regression
        return self