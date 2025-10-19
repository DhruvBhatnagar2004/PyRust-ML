"""
Rust-accelerated K-Means clustering implementation
"""

import numpy as np
from typing import Union, Optional
try:
    from ._rust import KMeans as _RustKMeans
    RUST_AVAILABLE = True
except ImportError:
    # Fallback for development without compiled Rust extension
    _RustKMeans = None
    RUST_AVAILABLE = False

if not RUST_AVAILABLE:
    from .fallback import PythonKMeans as _FallbackKMeans


class RustKMeans:
    """
    K-Means clustering implementation using Rust for high performance.
    
    Uses K-means++ initialization and parallel computation for efficiency.
    
    Attributes:
        n_clusters (int): Number of clusters
        max_iters (int): Maximum number of iterations
        tol (float): Tolerance for convergence
        fitted_ (bool): Whether the model has been fitted
    
    Example:
        >>> from pyrustml import RustKMeans
        >>> model = RustKMeans(n_clusters=3)
        >>> labels = model.fit_predict(X)
    """
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 300, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        
        if RUST_AVAILABLE:
            self._model = _RustKMeans(n_clusters, max_iters)
            self._using_rust = True
        else:
            self._model = _FallbackKMeans(n_clusters, max_iters, tol)
            self._using_rust = False
        
        self.fitted_ = False
        self.cluster_centers_ = None
        self.labels_ = None
    
    def fit(self, X: Union[np.ndarray, list]) -> 'RustKMeans':
        """
        Fit the K-Means clustering model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            self: Returns the instance itself
        """
        # Convert inputs to proper format for Rust or use directly for Python
        if self._using_rust:
            if isinstance(X, np.ndarray):
                X = X.tolist()
            self._model.fit(X)
            # Get cluster centers
            centroids = self._model.get_centroids()
            self.cluster_centers_ = np.array(centroids)
        else:
            self._model.fit(X)
            self.cluster_centers_ = self._model.cluster_centers_
        
        self.fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels of shape (n_samples,)
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if self._using_rust:
            if isinstance(X, np.ndarray):
                X = X.tolist()
            labels = self._model.predict(X)
            return np.array(labels)
        else:
            return self._model.predict(X)
    
    def fit_predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Fit the model and predict cluster labels.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels of shape (n_samples,)
        """
        if self._using_rust:
            if isinstance(X, np.ndarray):
                X = X.tolist()
            labels = self._model.fit_predict(X)
            self.fitted_ = True
            # Get cluster centers
            centroids = self._model.get_centroids()
            self.cluster_centers_ = np.array(centroids)
            self.labels_ = np.array(labels)
        else:
            labels = self._model.fit_predict(X)
            self.fitted_ = True
            self.cluster_centers_ = self._model.cluster_centers_
            self.labels_ = labels
        
        return self.labels_
    
    def inertia(self, X: Union[np.ndarray, list]) -> float:
        """
        Calculate within-cluster sum of squares (inertia).
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Within-cluster sum of squares
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before calculating inertia")
        
        if self._using_rust:
            if isinstance(X, np.ndarray):
                X = X.tolist()
            return self._model.inertia(X)
        else:
            return self._model.inertia(X)
    
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
            'n_clusters': self.n_clusters,
            'max_iters': self.max_iters,
            'tol': self.tol
        }
    
    def set_params(self, **params) -> 'RustKMeans':
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
        self._model = _RustKMeans(self.n_clusters, self.max_iters, self.tol)
        self.fitted_ = False
        return self