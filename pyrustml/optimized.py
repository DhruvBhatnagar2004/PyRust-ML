"""
Optimized implementations with Rust performance enhancements

This module provides access to the high-performance Rust implementations
with advanced optimizations including SIMD operations, parallel processing,
and memory pooling.
"""

import warnings
import numpy as np
from typing import Optional, Tuple, List

try:
    from ._rust import (
        OptimizedLinearRegression as RustOptimizedLinearRegression,
        OptimizedKMeans as RustOptimizedKMeans,
        benchmark_optimized_linear_regression,
        benchmark_optimized_kmeans,
    )
    RUST_AVAILABLE = True
except ImportError as e:
    RUST_AVAILABLE = False
    warnings.warn(f"Rust optimized implementations not available: {e}")


class OptimizedLinearRegression:
    """
    Optimized Linear Regression with Rust performance enhancements
    
    Features:
    - SIMD-optimized matrix operations
    - Parallel gradient computation for large datasets
    - Adaptive learning rate adjustment
    - Xavier weight initialization
    - L2 regularization support
    - Convergence monitoring
    
    Parameters:
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations
    tolerance : float, default=1e-6
        Tolerance for convergence
    regularization : float, default=0.0
        L2 regularization strength
    use_parallel : bool, default=True
        Use parallel processing for large datasets
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 tolerance: float = 1e-6, regularization: float = 0.0,
                 use_parallel: bool = True):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.regularization = regularization
        self.use_parallel = use_parallel
        
        if RUST_AVAILABLE:
            self.model = RustOptimizedLinearRegression(
                learning_rate, max_iter, tolerance, regularization, use_parallel
            )
        else:
            # Fallback to basic Python implementation
            from .fallback import LinearRegression as FallbackLinearRegression
            self.model = FallbackLinearRegression(learning_rate, max_iter)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OptimizedLinearRegression':
        """
        Fit the optimized linear regression model
        
        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        -------
        self : OptimizedLinearRegression
            Fitted model
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the optimized linear model
        
        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        -------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.model.predict(X)
    
    def get_convergence_history(self) -> np.ndarray:
        """Get convergence history for analysis"""
        if RUST_AVAILABLE:
            return self.model.get_convergence_history()
        else:
            return np.array([])
    
    def get_coefficients(self) -> Tuple[np.ndarray, float]:
        """Get model coefficients"""
        if RUST_AVAILABLE:
            return self.model.get_coefficients()
        else:
            return np.array([]), 0.0
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared score"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if RUST_AVAILABLE:
            return self.model.score(X, y)
        else:
            y_pred = self.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot


class OptimizedKMeans:
    """
    Optimized K-Means clustering with Rust performance enhancements
    
    Features:
    - K-means++ initialization for better clustering
    - SIMD-optimized distance computations
    - Parallel assignment for large datasets
    - Multiple initialization runs
    - Convergence monitoring
    - Silhouette score calculation
    
    Parameters:
    ----------
    n_clusters : int, default=3
        Number of clusters
    max_iter : int, default=300
        Maximum number of iterations
    tolerance : float, default=1e-6
        Tolerance for convergence
    n_init : int, default=10
        Number of initialization runs
    use_parallel : bool, default=True
        Use parallel processing for large datasets
    use_kmeans_plus_plus : bool, default=True
        Use K-means++ initialization
    """
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 tolerance: float = 1e-6, n_init: int = 10,
                 use_parallel: bool = True, use_kmeans_plus_plus: bool = True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_init = n_init
        self.use_parallel = use_parallel
        self.use_kmeans_plus_plus = use_kmeans_plus_plus
        
        if RUST_AVAILABLE:
            self.model = RustOptimizedKMeans(
                n_clusters, max_iter, tolerance, n_init, 
                use_parallel, use_kmeans_plus_plus
            )
        else:
            # Fallback to basic Python implementation
            from .fallback import KMeans as FallbackKMeans
            self.model = FallbackKMeans(n_clusters, max_iter)
    
    def fit(self, X: np.ndarray) -> 'OptimizedKMeans':
        """
        Fit the optimized K-means model
        
        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        -------
        self : OptimizedKMeans
            Fitted model
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        -------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.model.predict(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and predict cluster labels
        
        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        -------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.model.fit_predict(X)
    
    def get_centroids(self) -> np.ndarray:
        """Get cluster centroids"""
        if RUST_AVAILABLE:
            return self.model.get_centroids()
        else:
            return np.array([])
    
    def get_inertia(self) -> float:
        """Get final inertia"""
        if RUST_AVAILABLE:
            return self.model.get_inertia()
        else:
            return 0.0
    
    def get_convergence_history(self) -> np.ndarray:
        """Get convergence history"""
        if RUST_AVAILABLE:
            return self.model.get_convergence_history()
        else:
            return np.array([])
    
    def silhouette_score(self, X: np.ndarray) -> float:
        """Calculate silhouette score"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if RUST_AVAILABLE:
            return self.model.silhouette_score(X)
        else:
            return 0.0


def benchmark_performance(algorithm: str, X: np.ndarray, y: Optional[np.ndarray] = None,
                         n_runs: int = 5, **kwargs) -> Tuple[float, List[float]]:
    """
    Benchmark optimized algorithm performance
    
    Parameters:
    ----------
    algorithm : str
        Algorithm name ('linear_regression' or 'kmeans')
    X : array-like
        Input data
    y : array-like, optional
        Target data (for supervised learning)
    n_runs : int, default=5
        Number of benchmark runs
    **kwargs : dict
        Algorithm-specific parameters
        
    Returns:
    -------
    avg_time : float
        Average execution time
    times : list
        Individual run times
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust implementations not available for benchmarking")
    
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if algorithm.lower() == 'linear_regression':
        if y is None:
            raise ValueError("y is required for linear regression benchmarking")
        y = np.asarray(y, dtype=np.float64)
        return benchmark_optimized_linear_regression(X, y, n_runs)
    
    elif algorithm.lower() == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 3)
        return benchmark_optimized_kmeans(X, n_clusters, n_runs)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# Performance analysis utilities
class PerformanceAnalyzer:
    """Analyze and compare performance of optimized implementations"""
    
    @staticmethod
    def compare_implementations(X: np.ndarray, y: Optional[np.ndarray] = None,
                              algorithms: Optional[List[str]] = None) -> dict:
        """
        Compare performance between standard and optimized implementations
        
        Parameters:
        ----------
        X : array-like
            Input data
        y : array-like, optional
            Target data
        algorithms : list, optional
            Algorithms to compare
            
        Returns:
        -------
        results : dict
            Performance comparison results
        """
        if algorithms is None:
            algorithms = ['linear_regression', 'kmeans']
        
        results = {}
        
        for algo in algorithms:
            try:
                if algo == 'linear_regression' and y is not None:
                    avg_time, times = benchmark_performance(algo, X, y, n_runs=5)
                    results[algo] = {
                        'avg_time': avg_time,
                        'std_time': np.std(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'speedup': 'N/A'  # Would need baseline comparison
                    }
                elif algo == 'kmeans':
                    avg_time, times = benchmark_performance(algo, X, n_runs=5)
                    results[algo] = {
                        'avg_time': avg_time,
                        'std_time': np.std(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'speedup': 'N/A'
                    }
            except Exception as e:
                results[algo] = {'error': str(e)}
        
        return results
    
    @staticmethod
    def memory_analysis(X: np.ndarray) -> dict:
        """
        Analyze memory usage characteristics
        
        Parameters:
        ----------
        X : array-like
            Input data
            
        Returns:
        -------
        analysis : dict
            Memory analysis results
        """
        X = np.asarray(X, dtype=np.float64)
        
        memory_mb = X.nbytes / (1024 * 1024)
        
        return {
            'data_shape': X.shape,
            'data_size_mb': memory_mb,
            'dtype': str(X.dtype),
            'memory_layout': 'C' if X.flags.c_contiguous else 'F' if X.flags.f_contiguous else 'Neither',
            'estimated_working_memory_mb': memory_mb * 3,  # Rough estimate
        }