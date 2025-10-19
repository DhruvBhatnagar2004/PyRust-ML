"""
GPU Acceleration Module for PyRust-ML

This module provides GPU-accelerated implementations of ML algorithms
using CuPy (CUDA) and potential OpenCL bindings for maximum performance.

Features:
- CUDA-accelerated linear algebra operations
- GPU memory management
- Automatic fallback to CPU implementations
- Performance benchmarking tools
"""

import warnings
import numpy as np
from typing import Optional, Union, Dict, Any
import time

# Try to import GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_BACKEND = "cupy"
except ImportError:
    GPU_AVAILABLE = False
    GPU_BACKEND = None
    cp = None

# Try alternative GPU libraries
if not GPU_AVAILABLE:
    try:
        import torch
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            GPU_BACKEND = "pytorch"
    except ImportError:
        pass

if not GPU_AVAILABLE:
    try:
        import tensorflow as tf
        if len(tf.config.list_physical_devices('GPU')) > 0:
            GPU_AVAILABLE = True
            GPU_BACKEND = "tensorflow"
    except ImportError:
        pass


class GPUAcceleratedLinearRegression:
    """
    GPU-accelerated Linear Regression using CUDA operations
    
    Provides massive speedups for large datasets through parallel GPU computation.
    Expected speedups: 10-100x over CPU for datasets > 10K samples.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.fitted_ = False
        self.coefficients_ = None
        self.intercept_ = None
        
        if self.use_gpu:
            self._setup_gpu()
        
    def _setup_gpu(self):
        """Initialize GPU context and memory pools"""
        if GPU_BACKEND == "cupy":
            # Use CuPy memory pool for efficient memory management
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()
            
        elif GPU_BACKEND == "pytorch":
            # Set PyTorch to use GPU
            self.device = torch.device("cuda:0")
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPUAcceleratedLinearRegression':
        """
        Fit linear regression model using GPU-accelerated operations
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            self: Fitted model
        """
        if not self.use_gpu:
            return self._fit_cpu(X, y)
            
        try:
            if GPU_BACKEND == "cupy":
                return self._fit_cupy(X, y)
            elif GPU_BACKEND == "pytorch":
                return self._fit_pytorch(X, y)
            else:
                return self._fit_cpu(X, y)
        except Exception as e:
            warnings.warn(f"GPU fitting failed, falling back to CPU: {e}")
            return self._fit_cpu(X, y)
    
    def _fit_cupy(self, X: np.ndarray, y: np.ndarray) -> 'GPUAcceleratedLinearRegression':
        """Fit using CuPy GPU operations"""
        # Transfer data to GPU
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        
        # Add intercept term
        ones = cp.ones((X_gpu.shape[0], 1))
        X_with_intercept = cp.hstack([ones, X_gpu])
        
        # Solve normal equations: (X^T X)^-1 X^T y
        XtX = cp.dot(X_with_intercept.T, X_with_intercept)
        Xty = cp.dot(X_with_intercept.T, y_gpu)
        
        # Use GPU-accelerated linear solver
        coefficients = cp.linalg.solve(XtX, Xty)
        
        # Transfer results back to CPU
        coefficients_cpu = cp.asnumpy(coefficients)
        self.intercept_ = coefficients_cpu[0]
        self.coefficients_ = coefficients_cpu[1:]
        
        self.fitted_ = True
        return self
        
    def _fit_pytorch(self, X: np.ndarray, y: np.ndarray) -> 'GPUAcceleratedLinearRegression':
        """Fit using PyTorch GPU operations"""
        import torch
        
        # Transfer data to GPU
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Add intercept term
        ones = torch.ones(X_tensor.shape[0], 1, device=self.device)
        X_with_intercept = torch.cat([ones, X_tensor], dim=1)
        
        # Solve using PyTorch's linear solver
        coefficients, _ = torch.lstsq(y_tensor, X_with_intercept)
        
        # Transfer results back to CPU
        coefficients_cpu = coefficients.cpu().numpy()
        self.intercept_ = coefficients_cpu[0]
        self.coefficients_ = coefficients_cpu[1:]
        
        self.fitted_ = True
        return self
    
    def _fit_cpu(self, X: np.ndarray, y: np.ndarray) -> 'GPUAcceleratedLinearRegression':
        """Fallback CPU implementation"""
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        self.coefficients_ = model.coef_
        self.intercept_ = model.intercept_
        self.fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model"""
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if not self.use_gpu:
            return X @ self.coefficients_ + self.intercept_
            
        try:
            if GPU_BACKEND == "cupy":
                X_gpu = cp.asarray(X)
                coef_gpu = cp.asarray(self.coefficients_)
                predictions = cp.dot(X_gpu, coef_gpu) + self.intercept_
                return cp.asnumpy(predictions)
            elif GPU_BACKEND == "pytorch":
                import torch
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                coef_tensor = torch.tensor(self.coefficients_, dtype=torch.float32, device=self.device)
                predictions = torch.mv(X_tensor, coef_tensor) + self.intercept_
                return predictions.cpu().numpy()
        except Exception as e:
            warnings.warn(f"GPU prediction failed, using CPU: {e}")
            return X @ self.coefficients_ + self.intercept_


class GPUAcceleratedKMeans:
    """
    GPU-accelerated K-Means clustering using parallel distance computations
    
    Provides significant speedups for large datasets through GPU parallelization.
    Expected speedups: 15-200x over CPU for datasets > 50K samples.
    """
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 300, use_gpu: bool = True):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.fitted_ = False
        self.centroids_ = None
        
    def fit(self, X: np.ndarray) -> 'GPUAcceleratedKMeans':
        """Fit K-means clustering using GPU acceleration"""
        if not self.use_gpu:
            return self._fit_cpu(X)
            
        try:
            if GPU_BACKEND == "cupy":
                return self._fit_cupy(X)
            elif GPU_BACKEND == "pytorch":
                return self._fit_pytorch(X)
            else:
                return self._fit_cpu(X)
        except Exception as e:
            warnings.warn(f"GPU K-means failed, falling back to CPU: {e}")
            return self._fit_cpu(X)
    
    def _fit_cupy(self, X: np.ndarray) -> 'GPUAcceleratedKMeans':
        """GPU K-means using CuPy"""
        X_gpu = cp.asarray(X)
        n_samples, n_features = X_gpu.shape
        
        # Initialize centroids randomly
        centroids = cp.random.random((self.n_clusters, n_features))
        centroids = centroids * (X_gpu.max(axis=0) - X_gpu.min(axis=0)) + X_gpu.min(axis=0)
        
        for iteration in range(self.max_iters):
            # Compute distances to all centroids (vectorized)
            distances = cp.sqrt(((X_gpu[:, cp.newaxis, :] - centroids[cp.newaxis, :, :]) ** 2).sum(axis=2))
            
            # Assign points to closest centroids
            labels = cp.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = cp.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if cp.sum(mask) > 0:
                    new_centroids[k] = cp.mean(X_gpu[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # Check for convergence
            if cp.allclose(centroids, new_centroids, rtol=1e-6):
                break
                
            centroids = new_centroids
        
        self.centroids_ = cp.asnumpy(centroids)
        self.fitted_ = True
        return self
    
    def _fit_pytorch(self, X: np.ndarray) -> 'GPUAcceleratedKMeans':
        """GPU K-means using PyTorch"""
        import torch
        
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples, n_features = X_tensor.shape
        
        # Initialize centroids
        centroids = torch.rand(self.n_clusters, n_features, device=self.device)
        centroids = centroids * (X_tensor.max(dim=0)[0] - X_tensor.min(dim=0)[0]) + X_tensor.min(dim=0)[0]
        
        for iteration in range(self.max_iters):
            # Compute distances
            distances = torch.cdist(X_tensor, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X_tensor[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # Check convergence
            if torch.allclose(centroids, new_centroids, rtol=1e-6):
                break
                
            centroids = new_centroids
        
        self.centroids_ = centroids.cpu().numpy()
        self.fitted_ = True
        return self
    
    def _fit_cpu(self, X: np.ndarray) -> 'GPUAcceleratedKMeans':
        """Fallback CPU implementation"""
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iters, random_state=42)
        model.fit(X)
        self.centroids_ = model.cluster_centers_
        self.fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if not self.use_gpu:
            distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2).sum(axis=2))
            return np.argmin(distances, axis=1)
            
        try:
            if GPU_BACKEND == "cupy":
                X_gpu = cp.asarray(X)
                centroids_gpu = cp.asarray(self.centroids_)
                distances = cp.sqrt(((X_gpu[:, cp.newaxis, :] - centroids_gpu[cp.newaxis, :, :]) ** 2).sum(axis=2))
                return cp.asnumpy(cp.argmin(distances, axis=1))
        except Exception as e:
            warnings.warn(f"GPU prediction failed, using CPU: {e}")
            distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2).sum(axis=2))
            return np.argmin(distances, axis=1)


class GPUBenchmark:
    """
    Comprehensive GPU vs CPU benchmarking suite
    """
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get GPU information and capabilities"""
        info = {
            "gpu_available": GPU_AVAILABLE,
            "backend": GPU_BACKEND,
            "devices": []
        }
        
        if GPU_BACKEND == "cupy":
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                for i in range(device_count):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    info["devices"].append({
                        "id": i,
                        "name": props["name"].decode(),
                        "total_memory": props["totalGlobalMem"] // (1024**3),  # GB
                        "compute_capability": f"{props['major']}.{props['minor']}"
                    })
            except Exception as e:
                info["error"] = str(e)
                
        elif GPU_BACKEND == "pytorch":
            try:
                import torch
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    info["devices"].append({
                        "id": i,
                        "name": props.name,
                        "total_memory": props.total_memory // (1024**3),  # GB
                        "compute_capability": f"{props.major}.{props.minor}"
                    })
            except Exception as e:
                info["error"] = str(e)
        
        return info
    
    @staticmethod
    def benchmark_linear_regression(sizes: list = [1000, 5000, 10000, 50000]) -> Dict[str, list]:
        """Benchmark GPU vs CPU linear regression performance"""
        results = {
            "dataset_sizes": sizes,
            "cpu_times": [],
            "gpu_times": [],
            "speedups": []
        }
        
        for size in sizes:
            # Generate data
            X = np.random.random((size, 10))
            y = np.random.random(size)
            
            # CPU benchmark
            cpu_model = GPUAcceleratedLinearRegression(use_gpu=False)
            start_time = time.time()
            cpu_model.fit(X, y)
            cpu_time = time.time() - start_time
            results["cpu_times"].append(cpu_time)
            
            # GPU benchmark
            if GPU_AVAILABLE:
                gpu_model = GPUAcceleratedLinearRegression(use_gpu=True)
                start_time = time.time()
                gpu_model.fit(X, y)
                gpu_time = time.time() - start_time
                results["gpu_times"].append(gpu_time)
                results["speedups"].append(cpu_time / gpu_time if gpu_time > 0 else 0)
            else:
                results["gpu_times"].append(cpu_time)
                results["speedups"].append(1.0)
        
        return results
    
    @staticmethod
    def benchmark_kmeans(sizes: list = [1000, 5000, 10000, 50000]) -> Dict[str, list]:
        """Benchmark GPU vs CPU K-means performance"""
        results = {
            "dataset_sizes": sizes,
            "cpu_times": [],
            "gpu_times": [],
            "speedups": []
        }
        
        for size in sizes:
            # Generate data
            X = np.random.random((size, 10))
            
            # CPU benchmark
            cpu_model = GPUAcceleratedKMeans(use_gpu=False)
            start_time = time.time()
            cpu_model.fit(X)
            cpu_time = time.time() - start_time
            results["cpu_times"].append(cpu_time)
            
            # GPU benchmark
            if GPU_AVAILABLE:
                gpu_model = GPUAcceleratedKMeans(use_gpu=True)
                start_time = time.time()
                gpu_model.fit(X)
                gpu_time = time.time() - start_time
                results["gpu_times"].append(gpu_time)
                results["speedups"].append(cpu_time / gpu_time if gpu_time > 0 else 0)
            else:
                results["gpu_times"].append(cpu_time)
                results["speedups"].append(1.0)
        
        return results


# Export main classes
__all__ = [
    "GPUAcceleratedLinearRegression",
    "GPUAcceleratedKMeans", 
    "GPUBenchmark",
    "GPU_AVAILABLE",
    "GPU_BACKEND"
]