"""
PyRust-ML: High-Performance Machine Learning Toolkit Using Rust

A Python library that accelerates machine learning algorithms using Rust bindings.
"""

from .linear_regression import RustLinearRegression
from .svm import RustSVM
from .kmeans import RustKMeans
from .benchmarks import benchmark_models

# Import optimized implementations if available
try:
    from .optimized import (
        OptimizedLinearRegression,
        OptimizedKMeans,
        benchmark_performance,
        PerformanceAnalyzer
    )
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False

# Convenient aliases for main classes
LinearRegression = RustLinearRegression
SVM = RustSVM
KMeans = RustKMeans

__version__ = "0.2.0"
__author__ = "PyRust-ML Team"

__all__ = [
    "RustLinearRegression",
    "RustSVM", 
    "RustKMeans",
    "LinearRegression",  # Alias
    "SVM",              # Alias
    "KMeans",           # Alias
    "benchmark_models",
]

# Add optimized implementations if available
if OPTIMIZED_AVAILABLE:
    __all__.extend([
        "OptimizedLinearRegression",
        "OptimizedKMeans",
        "benchmark_performance",
        "PerformanceAnalyzer"
    ])