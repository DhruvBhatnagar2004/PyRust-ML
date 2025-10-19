"""
Benchmarking utilities for comparing Rust vs Python ML implementations
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LinearRegression as SkLearnLinearRegression
from sklearn.svm import SVC as SkLearnSVM
from sklearn.cluster import KMeans as SkLearnKMeans
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.datasets import make_regression, make_classification, make_blobs

try:
    from .linear_regression import RustLinearRegression
    from .svm import RustSVM
    from .kmeans import RustKMeans
except ImportError:
    # Fallback for when Rust extensions aren't available
    RustLinearRegression = None
    RustSVM = None
    RustKMeans = None


class BenchmarkResult:
    """Container for benchmark results"""
    
    def __init__(self, algorithm: str, implementation: str, 
                 fit_time: float, predict_time: float, 
                 accuracy: Optional[float] = None, 
                 r2_score: Optional[float] = None,
                 mse: Optional[float] = None):
        self.algorithm = algorithm
        self.implementation = implementation
        self.fit_time = fit_time
        self.predict_time = predict_time
        self.total_time = fit_time + predict_time
        self.accuracy = accuracy
        self.r2_score = r2_score
        self.mse = mse
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy DataFrame creation"""
        return {
            'Algorithm': self.algorithm,
            'Implementation': self.implementation,
            'Fit Time (s)': self.fit_time,
            'Predict Time (s)': self.predict_time,
            'Total Time (s)': self.total_time,
            'Accuracy': self.accuracy,
            'R² Score': self.r2_score,
            'MSE': self.mse
        }


def benchmark_linear_regression(X_train: np.ndarray, X_test: np.ndarray, 
                              y_train: np.ndarray, y_test: np.ndarray) -> List[BenchmarkResult]:
    """
    Benchmark linear regression implementations
    
    Args:
        X_train: Training features
        X_test: Test features  
        y_train: Training targets
        y_test: Test targets
        
    Returns:
        List of benchmark results
    """
    results = []
    
    # Benchmark Scikit-learn
    sklearn_model = SkLearnLinearRegression()
    
    # Fit timing
    start_time = time.time()
    sklearn_model.fit(X_train, y_train)
    sklearn_fit_time = time.time() - start_time
    
    # Predict timing
    start_time = time.time()
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_predict_time = time.time() - start_time
    
    sklearn_r2 = r2_score(y_test, sklearn_pred)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    
    results.append(BenchmarkResult(
        "Linear Regression", "Scikit-learn", 
        sklearn_fit_time, sklearn_predict_time,
        r2_score=sklearn_r2, mse=sklearn_mse
    ))
    
    # Benchmark Rust implementation (if available)
    if RustLinearRegression is not None:
        try:
            rust_model = RustLinearRegression()
            
            # Fit timing
            start_time = time.time()
            rust_model.fit(X_train, y_train)
            rust_fit_time = time.time() - start_time
            
            # Predict timing
            start_time = time.time()
            rust_pred = rust_model.predict(X_test)
            rust_predict_time = time.time() - start_time
            
            rust_r2 = r2_score(y_test, rust_pred)
            rust_mse = mean_squared_error(y_test, rust_pred)
            
            results.append(BenchmarkResult(
                "Linear Regression", "Rust", 
                rust_fit_time, rust_predict_time,
                r2_score=rust_r2, mse=rust_mse
            ))
        except Exception as e:
            print(f"Rust Linear Regression benchmark failed: {e}")
    
    return results


def benchmark_svm(X_train: np.ndarray, X_test: np.ndarray, 
                 y_train: np.ndarray, y_test: np.ndarray) -> List[BenchmarkResult]:
    """
    Benchmark SVM implementations
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets (binary classification)
        y_test: Test targets (binary classification)
        
    Returns:
        List of benchmark results
    """
    results = []
    
    # Benchmark Scikit-learn SVM (Linear kernel)
    sklearn_model = SkLearnSVM(kernel='linear', C=1.0)
    
    # Fit timing
    start_time = time.time()
    sklearn_model.fit(X_train, y_train)
    sklearn_fit_time = time.time() - start_time
    
    # Predict timing
    start_time = time.time()
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_predict_time = time.time() - start_time
    
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    results.append(BenchmarkResult(
        "SVM", "Scikit-learn", 
        sklearn_fit_time, sklearn_predict_time,
        accuracy=sklearn_accuracy
    ))
    
    # Benchmark Rust implementation (if available)
    if RustSVM is not None:
        try:
            # Convert labels to -1/1 for Rust SVM
            y_train_binary = np.where(y_train == 0, -1, 1)
            y_test_binary = np.where(y_test == 0, -1, 1)
            
            rust_model = RustSVM(kernel="linear", C=1.0)
            
            # Fit timing
            start_time = time.time()
            rust_model.fit(X_train, y_train_binary)
            rust_fit_time = time.time() - start_time
            
            # Predict timing
            start_time = time.time()
            rust_pred = rust_model.predict(X_test)
            rust_predict_time = time.time() - start_time
            
            # Convert back to 0/1 for accuracy calculation
            rust_pred_binary = np.where(rust_pred == -1, 0, 1)
            rust_accuracy = accuracy_score(y_test, rust_pred_binary)
            
            results.append(BenchmarkResult(
                "SVM", "Rust", 
                rust_fit_time, rust_predict_time,
                accuracy=rust_accuracy
            ))
        except Exception as e:
            print(f"Rust SVM benchmark failed: {e}")
    
    return results


def benchmark_kmeans(X_train: np.ndarray, n_clusters: int = 3) -> List[BenchmarkResult]:
    """
    Benchmark K-Means implementations
    
    Args:
        X_train: Training features
        n_clusters: Number of clusters
        
    Returns:
        List of benchmark results
    """
    results = []
    
    # Benchmark Scikit-learn K-Means
    sklearn_model = SkLearnKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Fit timing (K-means combines fit and predict)
    start_time = time.time()
    sklearn_labels = sklearn_model.fit_predict(X_train)
    sklearn_total_time = time.time() - start_time
    
    # For K-means, we'll consider the whole process as "fit" time
    results.append(BenchmarkResult(
        "K-Means", "Scikit-learn", 
        sklearn_total_time, 0.0
    ))
    
    # Benchmark Rust implementation (if available)
    if RustKMeans is not None:
        try:
            rust_model = RustKMeans(n_clusters=n_clusters, max_iters=300, tol=1e-4)
            
            # Fit timing
            start_time = time.time()
            rust_labels = rust_model.fit_predict(X_train)
            rust_total_time = time.time() - start_time
            
            results.append(BenchmarkResult(
                "K-Means", "Rust", 
                rust_total_time, 0.0
            ))
        except Exception as e:
            print(f"Rust K-Means benchmark failed: {e}")
    
    return results


def benchmark_models(dataset_size: int = 1000, n_features: int = 10, 
                    n_clusters: int = 3, random_state: int = 42,
                    algorithms: List[str] = None) -> pd.DataFrame:
    """
    Run comprehensive benchmarks across selected algorithms
    
    Args:
        dataset_size: Number of samples in the dataset
        n_features: Number of features
        n_clusters: Number of clusters for K-means
        random_state: Random seed for reproducibility
        algorithms: List of algorithms to benchmark. Options: ['Linear Regression', 'SVM', 'K-Means']
                   If None, benchmarks all algorithms.
        
    Returns:
        DataFrame with benchmark results
    """
    np.random.seed(random_state)
    all_results = []
    
    # Default to all algorithms if none specified
    if algorithms is None:
        algorithms = ['Linear Regression', 'SVM', 'K-Means']
    
    # Validate algorithm list
    if not algorithms:
        raise ValueError("At least one algorithm must be specified for benchmarking.")
    
    valid_algorithms = {'Linear Regression', 'SVM', 'K-Means'}
    invalid_algorithms = set(algorithms) - valid_algorithms
    if invalid_algorithms:
        raise ValueError(f"Invalid algorithms specified: {invalid_algorithms}. Valid options: {valid_algorithms}")
    
    print(f"Running benchmarks with {dataset_size} samples and {n_features} features...")
    print(f"Selected algorithms: {algorithms}")
    
    # Split data preparation
    split_idx = int(0.8 * dataset_size)
    
    # Linear Regression benchmark
    if 'Linear Regression' in algorithms:
        print("Benchmarking Linear Regression...")
        X_reg, y_reg = make_regression(n_samples=dataset_size, n_features=n_features, 
                                      noise=0.1, random_state=random_state)
        
        X_train_reg, X_test_reg = X_reg[:split_idx], X_reg[split_idx:]
        y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
        
        reg_results = benchmark_linear_regression(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
        all_results.extend(reg_results)
    
    # SVM benchmark
    if 'SVM' in algorithms:
        print("Benchmarking SVM...")
        X_svm, y_svm = make_classification(n_samples=dataset_size, n_features=n_features, 
                                          n_classes=2, n_redundant=0, random_state=random_state)
        
        X_train_svm, X_test_svm = X_svm[:split_idx], X_svm[split_idx:]
        y_train_svm, y_test_svm = y_svm[:split_idx], y_svm[split_idx:]
        
        svm_results = benchmark_svm(X_train_svm, X_test_svm, y_train_svm, y_test_svm)
        all_results.extend(svm_results)
    
    # K-Means benchmark
    if 'K-Means' in algorithms:
        print("Benchmarking K-Means...")
        X_kmeans, _ = make_blobs(n_samples=dataset_size, centers=n_clusters, 
                                n_features=n_features, random_state=random_state)
        
        kmeans_results = benchmark_kmeans(X_kmeans, n_clusters)
        all_results.extend(kmeans_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame([result.to_dict() for result in all_results])
    
    print("Benchmark completed!")
    return results_df


def calculate_speedup(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate speedup factors for Rust vs Python implementations
    
    Args:
        results_df: DataFrame with benchmark results
        
    Returns:
        DataFrame with speedup calculations
        
    Raises:
        ValueError: If missing implementations for proper comparison
    """
    speedup_data = []
    
    algorithms = results_df['Algorithm'].unique()
    
    for algorithm in algorithms:
        alg_data = results_df[results_df['Algorithm'] == algorithm]
        
        # Check that we have both implementations
        implementations = alg_data['Implementation'].unique()
        if 'Scikit-learn' not in implementations:
            raise ValueError(f"Missing Scikit-learn implementation for {algorithm}")
        if 'Rust' not in implementations:
            raise ValueError(f"Missing Rust implementation for {algorithm}")
        
        if len(alg_data) >= 2:
            sklearn_row = alg_data[alg_data['Implementation'] == 'Scikit-learn'].iloc[0]
            rust_row = alg_data[alg_data['Implementation'] == 'Rust'].iloc[0]
            
            fit_speedup = sklearn_row['Fit Time (s)'] / rust_row['Fit Time (s)']
            predict_speedup = sklearn_row['Predict Time (s)'] / rust_row['Predict Time (s)'] if rust_row['Predict Time (s)'] > 0 else float('inf')
            total_speedup = sklearn_row['Total Time (s)'] / rust_row['Total Time (s)']
            
            speedup_data.append({
                'Algorithm': algorithm,
                'Fit Speedup': fit_speedup,
                'Predict Speedup': predict_speedup,
                'Total Speedup': total_speedup
            })
    
    if not speedup_data:
        raise ValueError("No valid speedup calculations possible - missing implementation pairs")
    
    return pd.DataFrame(speedup_data)