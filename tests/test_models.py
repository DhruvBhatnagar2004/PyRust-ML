"""
Test suite for PyRust-ML algorithms
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

from pyrustml import RustLinearRegression, RustSVM, RustKMeans


class TestLinearRegression:
    """Test cases for Linear Regression"""
    
    def setup_method(self):
        """Set up test data"""
        self.X, self.y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def test_fit_predict(self):
        """Test basic fit and predict functionality"""
        model = RustLinearRegression()
        model.fit(self.X_train, self.y_train)
        
        assert model.fitted_ == True
        
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.y_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_score(self):
        """Test R-squared score calculation"""
        model = RustLinearRegression()
        model.fit(self.X_train, self.y_train)
        
        score = model.score(self.X_test, self.y_test)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0  # R² should be between 0 and 1 for good models
    
    def test_unfitted_model_error(self):
        """Test that unfitted model raises error"""
        model = RustLinearRegression()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(self.X_test)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.score(self.X_test, self.y_test)
    
    def test_input_validation(self):
        """Test input validation"""
        model = RustLinearRegression()
        
        # Mismatched X and y lengths
        with pytest.raises(ValueError, match="same number of samples"):
            model.fit(self.X_train[:10], self.y_train[:5])


class TestSVM:
    """Test cases for Support Vector Machine"""
    
    def setup_method(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def test_fit_predict(self):
        """Test basic fit and predict functionality"""
        model = RustSVM(learning_rate=0.01, max_iters=100)
        model.fit(self.X_train, self.y_train)
        
        assert model.fitted_ == True
        
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.y_test)
        assert set(np.unique(predictions)).issubset({-1, 1})  # SVM outputs -1 or 1
    
    def test_decision_function(self):
        """Test decision function"""
        model = RustSVM(learning_rate=0.01, max_iters=100)
        model.fit(self.X_train, self.y_train)
        
        decisions = model.decision_function(self.X_test)
        assert len(decisions) == len(self.y_test)
        assert isinstance(decisions, np.ndarray)
    
    def test_score(self):
        """Test accuracy score calculation"""
        model = RustSVM(learning_rate=0.01, max_iters=100)
        model.fit(self.X_train, self.y_train)
        
        score = model.score(self.X_test, self.y_test)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0  # Accuracy should be between 0 and 1
    
    def test_parameter_setting(self):
        """Test parameter setting"""
        model = RustSVM(learning_rate=0.1, lambda_reg=0.1, max_iters=500)
        params = model.get_params()
        
        assert params['learning_rate'] == 0.1
        assert params['lambda_reg'] == 0.1
        assert params['max_iters'] == 500


class TestKMeans:
    """Test cases for K-Means clustering"""
    
    def setup_method(self):
        """Set up test data"""
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)
    
    def test_fit_predict(self):
        """Test basic fit and predict functionality"""
        model = RustKMeans(n_clusters=3, max_iters=100)
        labels = model.fit_predict(self.X)
        
        assert model.fitted_ == True
        assert len(labels) == len(self.X)
        assert len(np.unique(labels)) <= 3  # Should have at most 3 clusters
        assert model.cluster_centers_ is not None
        assert model.cluster_centers_.shape == (3, 5)  # 3 clusters, 5 features
    
    def test_separate_fit_predict(self):
        """Test separate fit and predict calls"""
        model = RustKMeans(n_clusters=3, max_iters=100)
        model.fit(self.X)
        
        labels = model.predict(self.X)
        assert len(labels) == len(self.X)
    
    def test_inertia(self):
        """Test inertia calculation"""
        model = RustKMeans(n_clusters=3, max_iters=100)
        model.fit(self.X)
        
        inertia = model.inertia(self.X)
        assert isinstance(inertia, float)
        assert inertia >= 0.0  # Inertia should be non-negative
    
    def test_parameter_setting(self):
        """Test parameter setting"""
        model = RustKMeans(n_clusters=5, max_iters=200, tol=1e-3)
        params = model.get_params()
        
        assert params['n_clusters'] == 5
        assert params['max_iters'] == 200
        assert params['tol'] == 1e-3


class TestBenchmarking:
    """Test benchmarking functionality"""
    
    def test_benchmark_import(self):
        """Test that benchmarking module can be imported"""
        from pyrustml.benchmarks import benchmark_models
        assert callable(benchmark_models)
    
    def test_small_benchmark(self):
        """Test running a small benchmark"""
        from pyrustml.benchmarks import benchmark_models
        
        # Run a very small benchmark to verify it works
        results = benchmark_models(dataset_size=50, n_features=3, n_clusters=2)
        
        assert not results.empty
        assert 'Algorithm' in results.columns
        assert 'Implementation' in results.columns
        assert 'Total Time (s)' in results.columns


if __name__ == "__main__":
    pytest.main([__file__])