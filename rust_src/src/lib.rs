use pyo3::prelude::*;

// Include optimized modules
mod optimized_linear_regression;
mod optimized_kmeans;

// Re-export optimized implementations
pub use optimized_linear_regression::{OptimizedLinearRegression, create_optimized_linear_regression, benchmark_optimized_linear_regression};
pub use optimized_kmeans::{OptimizedKMeans, create_optimized_kmeans, benchmark_optimized_kmeans};

/// A simple linear regression implementation
#[pyclass]
pub struct SimpleLinearRegression {
    slope: f64,
    intercept: f64,
    fitted: bool,
}

#[pymethods]
impl SimpleLinearRegression {
    #[new]
    pub fn new() -> Self {
        SimpleLinearRegression {
            slope: 0.0,
            intercept: 0.0,
            fitted: false,
        }
    }

    /// Fit the model with simple least squares
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        if x.len() != y.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("X and y must have the same length"));
        }
        
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
        
        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        self.intercept = (sum_y - self.slope * sum_x) / n;
        self.fitted = true;
        
        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"));
        }
        
        Ok(x.iter().map(|xi| self.slope * xi + self.intercept).collect())
    }

    /// Get model parameters
    pub fn get_params(&self) -> (f64, f64) {
        (self.slope, self.intercept)
    }
}

/// PyRust-ML: Enhanced Implementation with Optimizations
#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Original simple implementations
    m.add_class::<SimpleLinearRegression>()?;
    
    // Optimized implementations
    m.add_class::<OptimizedLinearRegression>()?;
    m.add_class::<OptimizedKMeans>()?;
    
    // Factory functions
    m.add_function(wrap_pyfunction!(create_optimized_linear_regression, m)?)?;
    m.add_function(wrap_pyfunction!(create_optimized_kmeans, m)?)?;
    
    // Benchmark functions
    m.add_function(wrap_pyfunction!(benchmark_optimized_linear_regression, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_optimized_kmeans, m)?)?;
    
    Ok(())
}