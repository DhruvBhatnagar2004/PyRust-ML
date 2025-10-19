// Optimized Linear Regression with SIMD and advanced memory management
// Enhanced version with performance optimizations

use pyo3::prelude::*;
use pyo3::{PyResult, Python, Py, PyAny};
use numpy::{PyArray1, PyArray2, IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct OptimizedLinearRegression {
    weights: Option<Array1<f64>>,
    bias: f64,
    learning_rate: f64,
    max_iter: usize,
    tolerance: f64,
    regularization: f64,
    use_parallel: bool,
    convergence_history: Vec<f64>,
}

#[pymethods]
impl OptimizedLinearRegression {
    #[new]
    #[pyo3(signature = (learning_rate=0.01, max_iter=1000, tolerance=1e-6, regularization=0.0, use_parallel=true))]
    fn new(
        learning_rate: f64,
        max_iter: usize,
        tolerance: f64,
        regularization: f64,
        use_parallel: bool,
    ) -> Self {
        OptimizedLinearRegression {
            weights: None,
            bias: 0.0,
            learning_rate,
            max_iter,
            tolerance,
            regularization,
            use_parallel,
            convergence_history: Vec::new(),
        }
    }

    /// Fit the model with optimized gradient descent
    fn fit(&mut self, py: Python, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x = x.as_array();
        let y = y.as_array();
        
        let (n_samples, n_features) = x.dim();
        
        // Initialize weights with Xavier initialization for better convergence
        self.weights = Some(Array1::from_vec(
            (0..n_features)
                .map(|_| rand::random::<f64>() * (2.0 / n_features as f64).sqrt() - (1.0 / n_features as f64).sqrt())
                .collect()
        ));
        self.bias = 0.0;
        self.convergence_history.clear();
        
        let mut prev_cost = f64::INFINITY;
        
        for iteration in 0..self.max_iter {
            // Compute predictions using optimized dot product
            let predictions = self.predict_internal(&x);
            
            // Compute cost with L2 regularization
            let residuals = &predictions - &y;
            let cost = self.compute_cost(&residuals, n_samples);
            
            self.convergence_history.push(cost);
            
            // Check for convergence
            if (prev_cost - cost).abs() < self.tolerance {
                break;
            }
            prev_cost = cost;
            
            // Compute gradients using optimized operations
            self.update_weights(&x, &residuals, n_samples);
            
            // Early stopping if cost starts increasing significantly
            if iteration > 10 && cost > self.convergence_history[iteration - 5] * 1.1 {
                break;
            }
        }
        
        Ok(())
    }

    /// Predict using optimized matrix operations
    fn predict(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let x = x.as_array();
        let predictions = self.predict_internal(&x);
        Ok(predictions.into_pyarray(py).to_owned())
    }

    /// Get convergence history for analysis
    fn get_convergence_history(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        let history = Array1::from_vec(self.convergence_history.clone());
        Ok(history.into_pyarray(py).to_owned())
    }

    /// Get model coefficients
    fn get_coefficients(&self, py: Python) -> PyResult<(Py<PyArray1<f64>>, f64)> {
        match &self.weights {
            Some(weights) => {
                let weights_py = weights.clone().into_pyarray(py).to_owned();
                Ok((weights_py, self.bias))
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model has not been fitted yet"
            )),
        }
    }

    /// Calculate R-squared score
    fn score(&self, py: Python, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let x = x.as_array();
        let y = y.as_array();
        
        let predictions = self.predict_internal(&x);
        let y_mean = y.mean().unwrap();
        
        let ss_res: f64 = (&predictions - &y).mapv(|x| x * x).sum();
        let ss_tot: f64 = y.mapv(|x| (x - y_mean) * (x - y_mean)).sum();
        
        Ok(1.0 - ss_res / ss_tot)
    }
}

impl OptimizedLinearRegression {
    /// Internal prediction method with optimized operations
    fn predict_internal(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        match &self.weights {
            Some(weights) => {
                if self.use_parallel {
                    // Parallel computation for large datasets
                    self.parallel_matrix_multiply(x, weights)
                } else {
                    // Standard computation
                    x.dot(weights) + self.bias
                }
            }
            None => panic!("Model has not been fitted yet"),
        }
    }

    /// Optimized parallel matrix multiplication
    fn parallel_matrix_multiply(&self, x: &ArrayView2<f64>, weights: &Array1<f64>) -> Array1<f64> {
        let (n_samples, _) = x.dim();
        let mut result = Array1::zeros(n_samples);
        
        // Use parallel processing for large datasets
        if n_samples > 1000 {
            result.par_iter_mut().enumerate().for_each(|(i, res)| {
                let row = x.row(i);
                *res = row.dot(weights) + self.bias;
            });
        } else {
            // Use SIMD-optimized operations for smaller datasets
            result = x.dot(weights) + self.bias;
        }
        
        result
    }

    /// Compute cost with regularization
    fn compute_cost(&self, residuals: &Array1<f64>, n_samples: usize) -> f64 {
        let mse = residuals.mapv(|x| x * x).sum() / (2.0 * n_samples as f64);
        
        // Add L2 regularization
        let regularization_term = match &self.weights {
            Some(weights) => self.regularization * weights.mapv(|x| x * x).sum() / 2.0,
            None => 0.0,
        };
        
        mse + regularization_term
    }

    /// Update weights using optimized gradient computation
    fn update_weights(&mut self, x: &ArrayView2<f64>, residuals: &Array1<f64>, n_samples: usize) {
        if let Some(ref mut weights) = self.weights {
            // Compute gradients
            let weight_gradient = if self.use_parallel {
                self.parallel_gradient_computation(x, residuals, n_samples)
            } else {
                x.t().dot(residuals) / n_samples as f64
            };
            
            let bias_gradient = residuals.sum() / n_samples as f64;
            
            // Apply regularization to weight gradient
            let regularized_gradient = &weight_gradient + &(weights.mapv(|w| w * self.regularization));
            
            // Update parameters using adaptive learning rate
            let adaptive_lr = self.adaptive_learning_rate(residuals);
            *weights = weights - &(regularized_gradient * adaptive_lr);
            self.bias -= bias_gradient * adaptive_lr;
        }
    }

    /// Parallel gradient computation for large datasets
    fn parallel_gradient_computation(&self, x: &ArrayView2<f64>, residuals: &Array1<f64>, n_samples: usize) -> Array1<f64> {
        let n_features = x.ncols();
        let mut gradient = Array1::zeros(n_features);
        
        // Use parallel reduction for gradient computation
        gradient.par_iter_mut().enumerate().for_each(|(j, grad)| {
            *grad = x.column(j).dot(residuals) / n_samples as f64;
        });
        
        gradient
    }

    /// Adaptive learning rate based on gradient magnitude
    fn adaptive_learning_rate(&self, residuals: &Array1<f64>) -> f64 {
        let gradient_magnitude = residuals.mapv(|x| x * x).sum().sqrt();
        
        // Adjust learning rate based on gradient magnitude
        if gradient_magnitude > 1.0 {
            self.learning_rate * 0.9 // Reduce learning rate for large gradients
        } else if gradient_magnitude < 0.1 {
            self.learning_rate * 1.1 // Increase learning rate for small gradients
        } else {
            self.learning_rate
        }
    }
}

/// Create a new optimized linear regression instance
#[pyfunction]
pub fn create_optimized_linear_regression(
    learning_rate: Option<f64>,
    max_iter: Option<usize>,
    tolerance: Option<f64>,
    regularization: Option<f64>,
    use_parallel: Option<bool>,
) -> OptimizedLinearRegression {
    OptimizedLinearRegression::new(
        learning_rate.unwrap_or(0.01),
        max_iter.unwrap_or(1000),
        tolerance.unwrap_or(1e-6),
        regularization.unwrap_or(0.0),
        use_parallel.unwrap_or(true),
    )
}

/// Benchmark function for performance testing
#[pyfunction]
pub fn benchmark_optimized_linear_regression(
    py: Python,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    n_runs: usize,
) -> PyResult<(f64, Vec<f64>)> {
    let mut times = Vec::new();
    
    for _ in 0..n_runs {
        let start = std::time::Instant::now();
        
        let mut model = OptimizedLinearRegression::new(0.01, 1000, 1e-6, 0.0, true);
        model.fit(py, x.clone(), y.clone())?;
        
        let duration = start.elapsed();
        times.push(duration.as_secs_f64());
    }
    
    let avg_time = times.iter().sum::<f64>() / times.len() as f64;
    Ok((avg_time, times))
}