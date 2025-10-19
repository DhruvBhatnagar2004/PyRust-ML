use pyo3::prelude::*;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

#[pyclass]
pub struct SVM {
    weights: Option<Array1<f64>>,
    bias: f64,
    learning_rate: f64,
    lambda_reg: f64,
    max_iters: usize,
}

#[pymethods]
impl SVM {
    #[new]
    #[pyo3(signature = (learning_rate = 0.01, lambda_reg = 0.01, max_iters = 1000))]
    pub fn new(learning_rate: f64, lambda_reg: f64, max_iters: usize) -> Self {
        SVM {
            weights: None,
            bias: 0.0,
            learning_rate,
            lambda_reg,
            max_iters,
        }
    }

    /// Fit the SVM model using gradient descent
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let n_samples = x.len();
        let n_features = x[0].len();
        
        // Convert to ndarray and ensure labels are -1 or 1
        let x_matrix = Array2::from_shape_vec((n_samples, n_features), 
            x.into_iter().flatten().collect()).unwrap();
        let y_vector: Array1<f64> = y.into_iter()
            .map(|label| if label <= 0.0 { -1.0 } else { 1.0 })
            .collect::<Vec<f64>>()
            .into();
        
        // Initialize weights and bias
        let mut weights = Array1::zeros(n_features);
        let mut bias = 0.0;
        
        // Gradient descent
        for _ in 0..self.max_iters {
            let (dw, db) = self.compute_gradients(&x_matrix, &y_vector, &weights, bias);
            
            // Update weights and bias
            weights = weights - self.learning_rate * dw;
            bias = bias - self.learning_rate * db;
        }
        
        self.weights = Some(weights);
        self.bias = bias;
        
        Ok(())
    }

    /// Make predictions using the fitted model
    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let weights = self.weights.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        
        let n_samples = x.len();
        let n_features = x[0].len();
        
        let x_matrix = Array2::from_shape_vec((n_samples, n_features), 
            x.into_iter().flatten().collect()).unwrap();
        
        let predictions: Vec<f64> = x_matrix.rows()
            .into_par_iter()
            .map(|row| {
                let decision = weights.dot(&row) + self.bias;
                if decision >= 0.0 { 1.0 } else { -1.0 }
            })
            .collect();
        
        Ok(predictions)
    }

    /// Get decision function values (distances from hyperplane)
    pub fn decision_function(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let weights = self.weights.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        
        let n_samples = x.len();
        let n_features = x[0].len();
        
        let x_matrix = Array2::from_shape_vec((n_samples, n_features), 
            x.into_iter().flatten().collect()).unwrap();
        
        let decisions: Vec<f64> = x_matrix.rows()
            .into_par_iter()
            .map(|row| weights.dot(&row) + self.bias)
            .collect();
        
        Ok(decisions)
    }

    /// Calculate accuracy score
    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let predictions = self.predict(x)?;
        let y_binary: Vec<f64> = y.into_iter()
            .map(|label| if label <= 0.0 { -1.0 } else { 1.0 })
            .collect();
        
        let correct = predictions.iter()
            .zip(y_binary.iter())
            .filter(|(pred, actual)| (pred - actual).abs() < 1e-10)
            .count();
        
        Ok(correct as f64 / predictions.len() as f64)
    }

    /// Get model parameters
    pub fn get_params(&self) -> PyResult<(Vec<f64>, f64)> {
        let weights = self.weights.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        
        Ok((weights.to_vec(), self.bias))
    }
}

impl SVM {
    fn compute_gradients(&self, x: &Array2<f64>, y: &Array1<f64>, 
                        weights: &Array1<f64>, bias: f64) -> (Array1<f64>, f64) {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        
        let mut dw = Array1::zeros(n_features);
        let mut db = 0.0;
        
        for i in 0..n_samples {
            let xi = x.row(i);
            let yi = y[i];
            let decision = weights.dot(&xi) + bias;
            
            // Hinge loss gradient
            if yi * decision < 1.0 {
                // Misclassified or in margin
                for j in 0..n_features {
                    dw[j] += -yi * xi[j];
                }
                db += -yi;
            }
        }
        
        // Add regularization term
        for j in 0..n_features {
            dw[j] = dw[j] / n_samples as f64 + self.lambda_reg * weights[j];
        }
        db = db / n_samples as f64;
        
        (dw, db)
    }
}