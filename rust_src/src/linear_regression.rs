use pyo3::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rayon::prelude::*;

#[pyclass]
pub struct LinearRegression {
    weights: Option<Array1<f64>>,
    bias: f64,
}

#[pymethods]
impl LinearRegression {
    #[new]
    pub fn new() -> Self {
        LinearRegression {
            weights: None,
            bias: 0.0,
        }
    }

    /// Fit the linear regression model using Ordinary Least Squares
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let n_samples = x.len();
        let n_features = x[0].len();
        
        // Convert to ndarray
        let x_matrix = Array2::from_shape_vec((n_samples, n_features), 
            x.into_iter().flatten().collect()).unwrap();
        let y_vector = Array1::from_vec(y);
        
        // Add bias column (ones)
        let mut x_with_bias = Array2::zeros((n_samples, n_features + 1));
        x_with_bias.slice_mut(s![.., ..n_features]).assign(&x_matrix);
        x_with_bias.slice_mut(s![.., n_features]).fill(1.0);
        
        // Compute weights using normal equation: w = (X^T * X)^-1 * X^T * y
        let xt = x_with_bias.t();
        let xtx = xt.dot(&x_with_bias);
        let xty = xt.dot(&y_vector);
        
        // Solve the linear system (simplified - in production use proper linear algebra)
        let weights_with_bias = self.solve_linear_system(&xtx, &xty)?;
        
        self.weights = Some(weights_with_bias.slice(s![..n_features]).to_owned());
        self.bias = weights_with_bias[n_features];
        
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
                weights.dot(&row) + self.bias
            })
            .collect();
        
        Ok(predictions)
    }

    /// Calculate R-squared score
    pub fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        let predictions = self.predict(x)?;
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        
        let ss_res: f64 = y.iter().zip(predictions.iter())
            .map(|(actual, pred)| (actual - pred).powi(2))
            .sum();
        
        let ss_tot: f64 = y.iter()
            .map(|val| (val - y_mean).powi(2))
            .sum();
        
        Ok(1.0 - (ss_res / ss_tot))
    }
}

impl LinearRegression {
    /// Simplified linear system solver (Gaussian elimination)
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> PyResult<Array1<f64>> {
        let n = a.nrows();
        let mut aug_matrix = Array2::zeros((n, n + 1));
        aug_matrix.slice_mut(s![.., ..n]).assign(a);
        aug_matrix.slice_mut(s![.., n]).assign(b);
        
        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if aug_matrix[[k, i]].abs() > aug_matrix[[max_row, i]].abs() {
                    max_row = k;
                }
            }
            
            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let temp = aug_matrix[[i, j]];
                    aug_matrix[[i, j]] = aug_matrix[[max_row, j]];
                    aug_matrix[[max_row, j]] = temp;
                }
            }
            
            // Make all rows below this one 0 in current column
            for k in i + 1..n {
                if aug_matrix[[i, i]] != 0.0 {
                    let factor = aug_matrix[[k, i]] / aug_matrix[[i, i]];
                    for j in i..=n {
                        aug_matrix[[k, j]] -= factor * aug_matrix[[i, j]];
                    }
                }
            }
        }
        
        // Back substitution
        let mut solution = Array1::zeros(n);
        for i in (0..n).rev() {
            solution[i] = aug_matrix[[i, n]];
            for j in i + 1..n {
                solution[i] -= aug_matrix[[i, j]] * solution[j];
            }
            if aug_matrix[[i, i]] != 0.0 {
                solution[i] /= aug_matrix[[i, i]];
            }
        }
        
        Ok(solution)
    }
}