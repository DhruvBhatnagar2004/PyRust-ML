use pyo3::prelude::*;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use rand::prelude::*;

#[pyclass]
pub struct KMeans {
    n_clusters: usize,
    centroids: Option<Array2<f64>>,
    labels: Option<Vec<usize>>,
    max_iters: usize,
    tol: f64,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (n_clusters = 3, max_iters = 300, tol = 1e-4))]
    pub fn new(n_clusters: usize, max_iters: usize, tol: f64) -> Self {
        KMeans {
            n_clusters,
            centroids: None,
            labels: None,
            max_iters,
            tol,
        }
    }

    /// Fit the K-Means clustering model
    pub fn fit(&mut self, x: Vec<Vec<f64>>) -> PyResult<()> {
        let n_samples = x.len();
        let n_features = x[0].len();
        
        // Convert to ndarray
        let x_matrix = Array2::from_shape_vec((n_samples, n_features), 
            x.into_iter().flatten().collect()).unwrap();
        
        // Initialize centroids randomly
        let mut centroids = self.initialize_centroids(&x_matrix);
        let mut labels = vec![0; n_samples];
        
        for _ in 0..self.max_iters {
            let old_centroids = centroids.clone();
            
            // Assign points to nearest centroid
            labels = self.assign_clusters(&x_matrix, &centroids);
            
            // Update centroids
            centroids = self.update_centroids(&x_matrix, &labels);
            
            // Check for convergence
            if self.has_converged(&old_centroids, &centroids) {
                break;
            }
        }
        
        self.centroids = Some(centroids);
        self.labels = Some(labels);
        
        Ok(())
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<usize>> {
        let centroids = self.centroids.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        
        let n_samples = x.len();
        let n_features = x[0].len();
        
        let x_matrix = Array2::from_shape_vec((n_samples, n_features), 
            x.into_iter().flatten().collect()).unwrap();
        
        Ok(self.assign_clusters(&x_matrix, centroids))
    }

    /// Fit the model and return cluster labels
    pub fn fit_predict(&mut self, x: Vec<Vec<f64>>) -> PyResult<Vec<usize>> {
        self.fit(x.clone())?;
        Ok(self.labels.as_ref().unwrap().clone())
    }

    /// Get cluster centroids
    pub fn get_centroids(&self) -> PyResult<Vec<Vec<f64>>> {
        let centroids = self.centroids.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        
        let mut result = Vec::new();
        for row in centroids.rows() {
            result.push(row.to_vec());
        }
        
        Ok(result)
    }

    /// Calculate within-cluster sum of squares (inertia)
    pub fn inertia(&self, x: Vec<Vec<f64>>) -> PyResult<f64> {
        let centroids = self.centroids.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        let labels = self.labels.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        
        let n_samples = x.len();
        let n_features = x[0].len();
        
        let x_matrix = Array2::from_shape_vec((n_samples, n_features), 
            x.into_iter().flatten().collect()).unwrap();
        
        let inertia: f64 = x_matrix.rows()
            .into_iter()
            .zip(labels.iter())
            .map(|(point, &cluster)| {
                let centroid = centroids.row(cluster);
                point.iter()
                    .zip(centroid.iter())
                    .map(|(p, c)| (p - c).powi(2))
                    .sum::<f64>()
            })
            .sum();
        
        Ok(inertia)
    }
}

impl KMeans {
    fn initialize_centroids(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut rng = thread_rng();
        let n_features = x.ncols();
        let mut centroids = Array2::zeros((self.n_clusters, n_features));
        
        // K-means++ initialization
        let first_idx = rng.gen_range(0..x.nrows());
        centroids.row_mut(0).assign(&x.row(first_idx));
        
        for i in 1..self.n_clusters {
            let distances: Vec<f64> = (0..x.nrows())
                .map(|j| {
                    let point = x.row(j);
                    let min_dist = (0..i)
                        .map(|k| {
                            let centroid = centroids.row(k);
                            point.iter()
                                .zip(centroid.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                        })
                        .fold(f64::INFINITY, f64::min);
                    min_dist
                })
                .collect();
            
            let total_distance: f64 = distances.iter().sum();
            let mut cumulative = 0.0;
            let target = rng.gen::<f64>() * total_distance;
            
            for (j, &dist) in distances.iter().enumerate() {
                cumulative += dist;
                if cumulative >= target {
                    centroids.row_mut(i).assign(&x.row(j));
                    break;
                }
            }
        }
        
        centroids
    }
    
    fn assign_clusters(&self, x: &Array2<f64>, centroids: &Array2<f64>) -> Vec<usize> {
        x.rows()
            .into_par_iter()
            .map(|point| {
                let mut min_distance = f64::INFINITY;
                let mut cluster = 0;
                
                for (i, centroid) in centroids.rows().into_iter().enumerate() {
                    let distance: f64 = point.iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    
                    if distance < min_distance {
                        min_distance = distance;
                        cluster = i;
                    }
                }
                
                cluster
            })
            .collect()
    }
    
    fn update_centroids(&self, x: &Array2<f64>, labels: &[usize]) -> Array2<f64> {
        let n_features = x.ncols();
        let mut centroids = Array2::zeros((self.n_clusters, n_features));
        let mut counts = vec![0; self.n_clusters];
        
        for (point, &label) in x.rows().into_iter().zip(labels.iter()) {
            for (i, &val) in point.iter().enumerate() {
                centroids[[label, i]] += val;
            }
            counts[label] += 1;
        }
        
        for i in 0..self.n_clusters {
            if counts[i] > 0 {
                for j in 0..n_features {
                    centroids[[i, j]] /= counts[i] as f64;
                }
            }
        }
        
        centroids
    }
    
    fn has_converged(&self, old_centroids: &Array2<f64>, new_centroids: &Array2<f64>) -> bool {
        old_centroids.iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max) < self.tol
    }
}