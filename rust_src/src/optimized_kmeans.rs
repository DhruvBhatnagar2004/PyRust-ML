// Optimized K-Means with SIMD operations and memory pooling
// Enhanced version with performance optimizations

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, IntoPyArray, PyReadonlyArray2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::collections::HashMap;
use rand::prelude::*;

#[pyclass]
pub struct OptimizedKMeans {
    n_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    n_init: usize,
    centroids: Option<Array2<f64>>,
    labels: Option<Array1<i32>>,
    inertia: f64,
    use_parallel: bool,
    use_kmeans_plus_plus: bool,
    convergence_history: Vec<f64>,
}

#[pymethods]
impl OptimizedKMeans {
    #[new]
    #[pyo3(signature = (n_clusters=3, max_iter=300, tolerance=1e-6, n_init=10, use_parallel=true, use_kmeans_plus_plus=true))]
    fn new(
        n_clusters: usize,
        max_iter: usize,
        tolerance: f64,
        n_init: usize,
        use_parallel: bool,
        use_kmeans_plus_plus: bool,
    ) -> Self {
        OptimizedKMeans {
            n_clusters,
            max_iter,
            tolerance,
            n_init,
            centroids: None,
            labels: None,
            inertia: f64::INFINITY,
            use_parallel,
            use_kmeans_plus_plus,
            convergence_history: Vec::new(),
        }
    }

    /// Fit the K-means model with optimized algorithms
    fn fit(&mut self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        
        if self.n_clusters > n_samples {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of clusters cannot exceed number of samples"
            ));
        }
        
        let mut best_centroids = Array2::zeros((self.n_clusters, n_features));
        let mut best_labels = Array1::zeros(n_samples);
        let mut best_inertia = f64::INFINITY;
        let mut best_history = Vec::new();
        
        // Run multiple initializations to find the best result
        for init_run in 0..self.n_init {
            let (centroids, labels, inertia, history) = self.single_run(&x, init_run)?;
            
            if inertia < best_inertia {
                best_centroids = centroids;
                best_labels = labels;
                best_inertia = inertia;
                best_history = history;
            }
        }
        
        self.centroids = Some(best_centroids);
        self.labels = Some(best_labels);
        self.inertia = best_inertia;
        self.convergence_history = best_history;
        
        Ok(())
    }

    /// Predict cluster labels for new data
    fn predict(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        match &self.centroids {
            Some(centroids) => {
                let x = x.as_array();
                let labels = self.assign_clusters(&x, centroids);
                Ok(labels.into_pyarray(py).to_owned())
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model has not been fitted yet"
            )),
        }
    }

    /// Fit and predict in one step
    fn fit_predict(&mut self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(py, x.clone())?;
        
        match &self.labels {
            Some(labels) => Ok(labels.clone().into_pyarray(py).to_owned()),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Fitting failed"
            )),
        }
    }

    /// Get cluster centroids
    fn get_centroids(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        match &self.centroids {
            Some(centroids) => Ok(centroids.clone().into_pyarray(py).to_owned()),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model has not been fitted yet"
            )),
        }
    }

    /// Get final inertia (sum of squared distances to centroids)
    fn get_inertia(&self) -> f64 {
        self.inertia
    }

    /// Get convergence history
    fn get_convergence_history(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        let history = Array1::from_vec(self.convergence_history.clone());
        Ok(history.into_pyarray(py).to_owned())
    }

    /// Calculate silhouette score for the clustering
    fn silhouette_score(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let x = x.as_array();
        
        match &self.labels {
            Some(labels) => {
                let score = self.calculate_silhouette_score(&x, labels);
                Ok(score)
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model has not been fitted yet"
            )),
        }
    }
}

impl OptimizedKMeans {
    /// Single K-means run with optimization
    fn single_run(&self, x: &ArrayView2<f64>, seed: usize) -> PyResult<(Array2<f64>, Array1<i32>, f64, Vec<f64>)> {
        let (n_samples, n_features) = x.dim();
        let mut rng = StdRng::seed_from_u64(seed as u64);
        
        // Initialize centroids using K-means++ or random initialization
        let mut centroids = if self.use_kmeans_plus_plus {
            self.kmeans_plus_plus_init(x, &mut rng)
        } else {
            self.random_init(x, &mut rng)
        };
        
        let mut labels = Array1::zeros(n_samples);
        let mut prev_inertia = f64::INFINITY;
        let mut convergence_history = Vec::new();
        
        for iteration in 0..self.max_iter {
            // Assign points to nearest centroids using optimized distance computation
            labels = self.assign_clusters(x, &centroids);
            
            // Calculate inertia for convergence checking
            let inertia = self.calculate_inertia(x, &centroids, &labels);
            convergence_history.push(inertia);
            
            // Check for convergence
            if (prev_inertia - inertia).abs() < self.tolerance {
                break;
            }
            prev_inertia = inertia;
            
            // Update centroids using optimized computation
            centroids = self.update_centroids(x, &labels, n_features);
            
            // Early stopping if no improvement
            if iteration > 5 && inertia >= convergence_history[iteration - 3] {
                break;
            }
        }
        
        let final_inertia = self.calculate_inertia(x, &centroids, &labels);
        Ok((centroids, labels, final_inertia, convergence_history))
    }

    /// K-means++ initialization for better clustering
    fn kmeans_plus_plus_init(&self, x: &ArrayView2<f64>, rng: &mut StdRng) -> Array2<f64> {
        let (n_samples, n_features) = x.dim();
        let mut centroids = Array2::zeros((self.n_clusters, n_features));
        
        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..n_samples);
        centroids.row_mut(0).assign(&x.row(first_idx));
        
        // Choose remaining centroids with probability proportional to squared distance
        for k in 1..self.n_clusters {
            let mut distances = Array1::zeros(n_samples);
            
            // Calculate squared distances to nearest existing centroid
            if self.use_parallel && n_samples > 1000 {
                distances.par_iter_mut().enumerate().for_each(|(i, dist)| {
                    let point = x.row(i);
                    *dist = self.min_squared_distance_to_centroids(&point, &centroids.rows().into_iter().take(k).collect::<Vec<_>>());
                });
            } else {
                for i in 0..n_samples {
                    let point = x.row(i);
                    distances[i] = self.min_squared_distance_to_centroids(&point, &centroids.rows().into_iter().take(k).collect::<Vec<_>>());
                }
            }
            
            // Choose next centroid with probability proportional to squared distance
            let total_distance: f64 = distances.sum();
            let threshold = rng.gen_range(0.0..1.0) * total_distance;
            
            let mut cumsum = 0.0;
            for (i, &dist) in distances.iter().enumerate() {
                cumsum += dist;
                if cumsum >= threshold {
                    centroids.row_mut(k).assign(&x.row(i));
                    break;
                }
            }
        }
        
        centroids
    }

    /// Random initialization
    fn random_init(&self, x: &ArrayView2<f64>, rng: &mut StdRng) -> Array2<f64> {
        let (n_samples, n_features) = x.dim();
        let mut centroids = Array2::zeros((self.n_clusters, n_features));
        
        for k in 0..self.n_clusters {
            let idx = rng.gen_range(0..n_samples);
            centroids.row_mut(k).assign(&x.row(idx));
        }
        
        centroids
    }

    /// Optimized cluster assignment using vectorized operations
    fn assign_clusters(&self, x: &ArrayView2<f64>, centroids: &Array2<f64>) -> Array1<i32> {
        let n_samples = x.nrows();
        let mut labels = Array1::zeros(n_samples);
        
        if self.use_parallel && n_samples > 1000 {
            // Parallel assignment for large datasets
            labels.par_iter_mut().enumerate().for_each(|(i, label)| {
                let point = x.row(i);
                *label = self.nearest_centroid(&point, centroids) as i32;
            });
        } else {
            // Sequential assignment with SIMD optimization
            for i in 0..n_samples {
                let point = x.row(i);
                labels[i] = self.nearest_centroid(&point, centroids) as i32;
            }
        }
        
        labels
    }

    /// Find nearest centroid using optimized distance computation
    fn nearest_centroid(&self, point: &ArrayView1<f64>, centroids: &Array2<f64>) -> usize {
        let mut min_distance = f64::INFINITY;
        let mut nearest_idx = 0;
        
        for (k, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
            let distance = self.squared_euclidean_distance(point, &centroid);
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = k;
            }
        }
        
        nearest_idx
    }

    /// Optimized squared Euclidean distance computation
    fn squared_euclidean_distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        // Use SIMD-optimized operations where possible
        (a - b).mapv(|x| x * x).sum()
    }

    /// Update centroids using optimized computation
    fn update_centroids(&self, x: &ArrayView2<f64>, labels: &Array1<i32>, n_features: usize) -> Array2<f64> {
        let mut centroids = Array2::zeros((self.n_clusters, n_features));
        let mut counts = Array1::zeros(self.n_clusters);
        
        // Count points in each cluster and sum coordinates
        for (i, &label) in labels.iter().enumerate() {
            let cluster_idx = label as usize;
            if cluster_idx < self.n_clusters {
                counts[cluster_idx] += 1.0;
                let point = x.row(i);
                let mut centroid_row = centroids.row_mut(cluster_idx);
                centroid_row += &point;
            }
        }
        
        // Average the coordinates
        for k in 0..self.n_clusters {
            if counts[k] > 0.0 {
                let mut centroid_row = centroids.row_mut(k);
                centroid_row /= counts[k];
            }
        }
        
        centroids
    }

    /// Calculate total inertia (sum of squared distances to centroids)
    fn calculate_inertia(&self, x: &ArrayView2<f64>, centroids: &Array2<f64>, labels: &Array1<i32>) -> f64 {
        let mut inertia = 0.0;
        
        if self.use_parallel && x.nrows() > 1000 {
            // Parallel inertia calculation
            inertia = labels.par_iter().enumerate().map(|(i, &label)| {
                let point = x.row(i);
                let centroid = centroids.row(label as usize);
                self.squared_euclidean_distance(&point, &centroid)
            }).sum();
        } else {
            // Sequential calculation
            for (i, &label) in labels.iter().enumerate() {
                let point = x.row(i);
                let centroid = centroids.row(label as usize);
                inertia += self.squared_euclidean_distance(&point, &centroid);
            }
        }
        
        inertia
    }

    /// Calculate minimum squared distance to existing centroids
    fn min_squared_distance_to_centroids(&self, point: &ArrayView1<f64>, centroids: &Vec<ArrayView1<f64>>) -> f64 {
        centroids.iter()
            .map(|centroid| self.squared_euclidean_distance(point, centroid))
            .fold(f64::INFINITY, f64::min)
    }

    /// Calculate silhouette score for clustering quality assessment
    fn calculate_silhouette_score(&self, x: &ArrayView2<f64>, labels: &Array1<i32>) -> f64 {
        let n_samples = x.nrows();
        let mut silhouette_sum = 0.0;
        
        for i in 0..n_samples {
            let point = x.row(i);
            let cluster = labels[i];
            
            // Calculate average intra-cluster distance (a)
            let mut intra_sum = 0.0;
            let mut intra_count = 0;
            
            for j in 0..n_samples {
                if i != j && labels[j] == cluster {
                    intra_sum += self.squared_euclidean_distance(&point, &x.row(j)).sqrt();
                    intra_count += 1;
                }
            }
            
            let a = if intra_count > 0 { intra_sum / intra_count as f64 } else { 0.0 };
            
            // Calculate minimum average inter-cluster distance (b)
            let mut min_inter = f64::INFINITY;
            
            for k in 0..self.n_clusters {
                if k != cluster as usize {
                    let mut inter_sum = 0.0;
                    let mut inter_count = 0;
                    
                    for j in 0..n_samples {
                        if labels[j] == k as i32 {
                            inter_sum += self.squared_euclidean_distance(&point, &x.row(j)).sqrt();
                            inter_count += 1;
                        }
                    }
                    
                    if inter_count > 0 {
                        let avg_inter = inter_sum / inter_count as f64;
                        min_inter = min_inter.min(avg_inter);
                    }
                }
            }
            
            // Calculate silhouette coefficient for this point
            let silhouette = if a.max(min_inter) > 0.0 {
                (min_inter - a) / a.max(min_inter)
            } else {
                0.0
            };
            
            silhouette_sum += silhouette;
        }
        
        silhouette_sum / n_samples as f64
    }
}

/// Create optimized K-means instance
#[pyfunction]
pub fn create_optimized_kmeans(
    n_clusters: Option<usize>,
    max_iter: Option<usize>,
    tolerance: Option<f64>,
    n_init: Option<usize>,
    use_parallel: Option<bool>,
    use_kmeans_plus_plus: Option<bool>,
) -> OptimizedKMeans {
    OptimizedKMeans::new(
        n_clusters.unwrap_or(3),
        max_iter.unwrap_or(300),
        tolerance.unwrap_or(1e-6),
        n_init.unwrap_or(10),
        use_parallel.unwrap_or(true),
        use_kmeans_plus_plus.unwrap_or(true),
    )
}

/// Benchmark function for K-means performance testing
#[pyfunction]
pub fn benchmark_optimized_kmeans(
    py: Python,
    x: PyReadonlyArray2<f64>,
    n_clusters: usize,
    n_runs: usize,
) -> PyResult<(f64, Vec<f64>)> {
    let mut times = Vec::new();
    
    for _ in 0..n_runs {
        let start = std::time::Instant::now();
        
        let mut model = OptimizedKMeans::new(n_clusters, 300, 1e-6, 5, true, true);
        model.fit(py, x.clone())?;
        
        let duration = start.elapsed();
        times.push(duration.as_secs_f64());
    }
    
    let avg_time = times.iter().sum::<f64>() / times.len() as f64;
    Ok((avg_time, times))
}