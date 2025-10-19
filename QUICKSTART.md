# 🚀 PyRust-ML Quickstart

Get up and running with PyRust-ML in 5 minutes.

## ⚡ Quick Setup

```bash
# Clone and setup
git clone https://github.com/your-username/pyrust-ml.git
cd pyrust-ml
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Test installation
python test_quick.py

# Launch dashboard
streamlit run dashboard/professional_app.py
```

## 🐍 Basic Usage

```python
import numpy as np
from pyrustml import RustLinearRegression, RustSVM, RustKMeans

# Sample data
X, y = np.random.randn(1000, 5), np.random.randn(1000)

# Linear Regression
model = RustLinearRegression()
model.fit(X, y)
print(f"R² Score: {model.score(X, y):.4f}")

# SVM Classification
y_binary = np.random.choice([-1, 1], 1000)
svm = RustSVM(kernel='linear', C=1.0)
svm.fit(X, y_binary)
print(f"SVM Accuracy: {svm.score(X, y_binary):.4f}")

# K-Means Clustering
kmeans = RustKMeans(n_clusters=3)
labels = kmeans.fit_predict(X)
print(f"Inertia: {kmeans.inertia(X):.4f}")
```

## 📊 Dashboard Features

Access at `http://localhost:8502`:
- **Model Playground**: Interactive algorithm testing
- **Performance Benchmarks**: Rust vs Python comparisons
- **Advanced Analytics**: Real-time performance monitoring

## 🔥 Optional: Rust Acceleration

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Compile optimized extensions
cd rust_src && maturin develop --release
```

## 🧪 Testing

```bash
# Quick test
python test_quick.py

# Performance comparison
python test_rust_vs_python_performance.py
```

---

For detailed documentation, see [README.md](README.md)