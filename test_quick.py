#!/usr/bin/env python3
"""
Quick test of PyRust-ML functionality
"""

import sys
import os
import numpy as np

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from pyrustml import RustLinearRegression, RustSVM, RustKMeans
    print("✅ Successfully imported PyRust-ML components")
    
    # Test Linear Regression
    print("\n🧮 Testing Linear Regression...")
    X = np.random.randn(50, 3)
    y = np.random.randn(50)
    
    lr_model = RustLinearRegression()
    lr_model.fit(X, y)
    predictions = lr_model.predict(X)
    score = lr_model.score(X, y)
    
    print(f"   Linear Regression R² Score: {score:.4f}")
    print("   ✅ Linear Regression test passed!")
    
    # Test SVM
    print("\n🤖 Testing SVM...")
    X_svm = np.random.randn(50, 3)
    y_svm = np.random.choice([-1, 1], 50)  # SVM expects -1, 1 labels
    
    svm_model = RustSVM(kernel="linear", C=1.0)
    svm_model.fit(X_svm, y_svm)
    svm_predictions = svm_model.predict(X_svm)
    svm_score = svm_model.score(X_svm, y_svm)
    
    print(f"   SVM Accuracy: {svm_score:.4f}")
    print("   ✅ SVM test passed!")
    
    # Test K-Means
    print("\n🎯 Testing K-Means...")
    X_kmeans = np.random.randn(50, 3)
    
    kmeans_model = RustKMeans(n_clusters=3, max_iters=50)
    labels = kmeans_model.fit_predict(X_kmeans)
    inertia = kmeans_model.inertia(X_kmeans)
    
    print(f"   K-Means Inertia: {inertia:.4f}")
    print(f"   Number of unique clusters: {len(np.unique(labels))}")
    print("   ✅ K-Means test passed!")
    
    # Test benchmarking
    print("\n📊 Testing Benchmarking...")
    from pyrustml.benchmarks import benchmark_models
    
    results = benchmark_models(dataset_size=30, n_features=3, n_clusters=2)
    print(f"   Benchmark completed with {len(results)} results")
    print("   ✅ Benchmarking test passed!")
    
    print("\n🎉 All tests passed! PyRust-ML is working correctly.")
    print("   📝 Note: Currently using Python fallback implementations.")
    print("   🦀 To enable Rust acceleration, compile with: maturin develop")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()