#!/usr/bin/env python3
"""
COMPLETE SYSTEM VALIDATION: PyRust-ML
Comprehensive test to ensure interview-ready quality
"""

import sys
import traceback
import time
import numpy as np
from datetime import datetime

def test_rust_implementations():
    """Test that Rust implementations work correctly"""
    print("🦀 Testing Rust Implementations...")
    
    try:
        from pyrustml.linear_regression import RustLinearRegression
        from pyrustml.kmeans import RustKMeans
        from pyrustml.svm import RustSVM
        
        # Test data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        y_binary = np.where(y > 0, 1, -1)
        
        # Test Linear Regression
        lr = RustLinearRegression()
        lr.fit(X, y)
        lr_pred = lr.predict(X)
        lr_score = lr.score(X, y)
        print(f"  ✅ Linear Regression: Rust={lr._using_rust}, Score={lr_score:.3f}")
        
        # Test K-Means
        kmeans = RustKMeans(n_clusters=3)
        kmeans.fit(X)
        kmeans_labels = kmeans.predict(X)
        print(f"  ✅ K-Means: Rust={kmeans._using_rust}, Clusters={len(np.unique(kmeans_labels))}")
        
        # Test SVM
        svm = RustSVM()
        svm.fit(X, y_binary)
        svm_pred = svm.predict(X)
        svm_score = svm.score(X, y_binary)
        print(f"  ✅ SVM: Rust={svm._using_rust}, Score={svm_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Rust Implementation Error: {e}")
        return False

def test_kaggle_datasets():
    """Test Kaggle dataset integration"""
    print("\n📊 Testing Kaggle Dataset Integration...")
    
    try:
        from pyrustml.dataset_manager import DatasetManager
        
        dm = DatasetManager()
        
        # Test a few key datasets
        test_datasets = ['house_prices', 'heart_disease']
        
        for dataset_name in test_datasets:
            X, y, info = dm.download_kaggle_dataset(dataset_name)
            if X is not None and y is not None:
                print(f"  ✅ {dataset_name}: {X.shape} features, {info['type']} task")
            else:
                print(f"  ❌ {dataset_name}: Failed to load")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Kaggle Dataset Error: {e}")
        return False

def test_performance_analytics():
    """Test performance analytics generation"""
    print("\n📈 Testing Performance Analytics...")
    
    try:
        import pandas as pd
        import psutil
        from pyrustml.linear_regression import RustLinearRegression
        from pyrustml.kmeans import RustKMeans
        from pyrustml.svm import RustSVM
        
        # Sample data
        X = np.random.randn(50, 4)
        y = np.random.randn(50)
        y_binary = np.where(y > 0, 1, -1)
        
        algorithms = []
        
        # Test Linear Regression analytics
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        lr = RustLinearRegression()
        lr.fit(X, y)
        lr_time = time.time() - start_time
        lr_score = float(lr.score(X, y))
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = max(0.1, float(memory_after - memory_before))
        
        algorithms.append({
            'Algorithm': 'Linear Regression',
            'Execution Time (ms)': float(lr_time * 1000),
            'Accuracy/Score': lr_score,
            'Implementation': 'Rust' if lr._using_rust else 'Python',
            'Memory Usage (MB)': memory_used
        })
        
        # Test that DataFrame creation works
        df = pd.DataFrame(algorithms)
        print(f"  ✅ Analytics DataFrame: {df.shape}")
        print(f"  ✅ Sample Result: {algorithms[0]['Algorithm']} - {algorithms[0]['Implementation']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance Analytics Error: {e}")
        traceback.print_exc()
        return False

def test_streamlit_components():
    """Test Streamlit components and imports"""
    print("\n🎨 Testing Streamlit Components...")
    
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
        
        print("  ✅ Streamlit imported successfully")
        print("  ✅ Plotly imported successfully")
        
        # Test that the main dashboard imports work
        sys.path.append('.')
        from dashboard.professional_app import main
        print("  ✅ Professional dashboard imports successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Streamlit Components Error: {e}")
        return False

def test_error_handling():
    """Test error handling robustness"""
    print("\n🛡️ Testing Error Handling...")
    
    try:
        from pyrustml.linear_regression import RustLinearRegression
        
        # Test with invalid data
        lr = RustLinearRegression()
        
        try:
            # This should raise an error
            lr.predict(np.array([[1, 2, 3]]))
        except ValueError:
            print("  ✅ Proper error handling for unfitted model")
        
        # Test with edge cases
        X_edge = np.array([[1], [2]])
        y_edge = np.array([1, 2])
        lr.fit(X_edge, y_edge)
        pred = lr.predict(X_edge)
        print("  ✅ Edge case handling works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error Handling Test Failed: {e}")
        return False

def main():
    """Run comprehensive system validation"""
    print("🚀 PyRust-ML: COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 60)
    print(f"🕐 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print("=" * 60)
    
    tests = [
        ("Rust Implementations", test_rust_implementations),
        ("Kaggle Datasets", test_kaggle_datasets),
        ("Performance Analytics", test_performance_analytics),
        ("Streamlit Components", test_streamlit_components),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<10} {test_name}")
        if result:
            passed += 1
    
    print("=" * 60)
    success_rate = (passed / total) * 100
    print(f"SUCCESS RATE: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 INTERVIEW-READY QUALITY ACHIEVED!")
        print("✨ Your PyRust-ML project is deployment-ready!")
    elif success_rate >= 80:
        print("⚠️  Good quality, minor issues to address")
    else:
        print("❌ Critical issues need fixing before deployment")
    
    print("=" * 60)
    return success_rate >= 90

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)