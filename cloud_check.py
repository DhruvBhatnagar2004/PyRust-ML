"""
Cloud Deployment Checker
"""
import os
import sys
import platform

def check_cloud_environment():
    """Check if we're running in Streamlit Cloud"""
    # Common cloud environment indicators
    cloud_indicators = [
        'STREAMLIT_CLOUD' in os.environ,
        'STREAMLIT_SHARING' in os.environ,
        '/app/' in os.getcwd() if os.getcwd() else False,
        platform.system() == 'Linux' and 'streamlit' in str(sys.path),
    ]
    
    return any(cloud_indicators)

def get_rust_status():
    """Get Rust compilation status for current environment"""
    try:
        from pyrustml.linear_regression import RustLinearRegression
        test_lr = RustLinearRegression()
        
        # Check if Rust is actually working
        import numpy as np
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        test_lr.fit(X, y)
        
        return True
    except Exception as e:
        print(f"Rust check failed: {e}")
        return False

if __name__ == "__main__":
    print(f"Cloud Environment: {check_cloud_environment()}")
    print(f"Rust Available: {get_rust_status()}")
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")