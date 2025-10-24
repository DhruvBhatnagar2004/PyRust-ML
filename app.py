"""
PyRust-ML: Professional ML Toolkit Entry Point
Optimized for both local and cloud deployment with intelligent fallback
Version: 2.0 - Cloud Production Ready - October 2025
"""

import streamlit as st
import os
import sys
import warnings

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure page
st.set_page_config(
    page_title="PyRust-ML: High-Performance ML Toolkit",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    # Debug info in sidebar
    with st.sidebar:
        st.write("🔍 **Environment Debug**")
        st.write(f"Python: {sys.version.split()[0]}")
        st.write(f"Platform: {sys.platform}")
        st.write(f"Working Dir: {os.path.basename(os.getcwd())}")
        
        # Check cloud environment
        cloud_env = any([
            'STREAMLIT_CLOUD' in os.environ,
            '/app/' in os.getcwd(),
            'streamlit' in str(sys.path).lower()
        ])
        st.write(f"Cloud: {'Yes' if cloud_env else 'No'}")
        
        # Test Rust availability
        try:
            from pyrustml import RustLinearRegression
            lr = RustLinearRegression()
            rust_status = hasattr(lr, '_using_rust') and lr._using_rust
            if rust_status:
                st.write(f"Rust: ✅ Active")
            else:
                st.write(f"Rust: 🐍 Python Mode")
                st.caption("*Cloud deployment uses Python implementations for compatibility*")
        except Exception as e:
            st.write(f"Rust: ❌ Error")
            st.caption("*Using Python fallbacks*")
    
    try:
        # Import and run the professional dashboard
        from dashboard.professional_app import main as dashboard_main
        dashboard_main()
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.info("Please refresh the page or contact support.")

if __name__ == "__main__":
    main()