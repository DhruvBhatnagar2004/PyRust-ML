"""
PyRust-ML: Professional ML Toolkit Entry Point
Optimized for both local and cloud deployment with intelligent fallback
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
    try:
        # Import and run the professional dashboard
        from dashboard.professional_app import main as dashboard_main
        dashboard_main()
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.info("Please refresh the page or contact support.")

if __name__ == "__main__":
    main()