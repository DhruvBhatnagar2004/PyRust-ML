"""
PyRust-ML: Cloud Deployment Entry Point
Optimized for Streamlit Community Cloud deployment
"""

import streamlit as st
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Force fallback mode for cloud deployment
os.environ['PYRUST_ML_FORCE_FALLBACK'] = '1'

# Import and run the main dashboard
if __name__ == "__main__":
    from dashboard.professional_app import main
    main()