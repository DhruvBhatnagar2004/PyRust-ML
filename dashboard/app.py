"""
Streamlit Dashboard for PyRust-ML Benchmarking

A web interface to visualize performance comparisons between 
Rust and Python machine learning implementations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.datasets import make_regression, make_classification, make_blobs

# Import our benchmarking utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from pyrustml.benchmarks import benchmark_models, calculate_speedup
except ImportError:
    st.error("Could not import PyRust-ML. Please ensure the package is installed.")
    st.stop()


def main():
    st.set_page_config(
        page_title="PyRust-ML Benchmark Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ PyRust-ML Benchmark Dashboard")
    st.markdown("""
    Compare the performance of **Rust-accelerated** vs **Python (Scikit-learn)** machine learning algorithms.
    
    This dashboard demonstrates the speed improvements achieved by implementing core ML algorithms in Rust 
    with Python bindings using PyO3.
    """)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Benchmark Configuration")
    
    dataset_size = st.sidebar.slider(
        "Dataset Size", 
        min_value=100, 
        max_value=10000, 
        value=1000, 
        step=100,
        help="Number of samples in the dataset"
    )
    
    n_features = st.sidebar.slider(
        "Number of Features", 
        min_value=2, 
        max_value=50, 
        value=10, 
        step=1,
        help="Number of features in the dataset"
    )
    
    n_clusters = st.sidebar.slider(
        "Number of Clusters (K-Means)", 
        min_value=2, 
        max_value=10, 
        value=3, 
        step=1,
        help="Number of clusters for K-Means algorithm"
    )
    
    random_state = st.sidebar.number_input(
        "Random Seed", 
        min_value=0, 
        max_value=9999, 
        value=42,
        help="Random seed for reproducible results"
    )
    
    # Run benchmark button
    if st.sidebar.button("🚀 Run Benchmarks", type="primary"):
        run_benchmarks(dataset_size, n_features, n_clusters, random_state)
    
    # Load existing results if available
    if 'benchmark_results' in st.session_state:
        display_results(st.session_state.benchmark_results)
    else:
        st.info("👆 Configure your benchmark parameters and click 'Run Benchmarks' to start!")
        show_sample_results()


def run_benchmarks(dataset_size, n_features, n_clusters, random_state):
    """Run the benchmarks and store results in session state"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Initializing benchmarks...")
    progress_bar.progress(10)
    
    try:
        # Run benchmarks
        status_text.text("🔄 Running benchmarks... This may take a moment.")
        progress_bar.progress(50)
        
        results_df = benchmark_models(
            dataset_size=dataset_size,
            n_features=n_features, 
            n_clusters=n_clusters,
            random_state=random_state
        )
        
        progress_bar.progress(80)
        status_text.text("📊 Calculating speedup metrics...")
        
        # Calculate speedup
        speedup_df = calculate_speedup(results_df)
        
        progress_bar.progress(100)
        status_text.text("✅ Benchmarks completed!")
        
        # Store in session state
        st.session_state.benchmark_results = results_df
        st.session_state.speedup_results = speedup_df
        st.session_state.benchmark_config = {
            'dataset_size': dataset_size,
            'n_features': n_features,
            'n_clusters': n_clusters,
            'random_state': random_state
        }
        
        time.sleep(1)  # Brief pause to show completion
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_results(results_df)
        
    except Exception as e:
        st.error(f"❌ Benchmark failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def display_results(results_df):
    """Display benchmark results with visualizations"""
    
    st.header("📊 Benchmark Results")
    
    # Initialize chart counter if not exists
    if 'chart_counter' not in st.session_state:
        st.session_state.chart_counter = 0
    
    # Increment counter for unique IDs
    st.session_state.chart_counter += 1
    chart_id = st.session_state.chart_counter
    
    # Configuration info
    if 'benchmark_config' in st.session_state:
        config = st.session_state.benchmark_config
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dataset Size", f"{config['dataset_size']:,}")
        with col2:
            st.metric("Features", config['n_features'])
        with col3:
            st.metric("Clusters", config['n_clusters'])
        with col4:
            st.metric("Random Seed", config['random_state'])
    
    # Performance comparison charts
    st.subheader("⚡ Performance Comparison")
    
    # Execution time comparison
    fig_time = create_time_comparison_chart(results_df)
    st.plotly_chart(fig_time, width="stretch", key=f"time_comparison_chart_{chart_id}")
    
    # Speedup chart
    if 'speedup_results' in st.session_state:
        fig_speedup = create_speedup_chart(st.session_state.speedup_results)
        st.plotly_chart(fig_speedup, width="stretch", key=f"speedup_comparison_chart_{chart_id}")
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Format the results for better display
    display_df = results_df.copy()
    numeric_cols = ['Fit Time (s)', 'Predict Time (s)', 'Total Time (s)', 'Accuracy', 'R² Score', 'MSE']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else "N/A")
    
    st.dataframe(display_df, width="stretch")
    
    # Algorithm-specific metrics
    st.subheader("🎯 Algorithm Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Linear Regression metrics
        lr_data = results_df[results_df['Algorithm'] == 'Linear Regression']
        if not lr_data.empty:
            st.write("**Linear Regression**")
            for _, row in lr_data.iterrows():
                if pd.notnull(row['R² Score']):
                    st.metric(
                        f"R² Score ({row['Implementation']})", 
                        f"{row['R² Score']:.4f}"
                    )
    
    with col2:
        # SVM metrics
        svm_data = results_df[results_df['Algorithm'] == 'SVM']
        if not svm_data.empty:
            st.write("**SVM**")
            for _, row in svm_data.iterrows():
                if pd.notnull(row['Accuracy']):
                    st.metric(
                        f"Accuracy ({row['Implementation']})", 
                        f"{row['Accuracy']:.4f}"
                    )
    
    with col3:
        # K-Means info
        kmeans_data = results_df[results_df['Algorithm'] == 'K-Means']
        if not kmeans_data.empty:
            st.write("**K-Means**")
            for _, row in kmeans_data.iterrows():
                st.metric(
                    f"Fit Time ({row['Implementation']})", 
                    f"{row['Fit Time (s)']:.4f}s"
                )


def create_time_comparison_chart(results_df):
    """Create a bar chart comparing execution times"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Fit Time Comparison', 'Total Time Comparison'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    algorithms = results_df['Algorithm'].unique()
    
    for i, algorithm in enumerate(algorithms):
        alg_data = results_df[results_df['Algorithm'] == algorithm]
        
        # Fit time comparison
        fig.add_trace(
            go.Bar(
                name=f'{algorithm}',
                x=alg_data['Implementation'],
                y=alg_data['Fit Time (s)'],
                text=alg_data['Fit Time (s)'].apply(lambda x: f'{x:.4f}s'),
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # Total time comparison
        fig.add_trace(
            go.Bar(
                name=f'{algorithm}',
                x=alg_data['Implementation'],
                y=alg_data['Total Time (s)'],
                text=alg_data['Total Time (s)'].apply(lambda x: f'{x:.4f}s'),
                textposition='auto',
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title="⏱️ Execution Time Comparison",
        height=400,
        barmode='group'
    )
    
    fig.update_xaxes(title_text="Implementation", row=1, col=1)
    fig.update_xaxes(title_text="Implementation", row=1, col=2)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
    
    return fig


def create_speedup_chart(speedup_df):
    """Create a bar chart showing speedup factors"""
    
    fig = px.bar(
        speedup_df, 
        x='Algorithm', 
        y='Total Speedup',
        title='🚀 Rust vs Python Speedup (Higher is Better)',
        text='Total Speedup',
        color='Total Speedup',
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}x', textposition='outside')
    fig.update_layout(
        yaxis_title='Speedup Factor',
        height=400,
        showlegend=False
    )
    
    # Add a horizontal line at y=1 (no speedup)
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                  annotation_text="No Speedup (1x)")
    
    return fig


def show_sample_results():
    """Show sample results to demonstrate the dashboard"""
    
    st.subheader("📊 Sample Results")
    st.markdown("""
    Here's what the benchmark results might look like. The actual performance will depend on:
    - Dataset size and complexity
    - Hardware specifications
    - Compilation optimizations
    """)
    
    # Sample data
    sample_data = {
        'Algorithm': ['Linear Regression', 'Linear Regression', 'SVM', 'SVM', 'K-Means', 'K-Means'],
        'Implementation': ['Scikit-learn', 'Rust', 'Scikit-learn', 'Rust', 'Scikit-learn', 'Rust'],
        'Fit Time (s)': [0.0023, 0.0008, 0.0156, 0.0051, 0.0089, 0.0032],
        'Predict Time (s)': [0.0001, 0.0001, 0.0034, 0.0012, 0.0000, 0.0000],
        'Total Time (s)': [0.0024, 0.0009, 0.0190, 0.0063, 0.0089, 0.0032],
        'Accuracy': [None, None, 0.9450, 0.9420, None, None],
        'R² Score': [0.9234, 0.9231, None, None, None, None]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Show sample speedup
    speedup_data = {
        'Algorithm': ['Linear Regression', 'SVM', 'K-Means'],
        'Total Speedup': [2.67, 3.02, 2.78]
    }
    speedup_sample = pd.DataFrame(speedup_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sample Execution Times**")
        st.dataframe(sample_df[['Algorithm', 'Implementation', 'Total Time (s)']], width="stretch")
    
    with col2:
        st.write("**Sample Speedup Factors**")
        st.dataframe(speedup_sample, width="stretch")
    
    # Sample speedup chart
    fig_sample = px.bar(
        speedup_sample, 
        x='Algorithm', 
        y='Total Speedup',
        title='Sample Speedup Results',
        text='Total Speedup'
    )
    fig_sample.update_traces(texttemplate='%{text:.2f}x', textposition='outside')
    fig_sample.add_hline(y=1.0, line_dash="dash", line_color="red")
    
    st.plotly_chart(fig_sample, width="stretch", key="sample_speedup_chart_static")


if __name__ == "__main__":
    main()