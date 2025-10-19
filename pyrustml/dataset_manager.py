"""
Enhanced Dataset Management for PyRust-ML

Provides comprehensive dataset handling including:
- Built-in famous ML datasets
- Custom dataset upload and processing
- Data preprocessing and feature engineering
- Dataset statistics and visualization
"""

import pandas as pd
import numpy as np
import io
import streamlit as st
from typing import Tuple, Dict, List, Optional, Union
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    fetch_california_housing, make_regression, make_classification, make_blobs
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DatasetManager:
    """Comprehensive dataset management for ML experiments"""
    
    BUILTIN_DATASETS = {
        'iris': {
            'name': 'Iris Flower Dataset',
            'type': 'classification',
            'samples': 150,
            'features': 4,
            'classes': 3,
            'description': 'Classic dataset for flower species classification'
        },
        'wine': {
            'name': 'Wine Quality Dataset',
            'type': 'classification',
            'samples': 178,
            'features': 13,
            'classes': 3,
            'description': 'Wine quality classification based on chemical analysis'
        },
        'breast_cancer': {
            'name': 'Breast Cancer Wisconsin',
            'type': 'classification',
            'samples': 569,
            'features': 30,
            'classes': 2,
            'description': 'Breast cancer diagnosis (malignant/benign)'
        },
        'california_housing': {
            'name': 'California Housing Prices',
            'type': 'regression',
            'samples': 20640,
            'features': 8,
            'classes': None,
            'description': 'Median house values in California districts'
        },
        'digits': {
            'name': 'Handwritten Digits',
            'type': 'classification',
            'samples': 1797,
            'features': 64,
            'classes': 10,
            'description': 'Handwritten digit recognition (0-9)'
        }
    }
    
    def __init__(self):
        self.current_dataset = None
        self.dataset_info = None
        self.preprocessors = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
    
    def load_builtin_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Load a built-in dataset"""
        if dataset_name not in self.BUILTIN_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not available")
        
        if dataset_name == 'iris':
            data = load_iris()
            feature_names = data.feature_names
            target_names = data.target_names
        elif dataset_name == 'wine':
            data = load_wine()
            feature_names = data.feature_names
            target_names = data.target_names
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            feature_names = data.feature_names
            target_names = data.target_names
        elif dataset_name == 'california_housing':
            data = fetch_california_housing()
            feature_names = data.feature_names
            target_names = ['house_value']
        elif dataset_name == 'digits':
            data = load_digits()
            feature_names = [f'pixel_{i}' for i in range(64)]
            target_names = [str(i) for i in range(10)]
        
        # Create DataFrame
        X = pd.DataFrame(data.data, columns=feature_names)
        y = pd.Series(data.target, name='target')
        
        # Dataset info
        info = {
            'name': self.BUILTIN_DATASETS[dataset_name]['name'],
            'type': self.BUILTIN_DATASETS[dataset_name]['type'],
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': list(X.columns),
            'target_names': target_names,
            'description': self.BUILTIN_DATASETS[dataset_name]['description']
        }
        
        self.current_dataset = (X, y)
        self.dataset_info = info
        
        return X, y, info
    
    def load_custom_dataset(self, uploaded_file) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Load a custom dataset from uploaded file"""
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            # Basic validation
            if len(df) == 0:
                raise ValueError("Dataset is empty")
            
            if len(df.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns (features + target)")
            
            # Assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Handle non-numeric data
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < len(X.columns):
                st.warning(f"Non-numeric columns detected. Using only numeric features: {list(numeric_columns)}")
                X = X[numeric_columns]
            
            # Convert target to numeric if possible
            if y.dtype == 'object':
                try:
                    # Try to convert to category codes
                    y = pd.Categorical(y).codes
                    dataset_type = 'classification'
                except:
                    st.error("Could not convert target variable to numeric format")
                    return None, None, None
            else:
                # Determine if classification or regression
                if len(y.unique()) <= 20 and y.dtype in ['int64', 'int32']:
                    dataset_type = 'classification'
                else:
                    dataset_type = 'regression'
            
            # Dataset info
            info = {
                'name': uploaded_file.name,
                'type': dataset_type,
                'n_samples': len(X),
                'n_features': len(X.columns),
                'feature_names': list(X.columns),
                'target_names': list(y.unique()) if dataset_type == 'classification' else ['target'],
                'description': f'Custom dataset from {uploaded_file.name}'
            }
            
            self.current_dataset = (X, y)
            self.dataset_info = info
            
            return X, y, info
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None, None, None
    
    def generate_synthetic_dataset(self, task_type: str, n_samples: int = 1000, 
                                 n_features: int = 10, **kwargs) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Generate synthetic dataset for testing"""
        if task_type == 'regression':
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=kwargs.get('noise', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
            target_names = ['target']
            
        elif task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=kwargs.get('n_classes', 2),
                n_redundant=kwargs.get('n_redundant', 0),
                random_state=kwargs.get('random_state', 42)
            )
            target_names = [f'class_{i}' for i in range(kwargs.get('n_classes', 2))]
            
        elif task_type == 'clustering':
            X, y = make_blobs(
                n_samples=n_samples,
                centers=kwargs.get('n_clusters', 3),
                n_features=n_features,
                random_state=kwargs.get('random_state', 42)
            )
            target_names = [f'cluster_{i}' for i in range(kwargs.get('n_clusters', 3))]
        
        # Create DataFrames
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        # Dataset info
        info = {
            'name': f'Synthetic {task_type.title()} Dataset',
            'type': task_type,
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_names': feature_names,
            'target_names': target_names,
            'description': f'Synthetically generated {task_type} dataset'
        }
        
        self.current_dataset = (X_df, y_series)
        self.dataset_info = info
        
        return X_df, y_series, info
    
    def preprocess_data(self, X: pd.DataFrame, scaler_type: str = 'standard') -> pd.DataFrame:
        """Preprocess the dataset"""
        if scaler_type not in self.preprocessors:
            raise ValueError(f"Scaler {scaler_type} not available")
        
        scaler = self.preprocessors[scaler_type]
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train/test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def get_dataset_statistics(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Generate comprehensive dataset statistics"""
        stats = {
            'basic_info': {
                'n_samples': len(X),
                'n_features': len(X.columns),
                'memory_usage_mb': (X.memory_usage(deep=True).sum() + y.memory_usage(deep=True)) / 1024**2
            },
            'feature_stats': X.describe().to_dict(),
            'missing_values': X.isnull().sum().to_dict(),
            'target_stats': {
                'unique_values': len(y.unique()),
                'value_counts': y.value_counts().to_dict()
            }
        }
        
        return stats
    
    def create_data_visualization(self, X: pd.DataFrame, y: pd.Series, 
                                max_features: int = 5) -> List[go.Figure]:
        """Create visualizations for the dataset"""
        figures = []
        
        # Limit features for visualization
        features_to_plot = X.columns[:max_features]
        
        # Feature distribution plots
        fig_dist = make_subplots(
            rows=2, cols=min(3, len(features_to_plot)),
            subplot_titles=[f'Distribution of {col}' for col in features_to_plot[:6]]
        )
        
        for i, feature in enumerate(features_to_plot[:6]):
            row = 1 + i // 3
            col = 1 + i % 3
            
            fig_dist.add_trace(
                go.Histogram(x=X[feature], name=feature, showlegend=False),
                row=row, col=col
            )
        
        fig_dist.update_layout(title_text="Feature Distributions", height=500)
        figures.append(fig_dist)
        
        # Correlation heatmap (if not too many features)
        if len(X.columns) <= 20:
            corr_matrix = X.corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 8}
            ))
            fig_corr.update_layout(
                title="Feature Correlation Matrix",
                height=600,
                width=600
            )
            figures.append(fig_corr)
        
        # Target distribution
        if len(y.unique()) <= 20:  # Categorical
            fig_target = px.histogram(
                x=y, 
                title="Target Variable Distribution",
                labels={'x': 'Target', 'y': 'Count'}
            )
        else:  # Continuous
            fig_target = px.histogram(
                x=y, 
                title="Target Variable Distribution",
                labels={'x': 'Target', 'y': 'Count'},
                nbins=50
            )
        
        figures.append(fig_target)
        
        # Pairwise scatter plots (for small datasets)
        if len(features_to_plot) >= 2 and len(X) <= 2000:
            fig_scatter = px.scatter_matrix(
                X[features_to_plot[:4]], 
                color=y,
                title="Pairwise Feature Relationships",
                height=600
            )
            figures.append(fig_scatter)
        
        return figures
    
    def export_processed_data(self, X: pd.DataFrame, y: pd.Series) -> bytes:
        """Export processed data as CSV"""
        combined_data = X.copy()
        combined_data['target'] = y
        
        # Convert to CSV bytes
        output = io.StringIO()
        combined_data.to_csv(output, index=False)
        return output.getvalue().encode('utf-8')


# Streamlit integration functions
def render_dataset_selector():
    """Render the dataset selection interface"""
    st.sidebar.header("📊 Dataset Selection")
    
    dataset_source = st.sidebar.radio(
        "Choose Dataset Source:",
        ["Built-in Datasets", "Upload Custom Dataset", "Generate Synthetic"]
    )
    
    dataset_manager = DatasetManager()
    
    if dataset_source == "Built-in Datasets":
        dataset_name = st.sidebar.selectbox(
            "Select Dataset:",
            list(dataset_manager.BUILTIN_DATASETS.keys()),
            format_func=lambda x: dataset_manager.BUILTIN_DATASETS[x]['name']
        )
        
        if st.sidebar.button("Load Dataset"):
            X, y, info = dataset_manager.load_builtin_dataset(dataset_name)
            st.session_state.dataset = (X, y, info)
            st.session_state.dataset_manager = dataset_manager
            st.success(f"Loaded {info['name']} dataset!")
    
    elif dataset_source == "Upload Custom Dataset":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx'],
            help="Upload your dataset. Last column will be treated as target variable."
        )
        
        if uploaded_file is not None:
            if st.sidebar.button("Process Upload"):
                X, y, info = dataset_manager.load_custom_dataset(uploaded_file)
                if X is not None:
                    st.session_state.dataset = (X, y, info)
                    st.session_state.dataset_manager = dataset_manager
                    st.success(f"Loaded custom dataset: {info['name']}")
    
    elif dataset_source == "Generate Synthetic":
        task_type = st.sidebar.selectbox(
            "Task Type:",
            ["regression", "classification", "clustering"]
        )
        
        n_samples = st.sidebar.slider("Number of Samples:", 100, 10000, 1000)
        n_features = st.sidebar.slider("Number of Features:", 2, 50, 10)
        
        if task_type == "classification":
            n_classes = st.sidebar.slider("Number of Classes:", 2, 10, 3)
        elif task_type == "clustering":
            n_clusters = st.sidebar.slider("Number of Clusters:", 2, 10, 3)
        
        if st.sidebar.button("Generate Dataset"):
            kwargs = {'random_state': 42}
            if task_type == "classification":
                kwargs['n_classes'] = n_classes
            elif task_type == "clustering":
                kwargs['n_clusters'] = n_clusters
            
            X, y, info = dataset_manager.generate_synthetic_dataset(
                task_type, n_samples, n_features, **kwargs
            )
            st.session_state.dataset = (X, y, info)
            st.session_state.dataset_manager = dataset_manager
            st.success(f"Generated {info['name']}!")
    
    return dataset_manager


def render_dataset_overview():
    """Render dataset overview and statistics"""
    if 'dataset' not in st.session_state:
        st.info("👆 Please select a dataset from the sidebar to get started!")
        return None
    
    X, y, info = st.session_state.dataset
    dataset_manager = st.session_state.dataset_manager
    
    # Dataset info
    st.header(f"📊 {info['name']}")
    st.write(info['description'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Samples", f"{info['n_samples']:,}")
    with col2:
        st.metric("Features", info['n_features'])
    with col3:
        st.metric("Task Type", info['type'].title())
    with col4:
        if info['type'] == 'classification':
            st.metric("Classes", len(info['target_names']))
        else:
            st.metric("Target Range", f"{y.min():.2f} - {y.max():.2f}")
    
    # Data preview
    st.subheader("🔍 Data Preview")
    preview_rows = st.slider("Rows to preview:", 5, 50, 10)
    st.dataframe(pd.concat([X.head(preview_rows), y.head(preview_rows)], axis=1))
    
    # Statistics
    st.subheader("📈 Dataset Statistics")
    stats = dataset_manager.get_dataset_statistics(X, y)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Basic Information:**")
        st.json(stats['basic_info'])
    
    with col2:
        st.write("**Missing Values:**")
        missing_df = pd.DataFrame([stats['missing_values']]).T
        missing_df.columns = ['Missing Count']
        st.dataframe(missing_df)
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    figures = dataset_manager.create_data_visualization(X, y)
    
    for i, fig in enumerate(figures):
        st.plotly_chart(fig, width="stretch", key=f"dataset_viz_{i}")
    
    return X, y, info