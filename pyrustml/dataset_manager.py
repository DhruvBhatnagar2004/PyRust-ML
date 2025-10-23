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
            'description': 'Breast cancer diagnosis from cell nuclei measurements'
        },
        'california_housing': {
            'name': 'California Housing Prices',
            'type': 'regression',
            'samples': 20640,
            'features': 8,
            'target': 'continuous',
            'description': 'Housing prices in California districts'
        },
        'digits': {
            'name': 'Handwritten Digits',
            'type': 'classification',
            'samples': 1797,
            'features': 64,
            'classes': 10,
            'description': 'Handwritten digit recognition (8x8 pixel images)'
        }
    }
    
    # Popular Kaggle-style datasets that work excellently with PyRust-ML
    KAGGLE_COMPATIBLE_DATASETS = {
        'titanic': {
            'name': 'Titanic Passenger Survival',
            'task_type': 'classification',
            'description': 'Predict passenger survival on the Titanic - Classic ML competition',
            'target_column': 'survived',
            'difficulty': 'beginner',
            'preprocessing_suggestions': ['handle_missing', 'encode_categorical', 'feature_engineering']
        },
        'house_prices': {
            'name': 'House Price Prediction',
            'task_type': 'regression', 
            'description': 'Predict house prices - Advanced regression challenge',
            'target_column': 'price',
            'difficulty': 'intermediate',
            'preprocessing_suggestions': ['feature_scaling', 'outlier_removal', 'feature_selection']
        },
        'heart_disease': {
            'name': 'Heart Disease Detection',
            'task_type': 'classification',
            'description': 'Medical diagnosis prediction - Healthcare ML application',
            'target_column': 'target',
            'difficulty': 'intermediate',
            'preprocessing_suggestions': ['standardization', 'correlation_analysis']
        },
        'customer_segmentation': {
            'name': 'Customer Analytics',
            'task_type': 'classification',
            'description': 'Customer clustering for marketing analysis',
            'target_column': 'segment',
            'difficulty': 'intermediate',
            'preprocessing_suggestions': ['normalization', 'pca_analysis']
        },
        'iris': {
            'name': 'Iris Species Classification',
            'task_type': 'classification',
            'description': 'Classic multi-class classification benchmark',
            'target_column': 'species',
            'difficulty': 'beginner',
            'preprocessing_suggestions': ['simple_scaling']
        },
        'tips': {
            'name': 'Restaurant Tips Analysis',
            'task_type': 'regression',
            'description': 'Social behavior prediction - Tip amount analysis',
            'target_column': 'tip',
            'difficulty': 'beginner',
            'preprocessing_suggestions': ['categorical_encoding', 'feature_interaction']
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
        """Enhanced dataset loading with Kaggle dataset optimization"""
        try:
            # Read the file with enhanced options
            if uploaded_file.name.endswith('.csv'):
                # Try different encodings for international datasets
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except UnicodeDecodeError:
                        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            # Enhanced validation and preprocessing
            if len(df) == 0:
                raise ValueError("Dataset is empty")
            
            if len(df.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns (features + target)")
            
            # Smart column detection (common Kaggle patterns)
            kaggle_target_columns = ['target', 'label', 'class', 'y', 'outcome', 'result', 
                                   'survived', 'price', 'value', 'score', 'rating']
            
            # Auto-detect target column
            target_col = None
            for col in kaggle_target_columns:
                if col.lower() in [c.lower() for c in df.columns]:
                    target_col = [c for c in df.columns if c.lower() == col.lower()][0]
                    break
            
            # If no common target found, assume last column
            if target_col is None:
                target_col = df.columns[-1]
                st.info(f"Using '{target_col}' as target variable (last column)")
            else:
                st.success(f"Auto-detected target variable: '{target_col}'")
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Enhanced feature preprocessing
            original_shape = X.shape
            
            # Handle missing values intelligently
            if X.isnull().sum().sum() > 0:
                st.warning("Missing values detected - filling with median/mode")
                for col in X.columns:
                    if X[col].dtype in ['object', 'category']:
                        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')
                    else:
                        X[col] = X[col].fillna(X[col].median())
            
            # Smart feature selection and encoding
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns
            
            # Handle categorical features (Kaggle datasets often have these)
            if len(categorical_columns) > 0:
                st.info(f"Categorical columns detected: {list(categorical_columns[:3])}{'...' if len(categorical_columns) > 3 else ''}")
                
                # Simple categorical encoding for demo (could be enhanced)
                for col in categorical_columns:
                    if X[col].nunique() <= 10:  # Low cardinality
                        # One-hot encoding for low cardinality
                        dummies = pd.get_dummies(X[col], prefix=col)
                        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                    else:
                        # Label encoding for high cardinality
                        X[col] = pd.Categorical(X[col]).codes
            
            # Final numeric check
            X = X.select_dtypes(include=[np.number])
            
            if len(X.columns) == 0:
                raise ValueError("No numeric features available after preprocessing")
            
            st.info(f"Processed dataset: {original_shape} → {X.shape} (after preprocessing)")
            
            # Enhanced target processing
            if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                # Classification task
                unique_values = y.nunique()
                if unique_values > 50:
                    st.warning(f"High cardinality target ({unique_values} classes) - consider regression")
                
                y_encoded = pd.Categorical(y).codes
                dataset_type = 'classification'
                target_names = list(pd.Categorical(y).categories)
                st.success(f"Classification task detected: {unique_values} classes")
            else:
                # Regression or classification based on unique values
                unique_values = y.nunique()
                if unique_values <= 20 and y.dtype in ['int64', 'int32']:
                    dataset_type = 'classification'
                    y_encoded = y.astype(int)
                    target_names = [f"Class_{i}" for i in sorted(y.unique())]
                    st.success(f"Classification task: {unique_values} classes")
                else:
                    dataset_type = 'regression'
                    y_encoded = y.astype(float)
                    target_names = ['target_value']
                    st.success(f"Regression task: continuous target")
            
            # Enhanced dataset info with Kaggle-style metadata
            info = {
                'name': uploaded_file.name.replace('.csv', '').replace('.xlsx', ''),
                'type': dataset_type,
                'n_samples': len(X),
                'n_features': len(X.columns),
                'feature_names': list(X.columns),
                'target_names': target_names,
                'target_column': target_col,
                'preprocessing_applied': True,
                'missing_values_handled': True,
                'categorical_encoded': len(categorical_columns) > 0,
                'description': f'Real-world dataset: {uploaded_file.name} ({dataset_type})',
                'dataset_quality': 'Production-ready' if len(X) > 1000 else 'Small-scale demo'
            }
            
            self.current_dataset = (X, y_encoded)
            self.dataset_info = info
            
            return X, y_encoded, info
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None, None, None

    def download_kaggle_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Download and load popular Kaggle datasets directly"""
        
        if dataset_name not in self.KAGGLE_COMPATIBLE_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {list(self.KAGGLE_COMPATIBLE_DATASETS.keys())}")
        
        dataset_config = self.KAGGLE_COMPATIBLE_DATASETS[dataset_name]
        
        try:
            # Optional UI feedback
            try:
                import streamlit as st
                st.info(f"Loading {dataset_config['name']} dataset...")
            except:
                print(f"Loading {dataset_config['name']} dataset...")
            
            # Use direct URLs for popular datasets (no API key needed)
            dataset_urls = {
                'titanic': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
                'house_prices': 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
                'iris': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
                'tips': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv',
                'car_prices': 'https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv'
            }
            
            if dataset_name in dataset_urls:
                try:
                    df = pd.read_csv(dataset_urls[dataset_name])
                    st.success(f"Successfully downloaded {dataset_config['name']} dataset!")
                except Exception as e:
                    st.error(f"Could not download dataset: {e}")
                    return None, None, None
            else:
                # Generate synthetic data based on dataset configuration
                st.info(f"Generating realistic synthetic data for {dataset_config['name']}...")
                df = self._generate_synthetic_dataset(dataset_config)
            
            # Apply dataset-specific preprocessing
            target_col = dataset_config['target_column']
            
            # Handle special cases
            if dataset_name == 'titanic':
                # Titanic-specific preprocessing
                df = df.dropna(subset=['Age', 'Embarked']).reset_index(drop=True)
                df['Age'] = df['Age'].fillna(df['Age'].median())
                df['Fare'] = df['Fare'].fillna(df['Fare'].median())
                
                # Create feature engineering
                df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
                df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
                df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Elder'])
                
                # Select relevant features
                feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
                feature_cols = [col for col in feature_cols if col in df.columns]
                X = df[feature_cols].copy()
                y = df[target_col]
                
            elif dataset_name == 'house_prices':
                # House prices preprocessing
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                X = df[numeric_cols].drop(columns=[target_col] if target_col in numeric_cols else [])
                y = df[target_col] if target_col in df.columns else df[df.columns[-1]]
                
            else:
                # General preprocessing
                X = df.drop(columns=[target_col] if target_col in df.columns else [])
                y = df[target_col] if target_col in df.columns else df[df.columns[-1]]
            
            # Enhanced preprocessing pipeline
            X, y, processed_info = self._preprocess_kaggle_data(X, y, dataset_config)
            
            # Create comprehensive dataset info
            info = {
                'name': dataset_config['name'],
                'type': dataset_config['task_type'],
                'n_samples': len(X),
                'n_features': len(X.columns),
                'feature_names': list(X.columns),
                'target_names': processed_info['target_names'],
                'target_column': target_col,
                'description': dataset_config['description'],
                'preprocessing_suggestions': dataset_config['preprocessing_suggestions'],
                'difficulty': dataset_config['difficulty'],
                'source': 'Kaggle-compatible',
                'quality': 'Production-ready',
                'real_world': True
            }
            
            self.current_dataset = (X, y)
            self.dataset_info = info
            
            st.success(f"✅ {dataset_config['name']} loaded: {len(X)} samples, {len(X.columns)} features")
            return X, y, info
            
        except Exception as e:
            st.error(f"Error loading {dataset_name}: {str(e)}")
            return None, None, None
    
    def _generate_synthetic_dataset(self, config: Dict) -> pd.DataFrame:
        """Generate realistic synthetic data based on dataset configuration"""
        np.random.seed(42)  # For reproducibility
        
        n_samples = np.random.randint(1000, 5000)
        
        if config['task_type'] == 'classification':
            # Generate classification data
            n_features = np.random.randint(5, 15)
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(2, n_features//2),
                n_redundant=max(1, n_features//4),
                n_classes=np.random.randint(2, 5),
                random_state=42
            )
            
            # Create realistic feature names based on domain
            if 'heart' in config['name'].lower():
                feature_names = ['age', 'chest_pain', 'blood_pressure', 'cholesterol', 'max_heart_rate']
            elif 'customer' in config['name'].lower():
                feature_names = ['age', 'income', 'spending_score', 'loyalty_years', 'purchase_frequency']
            else:
                feature_names = [f'feature_{i}' for i in range(n_features)]
            
        else:
            # Generate regression data
            n_features = np.random.randint(8, 20)
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=0.1,
                random_state=42
            )
            
            if 'house' in config['name'].lower():
                feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 'yr_built']
            else:
                feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Pad feature names if needed
        while len(feature_names) < X.shape[1]:
            feature_names.append(f'feature_{len(feature_names)}')
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
        df[config['target_column']] = y
        
        return df
    
    def _preprocess_kaggle_data(self, X: pd.DataFrame, y: pd.Series, config: Dict) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Advanced preprocessing for Kaggle-style datasets"""
        
        processed_info = {'preprocessing_steps': []}
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            processed_info['preprocessing_steps'].append('Missing values imputed')
            for col in X.columns:
                if X[col].dtype in ['object', 'category']:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')
                else:
                    X[col] = X[col].fillna(X[col].median())
        
        # Enhanced categorical encoding
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            processed_info['preprocessing_steps'].append(f'Encoded {len(categorical_columns)} categorical features')
            
            for col in categorical_columns:
                if X[col].nunique() <= 5:  # One-hot for low cardinality
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                else:  # Label encoding for high cardinality
                    X[col] = pd.Categorical(X[col]).codes
        
        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number])
        
        # Feature scaling for better performance
        if len(X.columns) > 0:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            X = X_scaled
            processed_info['preprocessing_steps'].append('Features standardized')
        
        # Process target variable
        if config['task_type'] == 'classification':
            if y.dtype == 'object':
                y_categories = pd.Categorical(y)
                y = y_categories.codes
                target_names = list(y_categories.categories)
            else:
                target_names = [f"Class_{i}" for i in sorted(y.unique())]
        else:
            y = y.astype(float)
            target_names = ['target_value']
        
        processed_info['target_names'] = target_names
        processed_info['final_shape'] = X.shape
        
        return X, y, processed_info
    
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