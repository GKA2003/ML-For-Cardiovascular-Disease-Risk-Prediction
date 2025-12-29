"""
Data preprocessing module for Heart Disease ML Pipeline
Handles data loading, cleaning, and initial preprocessing
"""

import numpy as np
from scipy import stats
import pandas as pd
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    TRAIN_FILE, TEST_FILE, TARGET_COLUMN,
    RANDOM_SEED, TEST_SIZE
)
from src.utils import load_data, validate_dataframe, Timer

# Get module logger
module_logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing class for cardiovascular dataset.
    Handles loading, cleaning, validation, and initial preprocessing.
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Initialise the preprocessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.train_data = None
        self.test_data = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.data_info = {}
        
    def load_datasets(self, train_path: Optional[Path] = None, 
                     test_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets.
        
        Args:
            train_path: Path to training data (optional)
            test_path: Path to test data (optional)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        with Timer("Data loading"):
            # Use default paths if not provided
            train_path = train_path or RAW_DATA_DIR / TRAIN_FILE
            test_path = test_path or RAW_DATA_DIR / TEST_FILE
            
            # Load datasets
            self.train_data = load_data(train_path)
            self.test_data = load_data(test_path)

            # Store original shapes
            self.data_info['original_train_shape'] = self.train_data.shape
            self.data_info['original_test_shape'] = self.test_data.shape

            module_logger.info(f"Train data shape: {self.train_data.shape}")
            module_logger.info(f"Test data shape: {self.test_data.shape}")

            # Drop identifier columns from modelling data
            for name, df in (("train", self.train_data), ("test", self.test_data)):
                if "id" in df.columns:
                    df.drop(columns=["id"], inplace=True)
                    module_logger.info(f"Dropped 'id' column from {name} data (identifier not used as a feature)")

            # Validate target column exists in train data
            if TARGET_COLUMN not in self.train_data.columns:
                raise ValueError(f"Target column '{TARGET_COLUMN}' not found in training data")

            # Identify features (exclude target and identifier columns)
            identifier_features = ["id"]
            self.feature_names = [
                col for col in self.train_data.columns
                if col not in identifier_features and col != TARGET_COLUMN
            ]

            # Validate test data has same features
            validate_dataframe(self.test_data, self.feature_names, "Test data")
            
        return self.train_data, self.test_data
    
    def analyze_data_quality(self) -> Dict[str, pd.DataFrame]:
        """
        Analyse data quality including missing values, data types, and basic statistics.
        
        Returns:
            Dictionary containing quality analysis results
        """
        module_logger.info("Analysing data quality...")
        
        quality_report = {}
        
        # Missing values analysis
        train_missing = pd.DataFrame({
            'column': self.train_data.columns,
            'missing_count': self.train_data.isnull().sum(),
            'missing_percentage': (self.train_data.isnull().sum() / len(self.train_data)) * 100
        })
        train_missing = train_missing[train_missing['missing_count'] > 0]
        quality_report['train_missing'] = train_missing
        
        test_missing = pd.DataFrame({
            'column': self.test_data.columns,
            'missing_count': self.test_data.isnull().sum(),
            'missing_percentage': (self.test_data.isnull().sum() / len(self.test_data)) * 100
        })
        test_missing = test_missing[test_missing['missing_count'] > 0]
        quality_report['test_missing'] = test_missing
        
        # Data types analysis
        dtypes_df = pd.DataFrame({
            'column': self.feature_names,
            'train_dtype': [self.train_data[col].dtype for col in self.feature_names],
            'test_dtype': [self.test_data[col].dtype for col in self.feature_names],
            'train_unique': [self.train_data[col].nunique() for col in self.feature_names],
            'test_unique': [self.test_data[col].nunique() for col in self.feature_names]
        })
        quality_report['data_types'] = dtypes_df
        
        # Identify categorical vs numerical features
        self._identify_feature_types()
        
        # Basic statistics for numerical features
        if self.numerical_features:
            train_stats = self.train_data[self.numerical_features].describe()
            test_stats = self.test_data[self.numerical_features].describe()
            quality_report['train_numerical_stats'] = train_stats
            quality_report['test_numerical_stats'] = test_stats
        
        # Categorical features distribution
        if self.categorical_features:
            cat_distributions = {}
            for col in self.categorical_features:
                cat_distributions[col] = {
                    'train': self.train_data[col].value_counts(),
                    'test': self.test_data[col].value_counts()
                }
            quality_report['categorical_distributions'] = cat_distributions
        
        # Target variable analysis (class balance)
        target_dist = self.train_data[TARGET_COLUMN].value_counts()
        target_pct = self.train_data[TARGET_COLUMN].value_counts(normalize=True) * 100
        quality_report['target_distribution'] = pd.DataFrame({
            'count': target_dist,
            'percentage': target_pct
        })
        
        # Calculate class imbalance ratio
        imbalance_ratio = target_dist.max() / target_dist.min()
        self.data_info['imbalance_ratio'] = imbalance_ratio
        module_logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        return quality_report
    
    def _identify_feature_types(self) -> None:
        """Identify categorical and numerical features with identifier handling."""
        identifier_features = ['id']  # Explicit identifier columns
        categorical_features = []
        numerical_features = []

        force_numeric = [
            'age', 'totChol', 'sysBP', 'diaBP', 'BMI',
            'heartRate', 'glucose',
            'diabetes', 'prevalentHyp', 'BPMeds'
        ]

        for col in self.feature_names:
            # Skip identifier columns
            if col in identifier_features:
                continue

            if col in force_numeric:
                numerical_features.append(col)
                continue

            # Check actual data type
            if pd.api.types.is_numeric_dtype(self.train_data[col]):
                # Check if it might be categorical (low cardinality)
                unique_vals = self.train_data[col].nunique()
                unique_ratio = unique_vals / len(self.train_data)
                
                # Treat as categorical if low cardinality
                if unique_vals <= 10 or (unique_ratio < 0.05 and unique_vals < 50):
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                # String or object type - categorical
                categorical_features.append(col)
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        module_logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        module_logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")    
    

    def handle_missing_values(self, strategy: str = 'median', 
                            threshold: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values in the datasets with proper type conversion.
        Now includes outlier capping, distribution transformation, and high-cardinality feature binning.
        """
        module_logger.info(f"Handling missing values with strategy: {strategy}")
        
        train_df = self.train_data.copy()
        test_df = self.test_data.copy()
        
        # Drop columns with too many missing values
        for df in [train_df, test_df]:
            missing_pct = df.isnull().sum() / len(df)
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
            if cols_to_drop:
                module_logger.warning(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
                df.drop(columns=cols_to_drop, inplace=True)
        
        # Update feature lists after dropping columns
        self.feature_names = [col for col in self.feature_names if col in train_df.columns]
        self._identify_feature_types()
        
        # Imputation based on strategy
        if strategy == 'drop':
            # Drop rows with any missing values
            train_df = train_df.dropna()
            test_df = test_df.dropna()
        else:
            # Separate X and y for training data
            X_train = train_df[self.feature_names]
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df[self.feature_names]
            
            # Create imputers for different feature types
            if strategy == 'knn':
                # KNN imputation for all features
                imputer = KNNImputer(n_neighbors=5)
                X_train_imputed = pd.DataFrame(
                    imputer.fit_transform(X_train),
                    columns=self.feature_names,
                    index=X_train.index
                )
                X_test_imputed = pd.DataFrame(
                    imputer.transform(X_test),
                    columns=self.feature_names,
                    index=X_test.index
                )
            else:
                # Simple imputation with different strategies for different types
                transformers = []
                
                if self.numerical_features:
                    num_strategy = strategy if strategy in ['mean', 'median'] else 'median'
                    num_imputer = SimpleImputer(strategy=num_strategy)
                    transformers.append(('num', num_imputer, self.numerical_features))
                
                if self.categorical_features:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    transformers.append(('cat', cat_imputer, self.categorical_features))
                
                if transformers:
                    preprocessor = ColumnTransformer(
                        transformers=transformers, 
                        remainder='passthrough',
                        verbose_feature_names_out=False
                    )
                    
                    # Fit on training data and transform both sets
                    X_train_imputed = preprocessor.fit_transform(X_train)
                    X_test_imputed = preprocessor.transform(X_test)
                    
                    # Get feature names in correct order
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names_ordered = preprocessor.get_feature_names_out()
                    else:
                        feature_names_ordered = []
                        for name, transformer, features in transformers:
                            feature_names_ordered.extend(features)
                        all_features = X_train.columns.tolist()
                        transformed_features = set()
                        for _, _, features in transformers:
                            transformed_features.update(features)
                        remainder_features = [f for f in all_features if f not in transformed_features]
                        feature_names_ordered.extend(remainder_features)
                    
                    # Convert back to DataFrame
                    X_train_imputed = pd.DataFrame(
                        X_train_imputed,
                        columns=feature_names_ordered,
                        index=X_train.index
                    )
                    X_test_imputed = pd.DataFrame(
                        X_test_imputed,
                        columns=feature_names_ordered,
                        index=X_test.index
                    )
                else:
                    X_train_imputed = X_train
                    X_test_imputed = X_test
            
            # Reconstruct full dataframes
            train_df = pd.concat([X_train_imputed, y_train], axis=1)
            test_df = X_test_imputed
        
        # === IMPORTANT: CONVERT NUMERICAL FEATURES FIRST ===
        # Convert numerical features to proper numeric types BEFORE outlier handling
        for col in self.numerical_features:
            if col in train_df.columns:
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            if col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        
        # === OUTLIER HANDLING AND DISTRIBUTION TRANSFORMATION ===
        # Identify numerical features (excluding identifiers)
        valid_numerical = [col for col in self.numerical_features 
                        if col not in ['id'] and 
                        col in train_df.columns]
        
        if valid_numerical:
            module_logger.info("Applying outlier capping and distribution transformations")
            
            # 1. Outlier capping at 5th and 95th percentiles
            outlier_limits = {}
            for col in valid_numerical:
                # Only process if we have numeric data
                if pd.api.types.is_numeric_dtype(train_df[col]):
                    # Compute percentiles from training data only
                    low = train_df[col].quantile(0.05)
                    high = train_df[col].quantile(0.95)
                    outlier_limits[col] = (low, high)
                    
                    # Apply capping to both train and test
                    train_df[col] = train_df[col].clip(lower=low, upper=high)
                    if col in test_df.columns and pd.api.types.is_numeric_dtype(test_df[col]):
                        test_df[col] = test_df[col].clip(lower=low, upper=high)
            
            # 2. Apply log transformation to skewed features
            skewed_features = []
            for col in valid_numerical:
                # Only process if we have numeric data
                if pd.api.types.is_numeric_dtype(train_df[col]):
                    # Skip if constant values/binary
                    if train_df[col].nunique() <= 2:
                        # Identify features with significant skewness
                        skew_val = stats.skew(train_df[col].dropna())
                        if abs(skew_val) > 0.5:  # Threshold for moderate skew
                            skewed_features.append(col)
                            # Apply log1p transformation (handles zeros)
                            train_df[col] = np.log1p(train_df[col])
                            if col in test_df.columns and pd.api.types.is_numeric_dtype(test_df[col]):
                                test_df[col] = np.log1p(test_df[col])
            
            if skewed_features:
                module_logger.info(f"Applied log transformation to skewed features: {skewed_features}")
        
        # === HIGH-CARDINALITY FEATURE BINNING ===
        # Bin cigsPerDay feature to reduce cardinality
        if 'cigsPerDay' in self.categorical_features and 'cigsPerDay' in train_df.columns:
            module_logger.info("Binning high-cardinality feature: cigsPerDay")
            
            # Define bin edges and labels
            bins = [-1, 0, 10, 20, 80]
            labels = ['non_smoker', 'light', 'moderate', 'heavy']
            
            # Apply binning
            train_df['smoking_intensity'] = pd.cut(
                train_df['cigsPerDay'], bins=bins, labels=labels, include_lowest=True
            )
            if 'cigsPerDay' in test_df.columns:
                test_df['smoking_intensity'] = pd.cut(
                    test_df['cigsPerDay'], bins=bins, labels=labels, include_lowest=True
                )
            
            # Update feature metadata
            self.categorical_features.remove('cigsPerDay')
            self.categorical_features.append('smoking_intensity')
            self.feature_names = [f if f != 'cigsPerDay' else 'smoking_intensity' 
                                for f in self.feature_names]
            
            # Remove original column
            train_df = train_df.drop(columns=['cigsPerDay'], errors='ignore')
            test_df = test_df.drop(columns=['cigsPerDay'], errors='ignore')
        
        # Convert categorical features to category type
        for col in self.categorical_features:
            if col in train_df.columns:
                train_df[col] = train_df[col].astype('category')
            if col in test_df.columns:
                test_df[col] = test_df[col].astype('category')
        
        # Update stored data
        self.train_data = train_df
        self.test_data = test_df
        
        # Log results
        module_logger.info(f"Train data shape after imputation: {train_df.shape}")
        module_logger.info(f"Test data shape after imputation: {test_df.shape}")
        
        return train_df, test_df
    
    def detect_outliers(self, method: str = 'iqr', 
                       contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers in numerical features.
        
        Args:
            method: Detection method ('iqr', 'isolation_forest', 'zscore')
            contamination: Expected proportion of outliers (for isolation forest)
            
        Returns:
            DataFrame with outlier indicators
        """
        module_logger.info(f"Detecting outliers using {method} method")
        
        outliers_df = pd.DataFrame(index=self.train_data.index)
        X_numerical = self.train_data[self.numerical_features]
        
        if method == 'iqr':
            # IQR method
            Q1 = X_numerical.quantile(0.25)
            Q3 = X_numerical.quantile(0.75)
            IQR = Q3 - Q1
            
            for col in self.numerical_features:
                lower_bound = Q1[col] - 1.5 * IQR[col]
                upper_bound = Q3[col] + 1.5 * IQR[col]
                outliers_df[f'{col}_outlier'] = (
                    (X_numerical[col] < lower_bound) | 
                    (X_numerical[col] > upper_bound)
                )
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=self.random_state
            )
            outliers = iso_forest.fit_predict(X_numerical)
            outliers_df['isolation_forest_outlier'] = (outliers == -1)
        
        elif method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(X_numerical))
            threshold = 3
            outliers_df['zscore_outlier'] = (z_scores > threshold).any(axis=1)
        
        # Summary statistics
        outlier_counts = outliers_df.sum()
        outlier_pct = (outlier_counts / len(outliers_df)) * 100
        
        module_logger.info(f"Outliers detected: {outlier_counts.sum()} total")
        module_logger.info(f"Outlier percentages:\n{outlier_pct}")
        
        return outliers_df
    
    def create_train_validation_split(self, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train-validation split from training data.
        
        Args:
            stratify: Whether to use stratified splitting
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        X = self.train_data[self.feature_names]
        y = self.train_data[TARGET_COLUMN]
        
        stratify_col = y if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=TEST_SIZE,
            random_state=self.random_state,
            stratify=stratify_col
        )
        
        module_logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
        module_logger.info(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
        module_logger.info(f"Validation target distribution:\n{y_val.value_counts(normalize=True)}")
        
        return X_train, X_val, y_train, y_val
    
    def save_processed_data(self, suffix: str = "") -> None:
        """
        Save processed datasets to disk.
        
        Args:
            suffix: Optional suffix for filenames
        """
        train_filename = f"train_processed{suffix}.csv"
        test_filename = f"test_processed{suffix}.csv"
        
        train_path = PROCESSED_DATA_DIR / train_filename
        test_path = PROCESSED_DATA_DIR / test_filename
        
        self.train_data.to_csv(train_path, index=False)
        self.test_data.to_csv(test_path, index=False)
        
        module_logger.info(f"Processed data saved to {PROCESSED_DATA_DIR}")

# Convenience functions for quick preprocessing
def quick_preprocess(imputation_strategy: str = 'median') -> Tuple[pd.DataFrame, pd.DataFrame, DataPreprocessor]:
    """
    Quick preprocessing pipeline for getting started.
    
    Args:
        imputation_strategy: Strategy for handling missing values
        
    Returns:
        Tuple of (train_df, test_df, preprocessor)
    """
    preprocessor = DataPreprocessor()
    
    # Load data
    train_df, test_df = preprocessor.load_datasets()
    
    # Analyse quality
    quality_report = preprocessor.analyze_data_quality()
    
    # Handle missing values
    train_df, test_df = preprocessor.handle_missing_values(strategy=imputation_strategy)
    
    return train_df, test_df, preprocessor

if __name__ == "__main__":
    # Test the preprocessing pipeline
    module_logger.info("Testing data preprocessing pipeline...")
    
    train_df, test_df, preprocessor = quick_preprocess()
    
    # Detect outliers
    outliers = preprocessor.detect_outliers(method='iqr')
    
    # Create train-validation split
    X_train, X_val, y_train, y_val = preprocessor.create_train_validation_split()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    module_logger.info("Data preprocessing pipeline test completed!")