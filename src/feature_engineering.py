"""
Feature Engineering module for Heart Disease ML Pipeline
Creates new features and transforms existing ones to improve model performance
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    AGE_BINS, AGE_LABELS, RANDOM_SEED,
    PROCESSED_DATA_DIR
)
from src.utils import Timer, logger, validate_dataframe

# Get module logger
module_logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering class for cardiovascular dataset.
    Creates domain-specific features based on medical knowledge.
    """
    
    def __init__(self, categorical_features: List[str], 
                 numerical_features: List[str]):
        """
        Initialise the feature engineer.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
        """
        self.categorical_features = categorical_features.copy()
        self.numerical_features = numerical_features.copy()
        self.feature_names = None
        self.new_features = []
        self.scalers = {}
        self.encoders = {}
        self.categorical_mappings = {}  # Store mappings for consistency
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform features.
        
        Args:
            X: Input features DataFrame
            y: Target variable (optional)
            
        Returns:
            Transformed DataFrame with engineered features
        """
        module_logger.info("Starting feature engineering...")
        
        with Timer("Feature engineering"):
            # Create a copy to avoid modifying original
            X_engineered = X.copy()
            
            # 1. Create domain-specific features
            X_engineered = self._create_medical_features(X_engineered)
            
            # 2. Create interaction features
            X_engineered = self._create_interaction_features(X_engineered)
            
            # 3. Create binned features
            X_engineered = self._create_binned_features(X_engineered)
            
            # 4. Handle categorical encoding (fit encoders)
            X_engineered = self._encode_categorical_features(X_engineered, fit=True)
            
            # 5. Update feature lists
            self._update_feature_lists(X_engineered)
            
            # Store final feature names
            self.feature_names = X_engineered.columns.tolist()
            
            module_logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
            module_logger.info(f"New features created: {len(self.new_features)}")
            module_logger.info(f"Final numerical features: {len(self.numerical_features)}")
            module_logger.info(f"Final categorical features: {len(self.categorical_features)}")
            
        return X_engineered
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted parameters.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Transformed DataFrame
        """
        X_engineered = X.copy()
        
        # Apply same transformations
        X_engineered = self._create_medical_features(X_engineered)
        X_engineered = self._create_interaction_features(X_engineered)
        X_engineered = self._create_binned_features(X_engineered)
        X_engineered = self._encode_categorical_features(X_engineered, fit=False)
        
        # Ensure same columns as training
        if self.feature_names:
            # Add any missing columns
            for col in self.feature_names:
                if col not in X_engineered.columns:
                    X_engineered[col] = 0
            
            # Select and order columns
            X_engineered = X_engineered[self.feature_names]
        
        return X_engineered
    
    def _create_medical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create medically relevant features."""
        module_logger.info("Creating medical features...")
        
        # Pulse Pressure (systolic - diastolic BP)
        if 'sysBP' in X.columns and 'diaBP' in X.columns:
            X['pulse_pressure'] = X['sysBP'] - X['diaBP']
            self.new_features.append('pulse_pressure')
            
            # Mean Arterial Pressure
            X['mean_arterial_pressure'] = X['diaBP'] + (X['pulse_pressure'] / 3)
            self.new_features.append('mean_arterial_pressure')
        
        # Metabolic indicators
        if 'BMI' in X.columns and 'glucose' in X.columns:
            # Combined metabolic risk
            X['metabolic_risk'] = (X['BMI'] > 30).astype(int) + (X['glucose'] > 100).astype(int)
            self.new_features.append('metabolic_risk')
        
        # Cardiovascular risk indicators
        if 'totChol' in X.columns and 'age' in X.columns:
            # Age-adjusted cholesterol risk
            X['age_chol_risk'] = X['totChol'] / X['age']
            self.new_features.append('age_chol_risk')
        
        # Hypertension severity
        if 'sysBP' in X.columns and 'prevalentHyp' in X.columns:
            def hypertension_stage(row):
                if row['prevalentHyp'] == 0:
                    return 0
                elif row['sysBP'] < 130:
                    return 1  # Controlled
                elif row['sysBP'] < 140:
                    return 2  # Stage 1
                elif row['sysBP'] < 180:
                    return 3  # Stage 2
                else:
                    return 4  # Crisis
            
            X['hypertension_stage'] = X.apply(hypertension_stage, axis=1)
            self.new_features.append('hypertension_stage')
        
        # Heart rate categories
        if 'heartRate' in X.columns:
            def hr_category(hr):
                if hr < 60:
                    return 0  # Bradycardia
                elif hr <= 100:
                    return 1  # Normal
                else:
                    return 2  # Tachycardia
            
            X['hr_category'] = X['heartRate'].apply(hr_category)
            self.new_features.append('hr_category')
        
        return X
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        module_logger.info("Creating interaction features...")
        
        # Age and smoking interaction
        if 'age' in X.columns and 'is_smoking' in X.columns:
            X['age_smoking_interaction'] = X['age'] * (X['is_smoking'] == 'YES').astype(int)
            self.new_features.append('age_smoking_interaction')
        
        # BMI and diabetes interaction
        if 'BMI' in X.columns and 'diabetes' in X.columns:
            X['bmi_diabetes_interaction'] = X['BMI'] * X['diabetes']
            self.new_features.append('bmi_diabetes_interaction')
        
        # Blood pressure and medication interaction
        if 'sysBP' in X.columns and 'BPMeds' in X.columns:
            X['bp_meds_effectiveness'] = X['sysBP'] * (1 - X['BPMeds'])
            self.new_features.append('bp_meds_effectiveness')
        
        # Cholesterol and age interaction
        if 'totChol' in X.columns and 'age' in X.columns:
            X['chol_age_interaction'] = X['totChol'] * X['age'] / 100
            self.new_features.append('chol_age_interaction')
        
        # Multiple risk factors count
        risk_factors = []
        if 'is_smoking' in X.columns:
            risk_factors.append((X['is_smoking'] == 'YES').astype(int))
        if 'diabetes' in X.columns:
            risk_factors.append(X['diabetes'])
        if 'prevalentHyp' in X.columns:
            risk_factors.append(X['prevalentHyp'])
        if 'BMI' in X.columns:
            risk_factors.append((X['BMI'] > 30).astype(int))
        
        if risk_factors:
            X['risk_factor_count'] = sum(risk_factors)
            self.new_features.append('risk_factor_count')
        
        return X
    
    def _create_binned_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create binned versions of continuous features."""
        module_logger.info("Creating binned features...")
        
        # Age bins
        if 'age' in X.columns:
            X['age_group'] = pd.cut(X['age'], bins=AGE_BINS, labels=range(len(AGE_LABELS)), include_lowest=True)
            X['age_group'] = X['age_group'].astype(int)
            self.new_features.append('age_group')
        
        # BMI categories (WHO classification)
        if 'BMI' in X.columns:
            bmi_bins = [0, 18.5, 25, 30, 35, 40, 100]
            X['bmi_category'] = pd.cut(X['BMI'], bins=bmi_bins, labels=range(6), include_lowest=True)
            X['bmi_category'] = X['bmi_category'].astype(int)
            self.new_features.append('bmi_category')
        
        # Cholesterol levels (medical guidelines)
        if 'totChol' in X.columns:
            chol_bins = [0, 200, 240, 1000]
            X['chol_category'] = pd.cut(X['totChol'], bins=chol_bins, labels=range(3), include_lowest=True)
            X['chol_category'] = X['chol_category'].astype(int)
            self.new_features.append('chol_category')
        
        # Glucose levels
        if 'glucose' in X.columns:
            glucose_bins = [0, 70, 100, 126, 1000]
            X['glucose_category'] = pd.cut(X['glucose'], bins=glucose_bins, labels=range(4), include_lowest=True)
            X['glucose_category'] = X['glucose_category'].astype(int)
            self.new_features.append('glucose_category')
        
        # Handle cigsPerDay binning (if it exists and is categorical)
        if 'cigsPerDay' in X.columns:
            # Check if it's being treated as categorical
            if 'cigsPerDay' in self.categorical_features:
                module_logger.info("Binning high-cardinality feature: cigsPerDay")
                
                # Define bin edges and labels
                bins = [-1, 0, 10, 20, 80]
                
                # Apply binning and convert to numeric codes
                X['smoking_intensity'] = pd.cut(X['cigsPerDay'], bins=bins, labels=range(4), include_lowest=True)
                X['smoking_intensity'] = X['smoking_intensity'].astype(int)
                self.new_features.append('smoking_intensity')
                
                # Remove original column
                X = X.drop(columns=['cigsPerDay'], errors='ignore')
                
                # Update feature lists
                if 'cigsPerDay' in self.categorical_features:
                    self.categorical_features.remove('cigsPerDay')
                    self.categorical_features.append('smoking_intensity')
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features consistently."""
        module_logger.info("Encoding categorical features...")
        
        # Get all categorical columns that exist in the dataframe
        categorical_cols_to_encode = [col for col in self.categorical_features if col in X.columns]
        
        # Handle each categorical feature
        for col in categorical_cols_to_encode:
            if col in ['sex', 'is_smoking']:
                # Binary categorical features - use label encoding
                if fit:
                    self.encoders[col] = LabelEncoder()
                    X[f'{col}_encoded'] = self.encoders[col].fit_transform(X[col])
                    # Store mapping for reference
                    self.categorical_mappings[col] = dict(zip(
                        self.encoders[col].classes_, 
                        range(len(self.encoders[col].classes_))
                    ))
                else:
                    # Handle unseen categories during transform
                    try:
                        X[f'{col}_encoded'] = self.encoders[col].transform(X[col])
                    except ValueError as e:
                        module_logger.warning(f"Unseen category in {col}: {e}")
                        # Use most frequent class for unseen categories
                        X[f'{col}_encoded'] = 0
                
                # Remove original column
                X = X.drop(columns=[col], errors='ignore')
                
            elif col in X.columns:
                # Handle other categorical features
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    # For string categorical features, use ordinal encoding
                    if fit:
                        self.encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        X[[f'{col}_encoded']] = self.encoders[col].fit_transform(X[[col]])
                        # Store categories for reference
                        self.categorical_mappings[col] = dict(zip(
                            self.encoders[col].categories_[0], 
                            range(len(self.encoders[col].categories_[0]))
                        ))
                    else:
                        X[[f'{col}_encoded']] = self.encoders[col].transform(X[[col]])
                    
                    # Remove original column
                    X = X.drop(columns=[col], errors='ignore')
                else:
                    # Already numeric, keep as is
                    pass
        
        return X
    
    def _update_feature_lists(self, X: pd.DataFrame) -> None:
        """Update categorical and numerical feature lists after engineering."""
        # Get all columns
        all_columns = X.columns.tolist()
        
        # Update numerical features (include new numeric features)
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Update categorical features (encoded features are now numeric)
        categorical_columns = []
        for col in all_columns:
            if col.endswith('_encoded'):
                categorical_columns.append(col)
            elif col in self.categorical_features and col in X.columns:
                # Check if it's still categorical
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    categorical_columns.append(col)
        
        # Update lists
        self.numerical_features = [col for col in numeric_columns if col not in categorical_columns]
        self.categorical_features = categorical_columns
        
        module_logger.info(f"Updated feature lists after engineering:")
        module_logger.info(f"  Numerical: {len(self.numerical_features)} features")
        module_logger.info(f"  Categorical: {len(self.categorical_features)} features")
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names after engineering."""
        return self.feature_names
    
    def get_categorical_indices(self) -> List[int]:
        """Get indices of categorical features for algorithms that need them."""
        if not self.feature_names:
            return []
        
        categorical_indices = []
        for i, col in enumerate(self.feature_names):
            if col in self.categorical_features:
                categorical_indices.append(i)
        
        return categorical_indices
    
    def get_new_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of newly created features."""
        descriptions = {
            'pulse_pressure': 'Systolic BP - Diastolic BP',
            'mean_arterial_pressure': 'Diastolic BP + (Pulse Pressure / 3)',
            'metabolic_risk': 'Combined risk from BMI > 30 and glucose > 100',
            'age_chol_risk': 'Total cholesterol / age',
            'hypertension_stage': 'Hypertension severity (0-4 scale)',
            'hr_category': 'Heart rate category (0=brady, 1=normal, 2=tachy)',
            'age_smoking_interaction': 'Age × smoking status',
            'bmi_diabetes_interaction': 'BMI × diabetes status',
            'bp_meds_effectiveness': 'Systolic BP × (1 - BP medication)',
            'chol_age_interaction': 'Total cholesterol × age / 100',
            'risk_factor_count': 'Count of major risk factors',
            'age_group': 'Age group category (0-3)',
            'bmi_category': 'BMI category (0-5, WHO classification)',
            'chol_category': 'Cholesterol level category (0-2)',
            'glucose_category': 'Glucose level category (0-3)',
            'smoking_intensity': 'Smoking intensity category (0-3)'
        }
        
        return {feat: descriptions.get(feat, 'Unknown') for feat in self.new_features}
    
    def get_encoding_info(self) -> Dict[str, Dict]:
        """Get information about categorical encodings."""
        return {
            'mappings': self.categorical_mappings,
            'encoders': list(self.encoders.keys())
        }


class FeatureScaler:
    """
    Handles feature scaling with different strategies for different model types.
    """
    
    def __init__(self):
        """Initialise the feature scaler."""
        self.scalers = {}
        self.scaling_strategies = {
            'tree_based': None,  # No scaling needed
            'linear': StandardScaler(),
            'svm': StandardScaler(),
            'neural': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
    def fit_transform(self, X: pd.DataFrame, 
                     numerical_features: List[str],
                     strategy: str = 'linear') -> pd.DataFrame:
        """
        Fit and transform features with specified scaling strategy.
        
        Args:
            X: Input features
            numerical_features: List of numerical features to scale
            strategy: Scaling strategy to use
            
        Returns:
            Scaled DataFrame
        """
        if strategy == 'tree_based' or not numerical_features:
            return X
        
        X_scaled = X.copy()
        
        # Get scaler
        scaler = self.scaling_strategies.get(strategy, StandardScaler())
        self.scalers[strategy] = scaler
        
        # Filter numerical features that exist in the dataframe
        features_to_scale = [col for col in numerical_features if col in X.columns]
        
        if features_to_scale:
            # Fit and transform numerical features
            X_scaled[features_to_scale] = scaler.fit_transform(X[features_to_scale])
            module_logger.info(f"Applied {strategy} scaling to {len(features_to_scale)} features")
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame, 
                 numerical_features: List[str],
                 strategy: str = 'linear') -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Input features
            numerical_features: List of numerical features to scale
            strategy: Scaling strategy to use
            
        Returns:
            Scaled DataFrame
        """
        if strategy == 'tree_based' or not numerical_features:
            return X
        
        if strategy not in self.scalers:
            raise ValueError(f"Scaler for strategy '{strategy}' not fitted yet")
        
        X_scaled = X.copy()
        scaler = self.scalers[strategy]
        
        # Filter numerical features that exist in the dataframe
        features_to_scale = [col for col in numerical_features if col in X.columns]
        
        if features_to_scale:
            # Transform numerical features
            X_scaled[features_to_scale] = scaler.transform(X[features_to_scale])
        
        return X_scaled


def create_feature_engineering_pipeline(train_df: pd.DataFrame,
                                      categorical_features: List[str],
                                      numerical_features: List[str]) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Complete feature engineering pipeline.
    
    Args:
        train_df: Training DataFrame
        categorical_features: List of categorical features
        numerical_features: List of numerical features
        
    Returns:
        Tuple of (engineered_df, feature_engineer)
    """
    # Remove target column if present
    target_col = 'TenYearCHD'
    if target_col in train_df.columns:
        X = train_df.drop(target_col, axis=1)
        y = train_df[target_col]
    else:
        X = train_df
        y = None
    
    # Create feature engineer
    engineer = FeatureEngineer(categorical_features, numerical_features)
    
    # Fit and transform
    X_engineered = engineer.fit_transform(X, y)
    
    # Add target back if it was present
    if y is not None:
        X_engineered[target_col] = y
    
    # Log new features
    module_logger.info("\nNew features created:")
    for feat, desc in engineer.get_new_feature_descriptions().items():
        module_logger.info(f"  {feat}: {desc}")
    
    # Log encoding information
    encoding_info = engineer.get_encoding_info()
    module_logger.info(f"\nCategorical encodings applied:")
    for col, mapping in encoding_info['mappings'].items():
        module_logger.info(f"  {col}: {mapping}")
    
    return X_engineered, engineer


if __name__ == "__main__":
    # Test the feature engineering pipeline
    module_logger.info("Testing feature engineering pipeline...")
    
    # Load processed data
    from data_preprocessing import quick_preprocess
    train_df, test_df, preprocessor = quick_preprocess()
    
    # Apply feature engineering
    train_engineered, engineer = create_feature_engineering_pipeline(
        train_df,
        preprocessor.categorical_features,
        preprocessor.numerical_features
    )
    
    # Transform test data
    test_engineered = engineer.transform(test_df)
    
    module_logger.info(f"\nOriginal features: {len(preprocessor.feature_names)}")
    module_logger.info(f"Engineered features: {len(engineer.feature_names)}")
    module_logger.info(f"New features added: {len(engineer.new_features)}")
    
    # Check data types
    module_logger.info("\nFinal data types:")
    for col in train_engineered.columns:
        module_logger.info(f"  {col}: {train_engineered[col].dtype}")
    
    # Save engineered data
    train_engineered.to_csv(PROCESSED_DATA_DIR / "train_engineered.csv", index=False)
    test_engineered.to_csv(PROCESSED_DATA_DIR / "test_engineered.csv", index=False)
    
    module_logger.info("\nFeature engineering completed!")