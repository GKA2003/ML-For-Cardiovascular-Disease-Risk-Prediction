"""
Feature Engineering module for Heart Disease ML Pipeline
Creates new features and transforms existing ones to improve model performance
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.feature_names = None
        self.new_features = []
        self.scalers = {}
        self.encoders = {}
        
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
            
            # 4. Handle categorical encoding
            X_engineered = self._encode_categorical_features(X_engineered, fit=True)
            
            # Store final feature names
            self.feature_names = X_engineered.columns.tolist()
            
            module_logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
            module_logger.info(f"New features created: {len(self.new_features)}")
            
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
        
        # Smoking impact
        if 'is_smoking' in X.columns and 'cigsPerDay' in X.columns:
            # Pack years approximation (assuming 20 years of smoking on average)
            X['pack_years_approx'] = X.apply(
                lambda row: row['cigsPerDay'] * 20 / 20 if row['is_smoking'] == 'YES' else 0,
                axis=1
            )
            self.new_features.append('pack_years_approx')
        
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
            X['age_group'] = pd.cut(X['age'], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
            X['age_group'] = X['age_group'].cat.codes  # Convert to numeric
            self.new_features.append('age_group')
        
        # BMI categories (WHO classification)
        if 'BMI' in X.columns:
            bmi_bins = [0, 18.5, 25, 30, 35, 40, 100]
            bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II', 'Obese_III']
            X['bmi_category'] = pd.cut(X['BMI'], bins=bmi_bins, labels=bmi_labels, include_lowest=True)
            X['bmi_category'] = X['bmi_category'].cat.codes
            self.new_features.append('bmi_category')
        
        # Cholesterol levels (medical guidelines)
        if 'totChol' in X.columns:
            chol_bins = [0, 200, 240, 1000]
            chol_labels = ['Desirable', 'Borderline', 'High']
            X['chol_category'] = pd.cut(X['totChol'], bins=chol_bins, labels=chol_labels, include_lowest=True)
            X['chol_category'] = X['chol_category'].cat.codes
            self.new_features.append('chol_category')
        
        # Glucose levels
        if 'glucose' in X.columns:
            glucose_bins = [0, 70, 100, 126, 1000]
            glucose_labels = ['Low', 'Normal', 'Prediabetic', 'Diabetic']
            X['glucose_category'] = pd.cut(X['glucose'], bins=glucose_bins, labels=glucose_labels, include_lowest=True)
            X['glucose_category'] = X['glucose_category'].cat.codes
            self.new_features.append('glucose_category')
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        module_logger.info("Encoding categorical features...")
        
        # Handle sex encoding (binary)
        if 'sex' in X.columns:
            if fit:
                self.encoders['sex'] = LabelEncoder()
                X['sex_encoded'] = self.encoders['sex'].fit_transform(X['sex'])
            else:
                X['sex_encoded'] = self.encoders['sex'].transform(X['sex'])
            X = X.drop('sex', axis=1)
            
        # Handle is_smoking encoding (binary)
        if 'is_smoking' in X.columns:
            if fit:
                self.encoders['is_smoking'] = LabelEncoder()
                X['is_smoking_encoded'] = self.encoders['is_smoking'].fit_transform(X['is_smoking'])
            else:
                X['is_smoking_encoded'] = self.encoders['is_smoking'].transform(X['is_smoking'])
            X = X.drop('is_smoking', axis=1)
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names after engineering."""
        return self.feature_names
    
    def get_new_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of newly created features."""
        descriptions = {
            'pulse_pressure': 'Systolic BP - Diastolic BP',
            'mean_arterial_pressure': 'Diastolic BP + (Pulse Pressure / 3)',
            'metabolic_risk': 'Combined risk from BMI > 30 and glucose > 100',
            'age_chol_risk': 'Total cholesterol / age',
            'pack_years_approx': 'Approximate pack-years for smokers',
            'hypertension_stage': 'Hypertension severity (0-4 scale)',
            'hr_category': 'Heart rate category (0=brady, 1=normal, 2=tachy)',
            'age_smoking_interaction': 'Age × smoking status',
            'bmi_diabetes_interaction': 'BMI × diabetes status',
            'bp_meds_effectiveness': 'Systolic BP × (1 - BP medication)',
            'chol_age_interaction': 'Total cholesterol × age / 100',
            'risk_factor_count': 'Count of major risk factors',
            'age_group': 'Age group category',
            'bmi_category': 'BMI category (WHO classification)',
            'chol_category': 'Cholesterol level category',
            'glucose_category': 'Glucose level category'
        }
        
        return {feat: descriptions.get(feat, 'Unknown') for feat in self.new_features}


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
        
        # Fit and transform numerical features
        X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])
        
        module_logger.info(f"Applied {strategy} scaling to {len(numerical_features)} features")
        
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
        
        # Transform numerical features
        X_scaled[numerical_features] = scaler.transform(X[numerical_features])
        
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
    
    # Save engineered data
    train_engineered.to_csv(PROCESSED_DATA_DIR / "train_engineered.csv", index=False)
    test_engineered.to_csv(PROCESSED_DATA_DIR / "test_engineered.csv", index=False)
    
    module_logger.info("\nFeature engineering completed!")