"""
Model Training module for Heart Disease ML Pipeline
Implements training pipelines for all 9 models with hyperparameter tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Gradient boosting libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Interpretable ML
from interpret.glassbox import ExplainableBoostingClassifier

from src.config import (
    MODEL_CONFIGS, CV_FOLDS, RANDOM_SEED,
    BASELINE_MODELS_DIR, TUNED_MODELS_DIR
)
from src.utils import Timer, logger, save_model, get_memory_usage
from src.feature_engineering import FeatureScaler
from src.class_imbalance import ImbalanceHandler

# Get module logger
module_logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive model training class implementing all 9 algorithms.
    Handles training, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Initialise the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.training_times = {}
        self.memory_usage = {}
        self.best_params = {}
        self.cv_scores = {}
        self.feature_scaler = FeatureScaler()
        
    def get_model(self, model_name: str, **kwargs) -> Any:
        """
        Get model instance by name.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional parameters for the model
            
        Returns:
            Model instance
        """
        if model_name == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state, **kwargs)
        
        elif model_name == 'svm':
            return SVC(random_state=self.random_state, probability=True, **kwargs)
        
        elif model_name == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, **kwargs)
        
        elif model_name == 'extra_trees':
            return ExtraTreesClassifier(random_state=self.random_state, **kwargs)
        
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                **kwargs
            )
        
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(
                random_state=self.random_state,
                verbosity=-1,
                **kwargs
            )
        
        elif model_name == 'catboost':
            return cb.CatBoostClassifier(
                random_state=self.random_state,
                verbose=False,
                **kwargs
            )
        
        elif model_name == 'ebm':
            return ExplainableBoostingClassifier(
                random_state=self.random_state,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_param_grid(self, model_name: str, search_type: str = 'grid') -> Dict[str, List]:
        """
        Get hyperparameter search space for each model.
        
        Args:
            model_name: Name of the model
            search_type: Type of search ('grid' or 'random')
            
        Returns:
            Parameter grid dictionary
        """
        if model_name == 'logistic_regression':
            return {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
        
        elif model_name == 'svm':
            if search_type == 'grid':
                return {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
            else:
                return {
                    'C': np.logspace(-2, 2, 20),
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 20))
                }
        
        elif model_name == 'random_forest':
            if search_type == 'grid':
                return {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:
                return {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['auto', 'sqrt', 'log2']
                }
        
        elif model_name == 'extra_trees':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        
        elif model_name == 'xgboost':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        
        elif model_name == 'lightgbm':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 1.0]
            }
        
        elif model_name == 'catboost':
            return {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.3],
                'l2_leaf_reg': [1, 3, 5]
            }
        
        elif model_name == 'ebm':
            return {
                'max_bins': [256, 512],
                'max_interaction_bins': [32, 64],
                'interactions': [10, 25],
                'outer_bags': [8, 16],
                'inner_bags': [0, 8]
            }
        
        else:
            return {}
    
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           class_weight: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train baseline logistic regression model without tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            class_weight: Class weights for imbalanced data
            
        Returns:
            Dictionary with model and results
        """
        module_logger.info("Training baseline logistic regression model...")
        
        with Timer("Baseline model training") as timer:
            # Scale features for linear model
            X_train_scaled = self.feature_scaler.fit_transform(
                X_train, 
                X_train.select_dtypes(include=[np.number]).columns.tolist(),
                strategy='linear'
            )
            X_val_scaled = self.feature_scaler.transform(
                X_val,
                X_val.select_dtypes(include=[np.number]).columns.tolist(),
                strategy='linear'
            )
            
            # Train baseline model
            baseline_model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight=class_weight
            )
            
            baseline_model.fit(X_train_scaled, y_train)
            
            # Get predictions
            y_pred = baseline_model.predict(X_val_scaled)
            y_proba = baseline_model.predict_proba(X_val_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred, y_proba)
            
            # Store results
            self.models['baseline_logistic'] = baseline_model
            self.training_times['baseline_logistic'] = timer.elapsed
            self.memory_usage['baseline_logistic'] = get_memory_usage()
        
        # Log results
        module_logger.info("Baseline model performance:")
        for metric, value in metrics.items():
            module_logger.info(f"  {metric}: {value:.4f}")
        
        # Save baseline model
        metadata = {
            'model_type': 'logistic_regression',
            'is_baseline': True,
            'metrics': metrics,
            'training_time': timer.elapsed,
            'class_weight': class_weight
        }
        save_model(baseline_model, 'baseline_logistic', BASELINE_MODELS_DIR, metadata)
        
        return {
            'model': baseline_model,
            'metrics': metrics,
            'training_time': timer.elapsed,
            'coefficients': dict(zip(X_train.columns, baseline_model.coef_[0]))
        }
    
    def train_model_with_tuning(self, model_name: str,
                              X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series,
                              param_grid: Optional[Dict] = None,
                              search_type: str = 'grid',
                              cv_folds: int = CV_FOLDS,
                              class_weight: Optional[Dict] = None,
                              cat_features: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Train a model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            param_grid: Custom parameter grid (optional)
            search_type: Type of search ('grid' or 'random')
            cv_folds: Number of cross-validation folds
            class_weight: Class weights for imbalanced data
            cat_features: Indices of categorical features (for CatBoost)
            
        Returns:
            Dictionary with model and results
        """
        module_logger.info(f"Training {model_name} with hyperparameter tuning...")
        
        # Prepare data based on model type
        if model_name in ['svm', 'logistic_regression']:
            # Scale features for linear/kernel models
            X_train_prepared = self.feature_scaler.fit_transform(
                X_train,
                X_train.select_dtypes(include=[np.number]).columns.tolist(),
                strategy='svm' if model_name == 'svm' else 'linear'
            )
            X_val_prepared = self.feature_scaler.transform(
                X_val,
                X_val.select_dtypes(include=[np.number]).columns.tolist(),
                strategy='svm' if model_name == 'svm' else 'linear'
            )
        else:
            # Tree-based models don't need scaling
            X_train_prepared = X_train
            X_val_prepared = X_val
        
        # Get parameter grid
        if param_grid is None:
            param_grid = self.get_param_grid(model_name, search_type)
        
        # Handle class weights
        model_params = {}
        if class_weight and model_name in ['logistic_regression', 'svm', 'random_forest', 'extra_trees']:
            model_params['class_weight'] = class_weight
        elif class_weight and model_name == 'xgboost':
            model_params['scale_pos_weight'] = class_weight[1] / class_weight[0]
        elif class_weight and model_name == 'lightgbm':
            model_params['class_weight'] = class_weight
        elif class_weight and model_name == 'catboost':
            model_params['class_weights'] = class_weight
        
        # Special handling for CatBoost
        if model_name == 'catboost' and cat_features:
            model_params['cat_features'] = cat_features
        
        # Get base model
        base_model = self.get_model(model_name, **model_params)
        
        # Perform hyperparameter search
        with Timer(f"{model_name} hyperparameter tuning") as timer:
            if search_type == 'grid':
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_folds,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
            else:
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=20,
                    cv=cv_folds,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1,
                    random_state=self.random_state
                )
            
            # Fit the search
            search.fit(X_train_prepared, y_train)
            
            # Get best model
            best_model = search.best_estimator_
            self.best_params[model_name] = search.best_params_
            self.cv_scores[model_name] = {
                'mean': search.best_score_,
                'std': search.cv_results_['std_test_score'][search.best_index_]
            }
        
        # Evaluate on validation set
        y_pred = best_model.predict(X_val_prepared)
        y_proba = best_model.predict_proba(X_val_prepared)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_val, y_pred, y_proba)
        
        # Store results
        self.models[model_name] = best_model
        self.training_times[model_name] = timer.elapsed
        self.memory_usage[model_name] = get_memory_usage()
        
        # Log results
        module_logger.info(f"{model_name} best parameters: {self.best_params[model_name]}")
        module_logger.info(f"{model_name} CV score: {self.cv_scores[model_name]['mean']:.4f} "
                         f"(+/- {self.cv_scores[model_name]['std']:.4f})")
        module_logger.info(f"{model_name} validation performance:")
        for metric, value in metrics.items():
            module_logger.info(f"  {metric}: {value:.4f}")
        
        # Save tuned model
        metadata = {
            'model_type': model_name,
            'best_params': self.best_params[model_name],
            'cv_scores': self.cv_scores[model_name],
            'metrics': metrics,
            'training_time': timer.elapsed,
            'class_weight': class_weight
        }
        save_model(best_model, f"{model_name}_tuned", TUNED_MODELS_DIR, metadata)
        
        return {
            'model': best_model,
            'metrics': metrics,
            'best_params': self.best_params[model_name],
            'cv_scores': self.cv_scores[model_name],
            'training_time': timer.elapsed
        }
    
    def calculate_metrics(self, y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
        }
        
        return metrics
    
    def create_voting_ensemble(self, top_models: List[str],
                             X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             voting: str = 'soft') -> Dict[str, Any]:
        """
        Create a voting ensemble from top performing models.
        
        Args:
            top_models: List of top model names
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            voting: Type of voting ('hard' or 'soft')
            
        Returns:
            Dictionary with ensemble model and results
        """
        module_logger.info(f"Creating voting ensemble with models: {top_models}")
        
        # Prepare estimators
        estimators = []
        for model_name in top_models:
            if model_name in self.models:
                # Handle scaling for different models
                if model_name in ['svm', 'logistic_regression']:
                    # Create pipeline with scaling
                    scaler = StandardScaler()
                    model_pipeline = Pipeline([
                        ('scaler', scaler),
                        (model_name, self.models[model_name])
                    ])
                    estimators.append((model_name, model_pipeline))
                else:
                    estimators.append((model_name, self.models[model_name]))
        
        # Create ensemble
        with Timer("Voting ensemble training") as timer:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting=voting,
                n_jobs=-1
            )
            
            # Fit ensemble
            ensemble.fit(X_train, y_train)
            
            # Get predictions
            y_pred = ensemble.predict(X_val)
            y_proba = ensemble.predict_proba(X_val)[:, 1] if voting == 'soft' else None
            
            # Calculate metrics
            if y_proba is not None:
                metrics = self.calculate_metrics(y_val, y_pred, y_proba)
            else:
                # For hard voting, calculate metrics without probability-based ones
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1': f1_score(y_val, y_pred, zero_division=0),
                    'matthews_corrcoef': matthews_corrcoef(y_val, y_pred)
                }
        
        # Store results
        self.models['voting_ensemble'] = ensemble
        self.training_times['voting_ensemble'] = timer.elapsed
        self.memory_usage['voting_ensemble'] = get_memory_usage()
        
        # Log results
        module_logger.info("Voting ensemble performance:")
        for metric, value in metrics.items():
            module_logger.info(f"  {metric}: {value:.4f}")
        
        # Save ensemble
        metadata = {
            'model_type': 'voting_ensemble',
            'base_models': top_models,
            'voting': voting,
            'metrics': metrics,
            'training_time': timer.elapsed
        }
        save_model(ensemble, 'voting_ensemble', TUNED_MODELS_DIR, metadata)
        
        return {
            'model': ensemble,
            'metrics': metrics,
            'base_models': top_models,
            'training_time': timer.elapsed
        }
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.
        
        Returns:
            DataFrame with model summary
        """
        summary_data = []
        
        for model_name in self.models:
            summary_data.append({
                'model': model_name,
                'training_time': self.training_times.get(model_name, 0),
                'memory_usage_mb': self.memory_usage.get(model_name, 0),
                'cv_score_mean': self.cv_scores.get(model_name, {}).get('mean', None),
                'cv_score_std': self.cv_scores.get(model_name, {}).get('std', None),
                'best_params': str(self.best_params.get(model_name, {}))
            })
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Test the model training pipeline
    module_logger.info("Testing model training pipeline...")
    
    # Load preprocessed and engineered data
    from data_preprocessing import quick_preprocess
    from feature_engineering import create_feature_engineering_pipeline
    from sklearn.model_selection import train_test_split
    
    # Preprocess data
    train_df, test_df, preprocessor = quick_preprocess()
    
    # Apply feature engineering
    train_engineered, engineer = create_feature_engineering_pipeline(
        train_df,
        preprocessor.categorical_features,
        preprocessor.numerical_features
    )
    
    # Separate features and target
    X = train_engineered.drop('TenYearCHD', axis=1)
    y = train_engineered['TenYearCHD']
    
    # Create train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Calculate class weights
    imbalance_handler = ImbalanceHandler()
    class_weights = imbalance_handler.calculate_class_weights(y_train)
    
    # Initialise trainer
    trainer = ModelTrainer()
    
    # Train baseline model
    baseline_results = trainer.train_baseline_model(
        X_train, y_train, X_val, y_val, class_weights
    )
    
    module_logger.info("\nModel training pipeline test completed!")