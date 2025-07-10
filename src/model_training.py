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
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

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
from src.utils import Timer, logger, save_model, get_memory_usage, save_figure
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
    
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           class_weight: Optional[Dict] = None,
                           cv_folds: int = CV_FOLDS) -> Dict[str, Any]:
        """
        Train comprehensive baseline logistic regression model with cross-validation and visualizations.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            class_weight: Class weights for imbalanced data
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with model and results
        """
        module_logger.info("Training comprehensive baseline logistic regression model...")
        
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
            
            # Cross-validation evaluation
            module_logger.info("Performing cross-validation evaluation...")
            cv_results = self._perform_baseline_cv(
                baseline_model, X_train_scaled, y_train, cv_folds
            )
            
            # Get predictions on validation set
            y_pred = baseline_model.predict(X_val_scaled)
            y_proba = baseline_model.predict_proba(X_val_scaled)[:, 1]
            
            # Calculate validation metrics
            val_metrics = self.calculate_metrics(y_val, y_pred, y_proba)
            
            # Get feature coefficients
            coefficients = dict(zip(X_train.columns, baseline_model.coef_[0]))
            
            # Generate baseline visualizations
            module_logger.info("Generating baseline model visualizations...")
            viz_paths = self._create_baseline_visualizations(
                baseline_model, X_val_scaled, y_val, y_pred, y_proba, 
                coefficients, X_train.columns
            )
            
            # Perform threshold optimization for imbalanced data
            module_logger.info("Performing threshold optimization...")
            threshold_results = self._optimize_classification_threshold(y_val, y_proba)
            
            # Calculate calibration metrics
            module_logger.info("Assessing model calibration...")
            calibration_results = self._assess_model_calibration(y_val, y_proba)
            
            # Store results
            self.models['baseline_logistic'] = baseline_model
            self.training_times['baseline_logistic'] = timer.elapsed
            self.memory_usage['baseline_logistic'] = get_memory_usage()
            
            # Record memory usage
            memory_mb = self.memory_usage['baseline_logistic']
            module_logger.info(f"Memory usage: {memory_mb:.2f} MB")
        
        # Log results with formatted summary
        from src.utils import print_model_summary
        print_model_summary("Baseline Logistic Regression", val_metrics, timer.elapsed, coefficients)
        
        # Log cross-validation results
        module_logger.info("\nCross-validation results:")
        for metric, scores in cv_results.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            module_logger.info(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        # Prepare comprehensive metadata
        metadata = {
            'model_type': 'logistic_regression',
            'is_baseline': True,
            'validation_metrics': val_metrics,
            'cv_metrics': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'scores': [float(x) for x in v]} 
                          for k, v in cv_results.items()},
            'threshold_optimization': threshold_results,
            'calibration_assessment': calibration_results,
            'training_time': float(timer.elapsed),
            'memory_usage_mb': float(memory_mb),
            'class_weight': class_weight,
            'feature_count': len(X_train.columns),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'cv_folds': cv_folds,
            'class_distribution': {
                'train': y_train.value_counts().to_dict(),
                'validation': y_val.value_counts().to_dict()
            },
            'top_features': dict(sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:10]),
            'visualization_paths': [str(path) for path in viz_paths] + [threshold_results['visualization_path'], calibration_results['visualization_path']]
        }
        
        # Save baseline model
        try:
            model_path = save_model(baseline_model, 'baseline_logistic', BASELINE_MODELS_DIR, metadata)
            module_logger.info(f"Baseline model successfully saved to {model_path}")
        except Exception as e:
            module_logger.error(f"Failed to save baseline model: {str(e)}")
            raise
        
        return {
            'model': baseline_model,
            'validation_metrics': val_metrics,
            'cv_metrics': cv_results,
            'threshold_optimization': threshold_results,
            'calibration_assessment': calibration_results,
            'training_time': float(timer.elapsed),
            'memory_usage_mb': float(memory_mb),
            'coefficients': coefficients,
            'model_path': model_path,
            'visualization_paths': viz_paths + [threshold_results['visualization_path'], calibration_results['visualization_path']]
        }
    
    def _perform_baseline_cv(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                           cv_folds: int) -> Dict[str, List[float]]:
        """
        Perform cross-validation evaluation for baseline model.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with CV scores for each metric
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'average_precision': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model on fold
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Get predictions
            y_fold_pred = fold_model.predict(X_fold_val)
            y_fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]
            
            # Calculate metrics
            cv_results['accuracy'].append(accuracy_score(y_fold_val, y_fold_pred))
            cv_results['precision'].append(precision_score(y_fold_val, y_fold_pred, zero_division=0))
            cv_results['recall'].append(recall_score(y_fold_val, y_fold_pred, zero_division=0))
            cv_results['f1'].append(f1_score(y_fold_val, y_fold_pred, zero_division=0))
            cv_results['roc_auc'].append(roc_auc_score(y_fold_val, y_fold_proba))
            cv_results['average_precision'].append(average_precision_score(y_fold_val, y_fold_proba))
        
        return cv_results
    
    def _create_baseline_visualizations(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series,
                                      y_pred: np.ndarray, y_proba: np.ndarray,
                                      coefficients: Dict[str, float],
                                      feature_names: List[str]) -> List[Path]:
        """
        Create visualizations for baseline model.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target
            y_pred: Predictions
            y_proba: Prediction probabilities
            coefficients: Feature coefficients
            feature_names: List of feature names
            
        Returns:
            List of paths to saved visualizations
        """
        viz_paths = []
        
        # 1. ROC Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = roc_auc_score(y_val, y_proba)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Baseline Model - ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        viz_paths.append(save_figure(fig, 'baseline_roc_curve', 'baseline'))
        plt.close(fig)
        
        # 2. Precision-Recall Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_val, y_proba)
        avg_precision = average_precision_score(y_val, y_proba)
        
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
        ax.axhline(y=np.mean(y_val), color='k', linestyle='--', alpha=0.7, 
                  label=f'Random Classifier (AP = {np.mean(y_val):.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Baseline Model - Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        viz_paths.append(save_figure(fig, 'baseline_pr_curve', 'baseline'))
        plt.close(fig)
        
        # 3. Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_val, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No CHD', 'CHD'], yticklabels=['No CHD', 'CHD'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Baseline Model - Confusion Matrix')
        
        viz_paths.append(save_figure(fig, 'baseline_confusion_matrix', 'baseline'))
        plt.close(fig)
        
        # 4. Feature Coefficients (Top 15)
        fig, ax = plt.subplots(figsize=(10, 8))
        coef_df = pd.DataFrame({
            'feature': list(coefficients.keys()),
            'coefficient': list(coefficients.values())
        })
        coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
        coef_df = coef_df.nlargest(15, 'abs_coefficient')
        
        colors = ['red' if x < 0 else 'blue' for x in coef_df['coefficient']]
        bars = ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.7)
        
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Baseline Model - Top 15 Feature Coefficients')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, coef in zip(bars, coef_df['coefficient']):
            ax.text(coef + (0.02 if coef >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                   f'{coef:.3f}', ha='left' if coef >= 0 else 'right', va='center', fontsize=9)
        
        plt.tight_layout()
        viz_paths.append(save_figure(fig, 'baseline_feature_coefficients', 'baseline'))
        plt.close(fig)
        
        # 5. Model Performance Summary
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Metrics bar chart
        metrics = self.calculate_metrics(y_val, y_pred, y_proba)
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars1 = ax1.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Score')
        ax1.set_title('Baseline Model Performance Metrics')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars1, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Prediction distribution
        ax2.hist([y_proba[y_val == 0], y_proba[y_val == 1]], 
                bins=20, alpha=0.7, label=['No CHD', 'CHD'], color=['blue', 'red'])
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prediction Probability Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Class distribution
        class_counts = y_val.value_counts()
        ax3.pie(class_counts.values, labels=['No CHD', 'CHD'], autopct='%1.1f%%',
               colors=['skyblue', 'lightcoral'])
        ax3.set_title('Validation Set Class Distribution')
        
        # Training summary text
        ax4.axis('off')
        summary_text = f"""
        Baseline Model Summary
        ────────────────────────
        Algorithm: Logistic Regression
        Training Samples: {len(y_val) * 4}  # Approximate
        Validation Samples: {len(y_val)}
        Features: {len(feature_names)}
        
        Best Metrics:
        • ROC-AUC: {metrics['roc_auc']:.3f}
        • F1-Score: {metrics['f1']:.3f}
        • Precision: {metrics['precision']:.3f}
        • Recall: {metrics['recall']:.3f}
        
        Top Predictive Features:
        1. {coef_df.iloc[0]['feature']}
        2. {coef_df.iloc[1]['feature']}
        3. {coef_df.iloc[2]['feature']}
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        viz_paths.append(save_figure(fig, 'baseline_performance_summary', 'baseline'))
        plt.close(fig)
        
        module_logger.info(f"Generated {len(viz_paths)} baseline visualizations")
        return viz_paths
    
    def _optimize_classification_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Find optimal classification thresholds for different metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with optimal thresholds and corresponding metrics
        """
        from sklearn.metrics import precision_recall_curve
        
        # Test different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        
        results = {
            'f1_optimization': {'threshold': 0.5, 'score': 0.0},
            'balanced_accuracy_optimization': {'threshold': 0.5, 'score': 0.0},
            'youden_j_optimization': {'threshold': 0.5, 'score': 0.0}
        }
        
        best_f1 = 0
        best_balanced_acc = 0
        best_youden_j = 0
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            
            # Balanced accuracy
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
            
            # Youden's J statistic (sensitivity + specificity - 1)
            youden_j = sensitivity + specificity - 1
            
            # Update best thresholds
            if f1 > best_f1:
                best_f1 = f1
                results['f1_optimization'] = {'threshold': threshold, 'score': f1}
            
            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                results['balanced_accuracy_optimization'] = {
                    'threshold': threshold, 'score': balanced_acc,
                    'sensitivity': sensitivity, 'specificity': specificity
                }
            
            if youden_j > best_youden_j:
                best_youden_j = youden_j
                results['youden_j_optimization'] = {
                    'threshold': threshold, 'score': youden_j,
                    'sensitivity': sensitivity, 'specificity': specificity
                }
        
        # Create threshold optimization visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot threshold vs F1
        f1_scores = []
        balanced_accs = []
        youden_js = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_accs.append((sensitivity + specificity) / 2)
            youden_js.append(sensitivity + specificity - 1)
        
        # F1 Score vs Threshold
        ax1.plot(thresholds, f1_scores, 'b-', linewidth=2)
        ax1.axvline(results['f1_optimization']['threshold'], color='r', linestyle='--',
                   label=f"Optimal: {results['f1_optimization']['threshold']:.3f}")
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score vs Classification Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Balanced Accuracy vs Threshold
        ax2.plot(thresholds, balanced_accs, 'g-', linewidth=2)
        ax2.axvline(results['balanced_accuracy_optimization']['threshold'], color='r', linestyle='--',
                   label=f"Optimal: {results['balanced_accuracy_optimization']['threshold']:.3f}")
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Balanced Accuracy')
        ax2.set_title('Balanced Accuracy vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Youden's J vs Threshold
        ax3.plot(thresholds, youden_js, 'm-', linewidth=2)
        ax3.axvline(results['youden_j_optimization']['threshold'], color='r', linestyle='--',
                   label=f"Optimal: {results['youden_j_optimization']['threshold']:.3f}")
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel("Youden's J Statistic")
        ax3.set_title("Youden's J vs Threshold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Summary table
        ax4.axis('off')
        summary_text = f"""
        Threshold Optimization Results
        ══════════════════════════════
        
        F1 Score Optimization:
        • Optimal Threshold: {results['f1_optimization']['threshold']:.3f}
        • Best F1 Score: {results['f1_optimization']['score']:.3f}
        
        Balanced Accuracy Optimization:
        • Optimal Threshold: {results['balanced_accuracy_optimization']['threshold']:.3f}
        • Best Balanced Accuracy: {results['balanced_accuracy_optimization']['score']:.3f}
        • Sensitivity: {results['balanced_accuracy_optimization']['sensitivity']:.3f}
        • Specificity: {results['balanced_accuracy_optimization']['specificity']:.3f}
        
        Youden's J Optimization:
        • Optimal Threshold: {results['youden_j_optimization']['threshold']:.3f}
        • Best Youden's J: {results['youden_j_optimization']['score']:.3f}
        • Sensitivity: {results['youden_j_optimization']['sensitivity']:.3f}
        • Specificity: {results['youden_j_optimization']['specificity']:.3f}
        
        Recommendation for Medical Applications:
        Use Youden's J threshold for balanced sensitivity/specificity
        """
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        viz_path = save_figure(fig, 'baseline_threshold_optimization', 'baseline')
        plt.close(fig)
        
        results['visualization_path'] = str(viz_path)
        
        module_logger.info("Threshold optimization completed:")
        module_logger.info(f"  Best F1 threshold: {results['f1_optimization']['threshold']:.3f} "
                         f"(F1: {results['f1_optimization']['score']:.3f})")
        module_logger.info(f"  Best Balanced Accuracy threshold: {results['balanced_accuracy_optimization']['threshold']:.3f} "
                         f"(BA: {results['balanced_accuracy_optimization']['score']:.3f})")
        
        return results
    
    def _assess_model_calibration(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Assess model calibration using reliability diagrams and Brier score.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with calibration metrics
        """
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        # Calculate Brier score (lower is better)
        brier_score = brier_score_loss(y_true, y_proba)
        
        # Calculate calibration metrics
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        max_calibration_error = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Create calibration plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2, label="Model")
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title("Reliability Diagram (Calibration)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add calibration metrics text
        ax1.text(0.05, 0.95, f"Brier Score: {brier_score:.3f}\nMean Cal. Error: {calibration_error:.3f}\nMax Cal. Error: {max_calibration_error:.3f}",
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Histogram of predicted probabilities
        ax2.hist(y_proba[y_true == 0], bins=20, alpha=0.7, label='No CHD', color='blue', density=True)
        ax2.hist(y_proba[y_true == 1], bins=20, alpha=0.7, label='CHD', color='red', density=True)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = save_figure(fig, 'baseline_calibration_analysis', 'baseline')
        plt.close(fig)
        
        results = {
            'brier_score': float(brier_score),
            'mean_calibration_error': float(calibration_error),
            'max_calibration_error': float(max_calibration_error),
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'visualization_path': str(viz_path)
        }
        
        # Calibration assessment
        if calibration_error < 0.05:
            calibration_quality = "Excellent"
        elif calibration_error < 0.1:
            calibration_quality = "Good"
        elif calibration_error < 0.15:
            calibration_quality = "Fair"
        else:
            calibration_quality = "Poor"
        
        results['calibration_quality'] = calibration_quality
        
        module_logger.info("Model calibration assessment completed:")
        module_logger.info(f"  Brier Score: {brier_score:.3f}")
        module_logger.info(f"  Mean Calibration Error: {calibration_error:.3f}")
        module_logger.info(f"  Calibration Quality: {calibration_quality}")
        
        return results

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