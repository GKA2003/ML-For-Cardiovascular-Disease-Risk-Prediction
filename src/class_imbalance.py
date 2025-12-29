"""
Class Imbalance Handling module for Heart Disease ML Pipeline
Implements various strategies to handle imbalanced target distribution
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any
import logging
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

from src.config import (
    CV_FOLDS, RANDOM_SEED, TARGET_COLUMN,
    TABLES_DIR
)
from src.utils import save_figure

# Get module logger
module_logger = logging.getLogger(__name__)

class ImbalanceHandler:
    """
    Comprehensive class for handling imbalanced datasets.
    Implements multiple strategies and provides comparison tools.
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Initialise the imbalance handler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.sampling_strategies = {}
        self.class_weights = None
        self.imbalance_ratio = None
        self.threshold_results = {}
        
    def analyze_imbalance(self, y: pd.Series) -> Dict[str, Any]:
        """
        Analyse the class imbalance in the target variable.
        
        Args:
            y: Target variable
            
        Returns:
            Dictionary with imbalance analysis
        """
        module_logger.info("Analysing class imbalance...")
        
        # Class distribution
        class_counts = y.value_counts().sort_index()
        class_percentages = y.value_counts(normalize=True).sort_index() * 100
        
        # Imbalance ratio
        self.imbalance_ratio = class_counts.max() / class_counts.min()
        
        # Create visualisation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot
        class_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title('Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # Add value labels
        for i, v in enumerate(class_counts):
            ax1.text(i, v + 50, str(v), ha='center')
        
        # Pie chart
        class_percentages.plot(kind='pie', ax=ax2, autopct='%1.1f%%',
                              colors=['skyblue', 'salmon'])
        ax2.set_ylabel('')
        ax2.set_title('Class Percentage')
        
        plt.tight_layout()
        save_figure(fig, 'class_distribution', 'imbalance')
        
        analysis = {
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_percentages.to_dict(),
            'imbalance_ratio': self.imbalance_ratio,
            'minority_class': class_counts.idxmin(),
            'majority_class': class_counts.idxmax(),
            'total_samples': len(y)
        }
        
        module_logger.info(f"Imbalance ratio: {self.imbalance_ratio:.2f}")
        module_logger.info(f"Minority class: {analysis['minority_class']} ({class_percentages[analysis['minority_class']]:.1f}%)")
        
        return analysis
    
    def calculate_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """
        Calculate class weights for weighted learning.
        
        Args:
            y: Target variable
            
        Returns:
            Dictionary of class weights
        """
        module_logger.info("Calculating class weights...")
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))
        
        module_logger.info(f"Class weights: {self.class_weights}")
        
        return self.class_weights
    
    def _validate_data_for_smote(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare data for SMOTE.
        
        Args:
            X: Feature matrix
            
        Returns:
            Validated and converted DataFrame
        """
        module_logger.info("Validating data for SMOTE...")
        
        X_validated = X.copy()
        
        # Check for non-numeric columns
        non_numeric_cols = X_validated.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric_cols:
            module_logger.warning(f"Found non-numeric columns: {non_numeric_cols}")
            
            # Try to convert to numeric
            for col in non_numeric_cols:
                try:
                    X_validated[col] = pd.to_numeric(X_validated[col], errors='coerce')
                    module_logger.info(f"Converted {col} to numeric")
                except Exception as e:
                    module_logger.error(f"Could not convert {col} to numeric: {e}")
                    raise ValueError(f"Column '{col}' contains non-numeric data that cannot be converted: {X_validated[col].dtype}")
        
        # Check for missing values after conversion
        if X_validated.isnull().any().any():
            module_logger.warning("Found missing values after conversion, filling with 0")
            X_validated = X_validated.fillna(0)
        
        # Ensure all columns are numeric
        for col in X_validated.columns:
            if not pd.api.types.is_numeric_dtype(X_validated[col]):
                module_logger.error(f"Column {col} is still non-numeric: {X_validated[col].dtype}")
                raise ValueError(f"Column '{col}' could not be converted to numeric")
        
        module_logger.info("Data validation for SMOTE completed successfully")
        return X_validated
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series, 
                   variant: str = 'regular',
                   categorical_features: List[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE or its variants for oversampling.
        
        Args:
            X: Feature matrix
            y: Target variable
            variant: SMOTE variant ('regular', 'borderline', 'adasyn', 'nc')
            categorical_features: Indices of categorical features (for SMOTE-NC)
            
        Returns:
            Resampled X and y
        """
        module_logger.info(f"Applying SMOTE variant: {variant}")
        
        # Validate data first
        X_validated = self._validate_data_for_smote(X)
        
        # Select SMOTE variant
        if variant == 'regular':
            sampler = SMOTE(random_state=self.random_state)
        elif variant == 'borderline':
            sampler = BorderlineSMOTE(random_state=self.random_state)
        elif variant == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        elif variant == 'nc' and categorical_features is not None:
            # Validate categorical features indices
            valid_cat_features = [i for i in categorical_features if i < len(X_validated.columns)]
            if not valid_cat_features:
                module_logger.warning("No valid categorical features found, falling back to regular SMOTE")
                sampler = SMOTE(random_state=self.random_state)
            else:
                module_logger.info(f"Using SMOTE-NC with categorical features at indices: {valid_cat_features}")
                sampler = SMOTENC(
                    categorical_features=valid_cat_features,
                    random_state=self.random_state
                )
        else:
            module_logger.warning(f"Unknown SMOTE variant '{variant}' or missing categorical features, using regular SMOTE")
            sampler = SMOTE(random_state=self.random_state)
        
        try:
            # Apply resampling
            X_resampled, y_resampled = sampler.fit_resample(X_validated, y)
            
            # Convert back to DataFrame
            X_resampled = pd.DataFrame(X_resampled, columns=X_validated.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            # Log results
            original_counts = Counter(y)
            resampled_counts = Counter(y_resampled)
            module_logger.info(f"Original distribution: {dict(original_counts)}")
            module_logger.info(f"Resampled distribution: {dict(resampled_counts)}")
            
            self.sampling_strategies[variant] = {
                'original': dict(original_counts),
                'resampled': dict(resampled_counts)
            }
            
            return X_resampled, y_resampled
            
        except Exception as e:
            module_logger.error(f"SMOTE failed: {str(e)}")
            module_logger.info("Falling back to regular SMOTE without categorical features")
            
            # Fallback to regular SMOTE
            sampler = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X_validated, y)
            
            # Convert back to DataFrame
            X_resampled = pd.DataFrame(X_resampled, columns=X_validated.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            return X_resampled, y_resampled
    
    def apply_undersampling(self, X: pd.DataFrame, y: pd.Series,
                           method: str = 'random') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply undersampling techniques.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Undersampling method ('random', 'tomek')
            
        Returns:
            Resampled X and y
        """
        module_logger.info(f"Applying undersampling: {method}")
        
        # Validate data
        X_validated = self._validate_data_for_smote(X)
        
        if method == 'random':
            sampler = RandomUnderSampler(random_state=self.random_state)
        elif method == 'tomek':
            sampler = TomekLinks()
        else:
            raise ValueError(f"Unknown undersampling method: {method}")
        
        # Apply resampling
        X_resampled, y_resampled = sampler.fit_resample(X_validated, y)
        
        # Convert back to DataFrame
        X_resampled = pd.DataFrame(X_resampled, columns=X_validated.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        # Log results
        original_counts = Counter(y)
        resampled_counts = Counter(y_resampled)
        module_logger.info(f"Original distribution: {dict(original_counts)}")
        module_logger.info(f"Resampled distribution: {dict(resampled_counts)}")
        
        return X_resampled, y_resampled
    
    def apply_combined_sampling(self, X: pd.DataFrame, y: pd.Series,
                              method: str = 'smote_tomek') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply combined over/under sampling techniques.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Combined method ('smote_tomek', 'smote_enn')
            
        Returns:
            Resampled X and y
        """
        module_logger.info(f"Applying combined sampling: {method}")
        
        # Validate data
        X_validated = self._validate_data_for_smote(X)
        
        if method == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.random_state)
        elif method == 'smote_enn':
            sampler = SMOTEENN(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown combined method: {method}")
        
        # Apply resampling
        X_resampled, y_resampled = sampler.fit_resample(X_validated, y)
        
        # Convert back to DataFrame
        X_resampled = pd.DataFrame(X_resampled, columns=X_validated.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        # Log results
        original_counts = Counter(y)
        resampled_counts = Counter(y_resampled)
        module_logger.info(f"Original distribution: {dict(original_counts)}")
        module_logger.info(f"Resampled distribution: {dict(resampled_counts)}")
        
        return X_resampled, y_resampled
    
    def find_optimal_threshold(self, y_true: np.ndarray, 
                             y_proba: np.ndarray,
                             metric: str = 'f1') -> float:
        """
        Find optimal classification threshold for imbalanced data.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimise ('f1', 'balanced_accuracy', 'g_mean')
            
        Returns:
            Optimal threshold
        """
        from sklearn.metrics import f1_score, balanced_accuracy_score

        module_logger.info(f"Finding optimal threshold for {metric}...")
        
        # Try different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true, y_pred)
            elif metric == 'g_mean':
                # Geometric mean of sensitivity and specificity
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = np.sqrt(sensitivity * specificity)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        # Find best threshold
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        # Store results
        self.threshold_results[metric] = {
            'thresholds': thresholds,
            'scores': scores,
            'best_threshold': best_threshold,
            'best_score': best_score
        }
        
        # Plot threshold vs score
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, scores, 'b-', linewidth=2)
        plt.axvline(best_threshold, color='r', linestyle='--', 
                   label=f'Best threshold: {best_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Threshold Optimisation for {metric.capitalize()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_figure(plt.gcf(), f'threshold_optimisation_{metric}', 'imbalance')
        
        module_logger.info(f"Optimal threshold: {best_threshold:.3f} (score: {best_score:.3f})")
        
        return best_threshold
    
    def compare_sampling_methods(self, X: pd.DataFrame, y: pd.Series,
                               methods: List[str] = None,
                               categorical_indices: List[int] = None) -> pd.DataFrame:
        """
        Compare different sampling methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            methods: List of methods to compare
            categorical_indices: Indices of categorical features
            
        Returns:
            Comparison DataFrame
        """
        if methods is None:
            methods = ['none', 'smote', 'borderline_smote', 'adasyn', 
                      'random_under', 'smote_tomek']
            
            # Add SMOTE-NC if categorical features are available
            if categorical_indices and len(categorical_indices) > 0:
                methods.append('smote_nc')
        
        module_logger.info(f"Comparing sampling methods: {methods}")
        
        results = []
        
        for method in methods:
            module_logger.info(f"\nTesting method: {method}")
            
            try:
                # Apply sampling
                if method == 'none':
                    X_sampled, y_sampled = X, y
                elif method == 'smote':
                    X_sampled, y_sampled = self.apply_smote(X, y, 'regular')
                elif method == 'borderline_smote':
                    X_sampled, y_sampled = self.apply_smote(X, y, 'borderline')
                elif method == 'adasyn':
                    X_sampled, y_sampled = self.apply_smote(X, y, 'adasyn')
                elif method == 'random_under':
                    X_sampled, y_sampled = self.apply_undersampling(X, y, 'random')
                elif method == 'smote_tomek':
                    X_sampled, y_sampled = self.apply_combined_sampling(X, y, 'smote_tomek')
                elif method == 'smote_nc' and categorical_indices:
                    X_sampled, y_sampled = self.apply_smote(
                        X, y, 'nc', categorical_indices
                    )
                else:
                    module_logger.warning(f"Skipping method {method} (not applicable)")
                    continue
                
                # Calculate statistics
                class_counts = Counter(y_sampled)
                
                results.append({
                    'method': method,
                    'total_samples': len(y_sampled),
                    'class_0_count': class_counts[0],
                    'class_1_count': class_counts[1],
                    'class_0_pct': class_counts[0] / len(y_sampled) * 100,
                    'class_1_pct': class_counts[1] / len(y_sampled) * 100,
                    'imbalance_ratio': class_counts[0] / class_counts[1] if class_counts[1] > 0 else np.inf
                })
                
            except Exception as e:
                module_logger.error(f"Failed to apply method {method}: {str(e)}")
                continue
        
        comparison_df = pd.DataFrame(results)
        
        if len(comparison_df) > 0:
            # Create visualisation
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Sample counts
            methods_list = comparison_df['method'].tolist()
            x_pos = np.arange(len(methods_list))
            
            width = 0.35
            axes[0].bar(x_pos - width/2, comparison_df['class_0_count'], width, 
                   label='Class 0', color='skyblue')
            axes[0].bar(x_pos + width/2, comparison_df['class_1_count'], width, 
                   label='Class 1', color='salmon')
            
            axes[0].set_xlabel('Sampling Method')
            axes[0].set_ylabel('Sample Count')
            axes[0].set_title('Class Distribution by Sampling Method')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(methods_list, rotation=45)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Imbalance ratios
            axes[1].bar(x_pos, comparison_df['imbalance_ratio'], color='lightgreen')
            axes[1].set_xlabel('Sampling Method')
            axes[1].set_ylabel('Imbalance Ratio')
            axes[1].set_title('Imbalance Ratio by Sampling Method')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(methods_list, rotation=45)
            axes[1].axhline(y=1, color='r', linestyle='--', label='Perfect balance')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, 'sampling_methods_comparison', 'imbalance')
        
        return comparison_df

    def select_best_sampling_method(
            self,
            comparison_df: pd.DataFrame,
            *,
            prefer_over_retention: float = 0.15,
            exclude_none: bool = True
    ) -> str:
        """
        Choose the best sampling method from compare_sampling_methods() output.

        Strategy (fast, no extra model training):
        - Prefer methods that bring imbalance_ratio close to 1.0 (balanced).
        - Tie-break by retaining samples (avoid heavy undersampling).
        - If comparison_df has CV columns (cv_roc_auc_mean), use that as primary selector.
        """
        if comparison_df is None or comparison_df.empty:
            return "smote_nc"

        df = comparison_df.copy()

        if exclude_none and "method" in df.columns:
            df = df[df["method"] != "none"].copy()

        if df.empty:
            return "none"

        # If you later choose to compute quick CV scores, prefer them.
        if "cv_roc_auc_mean" in df.columns and df["cv_roc_auc_mean"].notna().any():
            df = df.sort_values(["cv_roc_auc_mean", "total_samples"], ascending=[False, False])
            return str(df.iloc[0]["method"])

        # Otherwise: balance closeness + retention
        # baseline totals (use 'none' row if present, else max total_samples)
        if (comparison_df["method"] == "none").any():
            baseline_total = float(comparison_df.loc[comparison_df["method"] == "none", "total_samples"].iloc[0])
        else:
            baseline_total = float(comparison_df["total_samples"].max())

        # closeness to perfect balance (imbalance_ratio==1 is best)
        df["balance_penalty"] = (df["imbalance_ratio"] - 1.0).abs()

        # retention: keep as much data as possible
        df["retention"] = df["total_samples"] / max(baseline_total, 1.0)

        # score: minimise penalty, but slightly reward retention
        # (prefer_over_retention controls how much retention matters)
        df["score"] = -df["balance_penalty"] + prefer_over_retention * df["retention"]

        df = df.sort_values(["score", "retention"], ascending=[False, False])

        return str(df.iloc[0]["method"])

    def create_stratified_folds(self, X: pd.DataFrame, y: pd.Series,
                              n_splits: int = CV_FOLDS) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified cross-validation folds.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of folds
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        module_logger.info(f"Creating {n_splits} stratified folds...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                            random_state=self.random_state)
        
        folds = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            folds.append((train_idx, val_idx))
            
            # Log fold statistics
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            train_dist = y_train_fold.value_counts(normalize=True) * 100
            val_dist = y_val_fold.value_counts(normalize=True) * 100
            
            module_logger.info(f"Fold {fold + 1}: Train distribution: {train_dist.to_dict()}, "
                             f"Val distribution: {val_dist.to_dict()}")
        
        return folds

    def get_sampler(self, strategy: str | None, categorical_indices: list[int] | None = None):
        """
        Return an imbalanced-learn sampler instance for use inside an imblearn Pipeline.
        """
        if strategy is None or strategy == "none":
            return None

        if strategy == "smote":
            return SMOTE(random_state=self.random_state)
        if strategy == "borderline_smote":
            return BorderlineSMOTE(random_state=self.random_state)
        if strategy == "adasyn":
            return ADASYN(random_state=self.random_state)
        if strategy == "random_under":
            return RandomUnderSampler(random_state=self.random_state)
        if strategy == "smote_tomek":
            return SMOTETomek(random_state=self.random_state)
        if strategy == "smote_nc":
            if not categorical_indices:
                raise ValueError("categorical_indices must be provided for SMOTE-NC.")
            return SMOTENC(categorical_features=categorical_indices, random_state=self.random_state)

        raise ValueError(f"Unknown sampling strategy: {strategy}")


def demonstrate_imbalance_handling(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Demonstrate various imbalance handling techniques.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Dictionary with results
    """
    handler = ImbalanceHandler()
    
    # Analyse imbalance
    imbalance_analysis = handler.analyze_imbalance(y)
    
    # Calculate class weights
    class_weights = handler.calculate_class_weights(y)
    
    # Identify categorical features by checking for encoded features
    categorical_indices = []
    for i, col in enumerate(X.columns):
        if col.endswith('_encoded') or col in ['age_group', 'bmi_category', 'chol_category', 
                                               'glucose_category', 'smoking_intensity', 
                                               'hypertension_stage', 'hr_category']:
            categorical_indices.append(i)
    
    module_logger.info(f"Identified categorical feature indices: {categorical_indices}")
    
    # Compare sampling methods
    comparison = handler.compare_sampling_methods(
        X, y, categorical_indices=categorical_indices
    )

    best_method = handler.select_best_sampling_method(comparison)
    module_logger.info(f"Selected best sampling method: {best_method}")
    
    # Create stratified folds
    folds = handler.create_stratified_folds(X, y)
    
    results = {
        'imbalance_analysis': imbalance_analysis,
        'class_weights': class_weights,
        'sampling_comparison': comparison,
        'cv_folds': folds,
        'handler': handler,
        'best_method': best_method
    }
    
    return results


if __name__ == "__main__":
    # Test the imbalance handling
    module_logger.info("Testing class imbalance handling...")
    
    # Load engineered data
    from data_preprocessing import quick_preprocess
    from feature_engineering import create_feature_engineering_pipeline
    
    # Preprocess data
    train_df, test_df, preprocessor = quick_preprocess()
    
    # Apply feature engineering
    train_engineered, engineer = create_feature_engineering_pipeline(
        train_df,
        preprocessor.categorical_features,
        preprocessor.numerical_features
    )
    
    # Separate features and target
    X = train_engineered.drop(TARGET_COLUMN, axis=1)
    y = train_engineered[TARGET_COLUMN]
    
    # Demonstrate imbalance handling
    results = demonstrate_imbalance_handling(X, y)
    
    # Save comparison results
    comparison_df = results['sampling_comparison']
    comparison_df.to_csv(TABLES_DIR / 'sampling_methods_comparison.csv', index=False)
    
    module_logger.info("\nClass imbalance handling completed!")
    module_logger.info(f"Comparison saved to: {TABLES_DIR / 'sampling_methods_comparison.csv'}")