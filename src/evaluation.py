"""
Model evaluation module
Handles performance metrics, statistical tests, and model comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)

from scipy import stats
from src.config import FIGURES_DIR
from src.utils import save_figure

def generate_performance_report(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Generate comprehensive performance report for multiple models
    
    Args:
        models: Dictionary of trained models {name: model}
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with performance metrics
    """
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Model': name,
            'AUC-ROC': roc_auc_score(y_test, y_proba),
            'AP': average_precision_score(y_test, y_proba),
            'F1': f1_score(y_test, y_pred),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        results.append(metrics)
    
    return pd.DataFrame(results)

# Additional functions for:
# - ROC curve visualization
# - Precision-recall curves
# - Statistical significance testing (DeLong test)
# - Feature importance comparison