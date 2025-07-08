"""
Model interpretation module
Handles SHAP, partial dependence, and clinical insights
"""

import pandas as pd
import shap
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from src.config import FIGURES_DIR
from src.utils import save_figure

def generate_shap_summary(model, X: pd.DataFrame, model_type: str = 'tree'):
    """
    Generate SHAP summary plot
    
    Args:
        model: Trained model
        X: Feature matrix
        model_type: 'tree', 'linear', or 'ebm'
    """
    if model_type == 'ebm' and isinstance(model, ExplainableBoostingClassifier):
        # EBM has built-in explainer
        global_exp = model.explain_global()
        global_exp.visualize()
    else:
        # Create SHAP explainer
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X)
        else:
            explainer = shap.KernelExplainer(model.predict, X)
        
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
    
    save_figure(plt.gcf(), 'shap_summary', 'interpretation')

# Additional functions for:
# - Individual prediction explanations
# - Partial dependence plots
# - Clinical risk score generation
# - Model card generation