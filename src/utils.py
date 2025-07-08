"""
Utility functions for the Heart Disease ML Pipeline
Provides common functionality used across different modules
"""

import os
import json
import pickle
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import (
    RANDOM_SEED, FIGURES_DIR, TABLES_DIR, 
    FIGURE_SIZE, STYLE, DPI, MODEL_FORMAT,
    LOG_LEVEL, LOG_FORMAT
)

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    logger.info(f"Random seeds set to {seed}")

# Set matplotlib style
def set_plot_style() -> None:
    """Set consistent plotting style for all visualisations."""
    plt.style.use(STYLE)
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    logger.info("Plot style configured")

# Data loading functions
def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load data from CSV file with error handling.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Loaded data from {file_path}: shape {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

# Model saving and loading
def save_model(model: Any, model_name: str, directory: Path, 
               metadata: Optional[Dict] = None) -> Path:
    """
    Save a trained model with metadata.
    
    Args:
        model: Trained model object
        model_name: Name for the saved model
        directory: Directory to save the model
        metadata: Optional metadata dictionary
        
    Returns:
        Path to saved model
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}"
    
    # Save model
    if MODEL_FORMAT == "joblib":
        model_path = directory / f"{filename}.joblib"
        joblib.dump(model, model_path)
    else:
        model_path = directory / f"{filename}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save metadata if provided
    if metadata:
        metadata['timestamp'] = timestamp
        metadata['model_file'] = str(model_path.name)
        metadata_path = directory / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    return model_path

def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a saved model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model object
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if model_path.suffix == ".joblib":
        model = joblib.load(model_path)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    logger.info(f"Model loaded from {model_path}")
    return model

# Data validation
def validate_dataframe(df: pd.DataFrame, expected_columns: List[str],
                      name: str = "DataFrame") -> None:
    """
    Validate DataFrame structure.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        name: Name for logging
        
    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{name} missing columns: {missing_columns}")
    
    logger.info(f"{name} validated: shape {df.shape}")

# Performance metrics formatting
def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """
    Format metrics dictionary for display.
    
    Args:
        metrics: Dictionary of metric values
        precision: Number of decimal places
        
    Returns:
        Formatted metrics dictionary
    """
    return {k: f"{v:.{precision}f}" for k, v in metrics.items()}

# Save figure with timestamp
def save_figure(fig: plt.Figure, name: str, 
                subdirectory: Optional[str] = None) -> Path:
    """
    Save matplotlib figure with timestamp.
    
    Args:
        fig: Matplotlib figure object
        name: Base name for the file
        subdirectory: Optional subdirectory within figures
        
    Returns:
        Path to saved figure
    """
    save_dir = FIGURES_DIR
    if subdirectory:
        save_dir = save_dir / subdirectory
        save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = save_dir / filename
    
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure saved to {filepath}")
    return filepath

# Create results table
def create_results_table(results: List[Dict[str, Any]], 
                        sort_by: str = "roc_auc",
                        ascending: bool = False) -> pd.DataFrame:
    """
    Create a formatted results table from model evaluation results.
    
    Args:
        results: List of result dictionaries
        sort_by: Metric to sort by
        ascending: Sort order
        
    Returns:
        Formatted DataFrame
    """
    df = pd.DataFrame(results)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    return df

# Timer context manager
class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"{self.name} started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {self.elapsed:.2f} seconds")

# Memory usage monitoring
def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Feature importance plotting
def plot_feature_importance(importance_df: pd.DataFrame, 
                           title: str = "Feature Importance",
                           top_n: int = 20) -> plt.Figure:
    """
    Create a horizontal bar plot of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        title: Plot title
        top_n: Number of top features to show
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    
    # Sort and select top features
    plot_df = importance_df.nlargest(top_n, 'importance')
    
    # Create horizontal bar plot
    ax.barh(plot_df['feature'], plot_df['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

# Statistical testing helpers
def calculate_confidence_interval(values: np.ndarray, 
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a set of values.
    
    Args:
        values: Array of values
        confidence: Confidence level
        
    Returns:
        Lower and upper bounds of confidence interval
    """
    from scipy import stats
    mean = np.mean(values)
    stderr = stats.sem(values)
    interval = stderr * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    return mean - interval, mean + interval

# Report generation helpers
def create_model_card(model_name: str, 
                     performance_metrics: Dict[str, float],
                     training_details: Dict[str, Any],
                     feature_importance: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Create a model card with comprehensive model information.
    
    Args:
        model_name: Name of the model
        performance_metrics: Dictionary of performance metrics
        training_details: Dictionary with training information
        feature_importance: Optional feature importance DataFrame
        
    Returns:
        Model card dictionary
    """
    model_card = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "performance_metrics": performance_metrics,
        "training_details": training_details,
        "feature_importance": feature_importance.to_dict() if feature_importance is not None else None,
        "framework_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": None  # Will be set when sklearn is imported
        }
    }
    
    try:
        import sklearn
        model_card["framework_versions"]["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    
    return model_card

# Initialise utilities
set_random_seeds()
set_plot_style()

logger.info("Utilities module loaded successfully")