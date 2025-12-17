"""
Project configuration settings for Heart Disease ML Pipeline
British English throughout
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
TUNED_MODELS_DIR = MODELS_DIR / "tuned"
FINAL_MODELS_DIR = MODELS_DIR / "final"

# Reports paths
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODEL_CARDS_DIR = REPORTS_DIR / "model_cards"

# Create directories if they don't exist
for dir_path in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    BASELINE_MODELS_DIR, TUNED_MODELS_DIR, FINAL_MODELS_DIR,
    FIGURES_DIR, TABLES_DIR, MODEL_CARDS_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Data file names
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TARGET_COLUMN = "TenYearCHD"

# Model configurations
MODEL_CONFIGS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "year": 1958,
        "type": "linear",
        "interpretability": "high"
    },
    "svm": {
        "name": "Support Vector Machine",
        "year": 1995,
        "type": "kernel",
        "interpretability": "low"
    },
    "random_forest": {
        "name": "Random Forest",
        "year": 2001,
        "type": "ensemble_bagging",
        "interpretability": "medium"
    },
    "extra_trees": {
        "name": "Extra Trees",
        "year": 2006,
        "type": "ensemble_bagging",
        "interpretability": "medium"
    },
    "xgboost": {
        "name": "XGBoost",
        "year": 2014,
        "type": "ensemble_boosting",
        "interpretability": "medium"
    },
    "lightgbm": {
        "name": "LightGBM",
        "year": 2017,
        "type": "ensemble_boosting",
        "interpretability": "medium"
    },
    "catboost": {
        "name": "CatBoost",
        "year": 2017,
        "type": "ensemble_boosting",
        "interpretability": "medium"
    },
    "ebm": {
        "name": "Explainable Boosting Machine",
        "year": 2019,
        "type": "gam",
        "interpretability": "high"
    },
    "voting": {
        "name": "Voting Ensemble",
        "year": "composite",
        "type": "ensemble_voting",
        "interpretability": "low"
    }
}

# Cross-validation settings
CV_FOLDS = 5
STRATIFIED = True

# Train-test split
TEST_SIZE = 0.2

# Class imbalance strategies
# Primary strategy chosen based on sampling comparison:
# - SMOTE-NC balances classes to ~50:50 while retaining all majority samples.
PRIMARY_IMBALANCE_STRATEGY = "smote_nc"
SECONDARY_IMBALANCE_STRATEGIES = ["class_weight"]

IMBALANCE_STRATEGIES = [
    PRIMARY_IMBALANCE_STRATEGY,
    *SECONDARY_IMBALANCE_STRATEGIES,
    "threshold_optimisation"
]


# Evaluation metrics
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "matthews_corrcoef"
]

# Feature engineering settings
AGE_BINS = [0, 40, 50, 60, 100]
AGE_LABELS = ["<40", "40-50", "50-60", "60+"]

# Visualisation settings
FIGURE_SIZE = (10, 6)
STYLE = "seaborn-v0_8-darkgrid"
COLOUR_PALETTE = "husl"
DPI = 300

# Model saving format
MODEL_FORMAT = "joblib"  # or "pickle"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"