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

RUN_EBM_IN_PHASE6 = False

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

PHASE6_MODELS = [
    "svm",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
    "ebm",
]

# NOTE: These keys assume the model is in a Pipeline step named "model"
ADVANCED_MODEL_PARAM_SPACES = {
    "svm": {
        "model__C": [1, 10, 50],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale", "auto", 0.01, 0.1],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__bootstrap": [True, False],
    },
    "extra_trees": {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "xgboost": {
        "model__n_estimators": [300, 600, 900],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 5, 10],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [1.0, 1.5, 2.0],
        "model__gamma": [0.0, 0.1, 0.2],
    },
    "lightgbm": {
        "model__n_estimators": [300, 600, 900],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__num_leaves": [31, 63, 127],
        "model__max_depth": [-1, 5, 10, 20],
        "model__min_child_samples": [10, 20, 50],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [0.0, 1.0, 2.0],
    },
    "catboost": {
        "model__iterations": [100, 300, 600],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__l2_leaf_reg": [1, 3, 5, 7],
        "model__border_count": [64, 128, 254],
        "model__bagging_temperature": [0.0, 0.5, 1.0],
    },
    "ebm": {
        "model__max_bins": [128, 256],
        "model__max_interaction_bins": [16, 32],
        "model__interactions": [0, 5],
        "model__outer_bags": [4],
        "model__inner_bags": [0],
    },
}

TUNING_DEFAULT_SEARCH_TYPE = "random"  # or "grid"
TUNING_N_ITER = 30
TUNING_SCORING = "roc_auc"
TUNING_N_JOBS = -1
TUNING_VERBOSE = 1

CALIBRATION_ENABLED = True
CALIBRATION_METHOD = "sigmoid"  # "isotonic" is heavier
CALIBRATION_CV = 5

THRESHOLD_OPTIMISATION_ENABLED = True
THRESHOLD_PRIMARY_STRATEGY = "f1"  # "f1" or "balanced_accuracy"

# Cross-validation settings
CV_FOLDS = 5
STRATIFIED = True

# Train-test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

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