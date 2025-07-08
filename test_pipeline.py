"""
Test script to demonstrate the Heart Disease ML Pipeline
Shows how to use the implemented modules with actual data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TARGET_COLUMN, RANDOM_SEED
from src.utils import set_random_seeds, logger
from src.data_preprocessing import DataPreprocessor, quick_preprocess
from src.eda import ExploratoryDataAnalyzer
from src.feature_engineering import FeatureEngineer, create_feature_engineering_pipeline
from src.class_imbalance import ImbalanceHandler, demonstrate_imbalance_handling
from src.model_training import ModelTrainer

def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    # Create sample data files if they don't exist
    # (In production, these would be the actual CSV files)
    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"
    
    # Check if files exist
    if not train_path.exists() or not test_path.exists():
        logger.warning("Data files not found. Please ensure train.csv and test.csv are in the data/raw/ directory")
        return None, None
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Train data loaded: {train_df.shape}")
    logger.info(f"Test data loaded: {test_df.shape}")
    
    return train_df, test_df

def test_preprocessing():
    """Test preprocessing pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Testing preprocessing pipeline...")
    logger.info("="*60)
    
    # Initialise preprocessor
    preprocessor = DataPreprocessor()
    
    # Load datasets
    train_df, test_df = preprocessor.load_datasets()
    
    # Analyse data quality
    quality_report = preprocessor.analyze_data_quality()
    
    # Print findings
    logger.info("\nData Quality Summary:")
    logger.info(f"Missing values in train: {len(quality_report['train_missing'])} columns")
    logger.info(f"Missing values in test: {len(quality_report['test_missing'])} columns")
    logger.info(f"Class imbalance ratio: {preprocessor.data_info['imbalance_ratio']:.2f}")
    
    # Handle missing values
    train_df_clean, test_df_clean = preprocessor.handle_missing_values(strategy='median')
    
    # Detect outliers
    outliers = preprocessor.detect_outliers(method='iqr')
    logger.info(f"Total outliers detected: {outliers.sum().sum()}")
    
    return preprocessor

def test_eda(preprocessor):
    """Test EDA functionality."""
    logger.info("\n" + "="*60)
    logger.info("Testing EDA pipeline...")
    logger.info("="*60)
    
    # Create EDA analyzer
    eda_analyzer = ExploratoryDataAnalyzer(preprocessor)
    
    # Perform complete EDA
    eda_results = eda_analyzer.perform_complete_eda()
    
    # Print key insights
    logger.info("\nEDA Key Insights:")
    
    # Target distribution
    if 'target_analysis' in eda_results:
        logger.info(f"Class distribution:")
        logger.info(eda_results['target_analysis']['distribution'].to_string())
    
    # Top correlations
    if 'correlations' in eda_results and 'target_correlations' in eda_results['correlations']:
        logger.info("\nTop 5 correlated features:")
        logger.info(eda_results['correlations']['target_correlations'].head(5).to_string())
    
    # Generate report
    eda_analyzer.generate_eda_report()
    logger.info("EDA report generated!")
    
    return eda_results

def demonstrate_quick_preprocessing():
    """Demonstrate the quick preprocessing function."""
    logger.info("\n" + "="*60)
    logger.info("Testing quick preprocessing...")
    logger.info("="*60)
    
    # Quick preprocessing with median imputation
    train_df, test_df, preprocessor = quick_preprocess(imputation_strategy='median')
    
    logger.info(f"Processed train shape: {train_df.shape}")
    logger.info(f"Processed test shape: {test_df.shape}")
    
    # Create train-validation split
    X_train, X_val, y_train, y_val = preprocessor.create_train_validation_split()
    
    logger.info(f"\nTrain-validation split:")
    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    return preprocessor

def test_feature_engineering(preprocessor):
    """Test feature engineering functionality."""
    logger.info("\n" + "="*60)
    logger.info("Testing feature engineering...")
    logger.info("="*60)
    
    # Apply feature engineering
    train_engineered, engineer = create_feature_engineering_pipeline(
        preprocessor.train_data,
        preprocessor.categorical_features,
        preprocessor.numerical_features
    )
    
    # Transform test data
    test_engineered = engineer.transform(preprocessor.test_data)
    
    logger.info(f"\nFeature engineering results:")
    logger.info(f"Original features: {len(preprocessor.feature_names)}")
    logger.info(f"Engineered features: {len(engineer.feature_names)}")
    logger.info(f"New features created: {len(engineer.new_features)}")
    
    logger.info("\nSample of new features:")
    for i, (feat, desc) in enumerate(engineer.get_new_feature_descriptions().items()):
        if i < 5:  # Show first 5
            logger.info(f"  - {feat}: {desc}")
    
    return train_engineered, test_engineered, engineer

def test_imbalance_handling(train_engineered):
    """Test class imbalance handling."""
    logger.info("\n" + "="*60)
    logger.info("Testing class imbalance handling...")
    logger.info("="*60)
    
    # Separate features and target
    X = train_engineered.drop(TARGET_COLUMN, axis=1)
    y = train_engineered[TARGET_COLUMN]
    
    # Test imbalance handling
    imbalance_results = demonstrate_imbalance_handling(X, y)
    
    logger.info("\nImbalance handling results:")
    logger.info(f"Imbalance ratio: {imbalance_results['imbalance_analysis']['imbalance_ratio']:.2f}")
    logger.info(f"Class weights: {imbalance_results['class_weights']}")
    
    # Show sampling comparison
    comparison_df = imbalance_results['sampling_comparison']
    logger.info("\nSampling methods comparison:")
    logger.info(comparison_df[['method', 'total_samples', 'imbalance_ratio']].to_string())
    
    return imbalance_results

def test_baseline_model(train_engineered, imbalance_results):
    """Test baseline model training."""
    logger.info("\n" + "="*60)
    logger.info("Testing baseline model training...")
    logger.info("="*60)
    
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = train_engineered.drop(TARGET_COLUMN, axis=1)
    y = train_engineered[TARGET_COLUMN]
    
    # Create train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Get class weights
    class_weights = imbalance_results['class_weights']
    
    # Initialise trainer
    trainer = ModelTrainer()
    
    # Train baseline model
    baseline_results = trainer.train_baseline_model(
        X_train, y_train, X_val, y_val, class_weights
    )
    
    logger.info("\nBaseline model results:")
    logger.info(f"Training time: {baseline_results['training_time']:.2f} seconds")
    logger.info("Validation metrics:")
    for metric, value in baseline_results['metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return trainer, baseline_results

def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("HEART DISEASE ML PIPELINE - TEST SCRIPT")
    logger.info("="*80)
    
    # Set random seed
    set_random_seeds(42)
    
    try:
        # Test data loading
        train_df, test_df = test_data_loading()
        if train_df is None:
            return
        
        # Test preprocessing
        preprocessor = test_preprocessing()
        
        # Test EDA
        eda_results = test_eda(preprocessor)
        
        # Test quick preprocessing
        preprocessor_quick = demonstrate_quick_preprocessing()
        
        # Test feature engineering
        train_engineered, test_engineered, engineer = test_feature_engineering(preprocessor)
        
        # Test imbalance handling
        imbalance_results = test_imbalance_handling(train_engineered)
        
        # Test baseline model
        trainer, baseline_results = test_baseline_model(train_engineered, imbalance_results)
        
        # Save processed data
        preprocessor.save_processed_data(suffix="_test")
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        # Summary of outputs
        logger.info("\nGenerated outputs:")
        logger.info(f"- Processed data saved in: {PROCESSED_DATA_DIR}")
        logger.info(f"- EDA figures saved in: {FIGURES_DIR}/eda/")
        logger.info(f"- Imbalance figures saved in: {FIGURES_DIR}/imbalance/")
        logger.info(f"- EDA report saved in: reports/tables/")
        logger.info(f"- Baseline model saved in: models/baseline/")
        logger.info("\nPipeline phases completed:")
        logger.info("✓ Phase 1: Data Loading")
        logger.info("✓ Phase 2: Exploratory Data Analysis")
        logger.info("✓ Phase 3: Feature Engineering")
        logger.info("✓ Phase 4: Class Imbalance Handling")
        logger.info("✓ Phase 5: Baseline Model Training")
        logger.info("\nNext steps: Run advanced model training and evaluation phases")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()