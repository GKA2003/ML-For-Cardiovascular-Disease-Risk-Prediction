"""
Main orchestrator for Heart Disease ML Pipeline
Coordinates all phases of the machine learning project
"""

import numpy as np
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import updated config vals
from src.config import (
    PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    TRAIN_FILE, TEST_FILE, RANDOM_SEED,
    FIGURES_DIR, TABLES_DIR, TARGET_COLUMN
)
from src.utils import set_random_seeds, Timer, logger
from src.data_preprocessing import DataPreprocessor
from src.eda import ExploratoryDataAnalyzer
from src.feature_engineering import FeatureEngineer, create_feature_engineering_pipeline
from src.class_imbalance import ImbalanceHandler, demonstrate_imbalance_handling
from src.model_training import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_phase_3_and_4(preprocessor, eda_results):
    """Run Phase 3 (Feature Engineering) and Phase 4 (Class Imbalance Handling)."""
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("="*60)
    
    with Timer("Phase 3"):
        # Load processed data
        train_df = preprocessor.train_data
        test_df = preprocessor.test_data
        
        # Apply feature engineering
        train_engineered, engineer = create_feature_engineering_pipeline(
            train_df,
            preprocessor.categorical_features,
            preprocessor.numerical_features
        )
        
        # Transform test data
        test_engineered = engineer.transform(test_df)
        
        # Log results
        logger.info(f"\nFeature engineering completed:")
        logger.info(f"Original features: {len(preprocessor.feature_names)}")
        logger.info(f"Engineered features: {len(engineer.feature_names)}")
        logger.info(f"New features created: {len(engineer.new_features)}")
        
        logger.info("\nNew features:")
        for feat, desc in engineer.get_new_feature_descriptions().items():
            logger.info(f"  - {feat}: {desc}")
        
        # Save engineered data
        train_engineered.to_csv(PROCESSED_DATA_DIR / "train_engineered.csv", index=False)
        test_engineered.to_csv(PROCESSED_DATA_DIR / "test_engineered.csv", index=False)
    
    # Phase 4: Class Imbalance Handling
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: CLASS IMBALANCE HANDLING")
    logger.info("="*60)
    
    with Timer("Phase 4"):
        # Separate features and target
        X = train_engineered.drop(TARGET_COLUMN, axis=1)
        y = train_engineered[TARGET_COLUMN]
        
        # Demonstrate imbalance handling techniques
        imbalance_results = demonstrate_imbalance_handling(X, y)
        
        # Log key results
        logger.info("\nImbalance handling results:")
        logger.info(f"Original imbalance ratio: {imbalance_results['imbalance_analysis']['imbalance_ratio']:.2f}")
        logger.info(f"Class weights calculated: {imbalance_results['class_weights']}")
        
        # Save comparison results
        comparison_df = imbalance_results['sampling_comparison']
        comparison_path = TABLES_DIR / 'sampling_methods_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nSampling methods comparison saved to: {comparison_path}")
    
    return train_engineered, test_engineered, engineer, imbalance_results

def run_phase_5_baseline(train_engineered, engineer, imbalance_results):
    """Run Phase 5 (Baseline Model Training)."""
    logger.info("\n" + "="*60)
    logger.info("PHASE 5: BASELINE MODEL DEVELOPMENT")
    logger.info("="*60)
    
    from sklearn.model_selection import train_test_split
    
    with Timer("Phase 5"):
        # Separate features and target
        X = train_engineered.drop(TARGET_COLUMN, axis=1)
        y = train_engineered[TARGET_COLUMN]
        
        # Create train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        logger.info(f"Train-validation split created:")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        
        # Get class weights
        class_weights = imbalance_results['class_weights']
        
        # Initialise model trainer
        trainer = ModelTrainer()
        
        # Train enhanced baseline model
        baseline_results = trainer.train_baseline_model(
            X_train, y_train, X_val, y_val, class_weights
        )
        
        # Log enhanced baseline results
        logger.info("\nEnhanced Baseline Logistic Regression Results:")
        logger.info(f"Training time: {baseline_results['training_time']:.2f} seconds")
        logger.info(f"Memory usage: {baseline_results['memory_usage_mb']:.2f} MB")
        
        logger.info("\nValidation metrics:")
        for metric, value in baseline_results['validation_metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\nCross-validation metrics (5-fold):")
        for metric, scores in baseline_results['cv_metrics'].items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logger.info(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        # Log feature importance
        coefficients = baseline_results['coefficients']
        sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        logger.info("\nTop 10 most important features (by coefficient magnitude):")
        for feat, coef in sorted_coefs:
            logger.info(f"  {feat}: {coef:.4f}")
        
        # Log visualizations
        logger.info(f"\nGenerated visualizations:")
        for viz_path in baseline_results['visualization_paths']:
            logger.info(f"  - {viz_path}")
        
        # Create comprehensive baseline report
        logger.info("\nCreating baseline model report...")
        baseline_report = {
            'model_name': 'Baseline Logistic Regression',
            'algorithm': 'Logistic Regression',
            'training_time': baseline_results['training_time'],
            'memory_usage_mb': baseline_results['memory_usage_mb'],
            'validation_metrics': baseline_results['validation_metrics'],
            'cv_metrics': baseline_results['cv_metrics'],
            'feature_importance': dict(sorted_coefs[:15]),  # Top 15 features
            'model_path': str(baseline_results['model_path']),
            'visualization_paths': [str(path) for path in baseline_results['visualization_paths']],
            'cross_validation_stability': {
                metric: {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'cv_coefficient': float(np.std(scores) / np.mean(scores)) if np.mean(scores) != 0 else 0
                }
                for metric, scores in baseline_results['cv_metrics'].items()
            }
        }
        
        # Save baseline report
        import json
        from src.utils import convert_numpy_types
        report_clean = convert_numpy_types(baseline_report)
        
        report_path = TABLES_DIR / f"baseline_model_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report_clean, f, indent=2)
        
        logger.info(f"Baseline model report saved to: {report_path}")
        
        # Assessment of baseline performance
        roc_auc = baseline_results['validation_metrics']['roc_auc']
        f1_score = baseline_results['validation_metrics']['f1']
        
        logger.info(f"\n" + "="*50)
        logger.info("BASELINE MODEL ASSESSMENT")
        logger.info("="*50)
        
        if roc_auc >= 0.75:
            logger.info("✅ Excellent baseline performance (ROC-AUC ≥ 0.75)")
        elif roc_auc >= 0.65:
            logger.info("✅ Good baseline performance (ROC-AUC ≥ 0.65)")
        elif roc_auc >= 0.55:
            logger.info("⚠️  Moderate baseline performance (ROC-AUC ≥ 0.55)")
        else:
            logger.info("❌ Poor baseline performance (ROC-AUC < 0.55)")
        
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"F1-Score: {f1_score:.4f}")
        
        # Cross-validation stability assessment
        roc_auc_cv_std = np.std(baseline_results['cv_metrics']['roc_auc'])
        if roc_auc_cv_std < 0.02:
            logger.info("✅ Very stable performance across CV folds")
        elif roc_auc_cv_std < 0.05:
            logger.info("✅ Stable performance across CV folds")
        else:
            logger.info("⚠️  Variable performance across CV folds")
        
        logger.info(f"CV ROC-AUC std: {roc_auc_cv_std:.4f}")
        
        # Feature importance insights
        top_feature = sorted_coefs[0]
        logger.info(f"\nMost predictive feature: {top_feature[0]} (coef: {top_feature[1]:.4f})")
        
        # Ready for next phase
        logger.info("\n" + "="*50)
        logger.info("BASELINE PHASE COMPLETED - READY FOR ADVANCED MODELS")
        logger.info("="*50)
    
    return trainer, X_train, X_val, y_train, y_val, baseline_results

def run_phase_1_and_2():
    """Run Phase 1 (Setup) and Phase 2 (EDA)."""
    logger.info("="*80)
    logger.info("HEART DISEASE PREDICTION ML PIPELINE")
    logger.info("Phase 1: Environment Setup & Data Loading")
    logger.info("Phase 2: Comprehensive EDA")
    logger.info("="*80)
    
    # Ensure random seeds are set
    set_random_seeds(RANDOM_SEED)
    
    # Phase 1: Data Loading and Initial Processing
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: DATA LOADING AND INITIAL PROCESSING")
    logger.info("="*60)
    
    with Timer("Phase 1"):
        # Initialise preprocessor
        preprocessor = DataPreprocessor(random_state=RANDOM_SEED)
        
        # Load datasets
        train_df, test_df = preprocessor.load_datasets()
        
        # Analyse data quality
        quality_report = preprocessor.analyze_data_quality()
        
        # Print initial findings
        logger.info("\nDATA QUALITY SUMMARY:")
        logger.info("-"*40)
        
        # Missing values summary
        if not quality_report['train_missing'].empty:
            logger.info("\nMissing values in training data:")
            logger.info(quality_report['train_missing'].to_string())
        else:
            logger.info("\nNo missing values in training data")
        
        if not quality_report['test_missing'].empty:
            logger.info("\nMissing values in test data:")
            logger.info(quality_report['test_missing'].to_string())
        else:
            logger.info("\nNo missing values in test data")
        
        # Target distribution
        logger.info("\nTarget variable distribution:")
        logger.info(quality_report['target_distribution'].to_string())
        logger.info(f"\nClass imbalance ratio: {preprocessor.data_info['imbalance_ratio']:.2f}")
        
        # Feature types
        logger.info(f"\nFeature types identified:")
        logger.info(f"Numerical features ({len(preprocessor.numerical_features)}): {preprocessor.numerical_features}")
        logger.info(f"Categorical features ({len(preprocessor.categorical_features)}): {preprocessor.categorical_features}")
        
        # Handle missing values if any exist
        if (quality_report['train_missing'].empty and quality_report['test_missing'].empty):
            logger.info("\nNo missing values to handle - skipping imputation step")
        else:
            logger.info("\nHandling missing values...")
            train_df, test_df = preprocessor.handle_missing_values(strategy='median')
        
        # Save initial processed data
        preprocessor.save_processed_data(suffix="_phase1")
    
    # Phase 2: Comprehensive EDA
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    logger.info("="*60)
    
    with Timer("Phase 2"):
        # Create EDA analyzer
        eda_analyzer = ExploratoryDataAnalyzer(preprocessor)
        
        # Perform complete EDA
        eda_results = eda_analyzer.perform_complete_eda()
        
        # Print key insights
        logger.info("\nKEY EDA INSIGHTS:")
        logger.info("-"*40)
        
        # Top correlations with target
        if 'correlations' in eda_results and 'target_correlations' in eda_results['correlations']:
            logger.info("\nTop 5 features correlated with target:")
            top_corr = eda_results['correlations']['target_correlations'].head(5)
            logger.info(top_corr.to_string())
        
        # Significant features from statistical tests
        if 'statistical_tests' in eda_results:
            if 'numerical' in eda_results['statistical_tests']:
                sig_numerical = eda_results['statistical_tests']['numerical'][
                    eda_results['statistical_tests']['numerical']['significant'] == True
                ]
                logger.info(f"\nStatistically significant numerical features: {len(sig_numerical)}")
                logger.info(sig_numerical[['feature', 'test', 'p_value']].to_string())
            
            if 'categorical' in eda_results['statistical_tests']:
                sig_categorical = eda_results['statistical_tests']['categorical'][
                    eda_results['statistical_tests']['categorical']['significant'] == True
                ]
                logger.info(f"\nStatistically significant categorical features: {len(sig_categorical)}")
                logger.info(sig_categorical[['feature', 'test', 'p_value']].to_string())
        
        # Outlier summary
        if 'outliers' in eda_results:
            logger.info(f"\nTotal samples with outliers: {eda_results['outliers']['total_samples_with_outliers']}")
            outlier_features = [(k, v) for k, v in eda_results['outliers']['counts'].items() if v > 0]
            outlier_features.sort(key=lambda x: x[1], reverse=True)
            logger.info("Features with most outliers:")
            for feat, count in outlier_features[:5]:
                logger.info(f"  {feat}: {count} outliers")
        
        # Generate comprehensive report
        eda_analyzer.generate_eda_report()
        logger.info("\nEDA report generated successfully!")
    
    logger.info("\n" + "="*60)
    logger.info("PHASES 1 & 2 COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("- Phase 3: Feature Engineering")
    logger.info("- Phase 4: Data Splitting & Class Imbalance Handling")
    logger.info("- Phase 5: Model Development Pipeline")
    logger.info("\nProcessed data saved in: " + str(PROCESSED_DATA_DIR))
    logger.info("EDA visualisations saved in: reports/figures/eda/")
    logger.info("EDA report saved in: reports/tables/")
    
    return preprocessor, eda_results

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='Heart Disease ML Pipeline')
    parser.add_argument('--phase', type=str, default='1-2',
                       help='Which phase(s) to run (e.g., "1-2", "3-4", "5", "all")')
    parser.add_argument('--skip-eda-plots', action='store_true',
                       help='Skip generating EDA visualisations')
    
    args = parser.parse_args()
    
    try:
        preprocessor = None
        eda_results = None
        train_engineered = None
        test_engineered = None
        engineer = None
        imbalance_results = None
        trainer = None
        
        if args.phase in ['1-2', 'all']:
            preprocessor, eda_results = run_phase_1_and_2()
        
        if args.phase in ['3-4', 'all']:
            if preprocessor is None:
                # Load saved data if running phases separately
                logger.info("Loading preprocessed data from Phase 1-2...")
                train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_processed_phase1.csv")
                test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_processed_phase1.csv")
                
                # Recreate preprocessor
                preprocessor = DataPreprocessor()
                preprocessor.train_data = train_df
                preprocessor.test_data = test_df
                preprocessor.feature_names = [col for col in train_df.columns if col != TARGET_COLUMN]
                preprocessor._identify_feature_types()
            
            train_engineered, test_engineered, engineer, imbalance_results = run_phase_3_and_4(
                preprocessor, eda_results
            )
        
        if args.phase in ['5', 'all']:
            if train_engineered is None:
                # Load engineered data if running phase separately
                logger.info("Loading engineered data from Phase 3-4...")
                train_engineered = pd.read_csv(PROCESSED_DATA_DIR / "train_engineered.csv")
                
                # Load imbalance results
                from src.class_imbalance import ImbalanceHandler
                handler = ImbalanceHandler()
                X = train_engineered.drop(TARGET_COLUMN, axis=1)
                y = train_engineered[TARGET_COLUMN]
                imbalance_results = {
                    'class_weights': handler.calculate_class_weights(y)
                }
            
            trainer, X_train, X_val, y_train, y_val, baseline_results = run_phase_5_baseline(
                train_engineered, engineer, imbalance_results
            )
            
            logger.info("\n" + "="*60)
            logger.info("BASELINE MODEL TRAINING COMPLETED")
            logger.info("="*60)
            logger.info("\nNext steps:")
            logger.info("- Phase 6: Advanced model training with hyperparameter tuning")
            logger.info("- Phase 7: Model evaluation and comparison")
            logger.info("- Phase 8: Feature importance and model interpretation")
        
        if args.phase not in ['1-2', '3-4', '5', 'all']:
            logger.error(f"Unknown phase: {args.phase}")
            logger.info("Valid options: '1-2', '3-4', '5', 'all'")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()