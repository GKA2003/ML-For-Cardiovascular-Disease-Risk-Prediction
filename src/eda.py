"""
Exploratory Data Analysis (EDA) module for Heart Disease ML Pipeline
Provides comprehensive statistical and visual analysis of the dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from pathlib import Path

from src.config import (
    FIGURES_DIR, TABLES_DIR, TARGET_COLUMN,
    FIGURE_SIZE, COLOUR_PALETTE
)
from src.utils import save_figure, Timer, logger
from src.data_preprocessing import DataPreprocessor

# Get module logger
module_logger = logging.getLogger(__name__)

class ExploratoryDataAnalyzer:
    """
    Comprehensive EDA class for cardiovascular dataset analysis.
    Generates statistical summaries, visualisations, and insights.
    """
    
    def __init__(self, preprocessor: DataPreprocessor):
        """
        Initialise the EDA analyzer.
        
        Args:
            preprocessor: DataPreprocessor instance with loaded data
        """
        self.preprocessor = preprocessor
        self.train_data = preprocessor.train_data
        self.test_data = preprocessor.test_data
        self.feature_names = preprocessor.feature_names
        self.categorical_features = preprocessor.categorical_features
        self.numerical_features = preprocessor.numerical_features
        self.eda_results = {}
        
    def perform_complete_eda(self) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis.
        
        Returns:
            Dictionary containing all EDA results
        """
        module_logger.info("Starting comprehensive EDA...")
        
        with Timer("Complete EDA"):
            # 1. Basic statistics
            self.eda_results['basic_stats'] = self._compute_basic_statistics()
            
            # 2. Missing value analysis
            self.eda_results['missing_analysis'] = self._analyze_missing_patterns()
            
            # 3. Target variable analysis
            self.eda_results['target_analysis'] = self._analyze_target_variable()
            
            # 4. Feature distributions
            self.eda_results['distributions'] = self._analyze_feature_distributions()
            
            # 5. Correlation analysis
            self.eda_results['correlations'] = self._analyze_correlations()
            
            # 6. Bivariate analysis
            self.eda_results['bivariate'] = self._perform_bivariate_analysis()
            
            # 7. Statistical tests
            self.eda_results['statistical_tests'] = self._perform_statistical_tests()
            
            # 8. Outlier analysis
            self.eda_results['outliers'] = self._analyze_outliers()
            
        module_logger.info("EDA completed successfully")
        return self.eda_results
    
    def _compute_basic_statistics(self) -> Dict[str, pd.DataFrame]:
        """Compute basic statistical summaries."""
        module_logger.info("Computing basic statistics...")
        
        stats_dict = {}
        
        # Overall dataset info
        dataset_info = pd.DataFrame({
            'Dataset': ['Training', 'Test'],
            'Samples': [len(self.train_data), len(self.test_data)],
            'Features': [len(self.feature_names), len(self.feature_names)],
            'Numerical': [len(self.numerical_features), len(self.numerical_features)],
            'Categorical': [len(self.categorical_features), len(self.categorical_features)]
        })
        stats_dict['dataset_info'] = dataset_info
        
        # Numerical features statistics
        if self.numerical_features:
            train_num_stats = self.train_data[self.numerical_features].describe()
            train_num_stats.loc['skewness'] = self.train_data[self.numerical_features].skew()
            train_num_stats.loc['kurtosis'] = self.train_data[self.numerical_features].kurtosis()
            stats_dict['train_numerical'] = train_num_stats
            
            test_num_stats = self.test_data[self.numerical_features].describe()
            test_num_stats.loc['skewness'] = self.test_data[self.numerical_features].skew()
            test_num_stats.loc['kurtosis'] = self.test_data[self.numerical_features].kurtosis()
            stats_dict['test_numerical'] = test_num_stats
        
        # Categorical features statistics
        if self.categorical_features:
            cat_stats = []
            for col in self.categorical_features:
                cat_stats.append({
                    'feature': col,
                    'unique_train': self.train_data[col].nunique(),
                    'unique_test': self.test_data[col].nunique(),
                    'mode_train': self.train_data[col].mode()[0] if not self.train_data[col].mode().empty else None,
                    'mode_test': self.test_data[col].mode()[0] if not self.test_data[col].mode().empty else None
                })
            stats_dict['categorical'] = pd.DataFrame(cat_stats)
        
        return stats_dict
    
    def _analyze_missing_patterns(self) -> Dict[str, Any]:
        """Analyse patterns in missing data."""
        module_logger.info("Analysing missing value patterns...")
        
        missing_results = {}
        
        # Missing value heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training data
        train_missing = self.train_data.isnull()
        if train_missing.any().any():
            sns.heatmap(train_missing, cbar=True, yticklabels=False, 
                       cmap='viridis', ax=ax1)
            ax1.set_title('Missing Values - Training Data')
        else:
            ax1.text(0.5, 0.5, 'No missing values', ha='center', va='center')
            ax1.set_title('Missing Values - Training Data')
        
        # Test data
        test_missing = self.test_data.isnull()
        if test_missing.any().any():
            sns.heatmap(test_missing, cbar=True, yticklabels=False, 
                       cmap='viridis', ax=ax2)
            ax2.set_title('Missing Values - Test Data')
        else:
            ax2.text(0.5, 0.5, 'No missing values', ha='center', va='center')
            ax2.set_title('Missing Values - Test Data')
        
        plt.tight_layout()
        save_figure(fig, 'missing_values_heatmap', 'eda')
        
        # Missing value statistics
        train_missing_stats = pd.DataFrame({
            'feature': self.train_data.columns,
            'missing_count': self.train_data.isnull().sum(),
            'missing_pct': (self.train_data.isnull().sum() / len(self.train_data)) * 100
        })
        train_missing_stats = train_missing_stats[train_missing_stats['missing_count'] > 0]
        missing_results['train_stats'] = train_missing_stats
        
        test_missing_stats = pd.DataFrame({
            'feature': self.test_data.columns,
            'missing_count': self.test_data.isnull().sum(),
            'missing_pct': (self.test_data.isnull().sum() / len(self.test_data)) * 100
        })
        test_missing_stats = test_missing_stats[test_missing_stats['missing_count'] > 0]
        missing_results['test_stats'] = test_missing_stats
        
        return missing_results
    
    def _analyze_target_variable(self) -> Dict[str, Any]:
        """Analyse the target variable distribution and class balance."""
        module_logger.info("Analysing target variable...")
        
        target_results = {}
        
        # Target distribution
        target_counts = self.train_data[TARGET_COLUMN].value_counts()
        target_pct = self.train_data[TARGET_COLUMN].value_counts(normalize=True) * 100
        
        target_results['distribution'] = pd.DataFrame({
            'class': target_counts.index,
            'count': target_counts.values,
            'percentage': target_pct.values
        })
        
        # Class imbalance ratio
        imbalance_ratio = target_counts.max() / target_counts.min()
        target_results['imbalance_ratio'] = imbalance_ratio
        
        # Visualisation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot
        target_counts.plot(kind='bar', ax=ax1, color=sns.color_palette('husl', 2))
        ax1.set_title('Target Variable Distribution')
        ax1.set_xlabel(TARGET_COLUMN)
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # Pie chart
        target_pct.plot(kind='pie', ax=ax2, autopct='%1.1f%%', 
                       colors=sns.color_palette('husl', 2))
        ax2.set_ylabel('')
        ax2.set_title('Target Variable Percentage')
        
        plt.tight_layout()
        save_figure(fig, 'target_distribution', 'eda')
        
        module_logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        return target_results
    
    def _analyze_feature_distributions(self) -> Dict[str, plt.Figure]:
        """Analyse and visualise feature distributions."""
        module_logger.info("Analysing feature distributions...")
        
        distribution_figs = {}
        
        # Numerical features
        if self.numerical_features:
            n_numerical = len(self.numerical_features)
            n_cols = 3
            n_rows = (n_numerical + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(self.numerical_features):
                ax = axes[idx]
                
                # Histogram with KDE
                self.train_data[col].hist(bins=30, ax=ax, alpha=0.6, 
                                         label='Train', density=True)
                self.train_data[col].plot(kind='density', ax=ax, label='Train KDE')
                
                # Add test data if available
                if col in self.test_data.columns:
                    self.test_data[col].hist(bins=30, ax=ax, alpha=0.6, 
                                            label='Test', density=True)
                    self.test_data[col].plot(kind='density', ax=ax, 
                                           label='Test KDE', linestyle='--')
                
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                ax.legend()
            
            # Hide unused subplots
            for idx in range(len(self.numerical_features), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            save_figure(fig, 'numerical_distributions', 'eda')
            distribution_figs['numerical'] = fig
        
        # Categorical features
        if self.categorical_features:
            n_categorical = len(self.categorical_features)
            n_cols = 2
            n_rows = (n_categorical + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, col in enumerate(self.categorical_features):
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
                
                # Count plot
                train_counts = self.train_data[col].value_counts()
                test_counts = self.test_data[col].value_counts() if col in self.test_data.columns else None
                
                x_labels = train_counts.index
                x_pos = np.arange(len(x_labels))
                
                ax.bar(x_pos - 0.2, train_counts.values, 0.4, 
                      label='Train', alpha=0.8)
                
                if test_counts is not None:
                    # Align test counts with train labels
                    test_values = [test_counts.get(label, 0) for label in x_labels]
                    ax.bar(x_pos + 0.2, test_values, 0.4, 
                          label='Test', alpha=0.8)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, rotation=45)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.legend()
            
            # Hide unused subplots
            total_subplots = n_rows * n_cols
            for idx in range(len(self.categorical_features), total_subplots):
                row = idx // n_cols
                col_idx = idx % n_cols
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            save_figure(fig, 'categorical_distributions', 'eda')
            distribution_figs['categorical'] = fig
        
        return distribution_figs
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyse feature correlations."""
        module_logger.info("Analysing correlations...")
        
        correlation_results = {}
        
        # Compute correlation matrix for numerical features
        if self.numerical_features:
            # Include target variable if numerical
            corr_features = self.numerical_features + [TARGET_COLUMN]
            corr_matrix = self.train_data[corr_features].corr()
            correlation_results['matrix'] = corr_matrix
            
            # Correlation heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap='coolwarm', center=0, square=True,
                       linewidths=1, cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            save_figure(plt.gcf(), 'correlation_heatmap', 'eda')
            
            # Top correlations with target
            target_corr = corr_matrix[TARGET_COLUMN].drop(TARGET_COLUMN)
            target_corr_abs = target_corr.abs().sort_values(ascending=False)
            
            correlation_results['target_correlations'] = pd.DataFrame({
                'feature': target_corr_abs.index,
                'correlation': target_corr[target_corr_abs.index].values,
                'abs_correlation': target_corr_abs.values
            })
            
            # Highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:  # Threshold for high correlation
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                correlation_results['high_correlations'] = pd.DataFrame(high_corr_pairs)
                module_logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        return correlation_results
    
    def _perform_bivariate_analysis(self) -> Dict[str, plt.Figure]:
        """Perform bivariate analysis between features and target."""
        module_logger.info("Performing bivariate analysis...")
        
        bivariate_figs = {}
        
        # Numerical features vs target
        if self.numerical_features:
            n_features = len(self.numerical_features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(self.numerical_features):
                ax = axes[idx]
                
                # Box plot by target class
                self.train_data.boxplot(column=col, by=TARGET_COLUMN, ax=ax)
                ax.set_title(f'{col} by {TARGET_COLUMN}')
                ax.set_xlabel(TARGET_COLUMN)
                ax.set_ylabel(col)
                plt.suptitle('')  # Remove default title
            
            # Hide unused subplots
            for idx in range(len(self.numerical_features), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            save_figure(fig, 'numerical_vs_target', 'eda')
            bivariate_figs['numerical_vs_target'] = fig
        
        # Categorical features vs target
        if self.categorical_features:
            n_features = len(self.categorical_features)
            n_cols = 2
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, col in enumerate(self.categorical_features):
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
                
                # Stacked bar plot
                crosstab = pd.crosstab(self.train_data[col], 
                                      self.train_data[TARGET_COLUMN], 
                                      normalize='index')
                crosstab.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title(f'{col} vs {TARGET_COLUMN} (Normalised)')
                ax.set_xlabel(col)
                ax.set_ylabel('Proportion')
                ax.legend(title=TARGET_COLUMN)
                ax.tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            total_subplots = n_rows * n_cols
            for idx in range(len(self.categorical_features), total_subplots):
                row = idx // n_cols
                col_idx = idx % n_cols
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            save_figure(fig, 'categorical_vs_target', 'eda')
            bivariate_figs['categorical_vs_target'] = fig
        
        return bivariate_figs
    
    def _perform_statistical_tests(self) -> Dict[str, pd.DataFrame]:
        """Perform statistical tests with effect size reporting."""
        module_logger.info("Performing statistical tests with effect size reporting...")
        
        # Helper function for Cliff's Delta
        def cliffs_delta(x, y):
            """Calculate Cliff's delta effect size."""
            nx, ny = len(x), len(y)
            wins = 0
            for i in x:
                for j in y:
                    if i > j:
                        wins += 1
                    elif i < j:
                        wins -= 1
            return wins / (nx * ny)
        
        # Helper function for Cramér's V
        def cramers_v(contingency_table):
            """Calculate Cramér's V effect size."""
            chi2 = stats.chi2_contingency(contingency_table)[0]
            n = contingency_table.sum().sum()
            phi2 = chi2 / n
            r, k = contingency_table.shape
            return np.sqrt(phi2 / min(k-1, r-1))
        
        test_results = {}
        
        # Tests for numerical features
        if self.numerical_features:
            numerical_tests = []
            
            for col in self.numerical_features:
                # ensure data is numeric
                if not pd.api.types.is_numeric_dtype(self.train_data[col]):
                    module_logger.warning(f"Skipping non-numeric feature: {col}")
                    continue

                # Separate by target class
                class_0 = self.train_data[self.train_data[TARGET_COLUMN] == 0][col].dropna()
                class_1 = self.train_data[self.train_data[TARGET_COLUMN] == 1][col].dropna()

                # Skip if not enough data
                if len(class_0) < 3 or len(class_1) < 3:
                    module_logger.warning(f"Insufficient data for '{col}' statistical test")
                    continue
                
                # Normality test (Shapiro-Wilk)
                _, p_normal_0 = stats.shapiro(class_0) if len(class_0) > 3 else (None, None)
                _, p_normal_1 = stats.shapiro(class_1) if len(class_1) > 3 else (None, None)
                
                # Choose appropriate test based on normality
                if p_normal_0 and p_normal_1 and p_normal_0 > 0.05 and p_normal_1 > 0.05:
                    # Use t-test for normal distributions
                    stat, p_value = stats.ttest_ind(class_0, class_1)
                    test_type = 't-test'
                else:
                    # Use Mann-Whitney U test for non-normal distributions
                    stat, p_value = stats.mannwhitneyu(class_0, class_1)
                    test_type = 'Mann-Whitney U'
                
                # Calculate Cliff's Delta effect size
                d_val = cliffs_delta(class_0, class_1)
                
                # Determine effect magnitude
                effect_magnitude = 'negligible'
                if abs(d_val) > 0.474:
                    effect_magnitude = 'large'
                elif abs(d_val) > 0.33:
                    effect_magnitude = 'medium'
                elif abs(d_val) > 0.147:
                    effect_magnitude = 'small'
                
                numerical_tests.append({
                    'feature': col,
                    'test': test_type,
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05 if p_value else None,
                    'effect_size': d_val,
                    'effect_magnitude': effect_magnitude
                })
            
            test_results['numerical'] = pd.DataFrame(numerical_tests)
        
        # Tests for categorical features
        if self.categorical_features:
            categorical_tests = []

            for col in self.categorical_features:
                # Skip high cardinality features (we've binned cigsPerDay)
                if self.train_data[col].nunique() > 20:
                    module_logger.warning(f"Skipping high cardinality feature '{col}' in categorical tests")
                    continue

                # Chi-square test
                crosstab = pd.crosstab(self.train_data[col], 
                                    self.train_data[TARGET_COLUMN])
                
                # Skip if any expected frequencies < 5
                min_expected = stats.chi2_contingency(crosstab)[3].min()
                if min_expected < 5:
                    module_logger.warning(f"Low expected frequencies for '{col}' - consider Fisher's exact test")
                
                chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)
                
                # Calculate Cramér's V effect size
                v_val = cramers_v(crosstab)
                
                # Determine effect magnitude
                effect_magnitude = 'negligible'
                if v_val > 0.5:
                    effect_magnitude = 'large'
                elif v_val > 0.3:
                    effect_magnitude = 'medium'
                elif v_val > 0.1:
                    effect_magnitude = 'small'
                
                categorical_tests.append({
                    'feature': col,
                    'test': 'Chi-square',
                    'statistic': chi2,
                    'p_value': p_value,
                    'dof': dof,
                    'significant': p_value < 0.05,
                    'effect_size': v_val,
                    'effect_magnitude': effect_magnitude
                })
            
            test_results['categorical'] = pd.DataFrame(categorical_tests)
        
        return test_results
    
    def _analyze_outliers(self) -> Dict[str, Any]:
        """Analyse outliers in numerical features. Skips identifiers."""
        module_logger.info("Analysing outliers...")
        
        outlier_results = {}

        # Filter out identifier columns
        valid_numerical = [col for col in self.numerical_features 
                            if col not in ['id'] and 
                            col in self.train_data.columns]
        
        if valid_numerical:
            # IQR method
            outlier_counts = {}
            outlier_indices = {}
            X_numerical = self.train_data[valid_numerical]
            
            for col in valid_numerical:
                Q1 = self.train_data[col].quantile(0.25)
                Q3 = self.train_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.train_data[
                    (self.train_data[col] < lower_bound) | 
                    (self.train_data[col] > upper_bound)
                ]
                
                outlier_counts[col] = len(outliers)
                outlier_indices[col] = outliers.index.tolist()
            
            outlier_results['counts'] = outlier_counts
            outlier_results['total_samples_with_outliers'] = len(
                set().union(*outlier_indices.values()) if outlier_indices else set()
            )
            
            # Visualise outliers
            n_features = len(self.numerical_features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(valid_numerical):
                ax = axes[idx]
                self.train_data.boxplot(column=col, ax=ax)
                ax.set_title(f'{col} - Outliers: {outlier_counts[col]}')
                ax.set_ylabel(col)
            
            # Hide unused subplots
            for idx in range(len(self.numerical_features), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            save_figure(fig, 'outlier_boxplots', 'eda')
            outlier_results['boxplot'] = fig
        
        return outlier_results
    
    def generate_eda_report(self) -> None:
        """Generate a comprehensive EDA report."""
        module_logger.info("Generating EDA report...")
        
        report_path = TABLES_DIR / f"eda_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("Heart Disease Prediction Dataset\n")
            f.write("="*80 + "\n\n")
            
            # Dataset Overview
            f.write("1. DATASET OVERVIEW\n")
            f.write("-"*40 + "\n")
            if 'basic_stats' in self.eda_results:
                f.write(self.eda_results['basic_stats']['dataset_info'].to_string())
                f.write("\n\n")
            
            # Target Variable Analysis
            f.write("2. TARGET VARIABLE ANALYSIS\n")
            f.write("-"*40 + "\n")
            if 'target_analysis' in self.eda_results:
                f.write(self.eda_results['target_analysis']['distribution'].to_string())
                f.write(f"\nImbalance Ratio: {self.eda_results['target_analysis']['imbalance_ratio']:.2f}")
                f.write("\n\n")
            
            # Feature Correlations
            f.write("3. TOP FEATURE CORRELATIONS WITH TARGET\n")
            f.write("-"*40 + "\n")
            if 'correlations' in self.eda_results and 'target_correlations' in self.eda_results['correlations']:
                top_corr = self.eda_results['correlations']['target_correlations'].head(10)
                f.write(top_corr.to_string())
                f.write("\n\n")
            
            # Statistical Tests
            f.write("4. STATISTICAL SIGNIFICANCE TESTS\n")
            f.write("-"*40 + "\n")
            if 'statistical_tests' in self.eda_results:
                if 'numerical' in self.eda_results['statistical_tests']:
                    f.write("Numerical Features:\n")
                    f.write(self.eda_results['statistical_tests']['numerical'].to_string())
                    f.write("\n\n")
                if 'categorical' in self.eda_results['statistical_tests']:
                    f.write("Categorical Features:\n")
                    f.write(self.eda_results['statistical_tests']['categorical'].to_string())
                    f.write("\n\n")
            
            # Outlier Analysis
            f.write("5. OUTLIER ANALYSIS\n")
            f.write("-"*40 + "\n")
            if 'outliers' in self.eda_results:
                outlier_df = pd.DataFrame([
                    {'feature': k, 'outlier_count': v} 
                    for k, v in self.eda_results['outliers']['counts'].items()
                ])
                f.write(outlier_df.to_string())
                f.write(f"\n\nTotal samples with outliers: {self.eda_results['outliers']['total_samples_with_outliers']}")
                f.write("\n\n")
        
        module_logger.info(f"EDA report saved to {report_path}")

if __name__ == "__main__":
    # Test the EDA pipeline
    module_logger.info("Testing EDA pipeline...")
    
    # First run preprocessing
    from data_preprocessing import quick_preprocess
    train_df, test_df, preprocessor = quick_preprocess()
    
    # Create EDA analyzer
    eda_analyzer = ExploratoryDataAnalyzer(preprocessor)
    
    # Perform complete EDA
    eda_results = eda_analyzer.perform_complete_eda()
    
    # Generate report
    eda_analyzer.generate_eda_report()
    
    module_logger.info("EDA pipeline test completed!")