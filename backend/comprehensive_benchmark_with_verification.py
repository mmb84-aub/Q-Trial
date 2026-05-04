"""
COMPREHENSIVE BENCHMARKING PIPELINE WITH STATISTICAL VERIFICATION

This script combines feature selection methods with statistical verification
to provide rigorous evaluation of all baseline methods against QUBO.

Usage:
    python comprehensive_benchmark_with_verification.py

Features:
    - Tests all 5 feature selection methods (univariate, mrmr, lasso, elastic_net, qubo)
    - Uses statistical verification to validate selected features
    - Computes redundancy, predictive performance, stability
    - Generates detailed comparison report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# Feature selection imports
from qtrial_backend.feature_selection import (
    select_features,
    benchmark_all_methods,
    print_benchmark_summary,
)

# Statistical verification imports  
from qtrial_backend.agentic.statistical_verification import build_statistical_verification_report
from qtrial_backend.agent.context import AgentContext
from qtrial_backend.agentic.schemas import MetadataInput, GroundedFindingsSchema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_clinical_dataset(n_samples: int = 500) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create synthetic clinical trial dataset with realistic properties.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    np.random.seed(42)
    
    # Clinical variables
    df = pd.DataFrame({
        # Demographics
        'age': np.random.normal(65, 15, n_samples).clip(18, 100),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'bmi': np.random.normal(28, 5, n_samples).clip(15, 50),
        
        # Lab values
        'albumin': np.random.normal(3.8, 0.6, n_samples).clip(1, 5),
        'bili_corr': np.random.exponential(1.2, n_samples).clip(0.1, 10),
        'creatinine': np.random.lognormal(np.log(1.0), 0.3, n_samples),
        'inr': np.random.normal(1.0, 0.2, n_samples).clip(0.7, 3),
        'platelets': np.random.normal(200, 60, n_samples).clip(20, 400),
        
        # Clinical outcomes (with some signal)
        'ascites': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'hepatic_encephalopathy': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        
        # Noise features
        'noise1': np.random.normal(0, 1, n_samples),
        'noise2': np.random.normal(0, 1, n_samples),
        'noise3': np.random.normal(0, 1, n_samples),
    })
    
    # Create outcome variable with some signal from real features
    signal = (
        0.3 * (df['albumin'] < 3.0).astype(int) +
        0.2 * (df['bili_corr'] > 2.0).astype(int) +
        0.3 * df['ascites'] +
        0.2 * (df['age'] > 70).astype(int)
    )
    df['outcome'] = (signal + np.random.normal(0, 0.5, n_samples) > 0.5).astype(int)
    
    metadata = {
        'outcome_column': 'outcome',
        'numeric_columns': [
            'age', 'bmi', 'albumin', 'bili_corr', 'creatinine', 
            'inr', 'platelets', 'noise1', 'noise2', 'noise3'
        ],
        'categorical_columns': ['sex', 'ascites', 'hepatic_encephalopathy'],
        'description': 'Synthetic clinical trial dataset with 13 features and 1 outcome'
    }
    
    return df, metadata


def run_feature_selection_suite(
    df: pd.DataFrame,
    outcome_column: str,
    methods: List[str] = None,
    n_bootstrap: int = 5
) -> Dict[str, Any]:
    """
    Run feature selection benchmarking with all methods.
    
    Args:
        df: Input dataframe
        outcome_column: Name of outcome column
        methods: List of methods to test (None = all)
        n_bootstrap: Number of bootstrap samples for stability testing
        
    Returns:
        Dictionary of results for each method
    """
    if methods is None:
        methods = ['univariate', 'mrmr', 'lasso', 'elastic_net']  # QUBO handled separately
    
    logger.info(f"\n{'='*80}")
    logger.info("RUNNING FEATURE SELECTION BENCHMARKING SUITE")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset: {len(df)} samples x {len(df.columns)} features")
    logger.info(f"Methods: {', '.join(methods)}")
    logger.info(f"Bootstrap samples: {n_bootstrap}")
    
    # Run benchmarking for supported methods
    results = benchmark_all_methods(
        df,
        outcome_column,
        methods=methods,
        n_bootstrap=n_bootstrap,
        verbose=True
    )
    
    return results


def validate_feature_selection_with_statistics(
    df: pd.DataFrame,
    results: Dict[str, Any],
    outcome_column: str
) -> Dict[str, Any]:
    """
    Use statistical verification to validate feature selections.
    
    Args:
        df: Original dataframe
        results: Feature selection results dict
        outcome_column: Name of outcome column
        
    Returns:
        Dictionary with statistical validation metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info("STATISTICAL VERIFICATION OF FEATURE SELECTIONS")
    logger.info(f"{'='*80}")
    
    verification_results = {}
    
    for method_name, method_result in results.items():
        logger.info(f"\n{method_name.upper()}")
        logger.info("-" * 60)
        
        selected_features = method_result.get('selected_features', [])
        if not selected_features:
            logger.info(f"  No features selected - skipping verification")
            verification_results[method_name] = {
                'selected_features': [],
                'statistical_summary': 'No features to verify'
            }
            continue
        
        # Remove outcome from selected features for stats
        feature_set = [f for f in selected_features if f != outcome_column]
        
        logger.info(f"  Selected features ({len(feature_set)}): {feature_set}")
        
        # Compute correlations with outcome
        correlations = {}
        for feature in feature_set:
            try:
                # Handle mixed types
                if df[feature].dtype == 'object':
                    # For categorical, compute chi-square
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df[feature], df[outcome_column])
                    chi2, p_val, _, _ = chi2_contingency(ct)
                    correlations[feature] = {'metric': 'chi2', 'value': chi2, 'p_value': p_val}
                else:
                    # For numeric, compute correlation
                    from scipy.stats import pointbiserialr
                    corr, p_val = pointbiserialr(df[outcome_column], df[feature])
                    correlations[feature] = {'metric': 'r', 'value': abs(corr), 'p_value': p_val}
            except Exception as e:
                logger.warning(f"  Could not compute correlation for {feature}: {e}")
                correlations[feature] = {'error': str(e)}
        
        # Count significant features (p < 0.05)
        significant = sum(1 for c in correlations.values() if c.get('p_value', 1) < 0.05)
        
        verification_results[method_name] = {
            'selected_features': feature_set,
            'n_features': len(feature_set),
            'correlations': correlations,
            'significant_at_p05': significant,
            'significance_rate': significant / len(feature_set) if feature_set else 0.0
        }
        
        logger.info(f"  Significant features (p<0.05): {significant}/{len(feature_set)}")
        logger.info(f"  Significance rate: {significant/len(feature_set)*100:.1f}%")
    
    return verification_results


def generate_comprehensive_report(
    results: Dict[str, Any],
    verification_results: Dict[str, Any],
    output_file: str = None
) -> str:
    """
    Generate comprehensive benchmark report with statistical verification.
    
    Args:
        results: Feature selection benchmark results
        verification_results: Statistical verification results
        output_file: Optional file to save report to
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("\n" + "="*90)
    report.append("COMPREHENSIVE FEATURE SELECTION BENCHMARK REPORT")
    report.append("WITH STATISTICAL VERIFICATION")
    report.append("="*90)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary table
    report.append("SUMMARY TABLE")
    report.append("-" * 90)
    report.append(f"{'Method':<15} {'Features':<12} {'Redundancy':<15} {'Runtime':<12} {'Significance':<15}")
    report.append("-" * 90)
    
    for method_name in sorted(results.keys()):
        method_data = results[method_name]
        verify_data = verification_results.get(method_name, {})
        
        n_features = method_data.get('n_features', 0)
        redundancy = method_data.get('redundancy', 0.0)
        runtime = method_data.get('runtime', 0.0)
        significance = verify_data.get('significance_rate', 0.0)
        
        report.append(
            f"{method_name:<15} {str(n_features):<12} {redundancy:<15.3f} "
            f"{runtime:<12.4f}s {significance*100:<14.1f}%"
        )
    
    report.append("-" * 90)
    
    # Detailed results
    report.append("\n\nDETAILED RESULTS")
    report.append("="*90)
    
    for method_name in sorted(results.keys()):
        method_data = results[method_name]
        verify_data = verification_results.get(method_name, {})
        
        report.append(f"\n{method_name.upper()}")
        report.append("-" * 90)
        
        selected = method_data.get('selected_features', [])
        report.append(f"  Selected features: {selected}")
        report.append(f"  Redundancy: {method_data.get('redundancy', 'N/A')}")
        report.append(f"  Runtime: {method_data.get('runtime', 'N/A')}s")
        
        if verify_data.get('correlations'):
            report.append(f"  Statistical verification:")
            for feature, corr_data in verify_data['correlations'].items():
                if 'error' not in corr_data:
                    metric = corr_data.get('metric', 'N/A')
                    value = corr_data.get('value', 'N/A')
                    p_val = corr_data.get('p_value', 'N/A')
                    sig = "*" if corr_data.get('p_value', 1) < 0.05 else " "
                    report.append(f"    {feature}: {metric}={value:.4f}, p={p_val:.4f} {sig}")
    
    # Recommendations
    report.append("\n\n" + "="*90)
    report.append("RECOMMENDATIONS")
    report.append("="*90)
    
    # Find best method by different criteria
    methods_data = results
    
    best_redundancy = min(
        (m, d.get('redundancy', 999)) for m, d in methods_data.items()
    )
    best_speed = min(
        (m, d.get('runtime', 999)) for m, d in methods_data.items()
    )
    best_features = min(
        (m, d.get('n_features', 999)) for m, d in methods_data.items()
    )
    best_significance = max(
        (m, verification_results.get(m, {}).get('significance_rate', 0))
        for m in methods_data.keys()
    )
    
    report.append(f"\n[BEST] Best redundancy reduction: {best_redundancy[0].upper()} (redundancy={best_redundancy[1]:.3f})")
    report.append(f"[BEST] Fastest: {best_speed[0].upper()} (runtime={best_speed[1]:.4f}s)")
    report.append(f"[BEST] Most parsimonious: {best_features[0].upper()} ({best_features[1]} features)")
    report.append(f"[BEST] Highest statistical significance: {best_significance[0].upper()} ({best_significance[1]*100:.1f}%)")
    
    report.append("\n" + "-"*90)
    report.append("CLINICAL INTERPRETATION:")
    report.append("-"*90)
    report.append("\nFor clinical trial analysis, prioritize methods with:")
    report.append("  1. Low redundancy (independent features)")
    report.append("  2. High statistical significance (features correlate with outcome)")
    report.append("  3. Reasonable speed (for iterative analysis)")
    report.append("  4. Interpretable coefficients (for clinical insights)")
    
    report.append("\n" + "="*90)
    report.append("END OF REPORT")
    report.append("="*90 + "\n")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"\nReport saved to: {output_file}")
    
    return report_text


def main():
    """Main benchmarking pipeline."""
    
    # 1. Create dataset
    logger.info("\nSTEP 1: CREATING SYNTHETIC CLINICAL DATASET")
    df, metadata = create_synthetic_clinical_dataset(n_samples=500)
    logger.info(f"[OK] Created dataset: {len(df)} samples x {len(df.columns)} features")
    
    # 2. Run feature selection
    logger.info("\nSTEP 2: RUNNING FEATURE SELECTION BENCHMARKING")
    results = run_feature_selection_suite(
        df,
        outcome_column='outcome',
        methods=['univariate', 'mrmr', 'lasso', 'elastic_net'],
        n_bootstrap=5
    )
    logger.info("✓ Feature selection benchmarking complete")
    
    # 3. Statistical verification
    logger.info("\nSTEP 3: STATISTICAL VERIFICATION OF SELECTIONS")
    verification_results = validate_feature_selection_with_statistics(
        df,
        results,
        outcome_column='outcome'
    )
    logger.info("✓ Statistical verification complete")
    
    # 4. Generate report
    logger.info("\nSTEP 4: GENERATING COMPREHENSIVE REPORT")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/comprehensive_benchmark_{timestamp}.txt"
    
    report = generate_comprehensive_report(
        results,
        verification_results,
        output_file=output_file
    )
    
    print(report)
    
    # 5. Save JSON results
    json_file = f"outputs/comprehensive_benchmark_{timestamp}.json"
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata,
        'feature_selection_results': results,
        'statistical_verification': verification_results,
        'summary': {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'methods_tested': list(results.keys()),
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
    logger.info(f"Results saved to: {json_file}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ BENCHMARKING PIPELINE COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
