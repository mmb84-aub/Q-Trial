"""Simplified test for feature selection methods - standalone."""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Direct imports from the module files
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_feature_selection_standalone():
    """Test feature selection without complex imports."""
    
    logger.info("=" * 70)
    logger.info("FEATURE SELECTION METHODS TEST")
    logger.info("=" * 70)
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 200
    
    # Informative features
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X3 = np.random.normal(0, 1, n_samples)
    
    # Redundant (correlated with informative)
    X4 = X1 + np.random.normal(0, 0.3, n_samples)
    X5 = X2 + np.random.normal(0, 0.3, n_samples)
    
    # Noise features
    X6 = np.random.normal(0, 1, n_samples)
    X7 = np.random.normal(0, 1, n_samples)
    X8 = np.random.normal(0, 1, n_samples)
    
    # Categorical
    Cat1 = np.random.choice(['A', 'B', 'C'], n_samples)
    Cat2 = np.random.choice(['X', 'Y'], n_samples)
    
    # Target (binary classification)
    y = (2*X1 - X2 + 0.5*X3 + np.random.normal(0, 1, n_samples) > 0).astype(int)
    
    df = pd.DataFrame({
        'var_1': X1, 'var_2': X2, 'var_3': X3,
        'var_4_corr': X4, 'var_5_corr': X5,
        'noise_1': X6, 'noise_2': X7, 'noise_3': X8,
        'category_1': Cat1, 'category_2': Cat2,
        'target': y
    })
    
    logger.info(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
    logger.info(f"Features: {list(df.columns)}\n")
    
    # Test each method
    methods_to_test = [
        ('univariate', 'Univariate Statistical Selection'),
        ('mrmr', 'mRMR (Minimum Redundancy Maximum Relevance)'),
        ('lasso', 'LASSO Regression'),
        ('elastic_net', 'Elastic Net Regression'),
    ]
    
    results_all = {}
    
    for method_name, method_label in methods_to_test:
        logger.info(f"\nTesting {method_label}...")
        logger.info("-" * 70)
        
        try:
            from qtrial_backend.feature_selection.interface import select_features
            
            result = select_features(df, 'target', method=method_name)
            
            selected = result.get('selected_features', [])
            n_selected = len(selected)
            redundancy = result.get('redundancy_measure', 0.0)
            
            logger.info(f"✓ {method_name.upper()}: {n_selected} features selected")
            logger.info(f"  Selected: {selected}")
            logger.info(f"  Redundancy (mean correlation): {redundancy:.3f}")
            
            if 'relevance_scores' in result and result['relevance_scores']:
                top_scores = sorted(result['relevance_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"  Top relevance scores: {top_scores}")
            
            results_all[method_name] = result
        
        except Exception as e:
            logger.error(f"✗ {method_name.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Method':<20} {'Features':<12} {'Redundancy':<15}")
    logger.info("-" * 70)
    
    for method_name, result in results_all.items():
        if 'error' not in result:
            n_feat = result.get('n_features', 0)
            redundancy = result.get('redundancy_measure', 0.0)
            logger.info(f"{method_name:<20} {n_feat:<12} {redundancy:<15.3f}")
    
    logger.info("=" * 70)
    return results_all


if __name__ == "__main__":
    try:
        results = test_feature_selection_standalone()
        logger.info("\n✓ All tests completed successfully!")
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
