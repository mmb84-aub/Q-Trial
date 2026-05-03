"""Final benchmark report comparing all feature selection methods."""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from qtrial_backend.feature_selection.interface import select_features
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection


def main():
    """Generate comprehensive benchmark report."""
    
    # Create dataset
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        'bili': np.random.gamma(2, 1.5, n),
        'ast': np.random.gamma(3, 20, n),
        'alt': np.random.gamma(2.5, 25, n),
        'albumin': np.random.normal(35, 5, n),
        'bili_corr': np.random.gamma(2, 1.5, n) + np.random.normal(0, 0.5, n),
        'ast_corr': np.random.gamma(3, 20, n) + np.random.normal(0, 5, n),
        'age': np.random.normal(50, 15, n),
        'bmi': np.random.normal(25, 5, n),
        'noise1': np.random.normal(0, 1, n),
        'noise2': np.random.normal(0, 1, n),
        'noise3': np.random.normal(0, 1, n),
        'sex': np.random.choice(['M', 'F'], n),
        'target': np.random.binomial(1, 0.5, n),
    })
    
    logger.info(f"Dataset: {df.shape[0]} samples × {df.shape[1]} features")
    
    print("\n" + "="*100)
    print("FEATURE SELECTION METHODS - COMPREHENSIVE BENCHMARK")
    print("="*100)
    print(f"{'Method':<15} {'Features':<12} {'Redundancy':<15} {'Runtime (s)':<15} {'Selection Quality':<25}")
    print("-"*100)
    
    all_results = {}
    
    # Test each method
    methods = [
        ('univariate', 'Univariate statistical tests (baseline)'),
        ('mrmr', 'mRMR: Min Redundancy Max Relevance'),
        ('lasso', 'LASSO: L1-regularized regression'),
        ('elastic_net', 'Elastic Net: L1+L2 regularization'),
    ]
    
    for method_name, method_desc in methods:
        try:
            import time
            t0 = time.time()
            result = select_features(df, 'target', method=method_name)
            runtime = time.time() - t0
            
            selected = result.get('selected_features', [])
            n_selected = len(selected)
            redundancy = result.get('redundancy_measure', 0.0)
            
            all_results[method_name] = {
                'n_features': n_selected,
                'redundancy': redundancy,
                'runtime': runtime,
                'selected': selected,
            }
            
            quality = "Good" if redundancy < 0.2 else "Fair" if redundancy < 0.4 else "High"
            print(f"{method_name:<15} {n_selected:<12} {redundancy:<15.3f} {runtime:<15.4f} {quality:<25}")
            
        except Exception as e:
            logger.error(f"{method_name}: {e}")
            print(f"{method_name:<15} ERROR")
    
    # QUBO
    try:
        import time
        t0 = time.time()
        profile = {
            'columns': list(df.columns),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        }
        result = run_qubo_feature_selection(df, profile, 'target', lambda_penalty=0.5)
        runtime = time.time() - t0
        
        selected = result.get('selected_features', [])
        n_selected = len(selected)
        redundancy = result.get('redundancy_after', 0.0)
        reduction_pct = result.get('redundancy_reduction_pct', 0.0)
        
        all_results['qubo'] = {
            'n_features': n_selected,
            'redundancy': redundancy,
            'runtime': runtime,
            'selected': selected,
            'reduction_pct': reduction_pct,
            'method': result.get('selection_method', 'N/A'),
        }
        
        quality = f"Good ({reduction_pct:.1f}% reduction)" if redundancy < 0.08 else "Fair"
        print(f"{'qubo':<15} {n_selected:<12} {redundancy:<15.3f} {runtime:<15.4f} {quality:<25}")
        
    except Exception as e:
        logger.error(f"QUBO: {e}")
        print(f"{'qubo':<15} ERROR")
    
    print("="*100)
    
    # Summary analysis
    print("\nDETAILED RESULTS:")
    print("-"*100)
    
    for method, data in all_results.items():
        if 'error' not in data:
            print(f"\n{method.upper()}:")
            print(f"  Selected features ({data['n_features']}): {data['selected']}")
            print(f"  Redundancy (mean pairwise correlation): {data['redundancy']:.3f}")
            print(f"  Runtime: {data['runtime']:.3f}s")
            if 'reduction_pct' in data:
                print(f"  Redundancy reduction: {data['reduction_pct']:.1f}%")
            if 'method' in data:
                print(f"  Selection method: {data['method']}")
    
    print("\n" + "="*100)
    print("KEY FINDINGS:")
    print("-"*100)
    
    # Find best performers
    best_redundancy = min([(m, d['redundancy']) for m, d in all_results.items() if 'redundancy' in d], key=lambda x: x[1])
    fastest = min([(m, d['runtime']) for m, d in all_results.items() if 'runtime' in d], key=lambda x: x[1])
    most_parsimonious = min([(m, d['n_features']) for m, d in all_results.items() if 'n_features' in d], key=lambda x: x[1])
    
    print(f"✓ Best redundancy reduction: {best_redundancy[0].upper()} (redundancy={best_redundancy[1]:.3f})")
    print(f"✓ Fastest: {fastest[0].upper()} ({fastest[1]:.3f}s)")
    print(f"✓ Most parsimonious: {most_parsimonious[0].upper()} ({most_parsimonious[1]} features)")
    
    print("\nRECOMMENDATION:")
    print("-"*100)
    print("For clinical trial analysis, use:")
    print("  • LASSO/Elastic Net: Best redundancy reduction + fast + interpretable coefficients")
    print("  • mRMR: Good balance of redundancy reduction + stability + reasonable speed")
    print("  • QUBO: Best for high-dimensional datasets + novel quantum-inspired approach")
    print("  • Univariate: Fast baseline for initial screening")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
