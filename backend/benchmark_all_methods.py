"""Complete benchmark comparing QUBO + all baseline methods."""

import sys
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qtrial_backend.feature_selection.interface import select_features
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection


def create_clinical_dataset(n_samples=300, random_state=42):
    """Create synthetic clinical trial dataset."""
    np.random.seed(random_state)
    
    # Simulate PBC-like dataset with mixed features
    data = {}
    
    # Liver function tests (informative)
    data['bili'] = np.random.gamma(2, 1.5, n_samples)
    data['ast'] = np.random.gamma(3, 20, n_samples)
    data['alt'] = np.random.gamma(2.5, 25, n_samples)
    data['albumin'] = np.random.normal(35, 5, n_samples)
    
    # Redundant features (correlated with above)
    data['bili_to_albumin'] = data['bili'] / (data['albumin'] / 10)
    data['ast_alt_ratio'] = data['ast'] / data['alt']
    
    # Demographics
    data['age'] = np.random.normal(50, 15, n_samples)
    data['bmi'] = np.random.normal(25, 5, n_samples)
    
    # Noise features
    for i in range(5):
        data[f'noise_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Categorical
    data['sex'] = np.random.choice(['M', 'F'], n_samples)
    data['treatment'] = np.random.choice(['A', 'B', 'C'], n_samples)
    
    # Outcome: disease progression (related to liver enzymes)
    outcome = (data['bili'] > np.median(data['bili'])).astype(int) | \
              (data['ast'] > np.median(data['ast'])).astype(int)
    outcome = outcome.astype(int)
    data['progression'] = outcome
    
    return pd.DataFrame(data), 'progression'


def benchmark_feature_selection(df, outcome_col, methods=['univariate', 'mrmr', 'lasso', 'qubo']):
    """Benchmark multiple feature selection methods."""
    
    results = {}
    
    for method in methods:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running {method.upper()}")
        logger.info(f"{'='*70}")
        
        try:
            import time
            t0 = time.time()
            
            if method == 'qubo':
                # QUBO needs a profile dict - create a minimal one
                profile = {
                    'columns': list(df.columns),
                    'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                    'categorical_columns': df.select_dtypes(exclude=[np.number]).columns.tolist(),
                }
                result = run_qubo_feature_selection(df, profile, outcome_col, lambda_penalty=0.5)
            else:
                result = select_features(df, outcome_col, method=method)
            
            runtime = time.time() - t0
            
            selected = result.get('selected_features', [])
            n_selected = len(selected)
            
            # Handle different redundancy metric names across methods
            if 'redundancy_measure' in result:
                redundancy = result.get('redundancy_measure', 0.0)
            elif 'redundancy_after' in result:
                redundancy = result.get('redundancy_after', 0.0)
            else:
                redundancy = 0.0
            
            # Calculate metrics
            metrics = {
                'method': method,
                'selected_features': selected,
                'n_features': n_selected,
                'redundancy': float(redundancy),
                'runtime': float(runtime),
                'n_candidates': result.get('n_candidates', df.shape[1] - 1),
            }
            
            # Add method-specific info
            if 'relevance_scores' in result and result['relevance_scores']:
                metrics['top_relevance'] = sorted(
                    result['relevance_scores'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            
            if method == 'qubo':
                metrics['solver'] = result.get('solver', 'N/A')
                metrics['lambda_penalty'] = result.get('lambda_penalty', 'N/A')
                metrics['selection_method'] = result.get('selection_method', 'N/A')
                metrics['redundancy_before'] = result.get('redundancy_before', 0.0)
                metrics['redundancy_reduction_pct'] = result.get('redundancy_reduction_pct', 0.0)
            
            results[method] = metrics
            
            logger.info(f"✓ {method}: {n_selected} features, redundancy={redundancy:.3f}, runtime={runtime:.2f}s")
            
        except Exception as e:
            logger.error(f"✗ {method} failed: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {'error': str(e)}
    
    return results


def print_comparison_table(results):
    """Print comparison table."""
    print("\n" + "="*100)
    print("FEATURE SELECTION BENCHMARK COMPARISON")
    print("="*100)
    print(f"{'Method':<15} {'Features':<12} {'Redundancy':<15} {'Runtime (s)':<15} {'Candidates':<12}")
    print("-"*100)
    
    for method in ['univariate', 'mrmr', 'lasso', 'qubo']:
        if method not in results:
            continue
        
        result = results[method]
        if 'error' in result:
            print(f"{method:<15} ERROR: {result['error'][:50]}")
        else:
            n_feat = result.get('n_features', 0)
            redundancy = result.get('redundancy', 0.0)
            runtime = result.get('runtime', 0.0)
            n_cand = result.get('n_candidates', 0)
            print(f"{method:<15} {n_feat:<12} {redundancy:<15.3f} {runtime:<15.4f} {n_cand:<12}")
    
    print("="*100)


def main():
    """Main benchmark routine."""
    logger.info("Creating synthetic clinical dataset...")
    df, outcome_col = create_clinical_dataset(n_samples=500)
    logger.info(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
    logger.info(f"Target: {outcome_col} (n_positive={df[outcome_col].sum()})")
    
    # Run benchmark
    methods = ['univariate', 'mrmr', 'lasso', 'elastic_net', 'qubo']
    results = benchmark_feature_selection(df, outcome_col, methods=methods)
    
    # Print comparison
    print_comparison_table(results)
    
    # Export results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert to JSON-serializable format
    json_results = {}
    for method, data in results.items():
        if 'error' not in data:
            json_results[method] = {
                k: v for k, v in data.items()
                if not isinstance(v, list) or k in ['selected_features', 'top_relevance']
            }
            if 'top_relevance' in json_results[method]:
                json_results[method]['top_relevance'] = [
                    {'feature': f, 'score': s} for f, s in json_results[method]['top_relevance']
                ]
        else:
            json_results[method] = data
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\nResults exported to: {output_file}")
    return results


if __name__ == "__main__":
    results = main()
