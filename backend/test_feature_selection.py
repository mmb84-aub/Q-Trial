"""Test and demo script for feature selection methods."""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add backend to path
backend_path = Path(__file__).parent
src_path = backend_path / "src"
sys.path.insert(0, str(src_path))

from qtrial_backend.feature_selection.interface import select_features
from qtrial_backend.feature_selection.benchmark import benchmark_all_methods, print_benchmark_summary, export_benchmark_results
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection


def create_sample_dataset(n_samples=200, n_features=30, n_informative=5, random_state=42):
    """
    Create a synthetic clinical dataset for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_informative: Number of truly informative features
        random_state: Random seed
    
    Returns:
        df: DataFrame with mixed types
        outcome_col: Name of outcome column
    """
    np.random.seed(random_state)
    
    data = {}
    
    # Create informative numeric features
    for i in range(n_informative):
        data[f"var_{i}"] = np.random.normal(0, 1, n_samples)
    
    # Create correlated features (redundant)
    for i in range(n_informative):
        for j in range(2):
            data[f"var_{n_informative + i}_{j}"] = data[f"var_{i}"] + np.random.normal(0, 0.5, n_samples)
    
    # Create noise features
    n_noise = n_features - n_informative - (n_informative * 2)
    for i in range(max(0, n_noise)):
        data[f"noise_{i}"] = np.random.normal(0, 1, n_samples)
    
    # Create categorical features
    for i in range(3):
        data[f"category_{i}"] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    
    # Add some missing values
    for col in data:
        missing_idx = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        col_data = data[col]
        if isinstance(col_data[0], (int, float)):
            col_data[missing_idx] = np.nan
    
    # Create outcome as linear combination of informative features
    outcome = np.zeros(n_samples)
    for i in range(n_informative):
        outcome += 2 * data[f"var_{i}"]
    outcome += np.random.normal(0, 2, n_samples)
    outcome = (outcome > outcome.median()).astype(int)  # Binary classification
    data["target"] = outcome
    
    df = pd.DataFrame(data)
    return df, "target"


def test_single_method(df, outcome_col, method, **kwargs):
    """Test a single feature selection method."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {method.upper()} feature selection")
    logger.info(f"{'='*60}")
    
    try:
        result = select_features(df, outcome_col, method=method, **kwargs)
        
        print(f"\nMethod: {method}")
        print(f"Selected features ({len(result['selected_features'])}):")
        for feat in result["selected_features"]:
            print(f"  - {feat}")
        
        print(f"\nRedundancy measure: {result.get('redundancy_measure', 'N/A'):.3f}")
        
        if "relevance_scores" in result and result["relevance_scores"]:
            print(f"\nTop relevance scores:")
            sorted_scores = sorted(
                result["relevance_scores"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for feat, score in sorted_scores:
                print(f"  - {feat}: {score:.3f}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in {method}: {e}", exc_info=True)
        return None


def main():
    """Main test routine."""
    logger.info("Creating synthetic dataset...")
    df, outcome_col = create_sample_dataset(n_samples=300, n_features=35)
    logger.info(f"Dataset shape: {df.shape}, Outcome column: {outcome_col}")
    
    # Test individual methods
    methods = ["univariate", "mrmr", "lasso", "elastic_net"]
    
    for method in methods:
        test_single_method(df, outcome_col, method)
    
    # Run full benchmark
    logger.info("\n" + "="*60)
    logger.info("Running FULL BENCHMARK across all methods...")
    logger.info("="*60)
    
    benchmark_results = benchmark_all_methods(
        df,
        outcome_col,
        methods=methods,
        task_type="classification",
        n_bootstrap=5,
        verbose=True,
    )
    
    # Print summary
    print_benchmark_summary(benchmark_results, task_type="classification")
    
    # Export results
    output_path = Path(__file__).parent.parent.parent / "outputs" / "feature_selection_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_benchmark_results(benchmark_results, str(output_path))
    logger.info(f"Results saved to {output_path}")
    
    return benchmark_results


if __name__ == "__main__":
    results = main()
