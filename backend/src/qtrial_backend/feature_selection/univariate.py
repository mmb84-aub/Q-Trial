"""Univariate statistical feature selection."""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif, f_regression
from .utils import handle_mixed_types, mean_pairwise_correlation

logger = logging.getLogger(__name__)


def univariate_selection(
    df,
    outcome_column,
    n_features=None,
    use_classification=True,
):
    """
    Univariate statistical feature selection.
    
    Ranks features by statistical association with target:
    - Classification: F-statistic (ANOVA)
    - Regression: F-statistic (linear regression)
    
    Selects top-k features by p-value significance.
    
    Args:
        df: Input DataFrame with mixed types
        outcome_column: Name of target column
        n_features: Number of features to select (auto-determined if None)
        use_classification: Classification (True) or regression (False) task
    
    Returns:
        Dict with keys:
        - selected_features: List of selected column names
        - relevance_scores: Dict mapping feature -> F-statistic
        - redundancy_measure: Mean pairwise correlation after selection
        - n_features: Number of selected features
        - method: "univariate"
        - p_values: Dict mapping feature -> p-value
    """
    start_time = pd.Timestamp.now()
    
    # Prepare data
    candidate_cols = [c for c in df.columns if c != outcome_column]
    prepared_df, encoders, numeric_cols, categorical_cols = handle_mixed_types(
        df[candidate_cols + [outcome_column]]
    )
    
    X = prepared_df[candidate_cols].values.astype(float)
    y = prepared_df[outcome_column].values.astype(float)
    
    n_candidates = X.shape[1]
    if n_features is None:
        n_features = max(4, int(np.ceil(np.sqrt(n_candidates))))
    n_features = min(n_features, n_candidates)
    
    logger.info(f"Univariate: starting with {n_candidates} candidates, targeting {n_features} features")
    
    # Compute F-statistics
    try:
        if use_classification:
            # For classification: F-statistic from one-way ANOVA
            f_stats, p_values = f_classif(X, y)
            task_name = "classification"
        else:
            # For regression: F-statistic from linear regression
            f_stats, p_values = f_regression(X, y)
            task_name = "regression"
        
        # Handle NaNs/infs
        f_stats = np.nan_to_num(f_stats, nan=0.0, posinf=0.0, neginf=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
        
        # Normalize F-statistics to [0, 1]
        f_stats_max = np.max(f_stats)
        if f_stats_max > 0:
            f_stats_normalized = f_stats / f_stats_max
        else:
            f_stats_normalized = np.ones_like(f_stats) / n_candidates
        
        # Select top-k by F-statistic (or equivalently, lowest p-values)
        selected_indices = np.argsort(f_stats)[-n_features:]
        
        # Build relevance scores
        relevance_scores = {}
        p_value_dict = {}
        for idx in selected_indices:
            relevance_scores[candidate_cols[idx]] = float(f_stats_normalized[idx])
            p_value_dict[candidate_cols[idx]] = float(p_values[idx])
        
        selected_features = [candidate_cols[i] for i in selected_indices]
        selected_features_with_outcome = [outcome_column] + selected_features
        
        # Compute redundancy after selection
        X_selected = X[:, selected_indices]
        redundancy_after = mean_pairwise_correlation(X_selected) if X_selected.shape[1] > 1 else 0.0
        
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(
            f"Univariate ({task_name}) complete: {len(selected_features)} features, "
            f"redundancy={redundancy_after:.3f}, time={elapsed:.2f}s"
        )
        
        return {
            "selected_features": selected_features_with_outcome,
            "relevance_scores": relevance_scores,
            "p_values": p_value_dict,
            "redundancy_measure": redundancy_after,
            "n_features": len(selected_features_with_outcome),
            "method": "univariate",
            "task_type": task_name,
        }
    
    except Exception as e:
        logger.error(f"Univariate selection failed: {e}")
        # Fallback: return all features
        all_features = [outcome_column] + candidate_cols
        return {
            "selected_features": all_features,
            "relevance_scores": {},
            "p_values": {},
            "redundancy_measure": 0.0,
            "n_features": len(all_features),
            "method": "univariate",
            "task_type": None,
            "error": str(e),
        }
