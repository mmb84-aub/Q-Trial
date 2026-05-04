"""mRMR (minimum Redundancy Maximum Relevance) feature selection."""

import logging
import numpy as np
import pandas as pd
from .utils import default_feature_count, handle_mixed_types, compute_mutual_information, mean_pairwise_correlation

logger = logging.getLogger(__name__)


def mrmr_selection(
    df,
    outcome_column,
    n_features=None,
    use_mutual_info=True,
):
    """
    mRMR (minimum Redundancy Maximum Relevance) feature selection.
    
    Iteratively selects features that maximize relevance while minimizing redundancy.
    Relevance measured via mutual information (information-theoretic).
    Redundancy measured via correlation.
    
    Args:
        df: Input DataFrame with mixed types
        outcome_column: Name of target column
        n_features: Number of features to select (auto-determined if None)
        use_mutual_info: Use mutual information (True) or correlation (False) for relevance
    
    Returns:
        Dict with keys:
        - selected_features: List of selected column names
        - relevance_scores: Dict mapping feature -> relevance score
        - redundancy_measure: Mean pairwise correlation after selection
        - n_features: Number of selected features
        - method: "mrmr"
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
        n_features = default_feature_count(n_candidates)
    n_features = min(n_features, n_candidates)
    
    logger.info(f"mRMR: starting with {n_candidates} candidates, targeting {n_features} features")
    
    # Compute relevance scores
    if use_mutual_info:
        relevance = compute_mutual_information(X, y)
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-8)
    else:
        # Use correlation with target as relevance
        relevance = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(n_candidates)])
        relevance = np.nan_to_num(relevance)
    
    # Compute redundancy matrix (pairwise correlations)
    redundancy = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(redundancy, 0)  # No self-redundancy
    
    # Greedy selection: start with highest relevance, then iteratively add features
    selected_indices = []
    relevance_scores_selected = {}
    
    # Step 1: Start with highest relevance feature
    best_idx = np.argmax(relevance)
    selected_indices.append(best_idx)
    relevance_scores_selected[candidate_cols[best_idx]] = float(relevance[best_idx])
    
    # Step 2: Iteratively add features
    remaining = set(range(n_candidates)) - set(selected_indices)
    
    for _ in range(n_features - 1):
        if not remaining:
            break
        
        best_score = -np.inf
        best_candidate = None
        
        for idx in remaining:
            # Score = relevance - lambda * average_redundancy_with_selected
            avg_redundancy = np.mean([redundancy[idx, sel_idx] for sel_idx in selected_indices])
            # mRMR score: maximize relevance, minimize redundancy
            score = relevance[idx] - avg_redundancy
            
            if score > best_score:
                best_score = score
                best_candidate = idx
        
        if best_candidate is not None:
            selected_indices.append(best_candidate)
            relevance_scores_selected[candidate_cols[best_candidate]] = float(relevance[best_candidate])
            remaining.remove(best_candidate)
    
    # Convert indices to feature names
    selected_features = [candidate_cols[i] for i in selected_indices]
    
    # Add outcome column back
    selected_features_with_outcome = [outcome_column] + selected_features
    
    # Compute redundancy after selection
    X_selected = X[:, selected_indices]
    redundancy_after = mean_pairwise_correlation(X_selected) if X_selected.shape[1] > 1 else 0.0
    
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"mRMR complete: {len(selected_features)} features, redundancy={redundancy_after:.3f}, time={elapsed:.2f}s")
    
    return {
        "selected_features": selected_features_with_outcome,
        "relevance_scores": relevance_scores_selected,
        "redundancy_measure": redundancy_after,
        "n_features": len(selected_features_with_outcome),
        "method": "mrmr",
        "use_mutual_info": use_mutual_info,
    }
