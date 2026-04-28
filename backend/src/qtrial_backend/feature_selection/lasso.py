"""LASSO/Elastic Net feature selection."""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from .utils import handle_mixed_types, mean_pairwise_correlation

logger = logging.getLogger(__name__)


def lasso_selection(
    df,
    outcome_column,
    n_features=None,
    use_elastic_net=False,
    cv_folds=5,
):
    """
    LASSO (or Elastic Net) feature selection using cross-validation.
    
    Fits regularized linear regression and selects features with non-zero coefficients.
    Automatically chooses regularization strength via cross-validation.
    Elastic Net adds L2 penalty for additional stability with correlated features.
    
    Args:
        df: Input DataFrame with mixed types
        outcome_column: Name of target column
        n_features: Maximum features to select (soft cap; actual depends on model)
        use_elastic_net: Use Elastic Net (L1+L2) instead of pure LASSO (L1)
        cv_folds: Number of cross-validation folds
    
    Returns:
        Dict with keys:
        - selected_features: List of selected column names (non-zero coeff features)
        - relevance_scores: Dict mapping feature -> absolute coefficient value
        - redundancy_measure: Mean pairwise correlation after selection
        - n_features: Number of selected features
        - method: "lasso" or "elastic_net"
        - optimal_alpha: Best regularization parameter found
    """
    start_time = pd.Timestamp.now()
    
    # Prepare data
    candidate_cols = [c for c in df.columns if c != outcome_column]
    prepared_df, encoders, numeric_cols, categorical_cols = handle_mixed_types(
        df[candidate_cols + [outcome_column]]
    )
    
    X = prepared_df[candidate_cols].values.astype(float)
    y = prepared_df[outcome_column].values.astype(float)
    
    # Standardize features (critical for LASSO)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_candidates = X.shape[1]
    logger.info(f"LASSO: starting with {n_candidates} candidates")
    
    # Fit LASSO or Elastic Net with cross-validation
    try:
        if use_elastic_net:
            model = ElasticNetCV(
                l1_ratio=0.5,  # 50% L1, 50% L2
                alphas=np.logspace(-4, 1, 100),
                cv=cv_folds,
                random_state=42,
                max_iter=5000,
            )
            method_name = "elastic_net"
        else:
            model = LassoCV(
                alphas=np.logspace(-4, 1, 100),
                cv=cv_folds,
                random_state=42,
                max_iter=5000,
            )
            method_name = "lasso"
        
        model.fit(X_scaled, y)
        coef_abs = np.abs(model.coef_)
        
        # Select features with non-zero coefficients
        selected_mask = coef_abs > 1e-6  # Threshold for "effectively non-zero"
        selected_indices = np.where(selected_mask)[0]
        
        if len(selected_indices) == 0:
            # If no features selected, take top-k by absolute coefficient
            if n_features is None:
                n_features = max(4, int(np.ceil(np.sqrt(n_candidates))))
            selected_indices = np.argsort(coef_abs)[-n_features:]
        
        # Build relevance scores from absolute coefficients
        relevance_scores = {}
        for idx in selected_indices:
            relevance_scores[candidate_cols[idx]] = float(coef_abs[idx])
        
        # Normalize relevance scores to [0, 1]
        if relevance_scores:
            max_coef = max(relevance_scores.values())
            if max_coef > 0:
                relevance_scores = {k: v / max_coef for k, v in relevance_scores.items()}
        
        selected_features = [candidate_cols[i] for i in selected_indices]
        selected_features_with_outcome = [outcome_column] + selected_features
        
        # Compute redundancy after selection
        X_selected = X[:, selected_indices]
        redundancy_after = mean_pairwise_correlation(X_selected) if X_selected.shape[1] > 1 else 0.0
        
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(
            f"{method_name.upper()} complete: {len(selected_features)} features, "
            f"alpha={model.alpha_:.6f}, redundancy={redundancy_after:.3f}, time={elapsed:.2f}s"
        )
        
        return {
            "selected_features": selected_features_with_outcome,
            "relevance_scores": relevance_scores,
            "redundancy_measure": redundancy_after,
            "n_features": len(selected_features_with_outcome),
            "method": method_name,
            "optimal_alpha": float(model.alpha_),
            "cv_score": float(model.score(X_scaled, y)),
        }
    
    except Exception as e:
        logger.error(f"LASSO fitting failed: {e}")
        # Fallback: return all features
        all_features = [outcome_column] + candidate_cols
        return {
            "selected_features": all_features,
            "relevance_scores": {},
            "redundancy_measure": 0.0,
            "n_features": len(all_features),
            "method": method_name,
            "optimal_alpha": None,
            "error": str(e),
        }
