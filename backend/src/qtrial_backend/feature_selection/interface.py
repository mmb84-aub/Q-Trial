"""Unified feature selection interface."""

import logging
from .mrmr import mrmr_selection
from .lasso import lasso_selection
from .univariate import univariate_selection

logger = logging.getLogger(__name__)


def select_features(
    df,
    outcome_column,
    method="mrmr",
    n_features=None,
    **kwargs,
):
    """
    Unified interface for feature selection.
    
    Allows switching between different methods with a single function call.
    Output format is standardized across all methods.
    
    Args:
        df: Input DataFrame
        outcome_column: Name of target column
        method: One of "mrmr", "lasso", "elastic_net", "univariate"
        n_features: Number of features to select (auto-determined if None)
        **kwargs: Method-specific arguments
    
    Returns:
        Dict with standardized output:
        - selected_features: List of selected column names
        - relevance_scores: Dict mapping feature -> score
        - redundancy_measure: Mean pairwise correlation
        - n_features: Number of selected features
        - method: Name of method used
        - [method-specific fields]
    
    Raises:
        ValueError: If method is not recognized
    """
    if method == "mrmr":
        return mrmr_selection(df, outcome_column, n_features, **kwargs)
    
    elif method == "lasso":
        kwargs["use_elastic_net"] = False
        return lasso_selection(df, outcome_column, n_features, **kwargs)
    
    elif method == "elastic_net":
        kwargs["use_elastic_net"] = True
        return lasso_selection(df, outcome_column, n_features, **kwargs)
    
    elif method == "univariate":
        return univariate_selection(df, outcome_column, n_features, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from: 'mrmr', 'lasso', 'elastic_net', 'univariate'"
        )
