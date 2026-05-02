"""Shared utilities for feature selection."""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def handle_mixed_types(df, numeric_cols=None, categorical_cols=None):
    """
    Prepare data for feature selection: encode categoricals, impute missing values.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names (auto-detected if None)
        categorical_cols: List of categorical column names (auto-detected if None)
    
    Returns:
        prepared_df: DataFrame with encoded categoricals and imputed values
        encoders: Dict of LabelEncoders for categorical columns
        numeric_cols: Final list of numeric columns
        categorical_cols: Final list of categorical columns
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    prepared_df = df.copy()
    encoders = {}
    
    # Encode categorical features
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values before encoding
        mask = prepared_df[col].notna()
        col_data = prepared_df[col].astype(str).values
        encoded_values = np.full(len(col_data), -1, dtype=float)
        encoded_values[mask] = le.fit_transform(col_data[mask])
        prepared_df[col] = encoded_values
        encoders[col] = le
    
    # Impute missing numeric values with median
    for col in numeric_cols:
        if prepared_df[col].isna().any():
            median_val = prepared_df[col].median()
            prepared_df[col] = prepared_df[col].fillna(median_val)
    
    return prepared_df, encoders, numeric_cols, categorical_cols


def compute_mutual_information(X, y):
    """
    Compute mutual information between each feature and target.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
    
    Returns:
        mi_scores: Array of MI scores
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    
    # Determine if classification or regression
    if len(np.unique(y)) < 20:  # Heuristic: classification
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:  # Regression
        mi_scores = mutual_info_regression(X, y, random_state=42)
    
    return mi_scores


def compute_feature_correlations(X):
    """
    Compute pairwise correlations as a matrix.
    
    Args:
        X: Feature matrix (n_samples, n_features)
    
    Returns:
        corr_matrix: Absolute correlation matrix
    """
    return np.abs(np.corrcoef(X.T))


def mean_pairwise_correlation(X, feature_indices=None):
    """
    Compute mean absolute pairwise correlation for selected features.
    
    Args:
        X: Feature matrix
        feature_indices: Indices of selected features (if None, use all)
    
    Returns:
        mean_corr: Mean absolute pairwise correlation
    """
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])
    
    if len(feature_indices) < 2:
        return 0.0
    
    X_subset = X[:, feature_indices]
    corr_matrix = np.abs(np.corrcoef(X_subset.T))
    # Extract upper triangle (excluding diagonal)
    upper_idx = np.triu_indices_from(corr_matrix, k=1)
    if len(upper_idx[0]) == 0:
        return 0.0
    return np.mean(corr_matrix[upper_idx])


def evaluate_predictive_performance(X_train, X_test, y_train, y_test, task_type='classification'):
    """
    Quick evaluation of predictive performance using simple model.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        task_type: 'classification' or 'regression'
    
    Returns:
        scores: Dict with performance metrics
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
    
    scores = {}
    
    try:
        if task_type == 'classification':
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            scores['accuracy'] = accuracy_score(y_test, y_pred)
            scores['auc'] = roc_auc_score(y_test, y_proba)
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            scores['r2'] = r2_score(y_test, y_pred)
            scores['mse'] = mean_squared_error(y_test, y_pred)
    except Exception as e:
        logger.warning(f"Could not evaluate predictive performance: {e}")
        scores = {}
    
    return scores
