"""
QUBO-based Feature Selection Module

Implements quantum-inspired feature selection using Quadratic Unconstrained Binary 
Optimization (QUBO) solved via classical simulated annealing.

Literature references:
- PMID 37173974: Leukocytes Classification using Quantum-Inspired Evolutionary Algorithm (2023)
- Romero et al. (2022): Machine Learning: Science and Technology 3, 015017
- Skolik et al. (2021): Quantum Machine Intelligence 3, 27
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import neal

logger = logging.getLogger(__name__)


def compute_relevance_scores(
    df: pd.DataFrame,
    outcome_column: Optional[str] = None,
    candidate_columns: Optional[list] = None,
) -> dict:
    """
    Step 1: Compute relevance scores for each candidate column.
    
    Uses the appropriate statistical measure based on data types:
    - Numeric vs Numeric: Absolute Pearson correlation
    - Categorical vs Numeric: Eta-squared from one-way ANOVA
    - Numeric vs Categorical: Point-biserial correlation
    - Categorical vs Categorical: Cramér's V from chi-square
    
    If no outcome column is designated, uses normalised variance as a proxy.
    
    Args:
        df: DataFrame with candidate columns
        outcome_column: Name of outcome column (optional)
        candidate_columns: List of column names to evaluate. If None, uses all non-outcome columns
    
    Returns:
        Dict mapping column name to relevance score in [0, 1]
    """
    if candidate_columns is None:
        candidate_columns = [c for c in df.columns if c != outcome_column]
    
    relevance_scores = {}
    
    # If no outcome column, use variance-based relevance
    if outcome_column is None:
        variances = df[candidate_columns].var()
        max_var = variances.max()
        if max_var > 0:
            relevance_scores = (variances / max_var).to_dict()
        else:
            relevance_scores = {col: 0.5 for col in candidate_columns}
        return relevance_scores
    
    outcome_dtype = df[outcome_column].dtype
    is_outcome_numeric = pd.api.types.is_numeric_dtype(outcome_dtype)
    
    for col in candidate_columns:
        col_dtype = df[col].dtype
        is_col_numeric = pd.api.types.is_numeric_dtype(col_dtype)
        
        try:
            if is_col_numeric and is_outcome_numeric:
                # Numeric vs Numeric: Pearson correlation
                score = abs(df[col].corr(df[outcome_column]))
            
            elif not is_col_numeric and is_outcome_numeric:
                # Categorical vs Numeric: Eta-squared from ANOVA
                groups = [group[col].dropna().values for name, group in df.groupby(outcome_column)]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) > 1:
                    f_stat, p_val = stats.f_oneway(*groups)
                    # Eta-squared as measure of effect size
                    grand_mean = df[outcome_column].mean()
                    ss_total = ((df[outcome_column] - grand_mean) ** 2).sum()
                    ss_between = sum(
                        len(df[df[col] == cat]) * (df[df[col] == cat][outcome_column].mean() - grand_mean) ** 2
                        for cat in df[col].unique()
                    )
                    score = abs(ss_between / ss_total) if ss_total > 0 else 0.0
                else:
                    score = 0.0
            
            elif is_col_numeric and not is_outcome_numeric:
                # Numeric vs Categorical: Point-biserial correlation
                # Convert outcome to numeric (0/1 for binary, or use first category)
                unique_outcomes = df[outcome_column].unique()
                if len(unique_outcomes) == 2:
                    outcome_numeric = (df[outcome_column] == unique_outcomes[0]).astype(int)
                    score = abs(stats.pointbiserialr(outcome_numeric, df[col])[0])
                else:
                    # For multiclass, use max absolute correlation with any class
                    scores = []
                    for outcome_val in unique_outcomes:
                        outcome_binary = (df[outcome_column] == outcome_val).astype(int)
                        r, _ = stats.pointbiserialr(outcome_binary, df[col])
                        scores.append(abs(r))
                    score = max(scores) if scores else 0.0
            
            else:
                # Categorical vs Categorical: Cramér's V
                contingency_table = pd.crosstab(df[col], df[outcome_column])
                chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                score = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
            
            relevance_scores[col] = score
        
        except Exception as e:
            logger.warning(f"Error computing relevance for column {col}: {e}")
            relevance_scores[col] = 0.0
    
    # Normalise to [0, 1]
    max_score = max(relevance_scores.values()) if relevance_scores else 1.0
    if max_score > 0:
        relevance_scores = {col: score / max_score for col, score in relevance_scores.items()}
    
    return relevance_scores


def compute_redundancy_matrix(
    df: pd.DataFrame,
    candidate_columns: list,
) -> np.ndarray:
    """
    Step 2: Compute pairwise redundancy matrix.
    
    Redundancy is measured as:
    - Numeric vs Numeric: Absolute Pearson correlation
    - Involving categorical: Cramér's V
    
    Args:
        df: DataFrame with candidate columns
        candidate_columns: List of column names
    
    Returns:
        M x M redundancy matrix where M = len(candidate_columns)
    """
    n = len(candidate_columns)
    redundancy = np.zeros((n, n))
    
    for i, col_i in enumerate(candidate_columns):
        for j, col_j in enumerate(candidate_columns):
            if i >= j:
                # Only compute upper triangle; set diagonal to 0
                continue
            
            is_i_numeric = pd.api.types.is_numeric_dtype(df[col_i].dtype)
            is_j_numeric = pd.api.types.is_numeric_dtype(df[col_j].dtype)
            
            try:
                if is_i_numeric and is_j_numeric:
                    # Numeric vs Numeric: Pearson correlation
                    score = abs(df[col_i].corr(df[col_j]))
                else:
                    # Involving categorical: Cramér's V
                    contingency_table = pd.crosstab(df[col_i], df[col_j])
                    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                    n_samples = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                    score = np.sqrt(chi2 / (n_samples * min_dim)) if min_dim > 0 and n_samples > 0 else 0.0
                
                redundancy[i, j] = score
            
            except Exception as e:
                logger.warning(f"Error computing redundancy for {col_i} vs {col_j}: {e}")
                redundancy[i, j] = 0.0
    
    return redundancy


def construct_qubo_matrix(
    relevance_scores: dict,
    redundancy_matrix: np.ndarray,
    candidate_columns: list,
    lambda_penalty: float = 0.5,
) -> dict:
    """
    Step 3: Construct QUBO matrix.
    
    Objective: Minimise -Σ(r_i * x_i) + λ * Σ(c_ij * x_i * x_j)
    
    Where:
    - r_i = relevance score for column i
    - c_ij = redundancy between columns i and j
    - λ = penalty weight
    
    Args:
        relevance_scores: Dict mapping column name to relevance score
        redundancy_matrix: M x M redundancy matrix
        candidate_columns: List of M column names
        lambda_penalty: Penalty weight for redundancy (default 0.5)
    
    Returns:
        QUBO dict with variable indices as keys
    """
    Q = {}
    
    # Diagonal entries: Q_ii = -r_i
    for i, col in enumerate(candidate_columns):
        Q[(i, i)] = -relevance_scores.get(col, 0.0)
    
    # Off-diagonal entries: Q_ij = λ * c_ij (upper triangular only)
    for i in range(len(candidate_columns)):
        for j in range(i + 1, len(candidate_columns)):
            Q[(i, j)] = lambda_penalty * redundancy_matrix[i, j]
    
    return Q


def solve_qubo(
    Q: dict,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
) -> dict:
    """
    Step 4: Solve QUBO using simulated annealing.
    
    Args:
        Q: QUBO matrix dict
        num_reads: Number of independent annealing attempts (default 1000)
        num_sweeps: Length of each annealing schedule (default 1000)
    
    Returns:
        Dict mapping variable index to binary value (0 or 1)
    """
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q, num_reads=num_reads, num_sweeps=num_sweeps)
    best_sample = response.first.sample
    
    return best_sample


def apply_hard_constraints(
    selected_indices: list,
    candidate_columns: list,
    relevance_scores: dict,
    outcome_column: Optional[str] = None,
    excluded_columns: Optional[list] = None,
) -> list:
    """
    Step 5: Apply hard constraints to the solver output.
    
    Rules:
    1. Always include outcome column if designated
    2. Never include already-excluded columns
    3. Minimum floor of 5 columns (fall back to top 10 by relevance)
    4. Maximum cap of 20 columns (take top 20 by relevance)
    
    Args:
        selected_indices: Indices from QUBO solver where value is 1
        candidate_columns: List of candidate column names
        relevance_scores: Dict mapping column to relevance score
        outcome_column: Name of outcome column (if any)
        excluded_columns: List of excluded columns (if any)
    
    Returns:
        List of selected column names
    """
    selected_cols = [candidate_columns[i] for i in selected_indices]
    
    # Rule 1: Always include outcome column if designated
    if outcome_column and outcome_column not in selected_cols:
        selected_cols.append(outcome_column)
    
    # Rule 2: Never include excluded columns (should already be filtered, but double-check)
    if excluded_columns:
        selected_cols = [col for col in selected_cols if col not in excluded_columns]
    
    # Rule 3: Minimum floor of 5 columns
    if len(selected_cols) < 5:
        logger.info(f"Selected columns ({len(selected_cols)}) below minimum of 5. Using top 10 by relevance.")
        # Order candidate columns by relevance
        sorted_cols = sorted(candidate_columns, key=lambda c: relevance_scores.get(c, 0.0), reverse=True)
        selected_cols = sorted_cols[:min(10, len(sorted_cols))]
        if outcome_column and outcome_column not in selected_cols:
            selected_cols = [outcome_column] + selected_cols[:9]
    
    # Rule 4: Maximum cap of 20 columns
    if len(selected_cols) > 20:
        logger.info(f"Selected columns ({len(selected_cols)}) exceed maximum of 20. Using top 20 by relevance.")
        # Sort selected by relevance and take top 20
        sorted_selected = sorted(selected_cols, key=lambda c: relevance_scores.get(c, 0.0), reverse=True)
        selected_cols = sorted_selected[:20]
        # Ensure outcome column is included
        if outcome_column and outcome_column not in selected_cols:
            selected_cols = [outcome_column] + sorted_selected[:19]
    
    return selected_cols


def mean_pairwise_correlation(df: pd.DataFrame, columns: list) -> float:
    """
    Compute mean absolute pairwise correlation for a set of columns.
    
    Args:
        df: DataFrame
        columns: List of column names (must all be numeric)
    
    Returns:
        Mean absolute correlation value
    """
    if len(columns) < 2:
        return 0.0
    
    try:
        corr = df[columns].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        return upper.stack().mean()
    except Exception as e:
        logger.warning(f"Error computing mean pairwise correlation: {e}")
        return 0.0


def run_qubo_feature_selection(
    df: pd.DataFrame,
    profile: dict,
    outcome_column: Optional[str] = None,
    lambda_penalty: float = 0.5,
) -> dict:
    """
    Step 7: Main orchestration function for QUBO feature selection.
    
    Runs the complete feature selection pipeline and returns a structured output dict.
    
    Args:
        df: Sanitised DataFrame with candidate columns
        profile: Static profile object from profiler
        outcome_column: Name of outcome column (optional)
        lambda_penalty: Penalty weight for redundancy (default 0.5)
    
    Returns:
        Dict with keys:
        - selected_columns: List of selected column names
        - excluded_columns: List of columns not selected
        - relevance_scores: Dict of column -> relevance score
        - redundancy_before: Mean correlation before selection
        - redundancy_after: Mean correlation after selection
        - redundancy_reduction: Fractional reduction
        - n_candidates: Number of columns entering solver
        - n_selected: Number of columns selected
        - solver: Solver name ("simulated_annealing")
        - lambda_penalty: Penalty weight used
        - num_reads: Number of reads used
        - num_sweeps: Number of sweeps used
        - selection_method: "qubo" or "relevance_fallback"
        - outcome_column: Name of outcome column or None
    """
    # Identify candidate columns (all numeric columns, excluding outcome)
    excluded = [outcome_column] if outcome_column else []
    candidate_columns = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c].dtype)]
    
    if not candidate_columns:
        logger.warning("No numeric candidate columns found. Returning empty selection.")
        return {
            "selected_columns": [outcome_column] if outcome_column else [],
            "excluded_columns": list(df.columns),
            "relevance_scores": {},
            "redundancy_before": 0.0,
            "redundancy_after": 0.0,
            "redundancy_reduction": 0.0,
            "n_candidates": 0,
            "n_selected": 1 if outcome_column else 0,
            "solver": "simulated_annealing",
            "lambda_penalty": lambda_penalty,
            "num_reads": 1000,
            "num_sweeps": 1000,
            "selection_method": "relevance_fallback",
            "outcome_column": outcome_column,
        }
    
    # Step 1: Compute relevance scores
    relevance_scores = compute_relevance_scores(df, outcome_column, candidate_columns)
    
    # Step 2: Compute redundancy matrix
    redundancy_matrix = compute_redundancy_matrix(df, candidate_columns)
    
    # Step 6: Measure redundancy before
    numeric_candidates = [c for c in candidate_columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
    redundancy_before = mean_pairwise_correlation(df, numeric_candidates)
    
    # Step 3: Construct QUBO matrix
    Q = construct_qubo_matrix(relevance_scores, redundancy_matrix, candidate_columns, lambda_penalty)
    
    # Step 4: Solve QUBO
    best_sample = solve_qubo(Q, num_reads=1000, num_sweeps=1000)
    selected_indices = [i for i, val in best_sample.items() if val == 1]
    
    # Step 5: Apply hard constraints
    selected_columns = apply_hard_constraints(
        selected_indices,
        candidate_columns,
        relevance_scores,
        outcome_column=outcome_column,
    )
    
    # Step 6: Validate against classical baseline
    numeric_selected = [c for c in selected_columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
    redundancy_after = mean_pairwise_correlation(df, numeric_selected)
    
    if redundancy_after > redundancy_before:
        logger.warning("Feature selection increased redundancy. Falling back to top-N by relevance.")
        # Fall back to top columns by relevance
        sorted_cols = sorted(candidate_columns, key=lambda c: relevance_scores.get(c, 0.0), reverse=True)
        selected_columns = sorted_cols[:min(10, len(sorted_cols))]
        if outcome_column and outcome_column not in selected_columns:
            selected_columns = [outcome_column] + selected_columns[:9]
        selection_method = "relevance_fallback"
        numeric_selected = [c for c in selected_columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
        redundancy_after = mean_pairwise_correlation(df, numeric_selected)
    else:
        selection_method = "qubo"
    
    # Compute reduction ratio
    redundancy_reduction = (redundancy_before - redundancy_after) / redundancy_before if redundancy_before > 0 else 0.0
    
    # Identify excluded columns
    excluded_columns = [col for col in df.columns if col not in selected_columns]
    
    # Round redundancy values to 2 decimal places for readability
    redundancy_before = round(redundancy_before, 4)
    redundancy_after = round(redundancy_after, 4)
    redundancy_reduction = round(redundancy_reduction, 4)
    
    # Round relevance scores to 2 decimal places
    relevance_scores = {col: round(score, 2) for col, score in relevance_scores.items()}
    
    return {
        "selected_columns": selected_columns,
        "excluded_columns": excluded_columns,
        "relevance_scores": relevance_scores,
        "redundancy_before": redundancy_before,
        "redundancy_after": redundancy_after,
        "redundancy_reduction": redundancy_reduction,
        "n_candidates": len(candidate_columns),
        "n_selected": len(selected_columns),
        "solver": "simulated_annealing",
        "lambda_penalty": lambda_penalty,
        "num_reads": 1000,
        "num_sweeps": 1000,
        "selection_method": selection_method,
        "outcome_column": outcome_column,
    }
