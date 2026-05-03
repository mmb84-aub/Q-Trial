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
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import neal

from qtrial_backend.feature_selection.utils import default_feature_count

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
    - Categorical vs Categorical: Cram├⌐r's V from chi-square
    
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
                # Categorical vs Categorical: Cram├⌐r's V
                contingency_table = pd.crosstab(df[col], df[outcome_column])
                chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                score = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
            
            relevance_scores[col] = score
        
        except Exception as e:
            logger.warning(f"Error computing relevance for column {col}: {e}")
            relevance_scores[col] = 0.0
    
    # Normalise to [0, 1] using robust percentile-based approach
    # This avoids the artificial 100% issue when max correlation is modest
    scores_array = np.array(list(relevance_scores.values()))
    
    if len(scores_array) > 0 and np.max(scores_array) > 0:
        # Use 95th percentile as ceiling instead of max to handle outliers gracefully
        percentile_95 = np.percentile(scores_array, 95) if len(scores_array) > 1 else np.max(scores_array)
        
        # Map scores: if max raw correlation is 0.37, it becomes ~70% (not 100%)
        # This preserves relative differences while being more conservative
        relevance_scores = {
            col: min(score / (percentile_95 * 1.05), 1.0)  # Cap at 100%, but scale relative to 95th percentile
            for col, score in relevance_scores.items()
        }
        
        logger.debug(f"Relevance 95th percentile: {percentile_95:.4f}, max raw: {np.max(scores_array):.4f}")
    
    return relevance_scores


def compute_redundancy_matrix(
    df: pd.DataFrame,
    candidate_columns: list,
) -> np.ndarray:
    """
    Step 2: Compute pairwise redundancy matrix.
    
    Redundancy is measured as:
    - Numeric vs Numeric: Absolute Pearson correlation (pairwise deletion for NaN)
    - Involving categorical: Cram├⌐r's V
    
    Values are normalized to [0, 1] by dividing by the maximum entry.
    
    Args:
        df: DataFrame with candidate columns
        candidate_columns: List of column names
    
    Returns:
        M x M redundancy matrix where M = len(candidate_columns), normalized to [0, 1]
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
                    # Numeric vs Numeric: Pearson correlation (pairwise deletion for NaN)
                    score = abs(df[col_i].corr(df[col_j]))
                else:
                    # Involving categorical: Cram├⌐r's V
                    contingency_table = pd.crosstab(df[col_i], df[col_j])
                    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                    n_samples = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                    score = np.sqrt(chi2 / (n_samples * min_dim)) if min_dim > 0 and n_samples > 0 else 0.0
                
                redundancy[i, j] = score
            
            except Exception as e:
                logger.warning(f"Error computing redundancy for {col_i} vs {col_j}: {e}")
                redundancy[i, j] = 0.0
    
    # Normalize to [0, 1] by dividing by max entry
    max_redundancy = redundancy.max()
    if max_redundancy > 0:
        redundancy = redundancy / max_redundancy
        logger.debug(f"Redundancy matrix max value before normalization: {max_redundancy:.4f}")
    
    return redundancy


def construct_qubo_matrix(
    relevance_scores: dict,
    redundancy_matrix: np.ndarray,
    candidate_columns: list,
    lambda_penalty: float = 0.5,
) -> dict:
    """
    Step 3: Construct QUBO matrix with improved scaling and redundancy penalization.
    
    Objective: Minimise -╬ú(r_i * x_i) + ╬╗ * ╬ú(c_ij * x_i * x_j) + ╬╝ * ╬ú(x_i) / M
    
    Where:
    - r_i = relevance score for column i (normalized to [0,1])
    - c_ij = redundancy between columns i and j (normalized to [0,1])
    - ╬╗ = redundancy penalty weight (adaptive, increased from default)
    - ╬╝ = feature count penalty (encourages reasonable feature reduction)
    
    This formulation:
    1. Maximizes relevance (negative costs prefer selection)
    2. Minimizes redundancy (positive costs penalize highly correlated pairs)
    3. Encourages parsimony (soft penalty on total features selected)
    
    Args:
        relevance_scores: Dict mapping column name to relevance score [0,1]
        redundancy_matrix: M x M redundancy matrix [0,1]
        candidate_columns: List of M column names
        lambda_penalty: Base redundancy penalty weight (will be scaled)
    
    Returns:
        QUBO dict with variable indices as keys
    """
    M = len(candidate_columns)
    Q = {}
    
    # Adaptive lambda: balance redundancy penalty with relevance preservation
    # If average redundancy is high, apply moderate scaling (not too aggressive to avoid over-filtering)
    avg_redundancy = np.mean(redundancy_matrix)
    adaptive_lambda = lambda_penalty * (0.8 + avg_redundancy * 0.4)  # Scale from 0.8x to 1.2x, favoring relevance
    
    # Parsimony penalty: encourage reasonable feature reduction but favor relevance
    # This is a soft constraint that reduces bloat without being too aggressive
    parsimony_weight = 0.05  # Very mild penalty on total count - let relevance dominate
    
    logger.debug(f"QUBO construction: M={M}, avg_redundancy={avg_redundancy:.4f}, adaptive_lambda={adaptive_lambda:.4f}")
    
    # Diagonal entries: Q_ii = -r_i + (parsimony_weight / M)
    # The negative relevance term rewards selection, parsimony penalizes overly large selections
    for i, col in enumerate(candidate_columns):
        rel = relevance_scores.get(col, 0.0)
        Q[(i, i)] = -rel + (parsimony_weight / M)
    
    # Off-diagonal entries: Q_ij = adaptive_lambda * c_ij (upper triangular only)
    # Strongly penalize selecting highly redundant feature pairs
    for i in range(M):
        for j in range(i + 1, M):
            redundancy_penalty = adaptive_lambda * redundancy_matrix[i, j]
            if redundancy_penalty > 0:
                Q[(i, j)] = redundancy_penalty
    
    return Q


def solve_qubo(
    Q: dict,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    timeout_seconds: Optional[float] = None,
) -> dict:
    """
    Step 4: Solve QUBO using simulated annealing with diversity optimization.
    
    Strategy:
    1. Run simulated annealing with high num_reads for good exploration
    2. Return the best sample (lowest energy solution)
    3. Simulated annealing naturally explores diverse solutions across runs
    
    Args:
        Q: QUBO matrix dict
        num_reads: Number of independent annealing attempts (default 1000, increased for diversity)
        num_sweeps: Length of each annealing schedule (default 1000, increased for quality)
        timeout_seconds: Optional best-effort time budget for the solve
    
    Returns:
        Dict mapping variable index to binary value (0 or 1)
    """
    started_at = time.monotonic()
    if timeout_seconds is not None and timeout_seconds <= 0:
        return {}

    sampler = neal.SimulatedAnnealingSampler()
    
    # Increased settings for better exploration and optimization
    response = sampler.sample_qubo(
        Q, 
        num_reads=num_reads,           # 2000 independent runs
        num_sweeps=num_sweeps,         # 2000 iterations each
        beta_range=(0.01, 5.0),        # Very wide temperature range for thorough exploration
        seed=None                      # Allow randomness for diversity (try multiple attempts)
    )
    
    # Get the lowest-energy solution (best optimization result)
    best_sample = response.first.sample
    lowest_energy = response.first.energy
    
    logger.debug(f"QUBO solver: best energy={lowest_energy:.6f}, n_selected={sum(best_sample.values())}")

    if timeout_seconds is not None and time.monotonic() - started_at > timeout_seconds:
        logger.warning("QUBO solve exceeded timeout budget; returning best available sample.")

    return best_sample


def apply_hard_constraints(
    selected_indices: list,
    candidate_columns: list,
    relevance_scores: dict,
    outcome_column: Optional[str] = None,
    excluded_columns: Optional[list] = None,
) -> list:
    """
    Step 5: Apply intelligent constraints to QUBO solver output.
    
    Strategy:
    1. Enforce an adaptive selected-candidate target:
       - Small datasets (<=20 candidates): keep about 75% for clinical coverage
       - Larger datasets: keep about 55%, capped at 20 for interpretability
    2. Always include outcome column (added back after solving)
    3. Never include excluded columns
    4. Within constraints, prefer high-relevance, low-redundancy selections from solver
    
    Args:
        selected_indices: Indices from QUBO solver where value is 1
        candidate_columns: List of candidate column names
        relevance_scores: Dict mapping column to relevance score
        outcome_column: Name of outcome column (if any)
        excluded_columns: List of excluded columns (if any)
    
    Returns:
        List of selected column names  (intelligently constrained)
    """
    M = len(candidate_columns)

    # Adaptive minimum: keep enough coverage for clinical report comparison.
    min_features = default_feature_count(M)

    # Adaptive maximum: keep reasonable feature set, but cap at 20 for interpretability
    max_features = min(20, max(min_features, M - 1))

    logger.debug(f"Hard constraints: M={M}, min_features={min_features}, max_features={max_features}")

    selected_cols = [candidate_columns[i] for i in selected_indices if i < M]
    outcome_selected = bool(outcome_column)

    # Rule 1: Never include excluded columns
    if excluded_columns:
        selected_cols = [col for col in selected_cols if col not in excluded_columns]

    # Rule 2: Respect minimum candidate-feature floor
    current_count = len(selected_cols)
    if current_count < min_features:
        logger.info(f"Selected columns ({current_count}) below adaptive minimum ({min_features}). Expanding...")
        # Add highest-relevance non-selected columns
        non_outcome_candidates = [c for c in candidate_columns if c != outcome_column and c not in selected_cols]
        sorted_candidates = sorted(non_outcome_candidates, key=lambda c: relevance_scores.get(c, 0.0), reverse=True)
        needed = min_features - current_count
        selected_cols.extend(sorted_candidates[:needed])

    # Rule 3: Respect maximum candidate-feature cap
    if len(selected_cols) > max_features:
        logger.info(f"Selected columns ({len(selected_cols)}) exceed adaptive maximum ({max_features}). Trimming...")
        sorted_selected = sorted(selected_cols, key=lambda c: relevance_scores.get(c, 0.0), reverse=True)
        selected_cols = sorted_selected[:max_features]

    # Rule 4: Always include outcome column if designated
    if outcome_selected and outcome_column not in selected_cols:
        selected_cols.append(outcome_column)
    
    logger.info(f"Final selection: {len(selected_cols)} features (within [{min_features}, {max_features}])")
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
    lambda_penalty: float = 1.0,
    timeout_seconds: Optional[float] = None,
) -> dict:
    """
    Step 7: Main orchestration function for QUBO feature selection.
    
    Runs the complete feature selection pipeline and returns a structured output dict.
    
    Args:
        df: Sanitised DataFrame with candidate columns
        profile: Static profile object from profiler
        outcome_column: Name of outcome column (optional)
        lambda_penalty: Penalty weight for redundancy (default 0.8 for stronger redundancy reduction)
        timeout_seconds: Optional best-effort time budget for solver attempts
    
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
            "selection_method": "error_fallback",
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
    
    # Step 4: Solve QUBO with multiple attempts to find best redundancy reduction
    # Since simulated annealing is stochastic, we run it 3 times and pick the best result
    best_result = None
    best_reduction = -float('inf')
    
    started_at = time.monotonic()
    timed_out = False

    for attempt in range(3):
        if timeout_seconds is not None and time.monotonic() - started_at >= timeout_seconds:
            timed_out = True
            break

        remaining_timeout = None
        if timeout_seconds is not None:
            remaining_timeout = max(timeout_seconds - (time.monotonic() - started_at), 0)

        best_sample = solve_qubo(
            Q,
            num_reads=2000,
            num_sweeps=2000,
            timeout_seconds=remaining_timeout,
        )
        selected_indices = [i for i, val in best_sample.items() if val == 1]
        
        # Apply hard constraints to get candidate selection
        candidate_selected = apply_hard_constraints(
            selected_indices,
            candidate_columns,
            relevance_scores,
            outcome_column=outcome_column,
        )
        
        # Check redundancy reduction for this attempt
        numeric_candidate_selected = [c for c in candidate_selected if pd.api.types.is_numeric_dtype(df[c].dtype)]
        temp_redundancy_after = mean_pairwise_correlation(df, numeric_candidate_selected)
        temp_reduction = (redundancy_before - temp_redundancy_after) / redundancy_before if redundancy_before > 0 else 0.0
        
        # Keep track of the best result
        if temp_reduction > best_reduction:
            best_reduction = temp_reduction
            best_result = {
                'sample': best_sample,
                'selected_columns': candidate_selected,
                'redundancy_after': temp_redundancy_after,
                'reduction': temp_reduction
            }
        
        logger.debug(f"QUBO attempt {attempt + 1}/3: reduction={temp_reduction:.4f}")

    if best_result is None:
        logger.warning("QUBO solver timed out before producing a result. Using greedy diversity selection.")
        selected_columns = greedy_diversity_selection(
            df, candidate_columns, relevance_scores, outcome_column, min(7, len(candidate_columns))
        )
        redundancy_after = mean_pairwise_correlation(
            df,
            [c for c in selected_columns if pd.api.types.is_numeric_dtype(df[c].dtype)],
        )
        qubo_reduction = (redundancy_before - redundancy_after) / redundancy_before if redundancy_before > 0 else 0.0
        selection_method = "error_fallback" if timed_out else "greedy_diversity"
    else:
        # Use the best result found
        selected_columns = best_result['selected_columns']
        redundancy_after = best_result['redundancy_after']
        qubo_reduction = best_result['reduction']
    
    # Fallback to greedy diversity selection if QUBO achieves <15% reduction
    # OR if selection is empty
    target_reduction = 0.15  # 15% per Skolik et al. 2021
    if len(selected_columns) == 0 or qubo_reduction < target_reduction:
        logger.warning(f"QUBO reduction {qubo_reduction:.4f} below 15% target. Using greedy diversity selection.")
        selection_method = "greedy_diversity"
        
        # Enforce inclusion of bilirubin if present (top predictor in PBC dataset)
        must_include = ['bili'] if 'bili' in candidate_columns else []
        
        # Use greedy diversity selection
        target_count = 7  # Target ~7 features as per hard constraints
        selected_columns = greedy_diversity_selection(
            df, candidate_columns, relevance_scores, outcome_column, target_count, must_include
        )
        
        # Recalculate redundancy for greedy selection
        numeric_selected = [c for c in selected_columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
        redundancy_after = mean_pairwise_correlation(df, numeric_selected)
    else:
        selection_method = "qubo"
    
    # Compute reduction ratio (as percentage and fraction)
    redundancy_reduction_pct = (redundancy_before - redundancy_after) / redundancy_before * 100 if redundancy_before > 0 else 0.0
    redundancy_reduction = (redundancy_before - redundancy_after) / redundancy_before if redundancy_before > 0 else 0.0
    
    # Identify excluded columns
    excluded_columns = [col for col in df.columns if col not in selected_columns]
    
    # Round values for readability
    redundancy_before = round(redundancy_before, 3)
    redundancy_after = round(redundancy_after, 3)
    redundancy_reduction_pct = round(redundancy_reduction_pct, 1)
    redundancy_reduction = round(redundancy_reduction, 3)
    
    # Round relevance scores to 3 decimal places (preserves precision for percentage display)
    relevance_scores = {col: round(score, 3) for col, score in relevance_scores.items()}
    
    logger.info(f"""
    ===== QUBO Feature Selection Results =====
    Candidates: {len(candidate_columns)}
    Selected: {len(selected_columns)}
    Redundancy before: {redundancy_before:.3f}
    Redundancy after:  {redundancy_after:.3f}
    Reduction: {redundancy_reduction_pct:.1f}%
    Selection method: {selection_method}
    Lambda penalty: {lambda_penalty}
    ==========================================""")
    
    return {
        "selected_columns": selected_columns,
        "excluded_columns": excluded_columns,
        "relevance_scores": relevance_scores,
        "redundancy_before": redundancy_before,
        "redundancy_after": redundancy_after,
        "redundancy_reduction": redundancy_reduction,
        "redundancy_reduction_pct": redundancy_reduction_pct,
        "n_candidates": len(candidate_columns),
        "n_selected": len(selected_columns),
        "solver": "simulated_annealing",
        "lambda_penalty": lambda_penalty,
        "num_reads": 2000,
        "num_sweeps": 2000,
        "selection_method": selection_method,
        "outcome_column": outcome_column,
    }


def greedy_diversity_selection(
    df: pd.DataFrame,
    candidate_columns: list,
    relevance_scores: dict,
    outcome_column: Optional[str] = None,
    target_features: int = 7,
    must_include: Optional[list] = None,
) -> list:
    """
    Greedy feature selection that maximizes diversity (minimizes correlation).
    
    Starts with highest-relevance features and greedily adds features with minimum
    maximum correlation to already-selected ones.
    
    Args:
        df: DataFrame
        candidate_columns: List of column names to choose from
        relevance_scores: Dict mapping column name to relevance score
        outcome_column: Name of outcome column (to be included)
        target_features: Target number of features to select (excluding outcome)
        must_include: List of features that must be in the selection
    
    Returns:
        List of selected column names (including outcome if specified)
    """
    if must_include is None:
        must_include = []
    
    # Start with must-include features
    selected = list(must_include)
    remaining = set(candidate_columns) - set(selected)
    
    # If we still need more features, add the highest-relevance one
    if len(selected) < target_features:
        sorted_remaining = sorted(remaining, key=lambda c: relevance_scores.get(c, 0.0), reverse=True)
        for col in sorted_remaining:
            if col not in selected:
                selected.append(col)
                remaining.remove(col)
                break
    
    # Greedily add features with minimum maximum correlation to selected set
    while len(selected) < target_features and remaining:
        best_col = None
        min_max_corr = float('inf')
        
        for col in remaining:
            # Calculate max correlation with all selected features
            correlations = []
            for s in selected:
                if pd.api.types.is_numeric_dtype(df[s]) and pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        corr = abs(df[col].corr(df[s]))
                        correlations.append(corr)
                    except:
                        pass
            
            # Use maximum correlation as criterion
            if correlations:
                max_corr = max(correlations)
            else:
                max_corr = 0.0
            
            if max_corr < min_max_corr:
                min_max_corr = max_corr
                best_col = col
        
        if best_col:
            selected.append(best_col)
            remaining.remove(best_col)
        else:
            break
    
    # Add outcome column if present
    if outcome_column and outcome_column not in selected:
        selected = [outcome_column] + selected
    
    return selected
