"""
Input: pd.DataFrame + list of MAR columns to impute + number of imputations m
Output: pooled statistics per column using Rubin's Rules
Purpose: Correct for MAR (Missing At Random) bias. Listwise deletion under MAR produces
  biased estimates. MICE generates m complete datasets and pools results.
Reference: Rubin (1987) Multiple Imputation for Nonresponse in Surveys;
  van Buuren & Groothuis-Oudshoorn (2011) mice: Multivariate Imputation by Chained
  Equations in R. Journal of Statistical Software, 45(3), 1–67.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class MICEParams(BaseModel):
    target_columns: list[str] = Field(
        description=(
            "Columns with missing values to impute. Must be numeric. "
            "Typically these are columns classified as MAR by little_mcar_test."
        )
    )
    m: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of independent imputations to perform (default 5). Range 2–20.",
    )


def _get_iterative_imputer(random_state: int):
    """Import IterativeImputer, handling the experimental flag for older sklearn."""
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    except ImportError:
        pass
    from sklearn.impute import IterativeImputer  # type: ignore[import]

    return IterativeImputer(max_iter=10, random_state=random_state)


def _mice_logic(
    df: pd.DataFrame, target_columns: list[str], m: int = 5
) -> dict:
    """Core MICE logic — callable directly with a DataFrame for programmatic use."""
    # Validate columns
    missing_cols = [c for c in target_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")

    non_numeric = [
        c for c in target_columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(f"MICE requires numeric columns. Non-numeric: {non_numeric}")

    # Use only numeric columns for imputation (IterativeImputer requires numeric)
    numeric_df = df.select_dtypes(include="number").copy()

    # Track original missing counts
    original_missing = {col: int(df[col].isna().sum()) for col in target_columns}

    # Run m independent imputations with different seeds
    seeds = list(range(42, 42 + m))
    imputed_means: list[np.ndarray] = []
    imputed_vars: list[np.ndarray] = []

    target_idx = [list(numeric_df.columns).index(c) for c in target_columns]

    for seed in seeds:
        imp = _get_iterative_imputer(seed)
        imputed_array = imp.fit_transform(numeric_df)
        imputed_vals = imputed_array[:, target_idx]  # shape (n_rows, n_target)
        imputed_means.append(np.mean(imputed_vals, axis=0))
        imputed_vars.append(np.var(imputed_vals, axis=0, ddof=1))

    # Pool using Rubin's Rules for each target column
    means_matrix = np.stack(imputed_means, axis=0)  # shape (m, n_target)
    vars_matrix = np.stack(imputed_vars, axis=0)     # shape (m, n_target)

    pooled_means = np.mean(means_matrix, axis=0)
    within_variance = np.mean(vars_matrix, axis=0)
    between_variance = np.var(means_matrix, axis=0, ddof=1) if m > 1 else np.zeros(len(target_columns))
    total_variance = within_variance + (1.0 + 1.0 / m) * between_variance
    pooled_se = np.sqrt(np.maximum(total_variance, 0.0))

    results = []
    for j, col in enumerate(target_columns):
        results.append(
            {
                "column": col,
                "original_missing_count": original_missing[col],
                "pooled_mean": round(float(pooled_means[j]), 4),
                "pooled_std": round(float(np.sqrt(max(within_variance[j], 0.0))), 4),
                "ci_lower": round(float(pooled_means[j] - 1.96 * pooled_se[j]), 4),
                "ci_upper": round(float(pooled_means[j] + 1.96 * pooled_se[j]), 4),
                "rubins_between_variance": round(float(between_variance[j]), 6),
                "rubins_within_variance": round(float(within_variance[j]), 6),
            }
        )

    return {
        "imputed_columns": results,
        "m_imputations": m,
        "method": "MICE (Multiple Imputation by Chained Equations)",
        "rubins_rules_applied": True,
        "imputed_dataframe": None,  # Full imputed df not serialised; use get_imputed_dataframe()
    }


def get_imputed_dataframe(df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """Return a single complete imputed DataFrame using random_state=42.

    This produces one multiply-imputed dataset for use in downstream analyses.
    For valid inference, call mice_imputation() and apply Rubin's Rules instead.
    """
    missing_cols = [c for c in target_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")

    numeric_df = df.select_dtypes(include="number").copy()
    imp = _get_iterative_imputer(42)
    imputed_array = imp.fit_transform(numeric_df)
    imputed_numeric = pd.DataFrame(
        imputed_array, columns=numeric_df.columns, index=df.index
    )

    result_df = df.copy()
    for col in target_columns:
        if col in imputed_numeric.columns:
            result_df[col] = imputed_numeric[col].values
    return result_df


@tool(
    name="mice_imputation",
    description=(
        "Multiple Imputation by Chained Equations (MICE) for Missing At Random (MAR) data. "
        "Runs m independent imputations using IterativeImputer, then pools results with "
        "Rubin's Rules to obtain valid pooled estimates and 95% confidence intervals. "
        "Call this when little_mcar_test classifies missingness as 'MAR or MNAR'. "
        "Returns pooled mean, within/between imputation variances, and 95% CI per column."
    ),
    params_model=MICEParams,
    category="stats",
)
def mice_imputation(params: MICEParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    missing = [c for c in params.target_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found: {missing}. Available: {ctx.column_names}"
        )
    return _mice_logic(df, params.target_columns, params.m)
