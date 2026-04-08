from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class MissingPatternsParams(BaseModel):
    columns: list[str] | None = Field(
        default=None, description="Columns to check. Null = all columns."
    )


@tool(
    name="missing_data_patterns",
    description=(
        "Analyse missing data patterns: per-column missing counts, "
        "pairwise co-missingness, and rows with the most null values."
    ),
    params_model=MissingPatternsParams,
    category="stats",
)
def missing_data_patterns(params: MissingPatternsParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    if params.columns:
        missing = [c for c in params.columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. Available: {ctx.column_names}"
            )
        df = df[params.columns]

    null_mask = df.isna()
    per_column = {
        col: {
            "null_count": int(null_mask[col].sum()),
            "null_pct": round(float(null_mask[col].mean() * 100), 2),
        }
        for col in df.columns
    }

    # Co-missingness: which columns tend to be missing together
    cols_with_missing = [c for c in df.columns if null_mask[c].any()]
    co_missing: dict = {}
    if len(cols_with_missing) > 1:
        subset = null_mask[cols_with_missing[:15]]  # Cap for size
        corr = subset.corr().round(3)
        co_missing = corr.where(pd.notnull(corr), None).to_dict()

    # Rows with most nulls
    null_per_row = null_mask.sum(axis=1)
    worst_rows = (
        null_per_row.nlargest(5)
        .to_dict()
    )

    return {
        "total_rows": int(df.shape[0]),
        "total_cols": int(df.shape[1]),
        "rows_with_any_null": int(null_mask.any(axis=1).sum()),
        "complete_rows": int((~null_mask.any(axis=1)).sum()),
        "per_column": per_column,
        "co_missingness_correlation": co_missing,
        "rows_with_most_nulls": {
            str(k): int(v) for k, v in worst_rows.items()
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Little's MCAR Test  (standalone function, not a @tool)
# ──────────────────────────────────────────────────────────────────────────────

def _little_mcar_manual(num_df: pd.DataFrame) -> tuple[float, int]:
    """Chi-square approximation of Little's MCAR statistic.

    Returns (chi2_statistic, degrees_of_freedom).
    Columns with no missing data are included in the covariance but not counted
    toward pattern degrees of freedom.
    """
    data = num_df.to_numpy(dtype=float)
    n_rows, n_cols = data.shape

    # Grand means (column-wise, ignoring NaN)
    grand_mean = np.nanmean(data, axis=0)  # shape (n_cols,)

    # Grand covariance (pairwise complete observations via pandas)
    cov_matrix = num_df.cov().to_numpy()  # shape (n_cols, n_cols)

    # Observed mask: True = observed (not NaN)
    observed_mask = ~np.isnan(data)  # shape (n_rows, n_cols)

    # Map each row to a unique pattern string
    pattern_keys = [tuple(row.tolist()) for row in observed_mask]
    unique_patterns = list({p: None for p in pattern_keys})  # preserve order, deduplicate

    chi2_stat = 0.0
    total_df_numerator = 0

    for pattern in unique_patterns:
        # Rows matching this pattern
        rows_in_pattern = [i for i, pk in enumerate(pattern_keys) if pk == pattern]
        n_k = len(rows_in_pattern)

        # Which columns are observed in this pattern
        obs_vars = [j for j, obs in enumerate(pattern) if obs]

        if len(obs_vars) == 0 or n_k == 0:
            continue

        # Mean of observed variables in this pattern
        pattern_data = data[np.ix_(rows_in_pattern, obs_vars)]
        pattern_mean = np.nanmean(pattern_data, axis=0)  # shape (len(obs_vars),)

        mu_k = grand_mean[obs_vars]  # overall mean for these vars
        diff = pattern_mean - mu_k

        # Sub-covariance matrix for observed variables
        cov_k = cov_matrix[np.ix_(obs_vars, obs_vars)]

        try:
            cov_inv = np.linalg.pinv(cov_k)
        except np.linalg.LinAlgError:
            continue

        chi2_stat += n_k * float(diff @ cov_inv @ diff)
        total_df_numerator += len(obs_vars)

    # Columns with any missingness contribute 1 df each (for estimated means)
    n_missing_cols = int(np.any(np.isnan(data), axis=0).sum())
    df_val = max(total_df_numerator - n_missing_cols, 1)

    return float(chi2_stat), df_val


def little_mcar_test(df: pd.DataFrame) -> dict:
    """Little's MCAR test: assess whether missingness is completely at random.

    Parameters
    ----------
    df:
        DataFrame to test. Only numeric columns are analysed.

    Returns
    -------
    dict with key 'little_mcar_test' containing chi-square statistic,
    degrees of freedom, p-value, classification ('MCAR' or 'MAR or MNAR'),
    and a one-sentence interpretation.

    References
    ----------
    Little, R. J. A. (1988). A test of missing completely at random for
    multivariate data with missing values. Journal of the American Statistical
    Association, 83(404), 1198–1202.
    """
    from scipy.stats import chi2 as chi2_dist

    num_df = df.select_dtypes(include="number")
    missing_cols = [c for c in num_df.columns if num_df[c].isna().any()]

    if not missing_cols:
        return {
            "little_mcar_test": {
                "chi_square": None,
                "degrees_of_freedom": None,
                "p_value": None,
                "classification": "MCAR",
                "n_patterns": 1,
                "missing_columns": [],
                "interpretation": (
                    "No missing data found — MCAR classification is not applicable."
                ),
            }
        }

    # Attempt pyampute first; fall back to manual chi-square approximation
    chi2_val: float | None = None
    df_val: int | None = None
    p_value: float | None = None

    try:
        from pyampute.exploration.mcar_statistical_tests import MCARTest  # type: ignore

        test = MCARTest(method="little")
        result = test.little_mcar_test(num_df.copy())
        # pyampute may return a named-tuple or numeric p-value depending on version
        if hasattr(result, "pvalue"):
            p_value = float(result.pvalue)
            chi2_val = float(getattr(result, "statistic", float("nan")))
            df_val = int(getattr(result, "df", 0)) or None
        else:
            p_value = float(result)
    except Exception:
        pass  # fall back to manual

    if p_value is None:
        chi2_val, df_val = _little_mcar_manual(num_df[missing_cols])
        p_value = float(1.0 - chi2_dist.cdf(chi2_val, df=df_val))

    # Classify
    classification = "MCAR" if p_value >= 0.05 else "MAR or MNAR"

    if classification == "MCAR":
        interpretation = (
            f"Fail to reject MCAR (p={p_value:.3f}): missingness appears completely "
            "at random — listwise deletion is unbiased."
        )
    else:
        interpretation = (
            f"Reject MCAR (p={p_value:.3f}): missingness is likely MAR or MNAR — "
            "consider MICE imputation to avoid biased estimates."
        )

    # Count unique patterns across missing columns only
    observed_mask = num_df[missing_cols].notna()
    n_patterns = int(observed_mask.drop_duplicates().shape[0])

    return {
        "little_mcar_test": {
            "chi_square": round(float(chi2_val), 4) if chi2_val is not None else None,
            "degrees_of_freedom": int(df_val) if df_val is not None else None,
            "p_value": round(float(p_value), 6),
            "classification": classification,
            "n_patterns": n_patterns,
            "missing_columns": missing_cols,
            "interpretation": interpretation,
        }
    }
