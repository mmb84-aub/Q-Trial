"""
Pairwise comparison tool — Stage 4 statistical tool.

Input:  pd.DataFrame + outcome_col + group_col + correction method
        (tukey|bonferroni|dunn|games_howell).
Output: PairwiseTestResult with all group pairs, raw and adjusted p-values,
        mean differences, and significance flags.
Does:   runs post-hoc pairwise comparisons after a significant omnibus test
        (ANOVA/Kruskal) to identify which specific group pairs differ.
"""
from __future__ import annotations

import itertools

import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class PairwiseTestParams(BaseModel):
    numeric_column: str = Field(description="Numeric column to compare across groups")
    group_column: str = Field(description="Categorical column defining the groups")
    alpha: float = Field(
        default=0.05,
        description="Significance level before Bonferroni correction (default 0.05).",
    )


@tool(
    name="pairwise_group_test",
    description=(
        "Compare a numeric column across all pairs of groups in a categorical column. "
        "Runs a Kruskal-Wallis overall test first, then pairwise Mann-Whitney U tests "
        "with Bonferroni correction. Returns a matrix of adjusted p-values and a "
        "significance flag for each pair. Use this when you have 3+ groups."
    ),
    params_model=PairwiseTestParams,
    category="stats",
)
def pairwise_group_test(params: PairwiseTestParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for col in (params.numeric_column, params.group_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {ctx.column_names}")

    labels = sorted(df[params.group_column].dropna().unique())
    if len(labels) < 2:
        raise ValueError(f"Need at least 2 groups in '{params.group_column}', found {len(labels)}.")

    # Track rows before and after listwise deletion per group
    groups: dict[str, pd.Series] = {}
    group_rows_dropped: dict[str, int] = {}
    group_n_before: dict[str, int] = {}
    for lbl in labels:
        raw = df.loc[df[params.group_column] == lbl, params.numeric_column]
        clean = raw.dropna()
        groups[str(lbl)] = clean
        group_n_before[str(lbl)] = int(len(raw))
        group_rows_dropped[str(lbl)] = int(raw.isna().sum())

    total_rows_dropped = sum(group_rows_dropped.values())
    group_arrays = [g.to_numpy(dtype=float) for g in groups.values()]

    # Overall Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(*group_arrays)

    # Pairwise Mann-Whitney U with Bonferroni
    pairs = list(itertools.combinations(groups.keys(), 2))
    n_comparisons = len(pairs)
    bonferroni_alpha = params.alpha / n_comparisons

    pairwise: list[dict] = []
    for a_lbl, b_lbl in pairs:
        a_arr = groups[a_lbl].to_numpy(dtype=float)
        b_arr = groups[b_lbl].to_numpy(dtype=float)
        if len(a_arr) < 2 or len(b_arr) < 2:
            pairwise.append({
                "group_a": a_lbl, "group_b": b_lbl,
                "skipped": "too few observations",
            })
            continue
        mw_stat, raw_p = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
        adj_p = min(float(raw_p) * n_comparisons, 1.0)  # Bonferroni
        pairwise.append({
            "group_a": a_lbl,
            "group_b": b_lbl,
            "n_a": int(len(a_arr)),
            "n_b": int(len(b_arr)),
            "mean_a": round(float(a_arr.mean()), 4),
            "mean_b": round(float(b_arr.mean()), 4),
            "median_a": round(float(pd.Series(a_arr).median()), 4),
            "median_b": round(float(pd.Series(b_arr).median()), 4),
            "mw_statistic": round(float(mw_stat), 4),
            "raw_p_value": round(float(raw_p), 6),
            "bonferroni_p_value": round(adj_p, 6),
            "significant_after_correction": bool(adj_p < params.alpha),
        })

    return {
        "numeric_column": params.numeric_column,
        "group_column": params.group_column,
        "n_groups": len(labels),
        "groups": list(groups.keys()),
        "group_sizes": {k: int(len(v)) for k, v in groups.items()},
        "kruskal_wallis": {
            "statistic": round(float(kw_stat), 4),
            "p_value": round(float(kw_p), 6),
            "significant": bool(kw_p < params.alpha),
        },
        "bonferroni_alpha": round(bonferroni_alpha, 6),
        "n_comparisons": n_comparisons,
        "pairwise_results": pairwise,
        "listwise_deletion": {
            "rows_dropped_per_group": group_rows_dropped,
            "total_rows_dropped": total_rows_dropped,
            "note": "Rows with missing values in numeric_column were excluded per group.",
        },
    }
