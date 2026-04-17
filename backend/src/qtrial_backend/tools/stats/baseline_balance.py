"""
Baseline balance checker — Stage 4 statistical tool.

Input:  pd.DataFrame + treatment_col + list of covariate columns.
Output: BaselineBalanceResult with standardised mean differences (SMD) per
        covariate and balance flags (|SMD| > 0.1).
Does:   verifies treatment arms are balanced at baseline; flags confounders
        that should be included in any adjusted analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class BaselineBalanceParams(BaseModel):
    treatment_column: str = Field(
        description="Column containing treatment/group assignment (e.g. 'trt')"
    )
    baseline_columns: list[str] = Field(
        description="Baseline characteristic columns to assess for balance"
    )
    smd_threshold: float = Field(
        default=0.1,
        description=(
            "SMD threshold above which a variable is flagged as imbalanced. "
            "Default 0.1 — the conventional threshold in RCT reporting."
        ),
    )


def _smd_continuous(a: np.ndarray, b: np.ndarray) -> float | None:
    """Standardised mean difference for continuous variables."""
    if len(a) < 2 or len(b) < 2:
        return None
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_sd = np.sqrt((var_a + var_b) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def _smd_binary(p_a: float, p_b: float) -> float | None:
    """SMD for binary proportions (Cohen's h-like)."""
    denom = np.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / 2)
    if denom == 0:
        return None
    return float((p_a - p_b) / denom)


@tool(
    name="baseline_balance",
    description=(
        "Produce a Table 1 / baseline balance table for a clinical trial. "
        "For continuous variables: mean ± SD per arm and standardised mean difference (SMD). "
        "For categorical variables: n (%) per arm and SMD on the predominant level. "
        "Variables with |SMD| > threshold are flagged as potentially imbalanced. "
        "|SMD| < 0.1 is the conventional well-balanced threshold in RCTs."
    ),
    params_model=BaselineBalanceParams,
    category="stats",
)
def baseline_balance(params: BaselineBalanceParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for col in [params.treatment_column] + params.baseline_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {ctx.column_names}")

    arms = sorted(df[params.treatment_column].dropna().unique())
    arm_labels = [str(a) for a in arms]
    group_sizes = {
        str(arm): int((df[params.treatment_column] == arm).sum()) for arm in arms
    }

    rows: list[dict] = []
    imbalanced: list[str] = []

    for col in params.baseline_columns:
        series = df[col]
        row: dict = {"variable": col, "arms": {}}

        if pd.api.types.is_numeric_dtype(series):
            arm_arrays: list[np.ndarray] = []
            for arm in arms:
                vals = df.loc[df[params.treatment_column] == arm, col].dropna()
                arm_arrays.append(vals.to_numpy(dtype=float))
                row["arms"][str(arm)] = {
                    "n": int(len(vals)),
                    "mean": round(float(vals.mean()), 4) if len(vals) else None,
                    "sd": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None,
                    "median": round(float(vals.median()), 4) if len(vals) else None,
                }

            smd = (
                _smd_continuous(arm_arrays[0], arm_arrays[1])
                if len(arms) == 2
                else None
            )
            row["type"] = "continuous"

        else:
            for arm in arms:
                arm_df = df[df[params.treatment_column] == arm]
                n_arm = int(arm_df[col].notna().sum())
                vc = arm_df[col].value_counts(normalize=True).round(4)
                row["arms"][str(arm)] = {
                    "n": n_arm,
                    "proportions": {str(k): float(v) for k, v in vc.items()},
                }

            # SMD for binary columns on most common level
            smd = None
            if len(arms) == 2 and series.nunique() <= 2:
                levels = series.dropna().unique()
                if len(levels) >= 1:
                    ref = levels[0]
                    p_a = float(
                        (df[df[params.treatment_column] == arms[0]][col] == ref).mean()
                    )
                    p_b = float(
                        (df[df[params.treatment_column] == arms[1]][col] == ref).mean()
                    )
                    smd = _smd_binary(p_a, p_b)
            row["type"] = "categorical"

        row["smd"] = round(smd, 4) if smd is not None else None
        is_imbalanced = smd is not None and abs(smd) > params.smd_threshold
        row["imbalanced"] = is_imbalanced
        if is_imbalanced:
            imbalanced.append(col)

        rows.append(row)

    return {
        "treatment_column": params.treatment_column,
        "arms": arm_labels,
        "group_sizes": group_sizes,
        "smd_threshold": params.smd_threshold,
        "n_variables": len(rows),
        "n_imbalanced": len(imbalanced),
        "imbalanced_variables": imbalanced,
        "interpretation": (
            "SMD is the standardised mean difference. |SMD| < 0.1 indicates "
            "good balance; 0.1-0.2 mild imbalance; > 0.2 notable imbalance."
        ),
        "table": rows,
    }
