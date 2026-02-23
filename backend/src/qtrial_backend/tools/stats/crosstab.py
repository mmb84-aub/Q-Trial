from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import chi2_contingency, fisher_exact

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class CrosstabParams(BaseModel):
    row_column: str = Field(description="Column for cross-tab rows")
    col_column: str = Field(description="Column for cross-tab columns")
    normalize: bool = Field(default=False, description="Return proportions")
    margins: bool = Field(default=False, description="Include row/col totals")


@tool(
    name="cross_tabulation",
    description=(
        "Compute a cross-tabulation of two columns with chi-square test. "
        "Automatically falls back to Fisher's exact test when any expected cell count < 5 "
        "(standard practice for small samples). "
        "Returns the contingency table, chi-square statistic, p-value, Cramér's V effect size, "
        "and a flag indicating whether Fisher's exact was used."
    ),
    params_model=CrosstabParams,
    category="stats",
)
def cross_tabulation(params: CrosstabParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for c in (params.row_column, params.col_column):
        if c not in df.columns:
            raise ValueError(
                f"Column '{c}' not found. Available: {ctx.column_names}"
            )

    ct = pd.crosstab(
        df[params.row_column],
        df[params.col_column],
        margins=params.margins,
        normalize="all" if params.normalize else False,
    )

    if params.normalize:
        ct = ct.round(4)

    # ── Significance test on the raw (non-normalised, no margins) table ───
    ct_raw = pd.crosstab(df[params.row_column], df[params.col_column])
    test_info: dict = {}
    try:
        chi2, p_chi2, dof, expected = chi2_contingency(ct_raw.values)
        min_expected = float(np.min(expected))
        use_fisher = min_expected < 5 and ct_raw.shape == (2, 2)

        if use_fisher:
            _, p_fisher = fisher_exact(ct_raw.values)
            test_info = {
                "test": "Fisher's exact",
                "p_value": round(float(p_fisher), 6),
                "reason": f"Min expected cell count = {min_expected:.2f} < 5",
            }
        else:
            # Cramér's V
            n_total = int(ct_raw.values.sum())
            k = min(ct_raw.shape) - 1
            cramers_v = float(np.sqrt(chi2 / (n_total * k))) if k > 0 and n_total > 0 else None
            test_info = {
                "test": "Chi-square",
                "statistic": round(float(chi2), 4),
                "p_value": round(float(p_chi2), 6),
                "degrees_of_freedom": int(dof),
                "min_expected_cell": round(min_expected, 2),
                "cramers_v": round(cramers_v, 4) if cramers_v is not None else None,
                "cramers_v_magnitude": (
                    "negligible" if cramers_v is not None and cramers_v < 0.1
                    else "small" if cramers_v is not None and cramers_v < 0.3
                    else "medium" if cramers_v is not None and cramers_v < 0.5
                    else "large"
                ) if cramers_v is not None else None,
            }
    except Exception as exc:
        test_info = {"test_error": str(exc)}

    return {
        "row_column": params.row_column,
        "col_column": params.col_column,
        "normalized": params.normalize,
        "n_total": int(len(df[[params.row_column, params.col_column]].dropna())),
        "significance_test": test_info,
        "table": {
            str(idx): {str(col): float(ct.loc[idx, col]) for col in ct.columns}
            for idx in ct.index
        },
    }
