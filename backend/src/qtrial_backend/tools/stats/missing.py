from __future__ import annotations

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
