from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class SampleRowsParams(BaseModel):
    n: int = Field(default=5, description="Number of rows to return")
    filter_column: str | None = Field(
        default=None, description="Column to filter on"
    )
    filter_value: str | None = Field(
        default=None, description="Value to filter for (string comparison)"
    )
    random: bool = Field(
        default=True, description="Random sample vs first N rows"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility (default 42).",
    )


@tool(
    name="sample_rows",
    description=(
        "Return sample rows from the dataset. "
        "Optionally filter by a column value. "
        "Useful for inspecting raw data when aggregate stats look unusual."
    ),
    params_model=SampleRowsParams,
    category="stats",
)
def sample_rows(params: SampleRowsParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe

    if params.filter_column:
        if params.filter_column not in df.columns:
            raise ValueError(
                f"Column '{params.filter_column}' not found. "
                f"Available: {ctx.column_names}"
            )
        df = df[df[params.filter_column].astype(str) == str(params.filter_value)]

    n = min(params.n, len(df), 20)  # Hard cap at 20 rows
    if n == 0:
        return {"rows": [], "total_matching": 0}

    if params.random and len(df) > n:
        sample = df.sample(n=n, random_state=params.seed)
    else:
        sample = df.head(n)

    return {
        "total_matching": int(len(df)),
        "n_returned": int(n),
        "rows": sample.where(pd.notnull(sample), None).to_dict(orient="records"),
    }
