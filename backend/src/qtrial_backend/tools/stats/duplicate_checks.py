"""
Duplicate detection tool — Stage 4 statistical tool.

Input:  pd.DataFrame + optional subset of columns to check.
Output: DuplicateCheckResult with total duplicate row count, duplicate rate,
        and sample duplicate rows grouped by key columns.
Does:   identifies exact and near-duplicate records that could inflate
        sample size estimates or bias statistical tests.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class DuplicateCheckParams(BaseModel):
    key_columns: Optional[list[str]] = Field(
        default=None,
        description=(
            "Columns that should uniquely identify a row (e.g. ['subject_id', 'visit_id']). "
            "Leave empty to check for exact full-row duplicates only."
        ),
    )
    subject_column: Optional[str] = Field(
        default=None,
        description="Column containing subject/patient ID for counting repeated subjects.",
    )


@tool(
    name="duplicate_checks",
    description=(
        "Detect duplicate rows in the dataset. "
        "Checks for (1) exact full-row duplicates, (2) duplicates on key columns "
        "such as subject_id + visit_id, and (3) repeated subject observations. "
        "Returns counts, percentages, and up to 10 example row indices per check."
    ),
    params_model=DuplicateCheckParams,
    category="stats",
)
def duplicate_checks(params: DuplicateCheckParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    n_total = len(df)
    result: dict = {"n_total_rows": n_total}

    # ── Exact full-row duplicates ─────────────────────────────────────
    exact_mask = df.duplicated(keep=False)
    exact_count = int(exact_mask.sum())
    result["exact_duplicates"] = {
        "count": exact_count,
        "pct": round(exact_count / n_total * 100, 2),
        "example_indices": df.index[exact_mask].tolist()[:10],
    }

    # ── Key column duplicates ─────────────────────────────────────────
    if params.key_columns:
        missing = [c for c in params.key_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Key columns not found: {missing}. Available: {ctx.column_names}"
            )
        key_mask = df.duplicated(subset=params.key_columns, keep=False)
        key_count = int(key_mask.sum())
        result["key_column_duplicates"] = {
            "key_columns": params.key_columns,
            "n_duplicated_rows": key_count,
            "pct": round(key_count / n_total * 100, 2),
            "n_unique_key_combinations": int(
                df.drop_duplicates(subset=params.key_columns).shape[0]
            ),
            "example_indices": df.index[key_mask].tolist()[:10],
        }

    # ── Subject-level analysis ────────────────────────────────────────
    if params.subject_column:
        if params.subject_column not in df.columns:
            raise ValueError(
                f"Subject column '{params.subject_column}' not found. "
                f"Available: {ctx.column_names}"
            )
        counts = df[params.subject_column].value_counts()
        repeated = counts[counts > 1]
        result["subject_analysis"] = {
            "subject_column": params.subject_column,
            "n_unique_subjects": int(counts.nunique()),
            "n_subjects_with_multiple_rows": int(len(repeated)),
            "max_rows_per_subject": int(counts.max()),
            "mean_rows_per_subject": round(float(counts.mean()), 2),
            "top_repeated_subjects": {
                str(k): int(v) for k, v in repeated.head(5).items()
            },
        }

    return result
