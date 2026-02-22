from __future__ import annotations

import re

import pandas as pd
from pydantic import BaseModel

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool

_DATE_PATTERNS = [
    r"^\d{4}-\d{2}-\d{2}$",   # ISO 8601: 2024-01-31
    r"^\d{2}/\d{2}/\d{4}$",   # US: 01/31/2024
    r"^\d{2}-\d{2}-\d{4}$",   # EU: 31-01-2024
    r"^\d{4}/\d{2}/\d{2}$",   # Alt ISO: 2024/01/31
]


class TypeCoercionParams(BaseModel):
    pass  # Scans all columns


@tool(
    name="type_coercion_suggestions",
    description=(
        "Scan all columns for likely type mismatches: "
        "numeric values stored as strings, potential date/time columns, "
        "binary indicators stored as floats, and high-cardinality numeric "
        "columns that may actually be identifiers. "
        "Returns actionable suggestions — does not modify data."
    ),
    params_model=TypeCoercionParams,
    category="stats",
)
def type_coercion_suggestions(params: TypeCoercionParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    suggestions: list[dict] = []

    for col in df.columns:
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            continue
        dtype = str(df[col].dtype)

        # ── Object columns ────────────────────────────────────────────
        if dtype == "object":
            # Try numeric conversion
            converted = pd.to_numeric(series, errors="coerce")
            n_ok = int(converted.notna().sum())
            if n_ok / n > 0.90:
                suggestions.append(
                    {
                        "column": col,
                        "current_dtype": dtype,
                        "suggestion": "convert_to_numeric",
                        "reason": f"{n_ok}/{n} ({n_ok/n*100:.1f}%) values parseable as numeric",
                        "n_would_fail": n - n_ok,
                    }
                )
                continue

            # Try date detection on sample
            sample_strs = series.head(30).astype(str)
            for pat in _DATE_PATTERNS:
                matches = int(sample_strs.str.match(pat).sum())
                if matches / min(30, n) > 0.80:
                    suggestions.append(
                        {
                            "column": col,
                            "current_dtype": dtype,
                            "suggestion": "convert_to_datetime",
                            "reason": f"Values match date pattern '{pat}'",
                        }
                    )
                    break

        # ── Float columns ─────────────────────────────────────────────
        elif "float" in dtype:
            unique_vals = set(series.dropna().unique())

            # Binary indicator stored as float
            if unique_vals <= {0.0, 1.0}:
                suggestions.append(
                    {
                        "column": col,
                        "current_dtype": dtype,
                        "suggestion": "convert_to_bool_or_int",
                        "reason": "Contains only 0.0 and 1.0 — likely a binary indicator",
                    }
                )
            # Whole-number float with very high cardinality → ID column
            elif series.apply(lambda x: float(x) == int(x)).all():
                if series.nunique() > max(n * 0.8, 50):
                    suggestions.append(
                        {
                            "column": col,
                            "current_dtype": dtype,
                            "suggestion": "may_be_identifier",
                            "reason": (
                                f"High cardinality ({series.nunique()} unique values), "
                                "all whole numbers — may be an ID column not suitable for modelling"
                            ),
                        }
                    )

        # ── Integer columns with very low cardinality ─────────────────
        elif "int" in dtype:
            n_unique = int(series.nunique())
            if n_unique <= 8 and n_unique < max(n * 0.05, 3):
                suggestions.append(
                    {
                        "column": col,
                        "current_dtype": dtype,
                        "suggestion": "consider_as_categorical",
                        "reason": (
                            f"Only {n_unique} unique integer values — "
                            "may represent a categorical variable (stage, grade, etc.)"
                        ),
                        "unique_values": sorted([int(v) for v in series.unique()]),
                    }
                )

    return {
        "n_columns_scanned": len(df.columns),
        "n_flagged": len(suggestions),
        "suggestions": suggestions,
    }
