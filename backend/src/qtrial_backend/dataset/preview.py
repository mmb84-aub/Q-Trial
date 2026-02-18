from __future__ import annotations

import pandas as pd


def build_dataset_preview(
    df: pd.DataFrame,
    max_rows: int = 25,
    max_cols: int = 30,
) -> dict:
    """
    Keep payload small & safe for LLM context:
    - schema
    - head rows
    - missingness
    - basic numeric stats
    """
    df2 = df.copy()

    # limit cols
    if df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]

    # convert dtypes to readable strings
    schema = {c: str(df2[c].dtype) for c in df2.columns}

    head = df2.head(max_rows)
    # make JSON-friendly (avoid NaN)
    head_records = head.where(pd.notnull(head), None).to_dict(orient="records")

    missing = (df2.isna().mean() * 100.0).round(2).to_dict()

    numeric_cols = df2.select_dtypes(include="number").columns.tolist()
    numeric_summary = {}
    if numeric_cols:
        desc = df2[numeric_cols].describe().transpose()
        numeric_summary = {
            col: {
                "count": float(desc.loc[col, "count"]),
                "mean": float(desc.loc[col, "mean"]) if pd.notnull(desc.loc[col, "mean"]) else None,
                "std": float(desc.loc[col, "std"]) if pd.notnull(desc.loc[col, "std"]) else None,
                "min": float(desc.loc[col, "min"]) if pd.notnull(desc.loc[col, "min"]) else None,
                "25%": float(desc.loc[col, "25%"]) if pd.notnull(desc.loc[col, "25%"]) else None,
                "50%": float(desc.loc[col, "50%"]) if pd.notnull(desc.loc[col, "50%"]) else None,
                "75%": float(desc.loc[col, "75%"]) if pd.notnull(desc.loc[col, "75%"]) else None,
                "max": float(desc.loc[col, "max"]) if pd.notnull(desc.loc[col, "max"]) else None,
            }
            for col in numeric_cols
        }

    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "schema": schema,
        "head": head_records,
        "missingness_pct": missing,
        "numeric_summary": numeric_summary,
        "notes": {
            "truncated_rows": int(max(0, df.shape[0] - max_rows)),
            "truncated_cols": int(max(0, df.shape[1] - max_cols)),
        },
    }
