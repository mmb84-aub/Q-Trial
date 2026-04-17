from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


ID_HINTS = {"id", "subject", "patient", "subjectid", "patientid", "subject_id", "patient_id"}


# ── main function ─────────────────────────────────────────────────────────────

def build_dataset_evidence(df: pd.DataFrame, quantum_evidence: dict[str, Any] | None = None) -> dict:
    """
    Compute deterministic, JSON-friendly evidence from a DataFrame.
    Does NOT call any LLM.  All values are Python primitives.
    
    Args:
        df: Input DataFrame
        quantum_evidence: Optional output from QUBO feature selection
    """
    evidence: dict[str, Any] = {}

    # 1) missingness_pct per column
    evidence["missingness_pct"] = {
        c: round(float(df[c].isna().mean() * 100), 2) for c in df.columns
    }

    # 2) duplicate check on likely ID columns
    id_cols = [c for c in df.columns if c.lower().replace(" ", "") in ID_HINTS]
    id_duplicates: dict[str, Any] = {}
    for col in id_cols:
        dup_count = int(df[col].dropna().duplicated().sum())
        id_duplicates[col] = {
            "duplicate_count": dup_count,
            "total_non_null": int(df[col].notna().sum()),
        }
    evidence["id_duplicates"] = id_duplicates

    # 3) constant columns (nunique == 1, or all null)
    constant_cols = []
    for c in df.columns:
        if df[c].dropna().nunique() <= 1:
            constant_cols.append(c)
    evidence["constant_columns"] = constant_cols

    # 4) cardinality per column
    evidence["cardinality"] = {
        c: int(df[c].nunique(dropna=True)) for c in df.columns
    }

    # 5) outlier flags for numeric columns (IQR method)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_flags: dict[str, Any] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        if outliers.empty:
            continue
        midpoint = (q1 + q3) / 2
        sorted_outliers = outliers.reindex(
            outliers.subtract(midpoint).abs().sort_values(ascending=False).index
        )
        top_values = [_safe_float(v) for v in sorted_outliers.head(3).values]
        outlier_flags[col] = {
            "count": int(len(outliers)),
            "pct_of_col": round(float(len(outliers) / len(series) * 100), 2),
            "lower_fence": _safe_float(lower),
            "upper_fence": _safe_float(upper),
            "top_outlier_values": [v for v in top_values if v is not None],
        }
    evidence["outlier_flags"] = outlier_flags

    # 6) top absolute correlations for numeric columns
    corr_pairs: list[dict[str, Any]] = []
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        seen: set[frozenset] = set()
        rows = []
        for col_a in corr_matrix.columns:
            for col_b in corr_matrix.columns:
                if col_a == col_b:
                    continue
                key = frozenset({col_a, col_b})
                if key in seen:
                    continue
                seen.add(key)
                val = _safe_float(corr_matrix.loc[col_a, col_b])
                if val is not None:
                    rows.append({"col_a": col_a, "col_b": col_b, "abs_corr": val})
        rows.sort(key=lambda x: x["abs_corr"], reverse=True)
        corr_pairs = rows[:10]
    evidence["top_correlations"] = corr_pairs

    # 7) value distribution for low-cardinality categorical columns
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    cat_distributions: dict[str, Any] = {}
    for col in cat_cols:
        if df[col].nunique(dropna=True) <= 20:
            counts = df[col].value_counts(dropna=False).head(10)
            cat_distributions[col] = {str(k): int(v) for k, v in counts.items()}
    evidence["categorical_distributions"] = cat_distributions

    # 8) Add quantum evidence if provided
    if quantum_evidence is not None:
        evidence["quantum_feature_selection"] = quantum_evidence

    return evidence


# ── citation formatter ────────────────────────────────────────────────────────

def format_citations(evidence: dict) -> dict[str, list[str]]:
    """
    Convert raw evidence into resolvable, human-readable citation strings.

    FORMAT RULES (enforced here, injected verbatim into agent prompts):

    top_correlations  → INDEXED list form ONLY:
        "top_correlations[0]: (albumin, status)=0.895"
        NEVER "top_correlations.albumin" or "top_correlations.albumin_status"
        because top_correlations is a LIST, not a dict keyed by column name.

    outlier_flags     → dotted column key (is a dict):
        "outlier_flags.bili: count=2, pct=13.33%, fences=[-0.675, 6.675], top=[14.5, 12.6]"

    missingness_pct   → dotted column key:
        "missingness_pct.bili=0.0%"

    id_duplicates     → dotted column key:
        "id_duplicates.id: duplicate_count=0, total_non_null=15"

    cardinality       → dotted column key:
        "cardinality.sex=2"

    categorical_distributions → dotted column key:
        "categorical_distributions.sex: f=12, m=3"

    constant_columns  → single entry (it is a list):
        "constant_columns: []"
    """
    citations: dict[str, list[str]] = {}

    # top_correlations: INDEXED because it is a list
    # "top_correlations.albumin" is INVALID — use "top_correlations[i]" ONLY
    corr_lines: list[str] = []
    for i, pair in enumerate(evidence.get("top_correlations", [])):
        corr_lines.append(
            f"top_correlations[{i}]: ({pair['col_a']}, {pair['col_b']})={pair['abs_corr']}"
        )
    citations["top_correlations"] = corr_lines

    # outlier_flags: dict keyed by column name
    outlier_lines: list[str] = []
    for col, info in evidence.get("outlier_flags", {}).items():
        outlier_lines.append(
            f"outlier_flags.{col}: count={info['count']}, "
            f"pct={info['pct_of_col']}%, "
            f"fences=[{info['lower_fence']}, {info['upper_fence']}], "
            f"top={info['top_outlier_values']}"
        )
    citations["outlier_flags"] = outlier_lines

    # missingness_pct: dict keyed by column
    miss_lines: list[str] = []
    for col, pct in evidence.get("missingness_pct", {}).items():
        miss_lines.append(f"missingness_pct.{col}={pct}%")
    citations["missingness_pct"] = miss_lines

    # id_duplicates: dict keyed by column
    dup_lines: list[str] = []
    for col, info in evidence.get("id_duplicates", {}).items():
        dup_lines.append(
            f"id_duplicates.{col}: duplicate_count={info['duplicate_count']}, "
            f"total_non_null={info['total_non_null']}"
        )
    citations["id_duplicates"] = dup_lines

    # cardinality: dict keyed by column
    card_lines: list[str] = []
    for col, n in evidence.get("cardinality", {}).items():
        card_lines.append(f"cardinality.{col}={n}")
    citations["cardinality"] = card_lines

    # categorical_distributions: dict keyed by column
    cat_lines: list[str] = []
    for col, dist in evidence.get("categorical_distributions", {}).items():
        pairs = ", ".join(f"{k}={v}" for k, v in dist.items())
        cat_lines.append(f"categorical_distributions.{col}: {pairs}")
    citations["categorical_distributions"] = cat_lines

    # constant_columns: list → single entry
    citations["constant_columns"] = [
        f"constant_columns: {evidence.get('constant_columns', [])}"
    ]

    return citations
