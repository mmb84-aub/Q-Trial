from __future__ import annotations

"""
Static report generator.

Runs a fixed, deterministic pipeline of statistical tools against a dataset
and renders the results as a Markdown document.  No LLM is involved — the
report is purely data-driven.
"""

import datetime
import re
from typing import Any

import pandas as pd

from qtrial_backend.agent.context import AgentContext


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call(func, params_cls, ctx: AgentContext, **kwargs) -> dict:
    """Construct params, run a tool function, return {} on any error."""
    try:
        params = params_cls(**kwargs)
        return func(params, ctx)
    except Exception as exc:
        return {"error": str(exc)}


def _fmt(val: Any, digits: int = 4) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    sep = " | ".join(["---"] * len(headers))
    head = " | ".join(headers)
    lines = [f"| {head} |", f"| {sep} |"]
    for row in rows:
        cells = " | ".join(_fmt(c) for c in row)
        lines.append(f"| {cells} |")
    return "\n".join(lines)


# ── Auto-detection ─────────────────────────────────────────────────────────────

_TREATMENT_HINTS = re.compile(
    r"\b(arm|group|trt|treat|treatment|cohort|alloc|rand|assign|intervention|branch)\b",
    re.IGNORECASE,
)
_TIME_HINTS = re.compile(
    r"\b(time|days|months|years|follow|duration|os|pfs|efs|rfs|dfs|ttf|tte|fu)\b",
    re.IGNORECASE,
)
_EVENT_HINTS = re.compile(
    r"\b(event|status|censored|dead|death|died|outcome|indicator|flag|relapse|tx)\b",
    re.IGNORECASE,
)


def _detect_treatment_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if _TREATMENT_HINTS.search(col) and 2 <= df[col].nunique() <= 6:
            return col
    # Fallback: any binary-ish column with 2-3 unique values
    for col in df.columns:
        if 2 <= df[col].nunique() <= 3:
            return col
    return None


def _detect_time_col(df: pd.DataFrame) -> str | None:
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if _TIME_HINTS.search(col):
            return col
    return None


def _detect_event_col(df: pd.DataFrame, time_col: str | None) -> str | None:
    # Look for binary 0/1 columns
    binary_candidates = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            binary_candidates.append(col)

    # Prefer those with event-hint names
    for col in binary_candidates:
        if _EVENT_HINTS.search(col):
            return col

    # Otherwise take the first binary candidate that isn't the time column
    for col in binary_candidates:
        if col != time_col:
            return col
    return None


# ── Section builders ───────────────────────────────────────────────────────────

def _section_overview(df: pd.DataFrame, dataset_name: str) -> str:
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    complete_rows = int(df.dropna().shape[0])
    lines = [
        "## 1. Dataset Overview",
        "",
        f"| Property | Value |",
        f"| --- | --- |",
        f"| File | {dataset_name} |",
        f"| Rows | {n_rows:,} |",
        f"| Columns | {n_cols} |",
        f"| Numeric columns | {len(numeric_cols)} |",
        f"| Categorical columns | {len(cat_cols)} |",
        f"| Complete rows (no nulls) | {complete_rows:,} ({complete_rows/n_rows*100:.1f}%) |",
        f"| Generated | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} |",
        "",
        "**Columns:**",
        "",
        _md_table(
            ["Column", "Type", "Non-null", "Unique"],
            [
                [
                    col,
                    str(df[col].dtype),
                    int(df[col].notna().sum()),
                    int(df[col].nunique()),
                ]
                for col in df.columns
            ],
        ),
    ]
    return "\n".join(lines)


def _section_data_quality(ctx: AgentContext) -> str:
    from qtrial_backend.tools.stats.duplicate_checks import duplicate_checks, DuplicateCheckParams
    from qtrial_backend.tools.stats.missing import missing_data_patterns, MissingPatternsParams
    from qtrial_backend.tools.stats.type_coercion import type_coercion_suggestions, TypeCoercionParams

    dup = _call(duplicate_checks, DuplicateCheckParams, ctx)
    miss = _call(missing_data_patterns, MissingPatternsParams, ctx)
    coerce = _call(type_coercion_suggestions, TypeCoercionParams, ctx)

    lines = ["## 2. Data Quality", ""]

    # Duplicates
    lines.append("### Duplicates")
    if "error" in dup:
        lines.append(f"> Error: {dup['error']}")
    else:
        ed = dup.get("exact_duplicates", {})
        lines.append(
            f"- Exact duplicate rows: **{ed.get('count', 0)}** ({ed.get('pct', 0):.2f}%)"
        )

    # Missingness
    lines += ["", "### Missingness"]
    if "error" in miss:
        lines.append(f"> Error: {miss['error']}")
    else:
        per_col = miss.get("per_column", {})
        cols_with_miss = {c: v for c, v in per_col.items() if v["null_pct"] > 0}
        if not cols_with_miss:
            lines.append("No missing values detected.")
        else:
            table_rows = [
                [col, v["null_count"], f"{v['null_pct']:.2f}%"]
                for col, v in sorted(cols_with_miss.items(), key=lambda x: -x[1]["null_pct"])
            ]
            lines.append(
                _md_table(["Column", "Null count", "Null %"], table_rows)
            )
            lines.append(
                f"\nRows with any null: **{miss.get('rows_with_any_null', '—')}** / "
                f"{miss.get('total_rows', '—')} "
                f"({miss.get('complete_rows', '—')} complete)"
            )

    # Type coercion hints
    lines += ["", "### Type coercion hints"]
    if "error" in coerce:
        lines.append(f"> Error: {coerce['error']}")
    else:
        suggestions = coerce.get("suggestions", [])
        if not suggestions:
            lines.append("No type coercion issues detected.")
        else:
            for s in suggestions[:10]:
                lines.append(f"- `{s.get('column')}`: {s.get('suggestion')} — {s.get('reason', '')}")

    return "\n".join(lines)


def _section_column_profiles(ctx: AgentContext) -> str:
    from qtrial_backend.tools.stats.column_stats import column_detailed_stats, ColumnStatsParams

    df = ctx.dataframe
    lines = ["## 3. Column Profiles", ""]

    numeric_rows = []
    cat_rows = []

    for col in df.columns:
        r = _call(column_detailed_stats, ColumnStatsParams, ctx, column=col)
        if "error" in r:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_rows.append([
                col,
                r.get("count", "—"),
                f"{r.get('null_pct', 0):.1f}%",
                _fmt(r.get("mean")),
                _fmt(r.get("std")),
                _fmt(r.get("min")),
                _fmt(r.get("median")),
                _fmt(r.get("max")),
                _fmt(r.get("skewness")),
            ])
        else:
            top = r.get("top_values", {})
            top_str = ", ".join(f"{k} ({v})" for k, v in list(top.items())[:3])
            cat_rows.append([
                col,
                r.get("count", "—"),
                f"{r.get('null_pct', 0):.1f}%",
                r.get("unique_count", "—"),
                top_str or "—",
            ])

    if numeric_rows:
        lines.append("### Numeric columns")
        lines.append("")
        lines.append(_md_table(
            ["Column", "N", "Null%", "Mean", "SD", "Min", "Median", "Max", "Skew"],
            numeric_rows,
        ))

    if cat_rows:
        lines += ["", "### Categorical columns", ""]
        lines.append(_md_table(
            ["Column", "N", "Null%", "Unique", "Top values"],
            cat_rows,
        ))

    return "\n".join(lines)


def _section_outliers(ctx: AgentContext) -> str:
    from qtrial_backend.tools.stats.outlier_detection import outlier_detection, OutlierParams

    r = _call(outlier_detection, OutlierParams, ctx)
    lines = ["## 4. Outlier Detection", ""]

    if "error" in r:
        lines.append(f"> {r['error']}")
        return "\n".join(lines)

    results = r.get("results", {})
    rows = []
    for col, info in results.items():
        if not isinstance(info, dict):
            continue
        if "skipped" in info:
            continue
        iqr_n = info.get("iqr_method", {}).get("n_outliers", 0)
        z_n = info.get("zscore_method", {}).get("n_outliers", 0)
        mad_n = (info.get("mad_method") or {}).get("n_outliers", 0)
        rows.append([col, iqr_n, z_n, mad_n])

    flagged = [row for row in rows if any(v > 0 for v in row[1:])]
    if not flagged:
        lines.append("No outliers detected across all numeric columns.")
    else:
        lines.append(_md_table(
            ["Column", "IQR outliers", "Z-score outliers", "MAD outliers"],
            flagged,
        ))

    return "\n".join(lines)


def _section_normality(ctx: AgentContext) -> str:
    from qtrial_backend.tools.stats.normality_test import normality_test, NormalityParams

    r = _call(normality_test, NormalityParams, ctx)
    lines = ["## 5. Normality Tests", ""]

    if "error" in r:
        lines.append(f"> {r['error']}")
        return "\n".join(lines)

    rows = []
    for col, info in r.items():
        if not isinstance(info, dict):
            continue
        if "skipped" in info:
            continue
        rows.append([
            col,
            info.get("test", "—"),
            info.get("n", "—"),
            _fmt(info.get("statistic"), 4),
            _fmt(info.get("p_value"), 4),
            "Yes" if info.get("is_normal") else "No",
        ])

    if not rows:
        lines.append("Insufficient data for normality testing.")
    else:
        lines.append(_md_table(
            ["Column", "Test", "N", "Statistic", "p-value", "Normal?"],
            rows,
        ))

    return "\n".join(lines)


def _section_correlation(ctx: AgentContext) -> str:
    from qtrial_backend.tools.stats.correlation import correlation_matrix, CorrelationParams

    df = ctx.dataframe
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    lines = ["## 6. Correlation Matrix (Spearman)", ""]

    if len(numeric_cols) < 2:
        lines.append("Fewer than 2 numeric columns — correlation matrix skipped.")
        return "\n".join(lines)

    r = _call(correlation_matrix, CorrelationParams, ctx, method="spearman")

    if "error" in r:
        lines.append(f"> {r['error']}")
        return "\n".join(lines)

    matrix = r.get("matrix", {})
    cols = r.get("columns", [])

    # Header row
    header = [""] + cols
    rows = []
    for c1 in cols:
        row = [c1]
        for c2 in cols:
            val = (matrix.get(c1) or {}).get(c2)
            row.append("—" if val is None else f"{val:.2f}")
        rows.append(row)

    lines.append(_md_table(header, rows))

    # Strong correlations
    strong = []
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j <= i:
                continue
            val = (matrix.get(c1) or {}).get(c2)
            if val is not None and abs(val) >= 0.6:
                strong.append((c1, c2, val))

    if strong:
        lines += ["", "**Pairs with |ρ| ≥ 0.6:**", ""]
        for c1, c2, val in sorted(strong, key=lambda x: -abs(x[2])):
            lines.append(f"- `{c1}` ↔ `{c2}`: ρ = {val:.3f}")

    return "\n".join(lines)


def _section_baseline_balance(ctx: AgentContext, treatment_col: str) -> str:
    from qtrial_backend.tools.stats.baseline_balance import baseline_balance, BaselineBalanceParams

    df = ctx.dataframe
    baseline_cols = [c for c in df.columns if c != treatment_col]

    lines = [f"## 7. Baseline Balance (treatment = `{treatment_col}`)", ""]

    r = _call(
        baseline_balance,
        BaselineBalanceParams,
        ctx,
        treatment_column=treatment_col,
        baseline_columns=baseline_cols,
    )

    if "error" in r:
        lines.append(f"> Could not compute baseline balance: {r['error']}")
        return "\n".join(lines)

    arms = r.get("arms", [])
    group_sizes = r.get("group_sizes", {})
    size_str = ", ".join(f"{arm}: n={group_sizes.get(arm, '?')}" for arm in arms)
    lines.append(f"Groups: {size_str}")
    lines.append(f"SMD threshold: {r.get('smd_threshold', 0.1)}")
    lines.append("")

    table_data = r.get("table", [])
    if not table_data:
        lines.append("No table data returned.")
        return "\n".join(lines)

    header = ["Variable", "Type"] + [f"{a} summary" for a in arms] + ["SMD", "Imbalanced?"]
    rows = []
    for row in table_data:
        arm_summaries = []
        for arm in arms:
            arm_info = row.get("arms", {}).get(str(arm), {})
            if row.get("type") == "continuous":
                mean = arm_info.get("mean")
                sd = arm_info.get("sd")
                arm_summaries.append(
                    f"{_fmt(mean, 2)} ± {_fmt(sd, 2)}" if mean is not None else "—"
                )
            else:
                props = arm_info.get("proportions", {})
                top_prop = list(props.items())[:1]
                if top_prop:
                    k, v = top_prop[0]
                    arm_summaries.append(f"{k}: {v*100:.1f}%")
                else:
                    arm_summaries.append("—")

        smd = row.get("smd")
        rows.append(
            [row.get("variable", "?"), row.get("type", "?")]
            + arm_summaries
            + [_fmt(smd, 3) if smd is not None else "—", "⚠ YES" if row.get("imbalanced") else "ok"]
        )

    lines.append(_md_table(header, rows))

    imbalanced = r.get("imbalanced_variables", [])
    if imbalanced:
        lines += ["", f"**⚠ Imbalanced variables ({len(imbalanced)}):** " + ", ".join(f"`{v}`" for v in imbalanced)]
    else:
        lines += ["", "**All variables are well-balanced (|SMD| < threshold).**"]

    return "\n".join(lines)


def _section_survival(ctx: AgentContext, time_col: str, event_col: str, group_col: str | None) -> str:
    from qtrial_backend.tools.stats.survival import survival_analysis, SurvivalParams

    lines = [f"## 8. Survival Analysis (time=`{time_col}`, event=`{event_col}`)", ""]

    r = _call(
        survival_analysis,
        SurvivalParams,
        ctx,
        time_column=time_col,
        event_column=event_col,
        group_column=group_col,
    )

    if "error" in r:
        lines.append(f"> Could not run survival analysis: {r['error']}")
        return "\n".join(lines)

    lines.append(f"- Total observations: **{r.get('n_total', '—')}**")
    lines.append(f"- Events: **{r.get('n_events', '—')}** ({r.get('event_rate_pct', '—')}%)")
    lines.append(f"- Censored: **{r.get('n_censored', '—')}**")
    lines.append("")

    overall = r.get("overall", {})
    lines.append(f"**Overall median survival:** {_fmt(overall.get('median_survival'), 1)}")
    surv_at = overall.get("survival_at_timepoints", {})
    if surv_at:
        lines.append("")
        lines.append(_md_table(
            ["Time point", "Survival probability"],
            [[f"t={t}", _fmt(v, 3)] for t, v in surv_at.items()],
        ))

    groups = r.get("groups", {})
    if groups:
        lines += ["", "**Per-group Kaplan-Meier:**", ""]
        group_rows = []
        for label, info in groups.items():
            group_rows.append([
                label,
                info.get("n", "—"),
                info.get("n_events", "—"),
                _fmt(info.get("median_survival"), 1),
            ])
        lines.append(_md_table(
            ["Group", "N", "Events", "Median survival"],
            group_rows,
        ))

        logrank_p = r.get("logrank_p_value")
        if logrank_p is not None:
            sig = "significant" if r.get("logrank_significant") else "not significant"
            lines.append(f"\nLog-rank test p-value: **{logrank_p:.4f}** ({sig} at α=0.05)")

    return "\n".join(lines)


# ── Main entry point ───────────────────────────────────────────────────────────

def build_static_report(df: pd.DataFrame, dataset_name: str) -> str:
    """
    Run the full static analysis pipeline and return a Markdown report string.
    This function is deterministic — no LLM is involved.
    """
    # Ensure tools are registered
    import qtrial_backend.tools  # noqa: F401

    ctx = AgentContext(dataframe=df, dataset_name=dataset_name)

    # Auto-detect clinical structure
    treatment_col = _detect_treatment_col(df)
    time_col = _detect_time_col(df)
    event_col = _detect_event_col(df, time_col) if time_col else None

    sections = [
        f"# Static Analysis Report — {dataset_name}",
        f"> Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · Fully deterministic, no LLM",
        "",
        _section_overview(df, dataset_name),
        "",
        _section_data_quality(ctx),
        "",
        _section_column_profiles(ctx),
        "",
        _section_outliers(ctx),
        "",
        _section_normality(ctx),
        "",
        _section_correlation(ctx),
    ]

    if treatment_col:
        sections += ["", _section_baseline_balance(ctx, treatment_col)]

    if time_col and event_col:
        sections += ["", _section_survival(ctx, time_col, event_col, treatment_col)]

    # Footer
    sections += [
        "",
        "---",
        "> **Next step:** run `analyze` (dynamic mode) to have the AI agent interpret these "
        "findings, run targeted tests, and compare against published literature.",
    ]

    return "\n".join(sections)
