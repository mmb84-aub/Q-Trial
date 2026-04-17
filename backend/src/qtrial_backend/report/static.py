from __future__ import annotations

"""
Static report generator.

Runs a fixed, deterministic pipeline of statistical tools against a dataset
and renders the results as a Markdown document.  No LLM is involved — the
report is purely data-driven.
"""

import datetime
import re
from typing import Any, Callable

import pandas as pd
from rich.console import Console

from qtrial_backend.agent.context import AgentContext

console = Console()


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


# ── Clinical trial three-stage analysis section ──────────────────────────────

def _build_clinical_config(
    df: pd.DataFrame,
    treatment_col: str | None,
    time_col: str | None,
    event_col: str | None,
) -> dict:
    """Derive clinical analysis config from auto-detected columns."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    exclude = {treatment_col, time_col, event_col} - {None}
    candidate_endpoints = [c for c in numeric_cols if c not in exclude]

    # Infer outcome type from available columns
    outcome_type = "continuous"
    if time_col and event_col:
        outcome_type = "survival"
    elif time_col:
        outcome_type = "longitudinal"

    config: dict = {"alpha": 0.05}
    if treatment_col:
        config["treatment_col"] = treatment_col
    if time_col:
        config["time_col"] = time_col
    if event_col:
        config["event_col"] = event_col
    config["outcome_type"] = outcome_type

    # Use first 3 numeric non-excluded columns as primary endpoints
    config["primary_endpoints"] = candidate_endpoints[:3]
    config["secondary_endpoints"] = candidate_endpoints[3:6]

    # Subgroup columns: categorical columns with 2-6 unique values
    subgroup_candidates = [
        c for c in df.columns
        if c not in exclude
        and not pd.api.types.is_numeric_dtype(df[c])
        and 2 <= df[c].nunique() <= 6
    ]
    config["subgroup_cols"] = subgroup_candidates[:3]

    return config


def _section_statistical_methodology() -> str:
    """
    Generate the Statistical Methodology chapter for the static report.

    Documents exactly what is implemented on this branch — no aspirational claims.
    References are to FDA/ICH guidance documents that motivated the design.
    """
    return """\
## Statistical Methodology

> **Why generic statistics are insufficient for clinical trials.**
> Standard t-tests and correlation coefficients were designed for independent
> observations from a single time point.  Clinical trial data violate these assumptions
> in three systematic ways: (1) multiplicity — testing multiple endpoints inflates the
> family-wise Type I error rate beyond the nominal α; (2) correlation — repeated
> measurements within the same subject are not independent; (3) missing data patterns —
> dropout in trials is rarely completely random (MCAR), so listwise deletion produces
> biased estimates.  The FDA (Statistical Principles for Clinical Trials, 1998; ICH E9)
> and EMA (ICH E9 R1 Addendum on Estimands, 2019) require that these three issues be
> addressed explicitly in the statistical analysis plan.

---

### Stage 1 — Data Integrity Methods

**Digit Preference Test**

Numeric columns are tested for non-uniform last-digit distributions using a
Pearson χ² goodness-of-fit test against the uniform null (each digit 0–9 equally
likely at the trailing integer position):

    H₀: P(last digit = k) = 1/10  for k ∈ {0,…,9}
    χ² = Σ (Oₖ − Eₖ)² / Eₖ  with 9 degrees of freedom

Columns with p < 0.05 are flagged as potentially affected by manual entry bias or
rounding artefacts (Austin et al., 2012).

**Baseline Balance — Standardised Mean Difference (SMD)**

For each baseline covariate, the SMD between treatment arms is computed:

- *Continuous:*   SMD = (μ_A − μ_B) / √[(σ²_A + σ²_B) / 2]
- *Binary:*        SMD = (p_A − p_B) / √[(p_A(1−p_A) + p_B(1−p_B)) / 2]

Variables with |SMD| > 0.1 are flagged as imbalanced (Normand et al., 2001;
CONSORT 2010 §15).  Unlike p-values, SMD is independent of sample size and
directly quantifies the practical magnitude of imbalance.

**Missing Data Classification — Little's MCAR Test**

Little's (1988) χ² test assesses whether the missing-data pattern is consistent
with Missing Completely At Random (MCAR):

    H₀: data are MCAR
    χ² = Σⱼ nⱼ (ȳⱼ − μ̂ⱼ)ᵀ Σ̂ⱼ⁻¹ (ȳⱼ − μ̂ⱼ),  df = Σⱼ kⱼ − k

*Important limitation:* this test can only distinguish MCAR from non-MCAR; it
cannot distinguish MAR from MNAR.  Classification: p ≥ 0.05 → "MCAR";
p < 0.05 → "Not MCAR" (MAR or MNAR — requires sensitivity analysis to distinguish).

---

### Stage 2 — Analysis Methods

**Mixed-Model Repeated Measures (MMRM)**

*Applies to:* longitudinal data with repeated observations per subject.

The MMRM is a linear mixed model with fixed effects for time, treatment, and
their interaction, plus a subject-level random intercept (compound-symmetry
covariance structure):

    y_ij = α + β_t·time_j + β_trt·treatment_i + β_{t×trt}·(time_j × treatment_i) + b_i + ε_ij
    b_i ~ N(0, σ²_b),  ε_ij ~ N(0, σ²_e)

Fitted using `statsmodels.formula.api.mixedlm` (REML).  Primary inference is on
the treatment × time interaction term.  The FDA recommends MMRM as the primary
analysis method for trials with repeated measurements (FDA Guidance, 2021).

*Limitation:* the implementation uses a random-intercept (compound-symmetry)
covariance structure rather than an unstructured covariance matrix.  Unstructured
covariance is more flexible but requires considerably more data.

**Constrained Longitudinal Data Analysis (cLDA)**

*Applies to:* randomised longitudinal trials where treatment is group-level and the
baseline (time = 0) is shared.

cLDA constrains the baseline mean to be equal across arms (no treatment main effect;
baseline differences are zero by randomisation):

    y_ij = α + β_j·I(time_j) + Σⱼ₌₁ γⱼ·(treatment_i × I(time_j > 0)) + b_i + ε_ij

This parameterisation avoids testing for a time-zero difference that cannot exist in
a properly randomised trial and increases statistical power (Liu et al., 2009).
Implemented as `outcome ~ C(time) + C(time):C(treatment)` in statsmodels, with the
treatment main effect intentionally omitted.

*Limitation:* same compound-symmetry constraint as MMRM above.

**Multiple Imputation by Chained Equations (MICE)**

*Applies to:* columns classified as "Not MCAR" by Little's test with <50% missing.

MICE generates m = 5 independent complete datasets using `sklearn.impute.IterativeImputer`
(BayesianRidge predictor, max_iter = 10), each with a different random seed (42–46).
Parameters are pooled across imputations using Rubin's (1987) Rules:

    Q̄ = (1/m) Σᵢ Qᵢ                              (pooled estimate)
    Ū = (1/m) Σᵢ Ûᵢ                              (within-imputation variance)
    B = (1/(m−1)) Σᵢ (Qᵢ − Q̄)²                  (between-imputation variance)
    T = Ū + (1 + 1/m) B                           (total variance)
    95% CI ≈ Q̄ ± 1.96√T

For model-based downstream analyses (MMRM, cLDA) where a single DataFrame is
required, one imputed dataset (seed = 42) is used.  Rubin's-Rules pooled statistics
are reported separately for disclosure.

**Effect Sizes with 95% CI**

Cohen's d with pooled standard deviation:

    d = (μ_A − μ_B) / σ_pooled,  where σ_pooled = √[((n_A−1)σ²_A + (n_B−1)σ²_B) / (n_A+n_B−2)]

95% CI computed via non-parametric bootstrap (1 000 resamples, 2.5th–97.5th percentile).

*For longitudinal endpoints:* because MMRM and cLDA do not return a marginal d directly,
d is computed cross-sectionally (collapsing all time points) as a conservative approximation.
This ignores the repeated-measures structure and will typically underestimate the true
treatment effect; findings are labelled with this limitation.

**Achieved Power**

Post-hoc power for each endpoint uses `statsmodels.stats.power.TTestIndPower` (two-sample)
or `TTestPower` (one-sample / paired), with the observed d and per-group n:

    power = 1 − β = P(reject H₀ | H₁: δ = d_obs)

Findings with power < 80% are flagged as potentially underpowered (ICH E9 §3.5).

---

### Stage 3 — Multiple Testing Correction

**Bonferroni correction (primary endpoints)**

Applied to all primary endpoints jointly.  Each raw p-value is multiplied by the
number of primary tests k:

    p_adj = min(k · p_raw, 1)

Controls the family-wise error rate (FWER) at α = 0.05.  Conservative when tests
are positively correlated; used for primary endpoints to minimise false positives
(FDA Draft Guidance on Multiple Endpoints, 2017).

**Benjamini-Hochberg FDR correction (secondary endpoints)**

Applied to secondary and exploratory endpoints.  Rank the m raw p-values
p_(1) ≤ … ≤ p_(m) and reject H_(i) if:

    p_(i) ≤ (i / m) · α

Controls the false discovery rate (FDR) at α = 0.05.  Less conservative than
Bonferroni for exploratory analyses (Benjamini & Hochberg, 1995).

**Hierarchical gatekeeping**

Secondary endpoint results are gated on primary endpoint significance:
if no primary endpoint survives Bonferroni correction, all secondary BH-FDR
results are marked `gated_out = True` (closed testing procedure, ICH E9 §2.2.5).

---

### Regulatory Alignment

| Framework stage | ICH E9 / FDA reference |
|---|---|
| Stage 1 (Integrity) | ICH E9 §4.6 — Data integrity; ICH E9 R1 §3.3 — Missing data |
| Stage 2 (Analysis) | ICH E9 §5.1–5.3 — Primary/secondary endpoints; FDA MMRM Guidance (2021) |
| Stage 3 (Correction) | ICH E9 §2.2.5 — Multiplicity; FDA Multiple Endpoints Guidance (2017) |

> **Disclaimer:** This implementation is a research and teaching tool.  It has not been
> validated for regulatory submission.  The MMRM and cLDA use compound-symmetry covariance
> (not ICH E9 R1 preferred unstructured covariance).  Power estimates assume independence
> across endpoints.  MICE uses BayesianRidge (not predictive mean matching).  Regulatory
> submissions require prospective statistical analysis plans, pre-specified sensitivity
> analyses, and independent statistical review.
"""


def _section_clinical_analysis(result: dict) -> str:
    """Render the three-stage clinical analysis result as Markdown."""
    lines = ["## Clinical Trial Statistical Framework (Three-Stage)", ""]

    # Stage 1
    s1 = result.get("stage_1_integrity", {})
    lines.append("### Stage 1 — Data Integrity")
    lines.append("")

    # Missingness classification
    mcar = s1.get("missingness_classification", {})
    mcar_inner = mcar.get("little_mcar_test", {}) if isinstance(mcar, dict) else {}
    if mcar_inner:
        lines.append(f"- **Missingness classification:** {mcar_inner.get('classification', 'N/A')}")
        lines.append(f"  - Little's MCAR p-value: {_fmt(mcar_inner.get('p_value'))}")
        lines.append(f"  - Interpretation: {mcar_inner.get('interpretation', '')}")
        if mcar_inner.get("test_determines"):
            lines.append(f"  - Test scope: {mcar_inner['test_determines']}")
        if mcar_inner.get("recommendation"):
            lines.append(f"  - Recommendation: {mcar_inner['recommendation']}")
        missing_cols = mcar_inner.get("missing_columns", [])
        if missing_cols:
            lines.append(f"  - Missing columns: {', '.join(f'`{c}`' for c in missing_cols)}")
    lines.append("")

    # Digit preference
    dp = s1.get("digit_preference", {})
    if isinstance(dp, dict) and not dp.get("error"):
        flagged = dp.get("flagged_columns", [])
        lines.append(f"- **Digit preference:** {dp.get('total_numeric_columns_tested', 0)} columns tested, "
                      f"{len(flagged)} flagged")
        if flagged:
            for f in flagged[:5]:
                lines.append(f"  - `{f['column']}`: χ²={_fmt(f.get('chi_square'))}, p={_fmt(f.get('p_value'))}")
    lines.append("")

    # Baseline balance
    bb = s1.get("baseline_balance", {})
    if isinstance(bb, dict) and not bb.get("error"):
        imbal = bb.get("imbalanced_variables", [])
        lines.append(f"- **Baseline balance:** {bb.get('n_variables', 0)} variables, "
                      f"{len(imbal)} imbalanced (|SMD| > {bb.get('smd_threshold', 0.1)})")
        if imbal:
            lines.append(f"  - Imbalanced: {', '.join(f'`{v}`' for v in imbal)}")
    lines.append("")

    # Integrity warnings
    warnings = s1.get("integrity_warnings", [])
    if warnings:
        lines.append("**Integrity warnings:**")
        for w in warnings:
            lines.append(f"- ⚠ {w}")
    else:
        lines.append("**Data integrity checks passed.**")
    lines.append("")

    # Stage 2
    s2 = result.get("stage_2_analysis", {})
    lines.append("### Stage 2 — Analysis")
    lines.append("")
    if s2.get("imputation_used"):
        lines.append(f"- **Imputation:** {s2.get('imputation_method', 'MICE')}")
        pooled = s2.get("mice_pooled_results")
        if isinstance(pooled, dict) and not pooled.get("error"):
            for col_result in pooled.get("imputed_columns", [])[:5]:
                lines.append(f"  - `{col_result['column']}`: pooled mean={_fmt(col_result.get('pooled_mean'))}, "
                             f"95% CI [{_fmt(col_result.get('ci_lower'))}, {_fmt(col_result.get('ci_upper'))}]")
    else:
        lines.append("- **Imputation:** not required (MCAR or no missing data)")
    lines.append("")

    # Primary analysis
    pa = s2.get("primary_analysis", {})
    if pa and not pa.get("error") and not pa.get("note"):
        lines.append("- **Primary analysis results:**")
        # Continuous / Binary endpoint tests
        for test_list_key in ("continuous_tests", "binary_tests"):
            for item in pa.get(test_list_key, []):
                ep = item.get("endpoint", "?")
                ht = item.get("hypothesis_test", {})
                es = item.get("effect_size", {})
                p_val = ht.get("p_value")
                cohen = es.get("cohen_d", {})
                d_val = cohen.get("value") if isinstance(cohen, dict) else None
                ci = cohen.get("ci_95") if isinstance(cohen, dict) else None
                ci_str = f" 95% CI [{_fmt(ci[0])}, {_fmt(ci[1])}]" if isinstance(ci, list) and len(ci) >= 2 else ""
                lines.append(f"  - `{ep}`: p={_fmt(p_val)}, d={_fmt(d_val)}{ci_str}")
        # MMRM (longitudinal primary analysis is the raw MMRM dict)
        if pa.get("model") and "MMRM" in str(pa.get("model", "")):
            lines.append(f"  - **MMRM:** p={_fmt(pa.get('p_value'))}, "
                          f"subjects={pa.get('n_subjects', '?')}, "
                          f"obs={pa.get('n_observations', '?')}")
            interp = pa.get("clinical_interpretation", "")
            if interp:
                lines.append(f"    {interp}")
    lines.append("")

    # ANCOVA
    ancova = s2.get("ancova")
    if isinstance(ancova, dict) and not ancova.get("error"):
        adj = ancova.get("adjusted_treatment_effect", {})
        lines.append(f"- **ANCOVA:** adjusted treatment p={_fmt(adj.get('p_value'))}, "
                      f"coeff={_fmt(adj.get('coefficient'))}, "
                      f"95% CI [{_fmt(adj.get('ci_lower'))}, {_fmt(adj.get('ci_upper'))}]")
    lines.append("")

    # cLDA
    clda = s2.get("clda")
    if isinstance(clda, dict) and not clda.get("error") and not clda.get("skipped"):
        lines.append(f"- **cLDA (constrained Longitudinal Data Analysis):**")
        lines.append(f"  - Baseline time: {clda.get('baseline_time', 'N/A')}")
        lines.append(f"  - Subjects: {clda.get('n_subjects', '?')}, "
                      f"Time points: {clda.get('n_timepoints', '?')}")
        lines.append(f"  - Treatment p-value: {_fmt(clda.get('p_value'))}")
        tbt = clda.get("treatment_by_time_effects", [])
        if tbt:
            for t in tbt[:5]:
                lines.append(
                    f"    - {t['term']}: coeff={_fmt(t.get('coefficient'))}, "
                    f"p={_fmt(t.get('p_value'))}"
                )
        interp = clda.get("clinical_interpretation", "")
        if interp:
            lines.append(f"  - {interp}")
        limits = clda.get("limitations", [])
        if limits:
            lines.append(f"  - **Limitations:** {'; '.join(limits)}")
    elif isinstance(clda, dict) and clda.get("skipped"):
        lines.append(f"- **cLDA:** skipped — {clda.get('reason', 'not applicable')}")
    lines.append("")

    # Stage 3
    s3 = result.get("stage_3_corrections", {})
    lines.append("### Stage 3 — Multiple Testing Correction")
    lines.append("")
    lines.append(f"- Primary correction: **{s3.get('primary_correction', 'N/A')}**")
    lines.append(f"- Secondary correction: **{s3.get('secondary_correction', 'N/A')}**")
    lines.append(f"- Hierarchical gate open: **{s3.get('hierarchical_gate_open', 'N/A')}**")
    lines.append("")

    findings = s3.get("corrected_findings", [])
    if findings:
        header = ["Endpoint", "Type", "Raw p", "Adjusted p", "Method", "Effect size", "OR",
                  "95% CI", "Power", "Req. n (80%)", "Significant"]
        rows = []
        for f in findings:
            ci = f.get("effect_size_ci", [0, 0])
            ci_str = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if isinstance(ci, list) and len(ci) >= 2 else "—"
            or_val = f.get("odds_ratio")
            req_n = f.get("n_required_80pct_power")
            rows.append([
                f.get("finding_id", "?"),
                f.get("endpoint_type", "?"),
                _fmt(f.get("raw_p_value")),
                _fmt(f.get("adjusted_p_value")),
                f.get("correction_method", "?"),
                _fmt(f.get("effect_size")),
                _fmt(or_val) if or_val is not None else "—",
                ci_str,
                _fmt(f.get("achieved_power")),
                str(int(req_n)) if req_n is not None else "—",
                "✓" if f.get("significant_after_correction") else "✗",
            ])
        lines.append(_md_table(header, rows))
    lines.append("")

    # Clinical summary
    summary = result.get("clinical_summary", "")
    if summary:
        lines.append(f"**Clinical summary:** {summary}")

    return "\n".join(lines)


# ── Main entry point ───────────────────────────────────────────────────────────

def build_static_report(
    df: pd.DataFrame,
    dataset_name: str,
    emit: Callable | None = None,
) -> tuple[str, str | None, dict | None]:
    """
    Run the full static analysis pipeline and return:
      1. Markdown report string (evidence-eligible sections only),
      2. Statistical Methodology chapter (display-only, not finding-eligible),
      3. Raw three-stage clinical analysis dict (or None).

    This function is deterministic — no LLM is involved.
    """
    # Ensure tools are registered
    import qtrial_backend.tools  # noqa: F401

    ctx = AgentContext(dataframe=df, dataset_name=dataset_name)

    console.print(
        f"[bold cyan]► Static Analysis:[/bold cyan] "
        f"Dataset [bold]{dataset_name}[/bold] "
        f"({len(df)} rows × {len(df.columns)} cols)"
    )

    # Auto-detect clinical structure
    treatment_col = _detect_treatment_col(df)
    time_col = _detect_time_col(df)
    event_col = _detect_event_col(df, time_col) if time_col else None

    _det = []
    if treatment_col: _det.append(f"treatment=[bold]{treatment_col}[/bold]")
    if time_col:      _det.append(f"time=[bold]{time_col}[/bold]")
    if event_col:     _det.append(f"event=[bold]{event_col}[/bold]")
    if _det:
        console.print("  [dim]Auto-detected: " + ", ".join(_det) + "[/dim]")

    def _run(label: str, fn, *args, **kwargs) -> str:
        console.print(f"  [yellow]▸[/yellow] {label}…")
        if emit is not None:
            try:
                emit({"type": "progress", "stage": "StaticAnalysis", "message": f"Running {label}…"})
            except Exception:
                pass
        result = fn(*args, **kwargs)
        console.print(f"    [green]✓[/green] {label}")
        if emit is not None:
            try:
                emit({"type": "progress", "stage": "StaticAnalysis", "message": f"✓ {label}"})
            except Exception:
                pass
        return result

    sections = [
        f"# Static Analysis Report — {dataset_name}",
        f"> Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · Fully deterministic, no LLM",
        "",
        _run("Overview",          _section_overview,       df, dataset_name),
        "",
        _run("Data Quality",      _section_data_quality,   ctx),
        "",
        _run("Column Profiles",   _section_column_profiles, ctx),
        "",
        _run("Outlier Detection", _section_outliers,       ctx),
        "",
        _run("Normality Tests",   _section_normality,      ctx),
        "",
        _run("Correlation Matrix",_section_correlation,    ctx),
    ]

    if treatment_col:
        sections += ["", _run("Baseline Balance", _section_baseline_balance, ctx, treatment_col)]

    if time_col and event_col:
        sections += ["", _run("Survival Analysis", _section_survival, ctx, time_col, event_col, treatment_col)]

    # ── Three-stage clinical trial analysis ───────────────────────────────
    clinical_analysis_result: dict | None = None
    methodology_chapter: str | None = None
    try:
        from qtrial_backend.tools.stats.clinical_stats import run_clinical_analysis

        clinical_config = _build_clinical_config(df, treatment_col, time_col, event_col)
        console.print(f"  [yellow]▸[/yellow] Clinical Trial Analysis (three-stage)…")
        if emit is not None:
            try:
                emit({"type": "progress", "stage": "StaticAnalysis",
                      "message": "Running Clinical Trial Analysis…"})
            except Exception:
                pass

        clinical_analysis_result = run_clinical_analysis(df, clinical_config)

        console.print(f"    [green]✓[/green] Clinical Trial Analysis")
        if emit is not None:
            try:
                emit({"type": "progress", "stage": "StaticAnalysis",
                      "message": "✓ Clinical Trial Analysis"})
            except Exception:
                pass

        methodology_chapter = _section_statistical_methodology()
        sections += ["", _section_clinical_analysis(clinical_analysis_result)]
    except Exception as exc:
        console.print(f"    [yellow]⚠ Clinical Trial Analysis skipped: {exc}[/yellow]")

    # Footer
    sections += [
        "",
        "---",
        "> **Next step:** run `analyze` (dynamic mode) to have the AI agent interpret these "
        "findings, run targeted tests, and compare against published literature.",
    ]

    console.print(f"  [bold green]✔ Static report complete[/bold green] ({sum(len(s) for s in sections)} chars)")
    return "\n".join(sections), methodology_chapter, clinical_analysis_result
