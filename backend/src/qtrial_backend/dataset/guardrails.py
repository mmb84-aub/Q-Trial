"""
Task 5 — Robustness guardrails for the Q-Trial agentic pipeline.

Four deterministic checks (no LLM) applied right after evidence computation:

1. low_cardinality_numeric  — numeric columns that look categorical
2. range_violation          — values outside physiologically plausible bounds
3. unit_plausibility        — magnitude mismatch suggesting unit errors
4. repeated_measures        — multiple rows per subject ID → longitudinal flag

Results are returned as a ``GuardrailReport`` and attached to the evidence
dict so every downstream agent can reference them.
"""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from qtrial_backend.dataset.evidence import ID_HINTS

# ── Physiological range catalogue ────────────────────────────────────────────
# Each entry: (pattern, min_plausible, max_plausible, unit_hint)
# Pattern is matched against the lowercased column name (substring match).
_RANGE_CATALOGUE: list[tuple[str, float, float, str]] = [
    ("age",         0,      130,    "years"),
    ("bili",        0,      100,    "mg/dL"),
    ("albumin",     0.5,    8,      "g/dL"),
    ("protime",     5,      60,     "seconds"),
    ("platelet",    10,     1500,   "×10³/μL"),
    ("ast",         0,      5000,   "U/L"),
    ("alt",         0,      5000,   "U/L"),
    ("alk_phos",    0,      15000,  "U/L"),
    ("alkaline",    0,      15000,  "U/L"),
    ("copper",      0,      2000,   "μg/day"),
    ("trig",        0,      3000,   "mg/dL"),
    ("chol",        0,      1500,   "mg/dL"),
    ("cholesterol", 0,      1500,   "mg/dL"),
    ("hemoglobin",  1,      25,     "g/dL"),
    ("hgb",         1,      25,     "g/dL"),
    ("creatinine",  0,      30,     "mg/dL"),
    ("glucose",     0,      2000,   "mg/dL"),
    ("sodium",      100,    180,    "mEq/L"),
    ("potassium",   1,      10,     "mEq/L"),
    ("bmi",         5,      80,     "kg/m²"),
    ("weight",      1,      400,    "kg"),
    ("height",      30,     250,    "cm"),
    ("sbp",         40,     250,    "mmHg"),
    ("dbp",         20,     180,    "mmHg"),
    ("hr",          20,     300,    "bpm"),
    ("pulse",       20,     300,    "bpm"),
]

# Unit plausibility rules: (pattern, expected_median_min, expected_median_max, warning_hint)
# If the column median falls OUTSIDE this range, flag it.
_UNIT_PLAUSIBILITY: list[tuple[str, float, float, str]] = [
    ("age",         1,    100,   "age >100 median: units may not be years"),
    ("bili",        0,    50,    "bilirubin median >50: may be μmol/L (×17.1) not mg/dL"),
    ("albumin",     1,    6,     "albumin median outside 1-6: may be in g/L instead of g/dL (÷10)"),
    ("protime",     8,    40,    "prothrombin median outside 8-40: check units (seconds vs %)"),
    ("platelet",    10,   1200,  "platelet median outside 10-1200: check units (×10³/μL vs /μL)"),
    ("chol",        0.5,  600,   "cholesterol median outside 0.5-600: may be mmol/L instead of mg/dL"),
    ("cholesterol", 0.5,  600,   "cholesterol median outside 0.5-600: may be mmol/L instead of mg/dL"),
    ("hemoglobin",  5,    22,    "hemoglobin median outside 5-22: may be g/L instead of g/dL"),
    ("hgb",         5,    22,    "hemoglobin median outside 5-22: may be g/L instead of g/dL"),
    ("creatinine",  0,    15,    "creatinine median outside 0-15: may be μmol/L instead of mg/dL"),
    ("weight",      2,    300,   "weight median outside 2-300: may not be in kg"),
    ("height",      50,   220,   "height median outside 50-220: may not be in cm"),
    ("bmi",         10,   60,    "BMI median outside 10-60: may be calculated differently"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _matches_pattern(col: str, pattern: str) -> bool:
    """
    Word-level match: pattern must appear as a complete word (delimited by
    underscores, hyphens, spaces, or column boundaries) in the normalised
    column name.  Prevents 'age' matching 'stage', 'alt' matching 'platelet', etc.
    """
    # Normalise column name to lowercase words separated by underscores
    normalised = re.sub(r"[\s\-]+", "_", col.lower())
    # Split into tokens and check exact token match
    tokens = normalised.split("_")
    if pattern in tokens:
        return True
    # Also accept if normalised starts or ends with pattern (e.g. "alk_phos" for "alk_phos")
    if normalised == pattern:
        return True
    # Accept multi-word patterns (e.g. "alk_phos" inside "serum_alk_phos")
    if f"_{pattern}" in normalised or f"{pattern}_" in normalised:
        return True
    return False


def _is_id_col(col: str) -> bool:
    # Exact match only (same logic as evidence.py) — avoids substring false positives
    # e.g. 'spiders' must NOT match because 'id' is a substring but not the full name
    return col.lower().replace(" ", "") in ID_HINTS


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ── Check 1: low-cardinality numeric columns ──────────────────────────────────

def _check_low_cardinality_numerics(df: pd.DataFrame) -> list[dict]:
    """
    Flag numeric columns that behave like categoricals:
    - between 2 and 6 distinct non-null values exclusively integers
    - not an obvious ID column
    """
    flags: list[dict] = []
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        if _is_id_col(col):
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        n_unique = int(series.nunique())
        if n_unique < 2 or n_unique > 6:
            continue
        # Check if all values are integers (within float tolerance)
        all_int = all(v == int(v) for v in series if _safe_float(v) is not None)
        if not all_int:
            continue
        unique_vals = sorted(series.unique().tolist())
        flags.append({
            "check_type": "low_cardinality_numeric",
            "column": col,
            "severity": "medium",
            "detail": (
                f"Numeric column '{col}' has only {n_unique} distinct integer "
                f"value(s): {unique_vals}. Likely a categorical/ordinal variable "
                f"encoded as a number."
            ),
            "suggested_action": (
                f"Confirm whether '{col}' is truly numeric or should be treated "
                "as a categorical variable. If categorical, provide a codebook "
                "via --metadata to label each code."
            ),
            "evidence": {"n_unique": n_unique, "unique_values": unique_vals},
        })
    return flags


# ── Check 2: physiological range violations ───────────────────────────────────

def _check_range_constraints(df: pd.DataFrame) -> list[dict]:
    """
    For columns matching known clinical name patterns, flag any rows
    where values fall outside the physiologically plausible range.
    """
    flags: list[dict] = []
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        for pattern, lo, hi, unit in _RANGE_CATALOGUE:
            if not _matches_pattern(col, pattern):
                continue
            series = df[col].dropna()
            if len(series) == 0:
                break
            violators = series[(series < lo) | (series > hi)]
            if violators.empty:
                break
            pct = round(float(len(violators) / len(series) * 100), 1)
            examples = [_safe_float(v) for v in violators.head(5)]
            severity = "high" if pct > 10 else "medium" if pct > 2 else "low"
            flags.append({
                "check_type": "range_violation",
                "column": col,
                "severity": severity,
                "detail": (
                    f"'{col}' has {len(violators)} value(s) ({pct}%) outside the "
                    f"plausible range [{lo}, {hi}] {unit}. "
                    f"Example violating values: {[v for v in examples if v is not None]}"
                ),
                "suggested_action": (
                    f"Verify that '{col}' is in units of {unit}. Check for data "
                    "entry errors, unit conversions, or extreme-but-real clinical values."
                ),
                "evidence": {
                    "expected_range": [lo, hi],
                    "unit_hint": unit,
                    "n_violations": int(len(violators)),
                    "pct_violations": pct,
                    "example_values": [v for v in examples if v is not None],
                },
            })
            break  # one catalogue match per column is sufficient
    return flags


# ── Check 3: unit plausibility (magnitude-based) ─────────────────────────────

def _check_unit_plausibility(df: pd.DataFrame) -> list[dict]:
    """
    Detect potential unit mismatches by comparing the column median to the
    expected median range for a given clinical variable.
    """
    flags: list[dict] = []
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        for pattern, med_lo, med_hi, warning in _UNIT_PLAUSIBILITY:
            if not _matches_pattern(col, pattern):
                continue
            series = df[col].dropna()
            if len(series) < 3:
                break
            median_val = _safe_float(series.median())
            if median_val is None:
                break
            if med_lo <= median_val <= med_hi:
                break  # looks fine
            flags.append({
                "check_type": "unit_plausibility",
                "column": col,
                "severity": "medium",
                "detail": (
                    f"'{col}' has a median of {round(median_val, 3)}, which falls "
                    f"outside the expected magnitude range [{med_lo}, {med_hi}]. "
                    f"Hint: {warning}"
                ),
                "suggested_action": (
                    f"Verify the units for '{col}'. The median ({round(median_val, 3)}) "
                    "suggests a possible unit conversion issue. Confirm units via metadata."
                ),
                "evidence": {
                    "median": round(median_val, 4),
                    "expected_median_range": [med_lo, med_hi],
                    "warning": warning,
                },
            })
            break
    return flags


# ── Check 4: repeated measures / longitudinal schema inference ────────────────

def _infer_repeated_measures(df: pd.DataFrame) -> dict | None:
    """
    Detect if any ID column has repeated rows, suggesting a longitudinal or
    repeated-measures study design rather than a one-row-per-subject design.

    Returns a dict with schema details, or None if no ID columns found or
    all IDs are unique.
    """
    id_cols = [c for c in df.columns if _is_id_col(c)]
    if not id_cols:
        return None

    for id_col in id_cols:
        counts = df[id_col].dropna().value_counts()
        max_repeats = int(counts.max()) if not counts.empty else 1
        n_subjects = int(len(counts))
        n_repeated = int((counts > 1).sum())

        if max_repeats <= 1:
            continue  # all unique → cross-sectional

        likely_longitudinal = max_repeats >= 3 or n_repeated / max(n_subjects, 1) > 0.5
        return {
            "id_column": id_col,
            "n_subjects": n_subjects,
            "total_rows": len(df),
            "max_repeats_per_subject": max_repeats,
            "n_subjects_with_repeats": n_repeated,
            "likely_longitudinal": likely_longitudinal,
            "detail": (
                f"ID column '{id_col}' has {n_subjects} unique subject(s) across "
                f"{len(df)} rows (max {max_repeats} rows per subject). "
                + (
                    "Likely a longitudinal / repeated-measures design."
                    if likely_longitudinal
                    else "Some subjects have repeated rows — confirm whether "
                         "this is a repeated-measures or wide-format dataset."
                )
            ),
        }

    return None  # ID columns found but all unique


# ── Public entry point ────────────────────────────────────────────────────────

def run_guardrails(df: pd.DataFrame) -> dict:
    """
    Run all four robustness guardrail checks against *df*.

    Returns a plain dict (JSON-serialisable) with keys:
        flags             — list of guardrail flag dicts
        repeated_measures — dict or None
        summary           — one-line human-readable summary
        counts_by_type    — dict[check_type -> count]
    """
    flags: list[dict] = []
    flags += _check_low_cardinality_numerics(df)
    flags += _check_range_constraints(df)
    flags += _check_unit_plausibility(df)

    repeated = _infer_repeated_measures(df)

    counts_by_type: dict[str, int] = {}
    for f in flags:
        counts_by_type[f["check_type"]] = counts_by_type.get(f["check_type"], 0) + 1

    high_count = sum(1 for f in flags if f["severity"] == "high")
    if not flags and repeated is None:
        summary = "All guardrail checks passed — no anomalies detected."
    else:
        parts = []
        if flags:
            parts.append(
                f"{len(flags)} guardrail flag(s) "
                f"({high_count} high-severity)"
            )
        if repeated:
            design = "longitudinal" if repeated["likely_longitudinal"] else "repeated-rows"
            parts.append(f"repeated-measures schema detected ({design})")
        summary = "; ".join(parts) + "."

    return {
        "flags": flags,
        "repeated_measures": repeated,
        "summary": summary,
        "counts_by_type": counts_by_type,
    }


def format_guardrail_citations(report: dict) -> list[str]:
    """
    Produce citation strings for guardrail flags in the same format as the
    rest of ``format_citations()``, so agents and the reasoning engine can
    reference them as ``guardrails[i]``.
    """
    lines: list[str] = []
    for i, flag in enumerate(report.get("flags", [])):
        lines.append(
            f"guardrails[{i}] ({flag['check_type']}, {flag['severity']}): "
            f"{flag['detail'][:160]}"
        )
    rm = report.get("repeated_measures")
    if rm:
        lines.append(f"guardrails.repeated_measures: {rm['detail'][:160]}")
    return lines
