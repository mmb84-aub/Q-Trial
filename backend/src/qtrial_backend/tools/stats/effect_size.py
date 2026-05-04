"""
Effect size computation tool — Stage 4 statistical tool.

Input:  pd.DataFrame + outcome_col + group_col (binary treatment indicator).
Output: EffectSizeResult with Cohen's d, Hedges' g, odds ratio, risk ratio,
        95% CIs, and clinical significance flags.
Does:   computes standardised effect sizes and ratio measures so the LLM agent
        can distinguish statistically significant but clinically trivial effects.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool

_COHEN_THRESHOLDS = [
    (0.2, "negligible"),
    (0.5, "small"),
    (0.8, "medium"),
    (float("inf"), "large"),
]

_CLIFF_THRESHOLDS = [
    (0.147, "negligible"),
    (0.33, "small"),
    (0.474, "medium"),
    (float("inf"), "large"),
]

_MAX_BOOTSTRAP_OBS = 3000
_MAX_CLIFF_PAIRS = 200_000


def _magnitude(value: float, thresholds: list) -> str:
    for cutoff, label in thresholds:
        if abs(value) < cutoff:
            return label
    return "large"


class EffectSizeParams(BaseModel):
    numeric_column: str = Field(description="Numeric column to compare")
    group_column: str = Field(description="Categorical column defining the two groups")
    group_a: str = Field(description="Label for group A")
    group_b: str = Field(description="Label for group B")
    method: str = Field(
        default="both",
        description=(
            "Effect size method: 'cohen_d' (parametric), "
            "'cliff_delta' (non-parametric), or 'both'."
        ),
    )
    bootstrap_ci: bool = Field(
        default=True,
        description="Compute 95% bootstrap confidence intervals (1000 resamples).",
    )
    compute_risk_measures: bool = Field(
        default=False,
        description=(
            "Compute Risk Ratio and Odds Ratio. Only meaningful when outcome_column "
            "is binary (0/1) and you are comparing two groups on a binary endpoint. "
            "Set to True when numeric_column is a binary outcome indicator."
        ),
    )


@tool(
    name="effect_size",
    description=(
        "Compute standardised effect size between two groups on a numeric column. "
        "Cohen's d: difference in means divided by pooled SD (parametric). "
        "Cliff's delta: probability that a random value from group A exceeds "
        "group B minus the reverse (non-parametric, range -1 to +1). "
        "Both include a magnitude label: negligible / small / medium / large."
    ),
    params_model=EffectSizeParams,
    category="stats",
)
def effect_size(params: EffectSizeParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for col in (params.numeric_column, params.group_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {ctx.column_names}")

    mask_a = df[params.group_column].astype(str) == str(params.group_a)
    mask_b = df[params.group_column].astype(str) == str(params.group_b)

    # Track rows before and after listwise deletion
    a_raw = df.loc[mask_a, params.numeric_column]
    b_raw = df.loc[mask_b, params.numeric_column]
    a = a_raw.dropna().to_numpy(dtype=float)
    b = b_raw.dropna().to_numpy(dtype=float)
    rows_dropped_a = int(a_raw.isna().sum())
    rows_dropped_b = int(b_raw.isna().sum())

    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            f"Not enough observations: group_a n={len(a)}, group_b n={len(b)}. "
            "Need at least 2 per group."
        )

    result: dict = {
        "numeric_column": params.numeric_column,
        "group_column": params.group_column,
        "group_a": {
            "label": str(params.group_a),
            "n": int(len(a)),
            "n_before_dropna": int(len(a_raw)),
            "rows_dropped": rows_dropped_a,
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)), 4),
        },
        "group_b": {
            "label": str(params.group_b),
            "n": int(len(b)),
            "n_before_dropna": int(len(b_raw)),
            "rows_dropped": rows_dropped_b,
            "mean": round(float(b.mean()), 4),
            "std": round(float(b.std(ddof=1)), 4),
        },
        "listwise_deletion": {
            "total_rows_dropped": rows_dropped_a + rows_dropped_b,
            "note": "Rows with missing values in numeric_column were excluded per group.",
        },
    }

    method = params.method.lower()
    rng = np.random.default_rng(42)
    notes: list[str] = []

    bootstrap_ci = params.bootstrap_ci
    if bootstrap_ci and (len(a) + len(b)) > _MAX_BOOTSTRAP_OBS:
        bootstrap_ci = False
        notes.append(
            "Bootstrap CIs skipped for runtime control because group sizes exceed "
            f"{_MAX_BOOTSTRAP_OBS:,} combined observations."
        )

    def _bounded_pair_samples(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) * len(y) <= _MAX_CLIFF_PAIRS:
            return x, y
        max_each = max(int(np.sqrt(_MAX_CLIFF_PAIRS)), 2)
        x_n = min(len(x), max_each)
        y_n = min(len(y), max_each)
        notes.append(
            "Cliff's delta used deterministic subsamples for runtime control "
            f"({x_n} from group A, {y_n} from group B)."
        )
        return (
            rng.choice(x, size=x_n, replace=False),
            rng.choice(y, size=y_n, replace=False),
        )

    def _bootstrap_ci(
        a: np.ndarray, b: np.ndarray, stat_fn, n_boot: int = 1000
    ) -> tuple[float, float]:
        boot_stats = [
            stat_fn(
                rng.choice(a, size=len(a), replace=True),
                rng.choice(b, size=len(b), replace=True),
            )
            for _ in range(n_boot)
        ]
        lo, hi = float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))
        return round(lo, 4), round(hi, 4)

    if method in ("cohen_d", "both"):
        n_a, n_b = len(a), len(b)
        pooled_std = np.sqrt(
            ((n_a - 1) * a.std(ddof=1) ** 2 + (n_b - 1) * b.std(ddof=1) ** 2)
            / (n_a + n_b - 2)
        )
        d = float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0
        d_entry: dict = {
            "value": round(d, 4),
            "magnitude": _magnitude(d, _COHEN_THRESHOLDS),
            "interpretation": (
                f"|d|={abs(d):.3f}: {_magnitude(d, _COHEN_THRESHOLDS)} effect. "
                "Positive = group A higher than group B."
            ),
        }
        if bootstrap_ci:
            def _d_fn(x: np.ndarray, y: np.ndarray) -> float:
                ps = np.sqrt(((len(x)-1)*x.std(ddof=1)**2 + (len(y)-1)*y.std(ddof=1)**2) / (len(x)+len(y)-2))
                return float((x.mean() - y.mean()) / ps) if ps > 0 else 0.0
            lo, hi = _bootstrap_ci(a, b, _d_fn)
            d_entry["ci_95"] = [lo, hi]
        result["cohen_d"] = d_entry

    if method in ("cliff_delta", "both"):
        def _cliff_fn(x: np.ndarray, y: np.ndarray) -> float:
            dom = sum(1 if xi > yi else -1 if xi < yi else 0 for xi in x for yi in y)
            return float(dom) / (len(x) * len(y))

        a_cliff, b_cliff = _bounded_pair_samples(a, b)
        delta = _cliff_fn(a_cliff, b_cliff)
        delta_entry: dict = {
            "value": round(delta, 4),
            "magnitude": _magnitude(delta, _CLIFF_THRESHOLDS),
            "interpretation": (
                f"δ={delta:.3f}: {_magnitude(delta, _CLIFF_THRESHOLDS)} effect. "
                "Positive = group A tends to be higher than group B."
            ),
        }
        if bootstrap_ci:
            lo, hi = _bootstrap_ci(a, b, _cliff_fn)
            delta_entry["ci_95"] = [lo, hi]
        result["cliff_delta"] = delta_entry

    # ── Risk measures (binary outcomes only) ─────────────────────────
    if params.compute_risk_measures:
        unique_vals = set(np.concatenate([a, b]))
        if unique_vals <= {0.0, 1.0}:
            p_a = float(a.mean())
            p_b = float(b.mean())
            rr = (p_a / p_b) if p_b > 0 else None
            or_val = ((p_a / (1 - p_a)) / (p_b / (1 - p_b))) if 0 < p_a < 1 and 0 < p_b < 1 else None
            rd = p_a - p_b
            nnt = (1.0 / abs(rd)) if rd != 0 else None
            result["risk_measures"] = {
                "p_a": round(p_a, 4),
                "p_b": round(p_b, 4),
                "risk_difference": round(rd, 4),
                "risk_ratio": round(rr, 4) if rr is not None else None,
                "odds_ratio": round(or_val, 4) if or_val is not None else None,
                "nnt": round(nnt, 2) if nnt is not None else None,
                "note": "Risk measures are only valid when numeric_column is a binary (0/1) endpoint.",
            }
        else:
            result["risk_measures"] = {
                "error": "compute_risk_measures=True but column is not binary (0/1). Skipped."
            }

    if notes:
        result["runtime_notes"] = list(dict.fromkeys(notes))

    return result
