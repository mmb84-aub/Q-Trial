"""
Multiple testing correction tool — Stage 4 statistical tool.

Input:  list of raw p-values + correction method (bonferroni|holm|fdr_bh|fdr_by).
Output: MultipleTestingResult with adjusted p-values, rejection flags, and FWER/FDR.
Does:   applies family-wise error rate (FWER) or false discovery rate (FDR)
        corrections when the agent has run multiple simultaneous hypothesis tests,
        per ICH E9 guidelines for clinical trial analysis.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class MultipleTestingParams(BaseModel):
    p_values: list[float] = Field(
        description="List of raw p-values to correct (between 0 and 1)."
    )
    labels: list[str] | None = Field(
        default=None,
        description=(
            "Label for each p-value (e.g. test name or variable). "
            "Must be the same length as p_values."
        ),
    )
    method: str = Field(
        default="BH",
        description=(
            "Correction method: "
            "'BH' (Benjamini-Hochberg FDR), "
            "'bonferroni' (FWER, conservative), "
            "'holm' (Holm-Bonferroni, less conservative than Bonferroni)."
        ),
    )
    alpha: float = Field(default=0.05, description="Significance level (default 0.05).")


def _bh_correction(p_vals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_vals)
    order = np.argsort(p_vals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    adj = p_vals * n / ranks  # raw adjustment
    # Enforce monotonicity: take cumulative minimum from largest rank down
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return np.minimum(adj, 1.0)


def _holm_correction(p_vals: np.ndarray, alpha: float) -> np.ndarray:
    """Holm-Bonferroni step-down correction."""
    n = len(p_vals)
    order = np.argsort(p_vals)
    adj = np.empty(n)
    running_min = 1.0
    for rank, idx in enumerate(order):
        adj[idx] = p_vals[idx] * (n - rank)
    # Enforce monotonicity from smallest upward
    for rank, idx in enumerate(order):
        adj[idx] = min(adj[idx], running_min)
        running_min = adj[idx]  # monotone non-decreasing already handled by min
    return np.minimum(adj, 1.0)


@tool(
    name="multiple_testing_correction",
    description=(
        "Apply multiple testing correction to a list of raw p-values. "
        "BH (Benjamini-Hochberg): controls False Discovery Rate — recommended for exploratory analysis. "
        "Bonferroni: conservative FWER control, multiply each p by the number of tests. "
        "Holm: less conservative than Bonferroni, still controls FWER. "
        "Returns corrected p-values and significance flags at the specified alpha."
    ),
    params_model=MultipleTestingParams,
    category="stats",
)
def multiple_testing_correction(params: MultipleTestingParams, ctx: AgentContext) -> dict:
    raw = np.array(params.p_values, dtype=float)
    n = len(raw)

    if params.labels and len(params.labels) != n:
        raise ValueError(
            f"labels length ({len(params.labels)}) must match p_values length ({n})."
        )
    labels = params.labels or [f"test_{i+1}" for i in range(n)]

    method_lower = params.method.lower()
    if method_lower == "bh":
        adj = _bh_correction(raw, params.alpha)
        method_name = "Benjamini-Hochberg (FDR)"
    elif method_lower == "bonferroni":
        adj = np.minimum(raw * n, 1.0)
        method_name = "Bonferroni (FWER)"
    elif method_lower == "holm":
        adj = _holm_correction(raw, params.alpha)
        method_name = "Holm-Bonferroni (FWER)"
    else:
        raise ValueError(
            f"method must be 'BH', 'bonferroni', or 'holm'. Got: '{params.method}'"
        )

    results = []
    for label, raw_p, adj_p in zip(labels, raw.tolist(), adj.tolist()):
        results.append(
            {
                "label": label,
                "raw_p_value": round(raw_p, 6),
                "adjusted_p_value": round(adj_p, 6),
                "significant": bool(adj_p < params.alpha),
            }
        )

    n_significant = int(sum(r["significant"] for r in results))

    return {
        "method": method_name,
        "alpha": params.alpha,
        "n_tests": n,
        "n_significant_after_correction": n_significant,
        "results": results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint classification + hierarchical gatekeeping  (standalone functions)
# ──────────────────────────────────────────────────────────────────────────────


def classify_endpoints(findings: list[dict], primary_ids: list[str]) -> list[dict]:
    """Tag each finding with 'primary' or 'secondary' endpoint type.

    Parameters
    ----------
    findings:
        List of finding dicts, each containing at least ``{"id": str, "p_value": float}``.
    primary_ids:
        List of finding IDs that correspond to primary endpoints.

    Returns
    -------
    The same list with an added ``"endpoint_type"`` key on each dict
    (``"primary"`` if ``finding["id"]`` is in *primary_ids*, else ``"secondary"``).
    """
    primary_set = set(primary_ids)
    out = []
    for finding in findings:
        tagged = dict(finding)
        tagged["endpoint_type"] = "primary" if finding.get("id") in primary_set else "secondary"
        out.append(tagged)
    return out


def hierarchical_testing(findings: list[dict]) -> dict:
    """Enforce hierarchical (gatekeeping) testing for primary → secondary endpoints.

    Parameters
    ----------
    findings:
        List of finding dicts already tagged by :func:`classify_endpoints`.  Each
        dict must contain:
        ``{"id", "p_value", "endpoint_type", "adjusted_p_value"}``.

    Logic
    -----
    1. If **any** primary endpoint has ``adjusted_p_value < 0.05`` → gate is OPEN.
    2. If NO primary endpoint reaches significance → all secondary results are
       marked ``gated_out = True`` (closed hierarchy).
    3. When the gate is open, BH-FDR is re-applied to secondary endpoints only.

    Returns
    -------
    dict with keys:
    ``primary_significant``, ``gate_open``, ``findings`` (extended), ``interpretation``.
    """
    primary = [f for f in findings if f.get("endpoint_type") == "primary"]
    secondary = [f for f in findings if f.get("endpoint_type") == "secondary"]

    primary_significant = any(
        float(f.get("adjusted_p_value", 1.0)) < 0.05 for f in primary
    )
    gate_open = primary_significant

    extended: list[dict] = []

    # Primary findings — gate does not apply to them; just propagate
    for f in primary:
        out = dict(f)
        out["gated_out"] = False
        out["hierarchical_adjusted_p"] = f.get("adjusted_p_value")
        out["reason"] = None
        extended.append(out)

    if not gate_open:
        # Close the gate: mark all secondary as gated_out
        for f in secondary:
            out = dict(f)
            out["gated_out"] = True
            out["hierarchical_adjusted_p"] = None
            out["reason"] = (
                "Primary endpoint did not reach significance — hierarchical gate closed"
            )
            extended.append(out)
        interpretation = (
            "No primary endpoint reached significance after correction. "
            "All secondary endpoint results are gated out and should not be "
            "interpreted as confirmatory."
        )
    else:
        # Re-apply BH-FDR to secondary endpoints
        if secondary:
            sec_p = np.array([float(f.get("p_value", 1.0)) for f in secondary])
            sec_adj = _bh_correction(sec_p, alpha=0.05)
            for f, adj_p in zip(secondary, sec_adj.tolist()):
                out = dict(f)
                out["gated_out"] = False
                out["hierarchical_adjusted_p"] = round(float(adj_p), 6)
                out["reason"] = None
                extended.append(out)
        interpretation = (
            "Primary endpoint is significant — hierarchical gate is open. "
            "Secondary endpoints are evaluated using BH-FDR correction."
        )

    return {
        "primary_significant": primary_significant,
        "gate_open": gate_open,
        "findings": extended,
        "interpretation": interpretation,
    }
