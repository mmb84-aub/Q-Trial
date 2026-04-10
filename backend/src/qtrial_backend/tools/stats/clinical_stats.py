"""
Input: pd.DataFrame + config dict with trial design parameters
Output: Three-stage clinical analysis result (integrity, analysis, corrections)
Purpose: Single entry point for a complete clinical trial statistical analysis:
  Stage 1 (Integrity), Stage 2 (Analysis), Stage 3 (Multiple-testing correction).
Reference: FDA Guidance for Industry — Statistical Principles for Clinical Trials (1998);
  ICH E9(R1) Addendum on Estimands (2019).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), returning {"error": str} on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"error": str(exc)}


def _make_ctx(df: pd.DataFrame):
    """Construct a minimal AgentContext wrapping *df* for existing @tool functions."""
    from qtrial_backend.agent.context import AgentContext

    return AgentContext(dataframe=df, dataset_name="clinical_analysis")


# ──────────────────────────────────────────────────────────────────────────────
# Public orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_clinical_analysis(df: pd.DataFrame, config: dict) -> dict:  # noqa: C901
    """Run a complete three-stage clinical trial statistical analysis.

    Parameters
    ----------
    df:
        Source DataFrame (rows = subjects, columns = variables).
    config:
        Analysis configuration dict.  All keys are optional:

        - ``treatment_col``        : str – treatment/arm column
        - ``primary_endpoints``    : list[str]
        - ``secondary_endpoints``  : list[str]
        - ``time_col``             : str – for longitudinal / MMRM
        - ``subject_col``          : str – patient ID for MMRM
        - ``event_col``            : str – event indicator for survival
        - ``outcome_type``         : "continuous" | "binary" | "survival" | "longitudinal"
        - ``subgroup_cols``        : list[str]
        - ``alpha``                : float (default 0.05)

    Returns
    -------
    dict with keys stage_1_integrity, stage_2_analysis, stage_3_corrections,
    and clinical_summary.
    """
    alpha: float = float(config.get("alpha", 0.05))
    treatment_col: str | None = config.get("treatment_col")
    primary_endpoints: list[str] = list(config.get("primary_endpoints") or [])
    secondary_endpoints: list[str] = list(config.get("secondary_endpoints") or [])
    time_col: str | None = config.get("time_col")
    subject_col: str | None = config.get("subject_col")
    event_col: str | None = config.get("event_col")
    outcome_type: str = str(config.get("outcome_type") or "continuous").lower()
    subgroup_cols: list[str] = list(config.get("subgroup_cols") or [])

    # ──────────────────────────────────────────────────────────────────────
    # STAGE 1 — DATA INTEGRITY
    # ──────────────────────────────────────────────────────────────────────
    from qtrial_backend.tools.stats.digit_preference import _digit_preference_logic
    from qtrial_backend.tools.stats.missing import little_mcar_test
    from qtrial_backend.tools.stats.baseline_balance import (
        baseline_balance as _baseline_balance,
        BaselineBalanceParams,
    )

    digit_pref = _safe(_digit_preference_logic, df)
    mcar_result = _safe(little_mcar_test, df)

    # Baseline balance: use all numeric columns that aren't endpoints/time/event/subject
    exclude_cols = set(
        primary_endpoints
        + secondary_endpoints
        + ([treatment_col] if treatment_col else [])
        + ([time_col] if time_col else [])
        + ([event_col] if event_col else [])
        + ([subject_col] if subject_col else [])
    )
    baseline_balance_result: dict = {}
    if treatment_col and treatment_col in df.columns:
        baseline_cols = [
            c
            for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        if baseline_cols:
            ctx = _make_ctx(df)
            bb_params = BaselineBalanceParams(
                treatment_column=treatment_col,
                baseline_columns=baseline_cols[:20],  # cap at 20 for readability
            )
            baseline_balance_result = _safe(_baseline_balance, bb_params, ctx)

    # Collect integrity warnings
    integrity_warnings: list[str] = []
    if isinstance(digit_pref, dict) and digit_pref.get("integrity_concern"):
        flagged = [f["column"] for f in digit_pref.get("flagged_columns", [])]
        integrity_warnings.append(
            f"Digit preference detected in columns: {flagged}. "
            "Possible manual entry bias or data fabrication."
        )
    if isinstance(mcar_result, dict):
        mcar_inner = mcar_result.get("little_mcar_test", {})
        if mcar_inner.get("classification") == "Not MCAR":
            integrity_warnings.append(
                f"Missingness: {mcar_inner.get('interpretation', 'MCAR rejected.')} "
                f"{mcar_inner.get('recommendation', '')}"
            )
    if isinstance(baseline_balance_result, dict) and baseline_balance_result.get(
        "imbalanced_variables"
    ):
        imbal = baseline_balance_result["imbalanced_variables"]
        integrity_warnings.append(
            f"Baseline imbalance detected (SMD > threshold) in: {imbal}"
        )

    data_quality_passed = len(integrity_warnings) == 0

    stage_1: dict = {
        "digit_preference": digit_pref,
        "baseline_balance": baseline_balance_result,
        "missingness_classification": mcar_result,
        "integrity_warnings": integrity_warnings,
        "data_quality_passed": data_quality_passed,
    }

    # ──────────────────────────────────────────────────────────────────────
    # STAGE 2 — ANALYSIS
    # ──────────────────────────────────────────────────────────────────────
    from qtrial_backend.tools.stats.mice_imputation import get_imputed_dataframe, _mice_logic
    from qtrial_backend.tools.stats.mmrm import _mmrm_logic
    from qtrial_backend.tools.stats.ancova import _ancova_logic
    from qtrial_backend.tools.stats.subgroup_analysis import _subgroup_logic
    from qtrial_backend.tools.stats.hypothesis_test import (
        hypothesis_test as _hypothesis_test,
        HypothesisTestParams,
    )
    from qtrial_backend.tools.stats.effect_size import (
        effect_size as _effect_size,
        EffectSizeParams,
    )
    from qtrial_backend.tools.stats.crosstab import (
        cross_tabulation as _crosstab,
        CrosstabParams,
    )
    from qtrial_backend.tools.stats.survival import (
        survival_analysis as _survival,
        SurvivalParams,
    )

    # Decide whether to use imputed data
    mcar_class = "MCAR"
    if isinstance(mcar_result, dict):
        mcar_class = (
            mcar_result.get("little_mcar_test", {}).get("classification", "MCAR")
        )

    missing_cols_for_imputation = []
    if isinstance(mcar_result, dict):
        missing_cols_for_imputation = (
            mcar_result.get("little_mcar_test", {}).get("missing_columns", [])
        )

    use_imputation = mcar_class == "Not MCAR" and bool(missing_cols_for_imputation)
    analysis_df = df
    mice_pooled_results: dict | None = None

    if use_imputation:
        # Run proper m=5 MICE with Rubin's Rules pooling
        try:
            mice_pooled_results = _mice_logic(df, missing_cols_for_imputation, m=5)
        except Exception as exc:
            mice_pooled_results = {"error": str(exc)}

        # Also produce one complete imputed DataFrame for downstream model-based
        # analyses (MMRM, ANCOVA, etc.) that require a full dataset.
        try:
            analysis_df = get_imputed_dataframe(df, missing_cols_for_imputation)
        except Exception as exc:
            use_imputation = False  # fall back to original
            stage_1["integrity_warnings"].append(
                f"MICE imputation failed ({exc}); using original data with listwise deletion."
            )

    # Determine treatment groups for two-group comparisons
    group_a: str | None = None
    group_b: str | None = None
    if treatment_col and treatment_col in analysis_df.columns:
        trt_vals = analysis_df[treatment_col].dropna().unique().tolist()
        if len(trt_vals) >= 2:
            group_a = str(trt_vals[0])
            group_b = str(trt_vals[1])

    ctx = _make_ctx(analysis_df)
    primary_analysis: dict = {}
    all_endpoints = primary_endpoints + secondary_endpoints

    # cLDA result placeholder (populated only for longitudinal data)
    clda_result: dict | None = None

    if outcome_type == "longitudinal":
        if time_col and treatment_col and primary_endpoints:
            primary_analysis = _safe(
                _mmrm_logic,
                analysis_df,
                primary_endpoints[0],
                time_col,
                treatment_col,
                subject_col,
                None,
            )

            # cLDA — complementary to MMRM for longitudinal repeated-measures
            from qtrial_backend.tools.stats.clda import _clda_logic

            clda_result = _safe(
                _clda_logic,
                analysis_df,
                primary_endpoints[0],
                time_col,
                treatment_col,
                subject_col,
            )
        else:
            primary_analysis = {
                "error": "longitudinal outcome requires time_col and primary_endpoints in config"
            }
            clda_result = {
                "skipped": True,
                "reason": "cLDA requires time_col, treatment_col, and primary_endpoints.",
            }

    elif outcome_type == "survival":
        if event_col and time_col:
            surv_params = SurvivalParams(
                time_column=time_col,
                event_column=event_col,
                group_column=treatment_col,
            )
            primary_analysis = _safe(_survival, surv_params, ctx)
        else:
            primary_analysis = {
                "error": "survival outcome requires time_col and event_col in config"
            }

    elif outcome_type == "binary":
        if treatment_col and primary_endpoints:
            binary_results = []
            for ep in primary_endpoints[:5]:  # limit to avoid runaway analysis
                ct_params = CrosstabParams(
                    row_column=treatment_col,
                    col_column=ep,
                )
                ct_res = _safe(_crosstab, ct_params, ctx)
                es_result: dict = {}
                if group_a and group_b:
                    es_params = EffectSizeParams(
                        numeric_column=ep,
                        group_column=treatment_col,
                        group_a=group_a,
                        group_b=group_b,
                        compute_risk_measures=True,
                    )
                    es_result = _safe(_effect_size, es_params, ctx)
                binary_results.append({"endpoint": ep, "crosstab": ct_res, "effect_size": es_result})
            primary_analysis = {"binary_tests": binary_results}
        else:
            primary_analysis = {
                "error": "binary outcome requires treatment_col and primary_endpoints in config"
            }

    else:  # continuous (default)
        if treatment_col and primary_endpoints and group_a and group_b:
            continuous_results = []
            for ep in primary_endpoints[:5]:
                ht_params = HypothesisTestParams(
                    numeric_column=ep,
                    group_column=treatment_col,
                    group_a=group_a,
                    group_b=group_b,
                    alpha=alpha,
                )
                ht_res = _safe(_hypothesis_test, ht_params, ctx)

                es_params = EffectSizeParams(
                    numeric_column=ep,
                    group_column=treatment_col,
                    group_a=group_a,
                    group_b=group_b,
                )
                es_res = _safe(_effect_size, es_params, ctx)
                continuous_results.append(
                    {"endpoint": ep, "hypothesis_test": ht_res, "effect_size": es_res}
                )
            primary_analysis = {"continuous_tests": continuous_results}
        else:
            primary_analysis = {
                "note": "treatment_col or primary_endpoints not provided; skipped primary analysis"
            }

    # Set cLDA skip reason for non-longitudinal outcome types
    if clda_result is None:
        clda_result = {
            "skipped": True,
            "reason": (
                f"cLDA is only applicable for longitudinal repeated-measures data "
                f"(current outcome_type: {outcome_type})."
            ),
        }

    # Subgroup analysis
    subgroup_result: dict | None = None
    if subgroup_cols and treatment_col and all_endpoints and group_a and group_b:
        ep_for_sg = primary_endpoints[0] if primary_endpoints else all_endpoints[0]
        subgroup_result = _safe(
            _subgroup_logic,
            analysis_df,
            ep_for_sg,
            treatment_col,
            [c for c in subgroup_cols if c in analysis_df.columns],
            "continuous" if outcome_type not in ("binary",) else "binary",
        )

    # ANCOVA (if primary endpoints + baseline numeric covariates available)
    ancova_result: dict | None = None
    if treatment_col and primary_endpoints and group_a:
        ancova_candidate_covs = [
            c
            for c in analysis_df.columns
            if c not in exclude_cols
            and pd.api.types.is_numeric_dtype(analysis_df[c])
            and analysis_df[c].notna().sum() > 10
        ][:5]  # cap at 5 covariates
        if ancova_candidate_covs and primary_endpoints[0] in analysis_df.columns:
            ancova_result = _safe(
                _ancova_logic,
                analysis_df,
                primary_endpoints[0],
                treatment_col,
                ancova_candidate_covs,
                False,
            )

    stage_2: dict = {
        "imputation_used": use_imputation,
        "imputation_method": "MICE (m=5, Rubin's Rules)" if use_imputation else None,
        "mice_pooled_results": mice_pooled_results,
        "primary_analysis": primary_analysis,
        "clda": clda_result,
        "subgroup_analysis": subgroup_result,
        "ancova": ancova_result,
    }

    # ──────────────────────────────────────────────────────────────────────
    # STAGE 3 — MULTIPLE TESTING CORRECTION
    # ──────────────────────────────────────────────────────────────────────
    from qtrial_backend.tools.stats.multiple_testing import (
        _bh_correction,
        classify_endpoints,
        hierarchical_testing,
    )
    from qtrial_backend.tools.stats.power_analysis import batch_power_analysis

    # Harvest all p-values and effect sizes from Stage 2 results
    raw_findings: list[dict] = []

    def _extract_p(res: dict, ep: str, ep_type: str, es_res: dict) -> dict | None:
        p_val = res.get("p_value")
        if p_val is None:
            return None
        # Attempt to extract Cohen's d from effect_size result
        cohen_key = "cohen_d"
        d_val = 0.0
        ci_lo, ci_hi = 0.0, 0.0
        if isinstance(es_res, dict) and cohen_key in es_res:
            d_info = es_res[cohen_key]
            if isinstance(d_info, dict):
                d_val = float(d_info.get("value", 0.0) or 0.0)
                # effect_size tool stores CIs as ci_95: [lo, hi]
                ci_95 = d_info.get("ci_95")
                if isinstance(ci_95, (list, tuple)) and len(ci_95) >= 2:
                    ci_lo = float(ci_95[0] or 0.0)
                    ci_hi = float(ci_95[1] or 0.0)
        # Estimate n per group
        n_pg = 10
        if isinstance(es_res, dict):
            ga = es_res.get("group_a", {})
            if isinstance(ga, dict):
                n_pg = max(int(ga.get("n", 10)), 2)
        return {
            "finding_id": ep,
            "id": ep,
            "endpoint_type": ep_type,
            "p_value": float(p_val),
            "adjusted_p_value": float(p_val),
            "effect_size": d_val,
            "effect_size_ci": [ci_lo, ci_hi],
            "n_per_group": n_pg,
            "alpha": alpha,
        }

    # Continuous results
    cont = primary_analysis.get("continuous_tests", [])
    for item in cont:
        ep = item.get("endpoint", "")
        ep_type = "primary" if ep in primary_endpoints else "secondary"
        finding = _extract_p(
            item.get("hypothesis_test", {}),
            ep,
            ep_type,
            item.get("effect_size", {}),
        )
        if finding:
            raw_findings.append(finding)

    # Binary results
    bin_tests = primary_analysis.get("binary_tests", [])
    for item in bin_tests:
        ep = item.get("endpoint", "")
        ep_type = "primary" if ep in primary_endpoints else "secondary"
        ct_res = item.get("crosstab", {})
        finding = _extract_p(ct_res, ep, ep_type, item.get("effect_size", {}))
        if finding:
            raw_findings.append(finding)

    # Survival (single p-value from log-rank)
    if outcome_type == "survival" and isinstance(primary_analysis, dict):
        logrank_p = primary_analysis.get("log_rank_p_value")
        if logrank_p is not None:
            raw_findings.append(
                {
                    "finding_id": "survival_primary",
                    "id": "survival_primary",
                    "endpoint_type": "primary",
                    "p_value": float(logrank_p),
                    "adjusted_p_value": float(logrank_p),
                    "effect_size": 0.0,
                    "effect_size_ci": [0.0, 0.0],
                    "n_per_group": max(int(len(df) // 2), 2),
                    "alpha": alpha,
                }
            )

    # Longitudinal: extract MMRM and cLDA findings for correction pipeline
    if outcome_type == "longitudinal":
        _long_label = primary_endpoints[0] if primary_endpoints else "longitudinal"

        # Cross-sectional Cohen's d as a conservative effect-size approximation for
        # longitudinal endpoints.  A proper repeated-measures Cohen's d requires the
        # ICC from the mixed model; this marginal estimate is labelled accordingly.
        _long_es_value = 0.0
        _long_es_ci: list = [0.0, 0.0]
        _long_n_per_group = max(int(len(analysis_df) // 2), 2)
        if group_a and group_b and treatment_col and primary_endpoints:
            _es_params = EffectSizeParams(
                numeric_column=primary_endpoints[0],
                group_column=treatment_col,
                group_a=group_a,
                group_b=group_b,
            )
            _long_es_raw = _safe(_effect_size, _es_params, ctx)
            if isinstance(_long_es_raw, dict) and "cohen_d" in _long_es_raw:
                _cd = _long_es_raw["cohen_d"]
                _long_es_value = float(_cd.get("value", 0.0))
                _long_es_ci = list(_cd.get("ci_95", [0.0, 0.0]))
                _long_n_per_group = max(
                    int(_long_es_raw.get("group_a", {}).get("n", len(analysis_df) // 2)), 2
                )

        # MMRM
        if isinstance(primary_analysis, dict) and not primary_analysis.get("error"):
            _mmrm_p = primary_analysis.get("p_value")
            if _mmrm_p is not None:
                raw_findings.append(
                    {
                        "finding_id": f"{_long_label}_mmrm",
                        "id": f"{_long_label}_mmrm",
                        "endpoint_type": "primary",
                        "p_value": float(_mmrm_p),
                        "adjusted_p_value": float(_mmrm_p),
                        "effect_size": _long_es_value,
                        "effect_size_ci": _long_es_ci,
                        "effect_size_note": (
                            "Cross-sectional Cohen's d (marginal approximation; "
                            "ignores repeated-measures structure)"
                        ),
                        "n_per_group": max(
                            int(primary_analysis.get("n_subjects", 10) / 2), 2
                        ),
                        "alpha": alpha,
                    }
                )
        # cLDA
        if (
            isinstance(clda_result, dict)
            and not clda_result.get("error")
            and not clda_result.get("skipped")
        ):
            _clda_p = clda_result.get("p_value")
            if _clda_p is not None:
                raw_findings.append(
                    {
                        "finding_id": f"{_long_label}_clda",
                        "id": f"{_long_label}_clda",
                        "endpoint_type": "primary",
                        "p_value": float(_clda_p),
                        "adjusted_p_value": float(_clda_p),
                        "effect_size": _long_es_value,
                        "effect_size_ci": _long_es_ci,
                        "effect_size_note": (
                            "Cross-sectional Cohen's d (marginal approximation; "
                            "ignores repeated-measures structure)"
                        ),
                        "n_per_group": max(
                            int(clda_result.get("n_subjects", 10) / 2), 2
                        ),
                        "alpha": alpha,
                    }
                )

    # Apply corrections
    primary_finding_ids = [f["id"] for f in raw_findings if f["endpoint_type"] == "primary"]
    secondary_finding_ids = [f["id"] for f in raw_findings if f["endpoint_type"] == "secondary"]

    primary_findings = [f for f in raw_findings if f["id"] in primary_finding_ids]
    secondary_findings = [f for f in raw_findings if f["id"] in secondary_finding_ids]

    # Bonferroni for primary
    if primary_findings:
        n_prim = len(primary_findings)
        for f in primary_findings:
            f["adjusted_p_value"] = min(float(f["p_value"]) * n_prim, 1.0)
            f["correction_method"] = "Bonferroni"

    # BH-FDR for secondary
    if secondary_findings:
        sec_raw = np.array([float(f["p_value"]) for f in secondary_findings])
        sec_adj = _bh_correction(sec_raw, alpha=alpha)
        for f, adj in zip(secondary_findings, sec_adj.tolist()):
            f["adjusted_p_value"] = float(adj)
            f["correction_method"] = "Benjamini-Hochberg FDR"

    all_corrected = primary_findings + secondary_findings

    # Hierarchical gatekeeping
    gate_result = hierarchical_testing(all_corrected)
    gate_open = gate_result.get("gate_open", True)

    # Power analysis for all findings
    power_findings = batch_power_analysis(all_corrected)

    # Build corrected_findings list
    corrected_findings: list[dict] = []
    power_map = {pf["finding_id"]: pf for pf in power_findings if "finding_id" in pf}

    for gf in gate_result.get("findings", all_corrected):
        fid = gf.get("id") or gf.get("finding_id", "")
        pwr = power_map.get(fid, {})
        adj_p = gf.get("hierarchical_adjusted_p") or gf.get("adjusted_p_value", gf.get("p_value", 1.0))
        cf: dict = {
            "finding_id": fid,
            "endpoint_type": gf.get("endpoint_type", "secondary"),
            "raw_p_value": round(float(gf.get("p_value", 1.0)), 6),
            "adjusted_p_value": round(float(adj_p or 1.0), 6),
            "correction_method": gf.get("correction_method", "none"),
            "effect_size": float(gf.get("effect_size", 0.0)),
            "effect_size_ci": gf.get("effect_size_ci", [0.0, 0.0]),
            "achieved_power": round(float(pwr.get("achieved_power", 0.0)), 4),
            "power_adequate": bool(pwr.get("adequately_powered", False)),
            "significant_after_correction": bool(
                not gf.get("gated_out", False)
                and float(adj_p or 1.0) < alpha
            ),
        }
        if "effect_size_note" in gf:
            cf["effect_size_note"] = gf["effect_size_note"]
        corrected_findings.append(cf)

    stage_3: dict = {
        "primary_correction": "Bonferroni",
        "secondary_correction": "Benjamini-Hochberg FDR",
        "hierarchical_gate_open": gate_open,
        "corrected_findings": corrected_findings,
    }

    # ──────────────────────────────────────────────────────────────────────
    # CLINICAL SUMMARY
    # ──────────────────────────────────────────────────────────────────────
    n_sig = sum(1 for f in corrected_findings if f.get("significant_after_correction"))
    n_total = len(corrected_findings)

    quality_str = (
        "Data integrity checks raised concerns."
        if not data_quality_passed
        else "Data integrity checks passed."
    )
    gate_str = (
        "Hierarchical gate is open (primary endpoint significant)."
        if gate_open
        else "Hierarchical gate is closed (primary endpoint not significant); secondary results are exploratory."
    )
    outcome_str = (
        f"{n_sig} of {n_total} endpoint(s) reached significance after correction."
        if corrected_findings
        else "No endpoints could be evaluated (insufficient configuration)."
    )

    clinical_summary = f"{quality_str} {outcome_str} {gate_str}"

    return {
        "framework": "Three-Stage Clinical Trial Analysis",
        "stage_1_integrity": stage_1,
        "stage_2_analysis": stage_2,
        "stage_3_corrections": stage_3,
        "clinical_summary": clinical_summary,
    }
