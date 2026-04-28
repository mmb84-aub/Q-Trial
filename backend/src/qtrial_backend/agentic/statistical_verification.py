"""
Deterministic statistical claim verification.

This module intentionally implements a narrow MVP. It maps only high-confidence
claims onto existing deterministic statistical tools and marks ambiguous claims
as not_verifiable rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.agentic.schemas import (
    StatisticalVerificationMetrics,
    StatisticalVerificationReport,
    VerifiedClaim,
)
from qtrial_backend.tools.stats.baseline_balance import BaselineBalanceParams, baseline_balance
from qtrial_backend.tools.stats.correlation import CorrelationParams, correlation_matrix
from qtrial_backend.tools.stats.crosstab import CrosstabParams, cross_tabulation
from qtrial_backend.tools.stats.effect_size import EffectSizeParams, effect_size
from qtrial_backend.tools.stats.hypothesis_test import HypothesisTestParams, hypothesis_test
from qtrial_backend.tools.stats.regression import RegressionParams, regression
from qtrial_backend.tools.stats.survival import SurvivalParams, survival_analysis


_P_VALUE_RE = re.compile(
    r"\bp(?:\s*[- ]?\s*value)?\s*(<=|>=|≤|≥|=|<|>)\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)
_NUM_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
_CI_RE = re.compile(
    rf"(?:95\s*%\s*)?(?:ci|confidence interval)\s*[:=]?\s*[\[(]?\s*{_NUM_RE}\s*(?:,|to)\s*{_NUM_RE}",
    re.IGNORECASE,
)
_EFFECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("odds_ratio", re.compile(rf"\b(?:or|odds ratio)\s*(?:=|:)?\s*{_NUM_RE}", re.IGNORECASE)),
    ("hazard_ratio", re.compile(rf"\b(?:hr|hazard ratio)\s*(?:=|:)?\s*{_NUM_RE}", re.IGNORECASE)),
    ("correlation_r", re.compile(rf"\b(?:r|correlation(?: coefficient)?)\s*(?:=|:)\s*{_NUM_RE}", re.IGNORECASE)),
    ("cohen_d", re.compile(rf"\b(?:cohen'?s?\s*d|d)\s*(?:=|:)\s*{_NUM_RE}", re.IGNORECASE)),
    ("smd", re.compile(rf"\b(?:smd|standardi[sz]ed mean difference)\s*(?:=|:)?\s*{_NUM_RE}", re.IGNORECASE)),
)
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_WORD_RE = re.compile(r"[a-z0-9]+")

_VAGUE_CLINICAL_TERMS = (
    "clinically important",
    "clinically meaningful",
    "robust",
    "useful",
    "promising",
    "concerning",
    "interesting",
)


@dataclass
class ColumnResolution:
    column: str | None
    role: str
    source: str
    confidence: str
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "role": self.role,
            "source": self.source,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }


@dataclass
class _ClaimCandidate:
    claim_id: str
    source: str
    text: str
    variable: str | None
    endpoint: str | None
    reported_p_value: float | None
    reported_p_operator: str | None
    claimed_significant: bool | None
    direction: str | None
    reported_effect_size: float | None
    reported_effect_size_label: str | None
    reported_ci_lower: float | None
    reported_ci_upper: float | None
    claim_family: str | None
    resolutions: list[ColumnResolution]


def build_statistical_verification_report(
    df: pd.DataFrame,
    qtrial_findings: list[Any] | None,
    analyst_report_text: str,
    analyst_report_name: str | None = None,
    metadata: Any | None = None,
    column_dict: dict[str, str] | None = None,
) -> StatisticalVerificationReport:
    """Build a narrow deterministic verification report for extracted claims."""
    ctx = AgentContext(dataframe=df, dataset_name="statistical_verification")
    candidates = _extract_candidates(
        df=df,
        analyst_report_text=analyst_report_text,
        analyst_report_name=analyst_report_name,
        qtrial_findings=qtrial_findings,
        metadata=metadata,
        column_dict=column_dict,
    )

    claims = [_verify_candidate(candidate, df, ctx, metadata, column_dict) for candidate in candidates]
    metrics = _build_metrics(claims)
    summary = _build_summary(metrics)
    return StatisticalVerificationReport(summary=summary, metrics=metrics, claims=claims)


def _extract_candidates(
    *,
    df: pd.DataFrame,
    analyst_report_text: str,
    analyst_report_name: str | None,
    qtrial_findings: list[Any] | None,
    metadata: Any | None,
    column_dict: dict[str, str] | None,
) -> list[_ClaimCandidate]:
    candidates: list[_ClaimCandidate] = []
    known_columns = list(df.columns)

    for idx, text in enumerate(_iter_claim_texts(analyst_report_text), start=1):
        candidates.append(
            _build_candidate(
                claim_id=f"analyst_{idx}",
                source=analyst_report_name or "analyst_report",
                text=text,
                known_columns=known_columns,
                df=df,
                metadata=metadata,
                column_dict=column_dict,
            )
        )

    for idx, finding in enumerate(qtrial_findings or [], start=1):
        text = _finding_text(finding)
        if not text:
            continue
        candidates.append(
            _build_candidate(
                claim_id=f"qtrial_{idx}",
                source="qtrial",
                text=text,
                known_columns=known_columns,
                df=df,
                metadata=metadata,
                column_dict=column_dict,
            )
        )

    return candidates


def _iter_claim_texts(raw_text: str) -> list[str]:
    texts: list[str] = []
    for line in raw_text.splitlines():
        stripped = _BULLET_PREFIX_RE.sub("", line.strip())
        if not stripped:
            continue
        if len(stripped) < 180:
            texts.append(stripped)
        else:
            texts.extend(part.strip() for part in _SENTENCE_SPLIT_RE.split(stripped) if part.strip())
    return [_clean_text(text) for text in texts if len(_clean_text(text).split()) >= 4]


def _build_candidate(
    *,
    claim_id: str,
    source: str,
    text: str,
    known_columns: list[str],
    df: pd.DataFrame,
    metadata: Any | None,
    column_dict: dict[str, str] | None,
) -> _ClaimCandidate:
    columns = _find_mentioned_columns(text, known_columns)
    endpoint_resolution = _resolve_endpoint_resolution(text, df, metadata, column_dict)
    endpoint = endpoint_resolution.column
    variable = _resolve_variable(text, columns, endpoint)
    if variable is None and endpoint and _mentions_treatment_or_group(text):
        variable = _resolve_group_column(text, df, metadata)
    reported_p_operator, reported_p_value = _extract_p_value(text)
    claimed_significant = _infer_claimed_significance(text, reported_p_operator, reported_p_value)
    direction = _infer_direction(text)
    reported_effect_label, reported_effect_size = _extract_reported_effect(text)
    reported_ci_lower, reported_ci_upper = _extract_reported_ci(text)
    claim_family = _classify_claim_family(text, df, columns, variable, endpoint, metadata)
    return _ClaimCandidate(
        claim_id=claim_id,
        source=source,
        text=text,
        variable=variable,
        endpoint=endpoint,
        reported_p_value=reported_p_value,
        reported_p_operator=reported_p_operator,
        claimed_significant=claimed_significant,
        direction=direction,
        reported_effect_size=reported_effect_size,
        reported_effect_size_label=reported_effect_label,
        reported_ci_lower=reported_ci_lower,
        reported_ci_upper=reported_ci_upper,
        claim_family=claim_family,
        resolutions=[endpoint_resolution],
    )


def _verify_candidate(
    candidate: _ClaimCandidate,
    df: pd.DataFrame,
    ctx: AgentContext,
    metadata: Any | None,
    column_dict: dict[str, str] | None,
) -> VerifiedClaim:
    if _is_vague(candidate.text):
        return _claim(candidate, "not_verifiable", rationale="Claim is vague or clinical-interpretive, not a concrete statistical assertion.")
    if candidate.claim_family is None:
        return _claim(candidate, "not_verifiable", rationale="Could not map claim to a supported MVP statistical family.")

    try:
        if candidate.claim_family == "cox_survival_predictor":
            return _verify_cox(candidate, df, ctx, metadata, column_dict)
        if candidate.claim_family == "kaplan_meier_logrank":
            return _verify_logrank(candidate, df, ctx, metadata, column_dict)
        if candidate.claim_family == "logistic_regression":
            return _verify_logistic(candidate, df, ctx)
        if candidate.claim_family == "correlation":
            return _verify_correlation(candidate, df, ctx)
        if candidate.claim_family == "continuous_group_difference":
            return _verify_continuous_group_difference(candidate, df, ctx)
        if candidate.claim_family == "binary_categorical_association":
            return _verify_categorical_association(candidate, df, ctx)
        if candidate.claim_family == "baseline_balance":
            return _verify_baseline_balance(candidate, df, ctx)
    except Exception as exc:
        return _claim(
            candidate,
            "unsupported",
            rationale=f"Mapped to {candidate.claim_family}, but deterministic verification failed: {exc}",
            test_used=candidate.claim_family,
        )

    return _claim(candidate, "unsupported", rationale="Mapped claim family is not implemented in the MVP.")


def _verify_categorical_association(
    candidate: _ClaimCandidate,
    df: pd.DataFrame,
    ctx: AgentContext,
) -> VerifiedClaim:
    if not candidate.variable or not candidate.endpoint:
        return _claim(candidate, "not_verifiable", rationale="Categorical association requires a variable and endpoint.")
    result = cross_tabulation(CrosstabParams(row_column=candidate.endpoint, col_column=candidate.variable), ctx)
    test_info = result.get("significance_test", {})
    p_value = _safe_float(test_info.get("p_value"))
    effect = _safe_float(test_info.get("cramers_v"))
    observed_direction = _binary_endpoint_direction(df, candidate.variable, candidate.endpoint)
    label, rationale = _label_from_significance(candidate, p_value, observed_direction=observed_direction)
    return _claim(
        candidate,
        label,
        rationale=rationale,
        test_used=str(test_info.get("test") or "cross_tabulation"),
        recomputed_p_value=p_value,
        effect_size=effect,
        effect_size_label="cramers_v" if effect is not None else None,
        effect_agreement="not_assessed",
        metadata={"tool_result": result},
    )


def _verify_continuous_group_difference(
    candidate: _ClaimCandidate,
    df: pd.DataFrame,
    ctx: AgentContext,
) -> VerifiedClaim:
    group_col = _resolve_group_column(candidate.text, df)
    if not candidate.variable or not group_col:
        return _claim(candidate, "not_verifiable", rationale="Group difference requires a numeric variable and group column.")
    groups = _two_group_labels(df[group_col])
    if groups is None:
        return _claim(candidate, "unsupported", rationale=f"Group column '{group_col}' does not have exactly two usable groups.")

    result = hypothesis_test(
        HypothesisTestParams(
            numeric_column=candidate.variable,
            group_column=group_col,
            group_a=groups[0],
            group_b=groups[1],
        ),
        ctx,
    )
    es = effect_size(
        EffectSizeParams(
            numeric_column=candidate.variable,
            group_column=group_col,
            group_a=groups[0],
            group_b=groups[1],
            method="both",
            bootstrap_ci=True,
        ),
        ctx,
    )
    p_value = _safe_float(result.get("p_value"))
    cohen = es.get("cohen_d", {}) if isinstance(es, dict) else {}
    ci = cohen.get("ci_95") if isinstance(cohen, dict) else None
    effect_value = _safe_float(cohen.get("value") if isinstance(cohen, dict) else None)
    ci_lower = _safe_float(ci[0]) if isinstance(ci, list) and len(ci) >= 2 else None
    ci_upper = _safe_float(ci[1]) if isinstance(ci, list) and len(ci) >= 2 else None
    observed_direction = _reverse_direction(_direction_from_effect(effect_value))
    label, rationale = _label_from_significance(candidate, p_value, observed_direction=observed_direction)
    label, rationale, effect_agreement = _apply_effect_agreement(
        candidate,
        label,
        rationale,
        effect_size=_reverse_effect(effect_value),
        effect_size_label="cohen_d",
        ci_lower=_reverse_effect(ci_upper),
        ci_upper=_reverse_effect(ci_lower),
    )
    return _claim(
        candidate,
        label,
        rationale=rationale,
        test_used=str(result.get("test") or "hypothesis_test"),
        recomputed_p_value=p_value,
        effect_size=effect_value,
        effect_size_label="cohen_d",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effect_agreement=effect_agreement,
        metadata={"tool_result": result, "effect_size_result": es, "group_column": group_col},
    )


def _verify_correlation(candidate: _ClaimCandidate, df: pd.DataFrame, ctx: AgentContext) -> VerifiedClaim:
    cols = _find_mentioned_columns(candidate.text, list(df.columns))
    numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) < 2:
        return _claim(candidate, "not_verifiable", rationale="Correlation claim requires two numeric dataset columns.")
    left, right = numeric_cols[:2]
    method = "spearman" if "spearman" in candidate.text.lower() else "pearson"
    result = correlation_matrix(CorrelationParams(columns=[left, right], method=method), ctx)
    p_value = _safe_float(result.get("p_values", {}).get(left, {}).get(right))
    corr = _safe_float(result.get("matrix", {}).get(left, {}).get(right))
    label, rationale = _label_from_significance(candidate, p_value, observed_direction=_direction_from_effect(corr))
    label, rationale, effect_agreement = _apply_effect_agreement(
        candidate,
        label,
        rationale,
        effect_size=corr,
        effect_size_label=f"{method}_r",
    )
    return _claim(
        candidate,
        label,
        rationale=rationale,
        test_used=f"{method}_correlation",
        recomputed_p_value=p_value,
        effect_size=corr,
        effect_size_label=f"{method}_r",
        effect_agreement=effect_agreement,
        metadata={"tool_result": result, "columns": [left, right]},
    )


def _verify_logistic(candidate: _ClaimCandidate, df: pd.DataFrame, ctx: AgentContext) -> VerifiedClaim:
    if not candidate.variable or not candidate.endpoint:
        return _claim(candidate, "not_verifiable", rationale="Logistic regression requires a predictor and binary endpoint.")
    if _implies_adjusted_model(candidate.text):
        return _claim(
            candidate,
            "unsupported",
            rationale="Claim implies an adjusted or multivariable model; MVP verification refuses to validate it with univariable logistic regression.",
            test_used="logistic_regression",
        )
    if not _is_binary_series(df[candidate.endpoint]):
        return _claim(candidate, "unsupported", rationale=f"Endpoint '{candidate.endpoint}' is not binary 0/1 for logistic regression.")
    result = regression(
        RegressionParams(
            model_type="logistic",
            outcome_column=candidate.endpoint,
            predictor_columns=[candidate.variable],
        ),
        ctx,
    )
    coef = _find_regression_term(result, candidate.variable, key="term")
    p_value = _safe_float(coef.get("p_value") if coef else None)
    odds_ratio = _safe_float(coef.get("odds_ratio") if coef else None)
    ci_lower = _safe_float(coef.get("or_ci_lower") if coef else None)
    ci_upper = _safe_float(coef.get("or_ci_upper") if coef else None)
    label, rationale = _label_from_significance(candidate, p_value, observed_direction=_direction_from_ratio(odds_ratio))
    label, rationale, effect_agreement = _apply_effect_agreement(
        candidate,
        label,
        rationale,
        effect_size=odds_ratio,
        effect_size_label="odds_ratio",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
    return _claim(
        candidate,
        label,
        rationale=rationale,
        test_used="logistic_regression",
        recomputed_p_value=p_value,
        effect_size=odds_ratio,
        effect_size_label="odds_ratio",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effect_agreement=effect_agreement,
        metadata={"tool_result": result},
    )


def _verify_cox(
    candidate: _ClaimCandidate,
    df: pd.DataFrame,
    ctx: AgentContext,
    metadata: Any | None,
    column_dict: dict[str, str] | None,
) -> VerifiedClaim:
    if not candidate.variable or not candidate.endpoint:
        return _claim(candidate, "not_verifiable", rationale="Cox verification requires a predictor and event endpoint.")
    if _implies_adjusted_model(candidate.text):
        return _claim(
            candidate,
            "unsupported",
            rationale="Claim implies an adjusted or multivariable model; MVP verification refuses to validate it with univariable Cox regression.",
            test_used="cox_regression",
        )
    time_resolution = _resolve_time_column_resolution(candidate.text, df, metadata, column_dict)
    time_col = time_resolution.column
    if not time_col:
        return _claim(candidate, "not_verifiable", rationale="Cox verification requires a uniquely resolved time/follow-up column.", extra_resolutions=[time_resolution])
    event_codes, ambiguous_event = _resolve_event_codes_for_survival(df[candidate.endpoint], metadata, candidate.text)
    if ambiguous_event:
        return _claim(candidate, "not_verifiable", rationale="Survival event coding is ambiguous; explicit event code metadata is required.", extra_resolutions=[time_resolution])
    result = regression(
        RegressionParams(
            model_type="cox",
            outcome_column=candidate.endpoint,
            time_column=time_col,
            predictor_columns=[candidate.variable],
            event_codes=event_codes,
        ),
        ctx,
    )
    coef = _find_regression_term(result, candidate.variable, key="covariate")
    p_value = _safe_float(coef.get("p_value") if coef else None)
    hazard_ratio = _safe_float(coef.get("hazard_ratio") if coef else None)
    ci_lower = _safe_float(coef.get("hr_ci_lower") if coef else None)
    ci_upper = _safe_float(coef.get("hr_ci_upper") if coef else None)
    label, rationale = _label_from_significance(candidate, p_value, observed_direction=_direction_from_ratio(hazard_ratio))
    label, rationale, effect_agreement = _apply_effect_agreement(
        candidate,
        label,
        rationale,
        effect_size=hazard_ratio,
        effect_size_label="hazard_ratio",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
    return _claim(
        candidate,
        label,
        rationale=rationale,
        test_used="cox_regression",
        recomputed_p_value=p_value,
        effect_size=hazard_ratio,
        effect_size_label="hazard_ratio",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effect_agreement=effect_agreement,
        metadata={"tool_result": result, "time_column": time_col},
        extra_resolutions=[time_resolution],
    )


def _verify_logrank(
    candidate: _ClaimCandidate,
    df: pd.DataFrame,
    ctx: AgentContext,
    metadata: Any | None,
    column_dict: dict[str, str] | None,
) -> VerifiedClaim:
    time_resolution = _resolve_time_column_resolution(candidate.text, df, metadata, column_dict)
    time_col = time_resolution.column
    event_resolution = _first_resolution(candidate, "endpoint")
    event_col = candidate.endpoint
    group_resolution = _resolve_group_column_resolution(candidate.text, df, metadata)
    group_col = group_resolution.column
    if not time_col or not event_col or not group_col:
        missing_parts = []
        if not time_col:
            missing_parts.append("time/follow-up column")
        if not event_col:
            missing_parts.append("event column")
        if not group_col:
            missing_parts.append("group column")
        return _claim(
            candidate,
            "not_verifiable",
            rationale=f"Log-rank verification requires uniquely resolved {', '.join(missing_parts)}.",
            extra_resolutions=[time_resolution, group_resolution],
        )
    event_codes, ambiguous_event = _resolve_event_codes_for_survival(df[event_col], metadata, candidate.text)
    if ambiguous_event:
        return _claim(candidate, "not_verifiable", rationale="Survival event coding is ambiguous; explicit event code metadata is required.", extra_resolutions=[time_resolution, group_resolution])
    result = survival_analysis(
        SurvivalParams(
            time_column=time_col,
            event_column=event_col,
            group_column=group_col,
            event_codes=event_codes,
        ),
        ctx,
    )
    p_value = _safe_float(result.get("logrank_p_value"))
    label, rationale = _label_from_significance(candidate, p_value, observed_direction=None)
    return _claim(
        candidate,
        label,
        rationale=rationale,
        test_used="kaplan_meier_logrank",
        recomputed_p_value=p_value,
        effect_agreement="not_assessed",
        metadata={"tool_result": result, "time_column": time_col, "group_column": group_col},
        extra_resolutions=[time_resolution, group_resolution, event_resolution] if event_resolution else [time_resolution, group_resolution],
    )


def _verify_baseline_balance(candidate: _ClaimCandidate, df: pd.DataFrame, ctx: AgentContext) -> VerifiedClaim:
    group_col = _resolve_group_column(candidate.text, df)
    variable = candidate.variable
    if not group_col:
        return _claim(candidate, "not_verifiable", rationale="Baseline balance claim requires a treatment/group column.")
    baseline_cols = [variable] if variable and variable != group_col else [
        col for col in df.columns if col != group_col and pd.api.types.is_numeric_dtype(df[col])
    ][:10]
    if not baseline_cols:
        return _claim(candidate, "not_verifiable", rationale="No baseline columns could be mapped.")
    result = baseline_balance(BaselineBalanceParams(treatment_column=group_col, baseline_columns=baseline_cols), ctx)
    imbalanced = bool(result.get("n_imbalanced", 0))
    claimed_balanced = "balanced" in candidate.text.lower() and "imbal" not in candidate.text.lower()
    if claimed_balanced:
        label = "contradicted" if imbalanced else "verified"
    else:
        label = "verified" if imbalanced else "contradicted"
    return _claim(
        candidate,
        label,
        rationale=f"Baseline balance recomputed with SMD threshold {result.get('smd_threshold')}.",
        test_used="baseline_balance",
        effect_agreement="not_assessed",
        metadata={"tool_result": result, "baseline_columns": baseline_cols, "group_column": group_col},
    )


def _label_from_significance(
    candidate: _ClaimCandidate,
    p_value: float | None,
    *,
    observed_direction: str | None = None,
) -> tuple[str, str]:
    if p_value is None:
        return "unsupported", "The selected deterministic test did not produce a p-value."
    recomputed_significant = p_value < 0.05
    claimed = candidate.claimed_significant
    if claimed is None:
        return "partial", f"Recomputed p={p_value:.6g}; claim did not clearly state significance."
    if claimed != recomputed_significant:
        return "contradicted", f"Claim significance={claimed}; recomputed significance={recomputed_significant} (p={p_value:.6g})."

    claimed_direction = candidate.direction if candidate.direction in {"positive", "negative"} else None
    if claimed_direction:
        if observed_direction in {"positive", "negative"}:
            if claimed_direction != observed_direction:
                return (
                    "contradicted",
                    f"Significance agrees, but claimed direction={claimed_direction} conflicts with recomputed direction={observed_direction} (p={p_value:.6g}).",
                )
        elif recomputed_significant:
            return "partial", f"Significance agrees, but recomputed output does not safely establish the claimed direction (p={p_value:.6g})."

    if candidate.reported_p_value is not None and not _reported_p_value_matches(candidate, p_value):
        op = candidate.reported_p_operator or "="
        return "partial", f"Significance agrees, but reported p{op}{candidate.reported_p_value:.6g} is not reproduced by recomputed p={p_value:.6g}."
    return "verified", f"Claim agrees with recomputed result (p={p_value:.6g})."


def _apply_effect_agreement(
    candidate: _ClaimCandidate,
    label: str,
    rationale: str,
    *,
    effect_size: float | None,
    effect_size_label: str | None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> tuple[str, str, str]:
    reported = candidate.reported_effect_size
    reported_label = candidate.reported_effect_size_label
    compatible = _effect_labels_compatible(reported_label, effect_size_label)
    if reported is None or not compatible or effect_size is None:
        return label, rationale, "not_assessed"

    if _opposite_effect_side(reported, effect_size, effect_size_label):
        return (
            "contradicted",
            f"{rationale} Reported {reported_label}={reported:.6g} is on the opposite side of the null from recomputed {effect_size_label}={effect_size:.6g}.",
            "conflicts",
        )

    if _material_effect_difference(reported, effect_size, effect_size_label):
        return (
            _downgrade_to_partial(label),
            f"{rationale} Reported {reported_label}={reported:.6g} materially differs from recomputed {effect_size_label}={effect_size:.6g}.",
            "partial",
        )

    if _reported_ci_excludes_null_but_recomputed_crosses(
        candidate.reported_ci_lower,
        candidate.reported_ci_upper,
        ci_lower,
        ci_upper,
        effect_size_label,
    ):
        return (
            _downgrade_to_partial(label),
            f"{rationale} Reported CI excludes the null, but recomputed CI crosses the null.",
            "partial",
        )

    return label, rationale, "agrees"


def _downgrade_to_partial(label: str) -> str:
    return "partial" if label == "verified" else label


def _claim(
    candidate: _ClaimCandidate,
    label: str,
    *,
    rationale: str,
    test_used: str | None = None,
    recomputed_p_value: float | None = None,
    effect_size: float | None = None,
    effect_size_label: str | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    effect_agreement: str | None = None,
    metadata: dict[str, Any] | None = None,
    extra_resolutions: list[ColumnResolution | None] | None = None,
) -> VerifiedClaim:
    resolutions = [
        resolution
        for resolution in [*candidate.resolutions, *(extra_resolutions or [])]
        if resolution is not None
    ]
    warnings = [warning for resolution in resolutions for warning in resolution.warnings]
    normalized_effect_agreement = effect_agreement or "not_assessed"
    return VerifiedClaim(
        claim_id=candidate.claim_id,
        source_text=candidate.text,
        label=label,  # type: ignore[arg-type]
        variable=candidate.variable,
        endpoint=candidate.endpoint,
        test_used=test_used,
        recomputed_p_value=recomputed_p_value,
        reported_p_value=candidate.reported_p_value,
        effect_size=effect_size,
        effect_size_label=effect_size_label,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        reported_effect_size=candidate.reported_effect_size,
        reported_effect_size_label=candidate.reported_effect_size_label,
        reported_ci_lower=candidate.reported_ci_lower,
        reported_ci_upper=candidate.reported_ci_upper,
        effect_agreement=normalized_effect_agreement,  # type: ignore[arg-type]
        confidence_warnings=warnings,
        rationale=rationale,
        metadata={
            "source": candidate.source,
            "claim_family": candidate.claim_family,
            "claimed_significant": candidate.claimed_significant,
            "reported_p_operator": candidate.reported_p_operator,
            "direction": candidate.direction,
            "reported_effect_size": candidate.reported_effect_size,
            "reported_effect_size_label": candidate.reported_effect_size_label,
            "reported_ci_lower": candidate.reported_ci_lower,
            "reported_ci_upper": candidate.reported_ci_upper,
            "effect_agreement": normalized_effect_agreement,
            "column_resolutions": [resolution.as_dict() for resolution in resolutions],
            **(metadata or {}),
        },
    )


def _build_metrics(claims: list[VerifiedClaim]) -> StatisticalVerificationMetrics:
    total = len(claims)
    verified = sum(1 for c in claims if c.label == "verified")
    contradicted = sum(1 for c in claims if c.label == "contradicted")
    partial = sum(1 for c in claims if c.label == "partial")
    unsupported = sum(1 for c in claims if c.label == "unsupported")
    not_verifiable = sum(1 for c in claims if c.label == "not_verifiable")
    return StatisticalVerificationMetrics(
        total_claims=total,
        verified_count=verified,
        contradicted_count=contradicted,
        partial_count=partial,
        unsupported_count=unsupported,
        not_verifiable_count=not_verifiable,
        verification_rate=round(verified / total, 4) if total else 0.0,
        contradiction_rate=round(contradicted / total, 4) if total else 0.0,
    )


def _build_summary(metrics: StatisticalVerificationMetrics) -> str:
    return (
        f"Verified {metrics.verified_count} of {metrics.total_claims} extracted claims; "
        f"{metrics.contradicted_count} contradicted, {metrics.partial_count} partial, "
        f"{metrics.unsupported_count} unsupported, {metrics.not_verifiable_count} not verifiable."
    )


def _first_resolution(candidate: _ClaimCandidate, role: str) -> ColumnResolution | None:
    for resolution in candidate.resolutions:
        if resolution.role == role:
            return resolution
    return None


def _classify_claim_family(
    text: str,
    df: pd.DataFrame,
    columns: list[str],
    variable: str | None,
    endpoint: str | None,
    metadata: Any | None,
) -> str | None:
    lowered = text.lower()
    if "baseline" in lowered and any(token in lowered for token in ("balance", "balanced", "imbalanced", "smd")):
        return "baseline_balance"
    if "correlat" in lowered and len([c for c in columns if pd.api.types.is_numeric_dtype(df[c])]) >= 2:
        return "correlation"
    if any(token in lowered for token in ("cox", "hazard ratio", "hazard")):
        return "cox_survival_predictor"
    if variable and endpoint and _resolve_time_column(df, metadata) and any(token in lowered for token in ("predict", "predictor", "associated", "association")) and endpoint != variable and "survival" in lowered:
        return "cox_survival_predictor"
    if any(token in lowered for token in ("log-rank", "log rank", "kaplan", "kaplan-meier")):
        return "kaplan_meier_logrank"
    if "survival" in lowered and _resolve_group_column(text, df, metadata) and _resolve_time_column(df, metadata):
        return "kaplan_meier_logrank"
    if variable and endpoint and _is_binary_series(df[endpoint]) and any(token in lowered for token in ("predict", "predictor", "associated", "association", "risk")):
        if pd.api.types.is_numeric_dtype(df[variable]) and not _is_binary_series(df[variable]):
            return "logistic_regression"
        return "binary_categorical_association"
    group_col = _resolve_group_column(text, df, metadata)
    if variable and group_col and pd.api.types.is_numeric_dtype(df[variable]) and any(token in lowered for token in ("difference", "higher", "lower", "increased", "decreased", "between")):
        return "continuous_group_difference"
    if len(columns) >= 2 and all(_is_binary_or_categorical(df[col]) for col in columns[:2]) and any(token in lowered for token in ("associated", "association", "relationship")):
        return "binary_categorical_association"
    return None


def _resolve_variable(text: str, columns: list[str], endpoint: str | None) -> str | None:
    for col in columns:
        if endpoint and col == endpoint:
            continue
        return col
    return None


def _resolve_endpoint_resolution(
    text: str,
    df: pd.DataFrame,
    metadata: Any | None,
    column_dict: dict[str, str] | None,
) -> ColumnResolution:
    lowered = text.lower()
    endpoint_candidates = _endpoint_candidates(df)

    explicit = [col for col in endpoint_candidates if _column_explicitly_mentioned(text, col)]
    if len(explicit) == 1:
        return ColumnResolution(explicit[0], "endpoint", "explicit_text", "exact", [])
    if len(explicit) > 1:
        return ColumnResolution(
            None,
            "endpoint",
            "explicit_text",
            "ambiguous",
            [f"Multiple endpoint columns were explicitly mentioned: {', '.join(explicit)}."],
        )

    meta_event_col = _metadata_value(metadata, "event_column")
    if isinstance(meta_event_col, str):
        resolved_event = _resolve_metadata_column(meta_event_col, list(df.columns))
        if resolved_event:
            return ColumnResolution(resolved_event, "endpoint", "metadata", "exact", [f"Endpoint resolved from metadata.event_column={meta_event_col}."])
        if any(token in lowered for token in ("event", "status", "outcome", "endpoint", "mortality", "death", "survival")):
            return ColumnResolution(
                None,
                "endpoint",
                "metadata",
                "unresolved",
                [f"metadata.event_column={meta_event_col} did not match a dataset column."],
            )

    meta_endpoint = _metadata_endpoint(metadata)
    if meta_endpoint:
        resolved_meta = _resolve_metadata_column(meta_endpoint, list(df.columns))
        if resolved_meta:
            return ColumnResolution(resolved_meta, "endpoint", "metadata", "exact", [f"Endpoint resolved from metadata.primary_endpoint={meta_endpoint}."])
        if any(token in lowered for token in ("outcome", "endpoint", "mortality", "death", "survival")):
            return ColumnResolution(
                None,
                "endpoint",
                "metadata",
                "unresolved",
                [f"metadata.primary_endpoint={meta_endpoint} did not match a dataset column."],
            )

    dictionary_matches = _resolve_from_column_dict(text, endpoint_candidates, column_dict)
    if len(dictionary_matches) == 1:
        return ColumnResolution(dictionary_matches[0], "endpoint", "data_dictionary", "unique_alias", [f"Endpoint resolved from data dictionary alias for {dictionary_matches[0]}."])
    if len(dictionary_matches) > 1:
        return ColumnResolution(
            None,
            "endpoint",
            "data_dictionary",
            "ambiguous",
            [f"Data dictionary matched multiple endpoint candidates: {', '.join(dictionary_matches)}."],
        )

    if any(token in lowered for token in ("mortality", "death", "dead", "deceased", "outcome", "endpoint", "survival")):
        if len(endpoint_candidates) == 1:
            return ColumnResolution(endpoint_candidates[0], "endpoint", "heuristic", "unique_alias", [f"Endpoint resolved by unique strong endpoint heuristic: {endpoint_candidates[0]}."])
        if len(endpoint_candidates) > 1:
            return ColumnResolution(
                None,
                "endpoint",
                "heuristic",
                "ambiguous",
                [f"Multiple plausible endpoint columns found: {', '.join(endpoint_candidates)}."],
            )

    return ColumnResolution(None, "endpoint", "heuristic", "unresolved", [])


def _resolve_group_column(text: str, df: pd.DataFrame, metadata: Any | None = None) -> str | None:
    return _resolve_group_column_resolution(text, df, metadata).column


def _resolve_group_column_resolution(text: str, df: pd.DataFrame, metadata: Any | None = None) -> ColumnResolution:
    mentioned = _find_mentioned_columns(text, list(df.columns))
    for col in mentioned:
        if _is_binary_or_categorical(df[col]) and not _looks_like_endpoint(col):
            return ColumnResolution(col, "group", "explicit_text", "exact", [])
    meta_group_col = _metadata_value(metadata, "group_column")
    if isinstance(meta_group_col, str):
        resolved_group = _resolve_metadata_column(meta_group_col, list(df.columns))
        if resolved_group and _is_binary_or_categorical(df[resolved_group]):
            return ColumnResolution(resolved_group, "group", "metadata", "exact", [f"Group column resolved from metadata.group_column={meta_group_col}."])
        return ColumnResolution(
            None,
            "group",
            "metadata",
            "unresolved",
            [f"metadata.group_column={meta_group_col} did not match a categorical dataset column."],
        )
    lowered = text.lower()
    preferred_tokens = ("treatment", "trt", "arm", "group", "therapy")
    if _mentions_treatment_or_group(text):
        for col in df.columns:
            if any(token in _canonical(col) for token in preferred_tokens) and _is_binary_or_categorical(df[col]):
                return ColumnResolution(col, "group", "heuristic", "unique_alias", [f"Group column resolved by treatment/group heuristic: {col}."])
    return ColumnResolution(None, "group", "heuristic", "unresolved", [])


def _resolve_time_column(df: pd.DataFrame, metadata: Any | None) -> str | None:
    return _resolve_time_column_resolution("", df, metadata, None).column


def _resolve_time_column_resolution(
    text: str,
    df: pd.DataFrame,
    metadata: Any | None,
    column_dict: dict[str, str] | None,
) -> ColumnResolution:
    time_candidates = _time_candidates(df)

    explicit = [col for col in time_candidates if _column_explicitly_mentioned(text, col)]
    if len(explicit) == 1:
        return ColumnResolution(explicit[0], "time", "explicit_text", "exact", [])
    if len(explicit) > 1:
        return ColumnResolution(None, "time", "explicit_text", "ambiguous", [f"Multiple time columns were explicitly mentioned: {', '.join(explicit)}."])

    if metadata is not None:
        candidate = _metadata_value(metadata, "time_column") or _metadata_value(metadata, "followup_time_column")
        if isinstance(candidate, str):
            resolved = _resolve_metadata_column(candidate, list(df.columns))
            if resolved:
                return ColumnResolution(resolved, "time", "metadata", "exact", [f"Time column resolved from metadata: {candidate}."])
            return ColumnResolution(None, "time", "metadata", "unresolved", [f"Metadata time column {candidate} did not match a dataset column."])

    dictionary_matches = _resolve_from_column_dict(text, time_candidates, column_dict)
    if len(dictionary_matches) == 1:
        return ColumnResolution(dictionary_matches[0], "time", "data_dictionary", "unique_alias", [f"Time column resolved from data dictionary alias for {dictionary_matches[0]}."])
    if len(dictionary_matches) > 1:
        return ColumnResolution(None, "time", "data_dictionary", "ambiguous", [f"Data dictionary matched multiple time candidates: {', '.join(dictionary_matches)}."])

    if len(time_candidates) == 1:
        return ColumnResolution(time_candidates[0], "time", "heuristic", "unique_alias", [f"Time column resolved by unique time-column heuristic: {time_candidates[0]}."])
    if len(time_candidates) > 1:
        return ColumnResolution(None, "time", "heuristic", "ambiguous", [f"Multiple plausible time columns found: {', '.join(time_candidates)}."])
    return ColumnResolution(None, "time", "heuristic", "unresolved", [])


def _metadata_endpoint(metadata: Any | None) -> str | None:
    value = _metadata_value(metadata, "primary_endpoint")
    return value if isinstance(value, str) else None


def _metadata_value(metadata: Any | None, key: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _resolve_metadata_column(value: str, columns: list[str]) -> str | None:
    if value in columns:
        return value
    canonical_value = _canonical(value)
    matches = [col for col in columns if _canonical(col) == canonical_value]
    return matches[0] if len(matches) == 1 else None


def _endpoint_candidates(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if _looks_like_endpoint(col)]


def _time_candidates(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        canonical = _canonical(col)
        if (canonical in {"time", "follow_up", "followup", "survival_time"} or "time" in canonical) and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def _column_explicitly_mentioned(text: str, column: str) -> bool:
    lowered = f" {text.lower()} "
    explicit_aliases = {column.lower(), _canonical(column), _canonical(column).replace("_", " ")}
    return any(_alias_in_text(alias, lowered) for alias in explicit_aliases)


def _resolve_from_column_dict(
    text: str,
    candidates: list[str],
    column_dict: dict[str, str] | None,
) -> list[str]:
    if not column_dict:
        return []
    text_tokens = _meaningful_tokens(text)
    if not text_tokens:
        return []
    matches: list[str] = []
    for col in candidates:
        description = column_dict.get(col) or column_dict.get(_canonical(col)) or ""
        aliases = _dictionary_aliases(col, description)
        if text_tokens & aliases:
            matches.append(col)
    return matches


def _dictionary_aliases(column: str, description: str) -> set[str]:
    tokens = _meaningful_tokens(f"{column} {description}")
    return {token for token in tokens if token not in {"column", "patient", "value", "values", "indicator", "status"}}


def _meaningful_tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "or", "was", "were", "with", "from", "for", "into", "this", "that",
        "had", "has", "have", "significant", "significantly", "associated", "association",
        "predictor", "predicts", "risk", "groups", "group", "treatment", "trial", "test",
    }
    return {token for token in _WORD_RE.findall(text.lower()) if len(token) >= 4 and token not in stop}


def _find_mentioned_columns(text: str, columns: list[str]) -> list[str]:
    lowered = f" {text.lower()} "
    found: list[str] = []
    for col in sorted(columns, key=len, reverse=True):
        aliases = _column_aliases(col)
        if any(_alias_in_text(alias, lowered) for alias in aliases):
            found.append(col)
    return found


def _column_aliases(column: str) -> set[str]:
    canonical = _canonical(column)
    aliases = {column.lower(), canonical, canonical.replace("_", " ")}
    if canonical in {"trt", "treatment", "treatment_group", "arm", "group"}:
        aliases.update({"treatment", "treatment group", "trial arm", "arm", "group"})
    if canonical == "death_event":
        aliases.update({"mortality", "death", "deaths", "death event"})
    if canonical == "serum_creatinine":
        aliases.update({"serum creatinine", "creatinine"})
    if canonical == "ejection_fraction":
        aliases.add("ejection fraction")
    if canonical == "serum_sodium":
        aliases.update({"serum sodium", "sodium"})
    return {alias for alias in aliases if alias}


def _alias_in_text(alias: str, lowered_text: str) -> bool:
    if not alias:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, lowered_text) is not None


def _looks_like_endpoint(column: str) -> bool:
    canonical = _canonical(column)
    return any(token in canonical for token in ("death", "mortality", "event", "outcome", "status"))


def _is_binary_series(series: pd.Series) -> bool:
    vals = set(pd.to_numeric(series.dropna(), errors="coerce").dropna().unique().tolist())
    return bool(vals) and vals <= {0, 1}


def _is_binary_or_categorical(series: pd.Series) -> bool:
    return bool(series.dropna().nunique() <= 8) or not pd.api.types.is_numeric_dtype(series)


def _two_group_labels(series: pd.Series) -> tuple[str, str] | None:
    labels = sorted(series.dropna().astype(str).unique().tolist())
    return (labels[0], labels[1]) if len(labels) == 2 else None


def _resolve_event_codes_for_survival(series: pd.Series, metadata: Any | None, text: str) -> tuple[list[int] | None, bool]:
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    values = sorted(set(int(v) for v in numeric.unique()))
    if not values or set(values) <= {0, 1}:
        return None, False

    configured = _metadata_event_codes(metadata, text)
    if configured is not None:
        return configured, False

    return None, True


def _metadata_event_codes(metadata: Any | None, text: str) -> list[int] | None:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        value = metadata.get("event_codes") or metadata.get("event_code")
    else:
        value = getattr(metadata, "event_codes", None) or getattr(metadata, "event_code", None)
    if isinstance(value, int):
        return [value]
    if isinstance(value, list | tuple | set):
        codes: list[int] = []
        for item in value:
            try:
                codes.append(int(item))
            except (TypeError, ValueError):
                return None
        return codes or None
    status_mapping = _metadata_value(metadata, "status_mapping")
    if isinstance(status_mapping, dict):
        target_terms = _event_target_terms(text)
        if not target_terms:
            return None
        matched_codes: list[int] = []
        for raw_code, raw_label in status_mapping.items():
            label_tokens = _meaningful_tokens(str(raw_label))
            if target_terms & label_tokens:
                try:
                    matched_codes.append(int(raw_code))
                except (TypeError, ValueError):
                    return None
        return matched_codes if len(matched_codes) == 1 else None
    return None


def _event_target_terms(text: str) -> set[str]:
    lowered = text.lower()
    if any(token in lowered for token in ("death", "mortality", "dead", "deceased")):
        return {"death", "mortality", "dead", "deceased"}
    if "transplant" in lowered:
        return {"transplant", "transplanted"}
    return set()


def _event_codes_for_series(series: pd.Series) -> list[int] | None:
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    values = sorted(set(int(v) for v in numeric.unique()))
    if not values or set(values) <= {0, 1}:
        return None
    if 2 in values:
        return [2]
    return [values[-1]]


def _find_regression_term(result: dict, variable: str, *, key: str) -> dict | None:
    target = _canonical(variable)
    for coef in result.get("coefficients", []):
        term = str(coef.get(key) or "")
        if _canonical(term) == target or _canonical(term).startswith(target):
            return coef
    return None


def _extract_p_value(text: str) -> tuple[str | None, float | None]:
    match = _P_VALUE_RE.search(text)
    if not match:
        return None, None
    op, raw = match.groups()
    op = {"≤": "<=", "≥": ">="}.get(op, op)
    value = _safe_float(raw)
    if value is None:
        return None, None
    return op, value


def _extract_reported_effect(text: str) -> tuple[str | None, float | None]:
    for label, pattern in _EFFECT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = _safe_float(match.group(1))
        if value is not None:
            return label, value
    return None, None


def _extract_reported_ci(text: str) -> tuple[float | None, float | None]:
    match = _CI_RE.search(text)
    if not match:
        return None, None
    left = _safe_float(match.group(1))
    right = _safe_float(match.group(2))
    if left is None or right is None:
        return None, None
    return (min(left, right), max(left, right))


def _infer_claimed_significance(
    text: str,
    reported_p_operator: str | None,
    reported_p_value: float | None,
) -> bool | None:
    lowered = text.lower()
    negative = (
        "not significant",
        "no significant",
        "non-significant",
        "no association",
        "no relationship",
        "did not show",
        "was not associated",
    )
    if any(phrase in lowered for phrase in negative):
        return False
    if reported_p_operator is not None and reported_p_value is not None:
        inferred = _significance_from_reported_p(reported_p_operator, reported_p_value)
        if inferred is not None:
            return inferred
    positive = ("significant", "associated", "association", "predicts", "predictor", "correlated", "differed")
    if any(phrase in lowered for phrase in positive):
        return True
    return None


def _infer_direction(text: str) -> str | None:
    lowered = text.lower()
    if any(term in lowered for term in ("not increased", "not higher", "not elevated", "not positive")):
        return None
    if any(term in lowered for term in ("not decreased", "not lower", "not reduced", "not negative")):
        return None
    if any(term in lowered for term in ("higher", "increased", "elevated", "positive")):
        return "positive"
    if any(term in lowered for term in ("lower", "decreased", "reduced", "negative")):
        return "negative"
    if any(term in lowered for term in ("no association", "no relationship", "not associated")):
        return "none"
    return None


def _is_vague(text: str) -> bool:
    lowered = text.lower()
    _, p_value = _extract_p_value(text)
    return any(term in lowered for term in _VAGUE_CLINICAL_TERMS) and p_value is None


def _mentions_treatment_or_group(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("treatment", "trt", "arm", "group", "therapy"))


def _finding_text(finding: Any) -> str:
    if isinstance(finding, str):
        return _clean_text(finding)
    for attr in ("finding_text", "finding_text_plain", "finding_text_raw", "comparison_claim_text", "source_text"):
        value = getattr(finding, attr, None)
        if isinstance(value, str) and value.strip():
            return _clean_text(value)
    if isinstance(finding, dict):
        for key in ("finding_text", "finding_text_plain", "finding_text_raw", "comparison_claim_text", "source_text"):
            value = finding.get(key)
            if isinstance(value, str) and value.strip():
                return _clean_text(value)
    return ""


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" -•\t")


def _canonical(value: str) -> str:
    words = _WORD_RE.findall(value.lower())
    return "_".join(words)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _significance_from_reported_p(operator: str, value: float, alpha: float = 0.05) -> bool | None:
    if operator == "=":
        return value < alpha
    if operator in {"<", "<="}:
        return True if value <= alpha else None
    if operator in {">", ">="}:
        return False if value >= alpha else None
    return None


def _reported_p_value_matches(candidate: _ClaimCandidate, recomputed_p: float) -> bool:
    if candidate.reported_p_value is None:
        return True
    operator = candidate.reported_p_operator or "="
    reported = candidate.reported_p_value
    if operator == "<":
        return recomputed_p < reported
    if operator == "<=":
        return recomputed_p <= reported
    if operator == ">":
        return recomputed_p > reported
    if operator == ">=":
        return recomputed_p >= reported
    return abs(reported - recomputed_p) <= max(0.001, reported * 0.25)


def _direction_from_effect(effect: float | None) -> str | None:
    if effect is None:
        return None
    if effect > 0:
        return "positive"
    if effect < 0:
        return "negative"
    return "none"


def _direction_from_ratio(ratio: float | None) -> str | None:
    if ratio is None:
        return None
    if ratio > 1:
        return "positive"
    if ratio < 1:
        return "negative"
    return "none"


def _reverse_effect(effect: float | None) -> float | None:
    return -effect if effect is not None else None


def _reverse_direction(direction: str | None) -> str | None:
    if direction == "positive":
        return "negative"
    if direction == "negative":
        return "positive"
    return direction


def _binary_endpoint_direction(df: pd.DataFrame, variable: str, endpoint: str) -> str | None:
    if not _is_binary_series(df[endpoint]) or not _is_binary_or_categorical(df[variable]):
        return None
    groups = _two_group_labels(df[variable])
    if groups is None:
        return None
    sub = df[[variable, endpoint]].dropna().copy()
    sub[endpoint] = pd.to_numeric(sub[endpoint], errors="coerce")
    low = sub.loc[sub[variable].astype(str) == groups[0], endpoint].mean()
    high = sub.loc[sub[variable].astype(str) == groups[1], endpoint].mean()
    if pd.isna(low) or pd.isna(high) or low == high:
        return None
    return "positive" if high > low else "negative"


def _effect_labels_compatible(reported_label: str | None, recomputed_label: str | None) -> bool:
    if reported_label is None or recomputed_label is None:
        return False
    reported = _canonical_effect_label(reported_label)
    recomputed = _canonical_effect_label(recomputed_label)
    return reported == recomputed


def _canonical_effect_label(label: str) -> str:
    normalized = _canonical(label)
    if normalized in {"odds_ratio", "or"}:
        return "odds_ratio"
    if normalized in {"hazard_ratio", "hr"}:
        return "hazard_ratio"
    if normalized in {"correlation_r", "pearson_r", "spearman_r", "kendall_r", "r", "correlation"}:
        return "correlation"
    if normalized in {"cohen_d", "cohens_d", "d"}:
        return "cohen_d"
    if normalized in {"smd", "standardized_mean_difference", "standardised_mean_difference"}:
        return "smd"
    return normalized


def _opposite_effect_side(reported: float, recomputed: float, effect_label: str | None) -> bool:
    canonical = _canonical_effect_label(effect_label or "")
    if canonical in {"odds_ratio", "hazard_ratio"}:
        return (reported - 1.0) * (recomputed - 1.0) < 0
    if canonical in {"correlation", "cohen_d", "smd"}:
        return reported * recomputed < 0
    return False


def _material_effect_difference(reported: float, recomputed: float, effect_label: str | None) -> bool:
    canonical = _canonical_effect_label(effect_label or "")
    if canonical in {"odds_ratio", "hazard_ratio"}:
        if reported <= 0 or recomputed <= 0:
            return False
        ratio = max(reported, recomputed) / min(reported, recomputed)
        return ratio >= 1.5
    if canonical in {"correlation", "cohen_d", "smd"}:
        return abs(reported - recomputed) >= max(0.2, abs(reported) * 0.5)
    return False


def _reported_ci_excludes_null_but_recomputed_crosses(
    reported_lower: float | None,
    reported_upper: float | None,
    recomputed_lower: float | None,
    recomputed_upper: float | None,
    effect_label: str | None,
) -> bool:
    if None in (reported_lower, reported_upper, recomputed_lower, recomputed_upper):
        return False
    null = 1.0 if _canonical_effect_label(effect_label or "") in {"odds_ratio", "hazard_ratio"} else 0.0
    reported_excludes = bool(reported_upper < null or reported_lower > null)
    recomputed_crosses = bool(recomputed_lower <= null <= recomputed_upper)
    return reported_excludes and recomputed_crosses


def _implies_adjusted_model(text: str) -> bool:
    lowered = text.lower()
    phrases = (
        "adjusted",
        "after adjustment",
        "after adjusting",
        "controlling for",
        "controlled for",
        "multivariable",
        "multivariate",
        "independent predictor",
        "independently associated",
    )
    return any(phrase in lowered for phrase in phrases)
