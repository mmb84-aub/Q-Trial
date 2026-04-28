"""
Automated report comparison pipeline.

Parses an uploaded human analyst report, normalizes both human and Q-Trial
findings into a shared representation, performs one-to-one matching, labels
agreement relationships, and computes summary metrics.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from math import sqrt
from dataclasses import dataclass
from typing import Any

from qtrial_backend.agentic.finding_categories import (
    classify_claim_type,
    classify_finding_category,
    is_analytical_category,
    is_comparison_claim_type,
)
from qtrial_backend.agentic.schemas import (
    ComparableFinding,
    ComparisonMetrics,
    ComparisonReport,
    FinalReportSchema,
    FindingMatch,
    HumanReportParseResult,
)
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName

_SUPPORTED_ANALYST_REPORT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".text",
    ".rst",
    ".json",
}

_FINDING_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_P_VALUE_RE = re.compile(r"\bp\s*([=<>])\s*(0?\.\d+|\d+\.\d+|\d+)", re.IGNORECASE)
_EFFECT_PATTERNS: list[tuple[str, str]] = [
    (r"\bHR\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", "hazard_ratio"),
    (r"\bOR\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", "odds_ratio"),
    (r"\bRR\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", "risk_ratio"),
    (r"\bCohen'?s?\s*d\s*[:=]?\s*([-+]?[0-9]+(?:\.[0-9]+)?)", "cohen_d"),
]
_CITATION_MARKERS = [
    re.compile(r"\[\d+(?:,\s*\d+)*\]"),
    re.compile(r"\(\s*(?:19|20)\d{2}\s*\)"),
    re.compile(r"\bPMID\s*:?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bdoi\s*:\s*\S+", re.IGNORECASE),
]
_SECTION_HEADERS = [
    "summary",
    "overview",
    "results",
    "findings",
    "conclusion",
    "recommendations",
    "discussion",
    "survival",
    "baseline",
    "analysis",
]
_ENDPOINT_ALIASES: dict[str, set[str]] = {
    "mortality": {
        "mortality",
        "death",
        "deaths",
        "death_event",
        "death event",
        "overall mortality",
        "death during follow up",
        "death during follow-up",
        "deceased during follow up",
        "deceased during follow-up",
    },
    "survival": {"survival", "overall survival", "os", "survival outcome", "overall outcome"},
    "progression_free_survival": {"progression free survival", "progression-free survival", "pfs"},
}
_VARIABLE_SYNONYMS: dict[str, set[str]] = {
    "age": {"older age", "older patients", "younger age", "younger patients", "patient age"},
    "serum_creatinine": {"serum creatinine", "creatinine"},
    "ejection_fraction": {"ejection fraction"},
    "serum_sodium": {"serum sodium", "sodium"},
    "creatinine_phosphokinase": {"cpk", "creatinine phosphokinase"},
    "death_event": {"death event", "mortality", "death during follow up", "death during follow-up"},
}
_DIRECTION_HINTS: dict[str, tuple[str, ...]] = {
    "positive": ("increase", "increased", "higher", "elevated", "associated with increased", "risk factor"),
    "negative": ("decrease", "decreased", "lower", "reduced", "protective", "associated with reduced"),
    "none": ("no relationship", "no association", "not associated", "unrelated", "no effect"),
}
_TOKEN_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are",
    "have", "has", "had", "into", "than", "then", "over", "under", "between",
    "during", "after", "before", "their", "there", "these", "those", "patients",
    "patient", "group", "groups", "study", "analysis", "trial", "data", "results",
    "finding", "findings", "significant", "statistically", "reported", "observed",
    "followup",
}
_RELATION_CLASSIFIER_SYSTEM = """\
You compare two clinical findings about the same study.

Classify the relationship as exactly one of:
- agree
- partial_agree
- contradict

Rules:
- agree: materially the same conclusion or same directional interpretation
- partial_agree: overlap exists but scope, certainty, endpoint framing, or evidence differs
- contradict: materially opposite claim, opposite significance, or opposite direction

Return JSON only:
{"relation":"agree|partial_agree|contradict","rationale":"..."}
"""


def analyst_report_extension_supported(
    filename: str | None,
    content_type: str | None = None,
) -> bool:
    if filename and _suffix(filename) in _SUPPORTED_ANALYST_REPORT_EXTENSIONS:
        return True
    if content_type:
        lowered = content_type.lower()
        if lowered.startswith("text/") or lowered == "application/json":
            return True
    return False


def parse_human_report_text(
    source_name: str,
    raw_text: str,
    known_variables: set[str] | None = None,
) -> HumanReportParseResult:
    section = "report"
    findings: list[ComparableFinding] = []
    total_candidates = 0

    for candidate in _iter_candidates(raw_text):
        total_candidates += 1
        cleaned = _clean_candidate(candidate)
        if _looks_like_section_header(cleaned):
            section = cleaned.rstrip(":").strip().lower()
            continue
        if not _is_finding_candidate(cleaned):
            continue

        finding = _build_comparable_finding(
            finding_id=f"human_{len(findings) + 1}",
            source="human",
            source_label=source_name,
            finding_text=cleaned,
            section=section,
            evidence_score=1.0 if _has_citation_marker(cleaned) else 0.0,
            evidence_label="cited" if _has_citation_marker(cleaned) else "uncited",
            known_variables=known_variables,
        )
        findings.append(finding)

    return HumanReportParseResult(
        source_name=source_name,
        findings=findings,
        total_candidates=total_candidates,
        discarded_candidates=max(total_candidates - len(findings), 0),
    )


def build_comparison_report(
    final_report: FinalReportSchema,
    analyst_report_name: str,
    analyst_report_text: str,
    provider: ProviderName,
) -> ComparisonReport:
    qtrial_findings = normalize_qtrial_findings(final_report)
    known_variables = {finding.variable for finding in qtrial_findings if finding.variable}
    human_parse = parse_human_report_text(
        analyst_report_name,
        analyst_report_text,
        known_variables=known_variables,
    )
    human_comparison_findings = [
        finding for finding in human_parse.findings if is_comparison_claim_type(finding.claim_type)
    ]
    matches, qtrial_only, human_only = _match_findings(qtrial_findings, human_comparison_findings, provider)
    contradictions = [m for m in matches if m.relation == "contradict"]

    metrics = _build_metrics(
        qtrial_findings=qtrial_findings,
        human_findings=human_comparison_findings,
        matches=matches,
        qtrial_only=qtrial_only,
        human_only=human_only,
    )
    summary = _build_summary(metrics)

    return ComparisonReport(
        analyst_report_name=analyst_report_name,
        summary=summary,
        metrics=metrics,
        matched_findings=matches,
        contradictions=contradictions,
        qtrial_only_findings=qtrial_only,
        human_only_findings=human_only,
        human_report_parse=human_parse,
    )


def normalize_qtrial_findings(final_report: FinalReportSchema) -> list[ComparableFinding]:
    findings: list[ComparableFinding] = []
    seen_texts: set[str] = set()

    grounded = final_report.grounded_findings
    if grounded is not None:
        for idx, finding in enumerate(grounded.findings, start=1):
            finding_category = getattr(finding, "finding_category", "analytical")
            claim_type = getattr(
                finding,
                "claim_type",
                classify_claim_type(
                    finding.finding_text_plain or finding.finding_text or finding.finding_text_raw or "",
                    finding_category=finding_category,
                ),
            )
            if not is_analytical_category(finding_category) or not is_comparison_claim_type(claim_type):
                continue
            evidence_score = 0.0
            evidence_label = "ungrounded"
            if finding.evidence_strength is not None:
                evidence_score = float(finding.evidence_strength.score) / 100.0
                evidence_label = finding.evidence_strength.label.lower()
            elif finding.citations:
                evidence_score = 0.6
                evidence_label = "cited"

            comparable = _build_comparable_finding(
                finding_id=f"qtrial_grounded_{idx}",
                source="qtrial",
                source_label="grounded_findings",
                finding_text=_best_qtrial_text(
                    comparison_claim_text=getattr(finding, "comparison_claim_text", None),
                    plain_text=finding.finding_text_plain,
                    finding_text=finding.finding_text,
                    raw_text=finding.finding_text_raw,
                ),
                section="grounded_findings",
                finding_category=finding_category,
                claim_type=claim_type,
                significance=_significance_from_grounding_status(finding.grounding_status),
                significant=None,
                evidence_score=evidence_score,
                evidence_label=evidence_label,
                citations_present=bool(finding.citations),
                metadata={
                    "grounding_status": finding.grounding_status,
                    "confidence_warning": finding.confidence_warning,
                    "comparison_claim_text": getattr(finding, "comparison_claim_text", None),
                    "finding_text_raw": finding.finding_text_raw,
                    "finding_text_plain": finding.finding_text_plain,
                },
            )
            _add_unique_finding(findings, seen_texts, comparable)

    clinical_analysis = final_report.clinical_analysis or {}
    corrected = (
        clinical_analysis.get("stage_3_corrections", {}) if isinstance(clinical_analysis, dict) else {}
    ).get("corrected_findings", [])
    if isinstance(corrected, list):
        for idx, finding in enumerate(corrected, start=1):
            variable = _canonicalize_variable_name(str(finding.get("finding_id", "")).strip() or None)
            endpoint_hint = (
                _infer_endpoint(str(finding.get("endpoint") or "").strip())
                or _infer_endpoint(str(finding.get("finding_id", "")).strip())
            )
            finding_category = str(
                finding.get("finding_category")
                or classify_finding_category(
                    str(finding.get("finding_text_plain") or finding.get("finding_text_raw") or ""),
                    variable=variable,
                    endpoint=endpoint_hint,
                    analysis_type=str(finding.get("analysis_type") or "association"),
                )
            )
            claim_type = str(
                finding.get("claim_type")
                or classify_claim_type(
                    str(finding.get("finding_text_plain") or finding.get("finding_text_raw") or ""),
                    finding_category=finding_category,
                    variable=variable,
                    endpoint=endpoint_hint,
                    significant=_bool_or_none(finding.get("significant_after_correction")),
                    p_value=_safe_float(finding.get("adjusted_p_value") or finding.get("raw_p_value")),
                )
            )
            if not is_analytical_category(finding_category) or not is_comparison_claim_type(claim_type):
                continue
            if finding_category in {"endpoint_result", "survival_result"} and variable in {"death_event", "mortality", "survival", "survival_primary"}:
                variable = None
            summary_parts = [variable or endpoint_hint or f"finding {idx}"]
            raw_p = finding.get("raw_p_value")
            adj_p = finding.get("adjusted_p_value")
            odds_ratio = _safe_float(finding.get("odds_ratio"))
            if raw_p is not None:
                summary_parts.append(f"raw p={raw_p}")
            if adj_p is not None:
                summary_parts.append(f"adjusted p={adj_p}")
            effect = finding.get("effect_size")
            if effect is not None:
                summary_parts.append(f"effect size={effect}")
            if odds_ratio is not None:
                summary_parts.append(f"OR={odds_ratio}")
            if finding.get("significant_after_correction"):
                summary_parts.append("significant after correction")
            else:
                summary_parts.append("not significant after correction")

            final_p_value = _safe_float(adj_p if adj_p is not None else raw_p)
            raw_summary = "; ".join(summary_parts)
            direction = _infer_direction(
                finding_text=raw_summary,
                effect_size=odds_ratio if odds_ratio is not None else _safe_float(effect),
                effect_size_label="odds_ratio" if odds_ratio is not None else "effect_size",
            )
            plain_text = str(finding.get("finding_text_plain") or "").strip()
            raw_text = str(finding.get("finding_text_raw") or raw_summary).strip()
            comparable = _build_comparable_finding(
                finding_id=f"qtrial_corrected_{idx}",
                source="qtrial",
                source_label="clinical_analysis",
                finding_text=_best_qtrial_text(
                    comparison_claim_text=str(finding.get("comparison_claim_text") or "").strip() or None,
                    plain_text=plain_text or None,
                    finding_text=str(finding.get("finding_text") or "").strip() or None,
                    raw_text=raw_text,
                ),
                section="clinical_analysis",
                finding_category=finding_category,
                claim_type=claim_type,
                variable=variable,
                endpoint=endpoint_hint,
                significance=(
                    "significant" if finding.get("significant_after_correction") else "not_significant"
                ),
                significant=_bool_or_none(finding.get("significant_after_correction")),
                p_value=final_p_value,
                effect_size=odds_ratio if odds_ratio is not None else _safe_float(effect),
                effect_size_label="odds_ratio" if odds_ratio is not None else "effect_size",
                direction=direction,
                evidence_score=1.0 if finding.get("significant_after_correction") else 0.75,
                evidence_label="corrected_statistical_result",
                citations_present=True,
                metadata={
                    "endpoint_type": finding.get("endpoint_type"),
                    "raw_p_value": raw_p,
                    "adjusted_p_value": adj_p,
                    "odds_ratio": odds_ratio,
                    "power_adequate": finding.get("power_adequate"),
                    "comparison_claim_text": str(finding.get("comparison_claim_text") or "").strip() or None,
                    "finding_text_raw": raw_text,
                    "finding_text_plain": plain_text or None,
                },
            )
            _add_unique_finding(findings, seen_texts, comparable)

    return findings


def _match_findings(
    qtrial_findings: list[ComparableFinding],
    human_findings: list[ComparableFinding],
    provider: ProviderName,
) -> tuple[list[FindingMatch], list[ComparableFinding], list[ComparableFinding]]:
    if not qtrial_findings or not human_findings:
        return [], list(qtrial_findings), list(human_findings)

    candidate_pairs: list[_CandidatePair] = []
    for qi, qfinding in enumerate(qtrial_findings):
        for hi, hfinding in enumerate(human_findings):
            score = _candidate_match_score(qfinding, hfinding)
            if score >= 0.28:
                candidate_pairs.append(_CandidatePair(qi=qi, hi=hi, score=score))

    candidate_pairs.sort(key=lambda item: item.score, reverse=True)
    used_q: set[int] = set()
    used_h: set[int] = set()
    matches: list[FindingMatch] = []

    for pair in candidate_pairs:
        if pair.qi in used_q or pair.hi in used_h:
            continue
        qfinding = qtrial_findings[pair.qi]
        hfinding = human_findings[pair.hi]
        relation, rationale = _classify_relation(qfinding, hfinding, provider)
        matched_by = _matched_by_label(qfinding, hfinding)
        matches.append(
            FindingMatch(
                qtrial_finding=qfinding,
                human_finding=hfinding,
                relation=relation,
                match_score=round(pair.score, 4),
                rationale=rationale,
                qtrial_evidence_stronger=qfinding.evidence_score > hfinding.evidence_score,
                matched_by=matched_by,
                variable_detected=bool(qfinding.variable and hfinding.variable),
                endpoint_detected=bool(qfinding.endpoint and hfinding.endpoint),
                text_used_for_matching={
                    "qtrial": qfinding.finding_text,
                    "human": hfinding.finding_text,
                },
            )
        )
        used_q.add(pair.qi)
        used_h.add(pair.hi)

    qtrial_only = [finding for idx, finding in enumerate(qtrial_findings) if idx not in used_q]
    human_only = [finding for idx, finding in enumerate(human_findings) if idx not in used_h]
    return matches, qtrial_only, human_only


def _classify_relation(
    qfinding: ComparableFinding,
    hfinding: ComparableFinding,
    provider: ProviderName,
) -> tuple[str, str]:
    deterministic = _deterministic_relation(qfinding, hfinding)
    if deterministic is not None:
        return deterministic

    try:
        client = get_client(provider)
        req = LLMRequest(
            system_prompt=_RELATION_CLASSIFIER_SYSTEM,
            user_prompt=_build_relation_prompt(qfinding, hfinding),
            payload={"temperature": 0},
        )
        resp = client.generate(req)
        raw = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        relation = str(data.get("relation", "")).strip()
        if relation in {"agree", "partial_agree", "contradict"}:
            return relation, str(data.get("rationale", "")).strip()
    except Exception:
        pass

    overlap = _token_overlap_ratio(qfinding.normalized_text, hfinding.normalized_text)
    if (
        qfinding.significance != "unclear"
        and hfinding.significance != "unclear"
        and qfinding.significance != hfinding.significance
    ):
        return "contradict", _contradiction_rationale(qfinding, hfinding)
    if overlap >= 0.5:
        return "partial_agree", _partial_agreement_rationale(qfinding, hfinding)
    return "partial_agree", _partial_agreement_rationale(qfinding, hfinding)


def _deterministic_relation(
    qfinding: ComparableFinding,
    hfinding: ComparableFinding,
) -> tuple[str, str] | None:
    same_variable = bool(qfinding.variable and hfinding.variable and qfinding.variable == hfinding.variable)
    if qfinding.variable and hfinding.variable and not same_variable:
        return None
    same_endpoint = _endpoints_compatible(qfinding.endpoint, hfinding.endpoint)
    overlap = _token_overlap_ratio(qfinding.normalized_text, hfinding.normalized_text)
    if same_variable:
        if (
            qfinding.significant is not None
            and hfinding.significant is not None
            and qfinding.significant != hfinding.significant
        ):
            return "contradict", _contradiction_rationale(qfinding, hfinding)
        if _bools_match(qfinding.significant, hfinding.significant) and _directions_compatible(
            qfinding.direction,
            hfinding.direction,
        ):
            if same_endpoint:
                return "agree", _agreement_rationale(qfinding, hfinding)
            if qfinding.endpoint is None or hfinding.endpoint is None:
                return "agree", _agreement_rationale(qfinding, hfinding)
        if same_endpoint or qfinding.endpoint is None or hfinding.endpoint is None:
            return "partial_agree", _partial_agreement_rationale(qfinding, hfinding)
        return "partial_agree", _partial_agreement_rationale(qfinding, hfinding)
    return None


def _build_metrics(
    qtrial_findings: list[ComparableFinding],
    human_findings: list[ComparableFinding],
    matches: list[FindingMatch],
    qtrial_only: list[ComparableFinding],
    human_only: list[ComparableFinding],
) -> ComparisonMetrics:
    matched_pairs = len(matches)
    agreement_count = sum(1 for match in matches if match.relation == "agree")
    partial_count = sum(1 for match in matches if match.relation == "partial_agree")
    contradiction_count = sum(1 for match in matches if match.relation == "contradict")
    evidence_upgrades = sum(1 for match in matches if match.qtrial_evidence_stronger)
    mcc_value, mcc_interpretation = _compute_mcc(matches)

    total_qtrial = len(qtrial_findings)
    total_human = len(human_findings)
    return ComparisonMetrics(
        total_qtrial_findings=total_qtrial,
        total_human_findings=total_human,
        matched_pairs=matched_pairs,
        qtrial_only_count=len(qtrial_only),
        human_only_count=len(human_only),
        recall_against_human=_ratio(matched_pairs, total_human),
        novel_rate=_ratio(len(qtrial_only), total_qtrial),
        agreement_count=agreement_count,
        partial_agreement_count=partial_count,
        contradiction_count=contradiction_count,
        agreement_rate_over_matched=_ratio(agreement_count, matched_pairs),
        contradiction_rate_over_matched=_ratio(contradiction_count, matched_pairs),
        evidence_upgrade_rate=_ratio(evidence_upgrades, matched_pairs),
        mcc=mcc_value,
        mcc_interpretation=mcc_interpretation,
    )


def _build_summary(metrics: ComparisonMetrics) -> str:
    summary = (
        f"Matched {metrics.matched_pairs} of {metrics.total_human_findings} human findings. "
        f"Q-Trial surfaced {metrics.qtrial_only_count} additional findings, "
        f"agreed on {metrics.agreement_count}, partially agreed on {metrics.partial_agreement_count}, "
        f"and contradicted {metrics.contradiction_count} matched findings."
    )
    if metrics.mcc is not None and metrics.mcc_interpretation:
        summary += f" MCC on binary-significance matches was {metrics.mcc:.2f} ({metrics.mcc_interpretation})."
    return summary


def _build_relation_prompt(qfinding: ComparableFinding, hfinding: ComparableFinding) -> str:
    return (
        "Q-Trial finding:\n"
        f"- text: {qfinding.finding_text}\n"
        f"- variable: {qfinding.variable or 'unknown'}\n"
        f"- endpoint: {qfinding.endpoint or 'unknown'}\n"
        f"- direction: {qfinding.direction}\n"
        f"- significant: {qfinding.significant}\n"
        f"- significance: {qfinding.significance}\n"
        f"- p_value: {qfinding.p_value}\n"
        f"- evidence: {qfinding.evidence_label} ({qfinding.evidence_score:.2f})\n\n"
        "Human finding:\n"
        f"- text: {hfinding.finding_text}\n"
        f"- variable: {hfinding.variable or 'unknown'}\n"
        f"- endpoint: {hfinding.endpoint or 'unknown'}\n"
        f"- direction: {hfinding.direction}\n"
        f"- significant: {hfinding.significant}\n"
        f"- significance: {hfinding.significance}\n"
        f"- p_value: {hfinding.p_value}\n"
        f"- evidence: {hfinding.evidence_label} ({hfinding.evidence_score:.2f})"
    )


def _candidate_match_score(qfinding: ComparableFinding, hfinding: ComparableFinding) -> float:
    token_overlap = _token_overlap_ratio(qfinding.normalized_text, hfinding.normalized_text)
    same_variable = bool(qfinding.variable and hfinding.variable and qfinding.variable == hfinding.variable)
    if qfinding.variable and hfinding.variable and not same_variable:
        return 0.0
    same_endpoint = _endpoints_compatible(qfinding.endpoint, hfinding.endpoint)
    if token_overlap == 0.0 and not same_variable and not same_endpoint:
        return 0.0
    if token_overlap < 0.08 and not same_variable and not same_endpoint:
        return 0.0

    score = 0.0
    if same_variable:
        score += 0.62
    if same_endpoint:
        score += 0.2 if same_variable else 0.12
    elif qfinding.endpoint and hfinding.endpoint:
        score += 0.05 * _token_overlap_ratio(qfinding.endpoint, hfinding.endpoint)

    if _bools_match(qfinding.significant, hfinding.significant) and qfinding.significant is not None:
        score += 0.08
    elif (
        qfinding.significant is not None
        and hfinding.significant is not None
        and qfinding.significant != hfinding.significant
    ):
        score += 0.04

    if _directions_compatible(qfinding.direction, hfinding.direction):
        score += 0.05

    if qfinding.p_value is not None and hfinding.p_value is not None:
        score += 0.08 if abs(qfinding.p_value - hfinding.p_value) <= 0.02 else 0.03

    if not same_variable:
        score += token_overlap * 0.35
    else:
        score += token_overlap * 0.15

    return min(score, 1.0)


def _build_comparable_finding(
    finding_id: str,
    source: str,
    source_label: str,
    finding_text: str,
    section: str | None,
    finding_category: str | None = None,
    claim_type: str | None = None,
    variable: str | None = None,
    endpoint: str | None = None,
    direction: str = "unknown",
    significant: bool | None = None,
    significance: str = "unclear",
    p_value: float | None = None,
    effect_size: float | None = None,
    effect_size_label: str | None = None,
    evidence_score: float = 0.0,
    evidence_label: str = "",
    citations_present: bool | None = None,
    metadata: dict[str, Any] | None = None,
    known_variables: set[str] | None = None,
) -> ComparableFinding:
    normalized_text = _normalize_text(finding_text)
    variable_value = variable or _infer_variable(finding_text, known_variables=known_variables)
    endpoint_value = endpoint or _infer_endpoint(finding_text)
    parsed_p_value = p_value if p_value is not None else _extract_p_value(finding_text)
    parsed_significant = (
        significant if significant is not None else _infer_significant(finding_text, p_value=parsed_p_value)
    )
    parsed_significance = _significance_label(parsed_significant, significance)
    parsed_effect_size, parsed_effect_label = _extract_effect_size(finding_text)
    final_effect_size = effect_size if effect_size is not None else parsed_effect_size
    final_effect_label = effect_size_label or parsed_effect_label
    parsed_direction = direction if direction != "unknown" else _infer_direction(
        finding_text,
        effect_size=final_effect_size,
        effect_size_label=final_effect_label,
    )
    parsed_claim_type = claim_type or classify_claim_type(
        finding_text,
        finding_category=finding_category,
        variable=variable_value,
        endpoint=endpoint_value,
        significant=parsed_significant,
        p_value=parsed_p_value,
    )
    return ComparableFinding(
        finding_id=finding_id,
        source=source,  # type: ignore[arg-type]
        source_label=source_label,
        finding_text=finding_text.strip(),
        normalized_text=normalized_text,
        section=section,
        finding_category=finding_category,
        claim_type=parsed_claim_type,  # type: ignore[arg-type]
        variable=variable_value,
        endpoint=endpoint_value,
        direction=parsed_direction,  # type: ignore[arg-type]
        significant=parsed_significant,
        significance=parsed_significance,  # type: ignore[arg-type]
        p_value=parsed_p_value,
        effect_size=final_effect_size,
        effect_size_label=final_effect_label,
        evidence_score=max(0.0, min(evidence_score, 1.0)),
        evidence_label=evidence_label,
        citations_present=_has_citation_marker(finding_text) if citations_present is None else citations_present,
        metadata=metadata or {},
    )


def _iter_candidates(raw_text: str) -> list[str]:
    candidates: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _BULLET_PREFIX_RE.match(stripped):
            candidates.append(_BULLET_PREFIX_RE.sub("", stripped))
            continue
        if len(stripped) < 160:
            candidates.append(stripped)
            continue
        candidates.extend(part for part in _FINDING_SPLIT_RE.split(stripped) if part.strip())
    return candidates


def _clean_candidate(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip(" -•\t")


def _is_finding_candidate(text: str) -> bool:
    if len(text) < 24:
        return False
    if len(text.split()) < 4:
        return False
    if _looks_like_section_header(text):
        return True
    keywords = (
        "p=", "p <", "p<", "significant", "not significant", "hazard ratio",
        "odds ratio", "survival", "mortality", "baseline", "improved", "reduced",
        "increased", "decreased", "difference", "associated", "correlated",
        "confidence interval", "ci", "endpoint", "effect size", "probability", "risk",
    )
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords) or _extract_p_value(text) is not None


def _looks_like_section_header(text: str) -> bool:
    cleaned = text.strip().rstrip(":").lower()
    if cleaned in _SECTION_HEADERS:
        return True
    if len(cleaned.split()) <= 4 and cleaned.isalpha():
        return cleaned in _SECTION_HEADERS
    return False


def _normalize_text(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [token for token in lowered.split() if len(token) > 2 and token not in _TOKEN_STOPWORDS]
    return " ".join(tokens)


def _infer_endpoint(text: str) -> str | None:
    lowered = text.lower()
    for canonical, aliases in _ENDPOINT_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return canonical
    canonicalized = _canonicalize_variable_name(text)
    if canonicalized == "death_event":
        return "mortality"
    if canonicalized == "survival":
        return "survival"
    match = re.search(r"\b(?:endpoint|outcome|variable|column)\s+([a-z0-9_.-]+)", lowered)
    if match:
        inferred = _canonicalize_variable_name(match.group(1))
        if inferred == "death_event":
            return "mortality"
        return inferred
    return None


def _infer_significant(text: str, p_value: float | None = None) -> bool | None:
    lowered = text.lower()
    if (
        "not significant" in lowered
        or "did not show a statistically significant association" in lowered
        or "did not show statistically significant association" in lowered
        or "did not show a significant association" in lowered
        or "did not show significant association" in lowered
        or "was not statistically significant" in lowered
        or "were not statistically significant" in lowered
        or "not a statistically significant" in lowered
        or "no significant" in lowered
        or "non-significant" in lowered
        or "no relationship" in lowered
        or "no association" in lowered
    ):
        return False
    if p_value is not None:
        return p_value < 0.05
    if "statistically significant" in lowered or "significant" in lowered:
        return True
    if any(
        phrase in lowered
        for phrase in (
            "associated with",
            "predicts",
            "predictor of",
            "linked to",
            "correlated with",
            "higher probability of",
            "lower probability of",
            "higher risk of",
            "lower risk of",
        )
    ):
        return True
    return None


def _extract_p_value(text: str) -> float | None:
    match = _P_VALUE_RE.search(text)
    if not match:
        return None
    comparator, value = match.groups()
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if comparator == "<" and parsed > 0:
        return parsed - 1e-6
    return parsed


def _extract_effect_size(text: str) -> tuple[float | None, str | None]:
    for pattern, label in _EFFECT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _safe_float(match.group(1)), label
    return None, None


def _has_citation_marker(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CITATION_MARKERS)


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    shared = sum((left_counts & right_counts).values())
    return shared / max(len(left_tokens), len(right_tokens))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _compute_mcc(matches: list[FindingMatch]) -> tuple[float | None, str | None]:
    """
    Compute MCC on matched pairs with explicit binary significance labels.

    Mapping:
    - y_true = human finding significant / not significant
    - y_pred = Q-Trial finding significant / not significant
    - partial_agree pairs are excluded
    - pairs with unclear significance are excluded

    This yields a transparent binary agreement score without forcing partial matches
    into a pseudo-binary label. If the confusion matrix is degenerate, MCC is None.
    """
    tp = tn = fp = fn = 0
    for match in matches:
        if match.relation == "partial_agree":
            continue
        y_pred = match.qtrial_finding.significant
        y_true = match.human_finding.significant
        if y_pred is None or y_true is None:
            continue
        if y_pred and y_true:
            tp += 1
        elif (not y_pred) and (not y_true):
            tn += 1
        elif y_pred and (not y_true):
            fp += 1
        else:
            fn += 1

    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator <= 0:
        return None, None

    value = ((tp * tn) - (fp * fn)) / sqrt(denominator)
    value = round(value, 4)
    return value, _interpret_mcc(value)


def _interpret_mcc(value: float) -> str:
    if value > 0.8:
        return "near-perfect agreement"
    if value >= 0.6:
        return "substantial agreement"
    if value >= 0.4:
        return "moderate agreement"
    return "poor agreement"


def _significance_from_grounding_status(status: str) -> str:
    return "unclear"


def _best_qtrial_text(
    comparison_claim_text: str | None,
    plain_text: str | None,
    finding_text: str | None,
    raw_text: str | None,
) -> str:
    for candidate in (comparison_claim_text, plain_text, finding_text, raw_text):
        if candidate and candidate.strip():
            return candidate.strip()
    return ""


def _infer_variable(text: str, known_variables: set[str] | None = None) -> str | None:
    lowered = text.lower()
    if known_variables:
        alias_map = _build_variable_alias_map(known_variables)
        for alias, canonical in alias_map.items():
            if alias in lowered:
                return canonical

    leading_phrase = re.match(
        r"^\s*([a-z][a-z0-9_ ]{2,40}?)\s+(?:show|shows|showed|predicts|predicted|is|was|were|are|has|have|associated|increased|decreased|reduced)\b",
        lowered,
    )
    if leading_phrase:
        return _canonicalize_variable_name(leading_phrase.group(1))

    match = re.match(r"^\s*([a-z][a-z0-9_ ]{2,40})\s*[:=-]", lowered)
    if match:
        return _canonicalize_variable_name(match.group(1))

    directional = re.search(
        r"\b(?:higher|lower|elevated|reduced|increased|decreased)\s+([a-z][a-z0-9_ ]{2,40}?)\s+(?:is|was|were|are|predicts|predicted|associated|linked)",
        lowered,
    )
    if directional:
        return _canonicalize_variable_name(directional.group(1))

    bare_token = re.search(r"\b([a-z][a-z0-9]+(?:_[a-z0-9]+)+)\b", lowered)
    if bare_token:
        return _canonicalize_variable_name(bare_token.group(1))
    return None


def _build_variable_alias_map(variables: set[str]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for variable in sorted(variables, key=len, reverse=True):
        canonical = _canonicalize_variable_name(variable)
        if not canonical:
            continue
        aliases = {
            canonical,
            canonical.replace("_", " "),
            canonical.replace("_", ""),
        }
        aliases.update(_VARIABLE_SYNONYMS.get(canonical, set()))
        for alias in aliases:
            alias_map[alias.lower()] = canonical
    return alias_map


def _canonicalize_variable_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^a-z0-9_ ]", " ", value.lower())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    if not cleaned:
        return None
    for canonical, aliases in _VARIABLE_SYNONYMS.items():
        normalized_aliases = {alias.replace(" ", "_") for alias in aliases}
        if cleaned == canonical or cleaned in normalized_aliases:
            return canonical
    if cleaned in {"mortality", "death", "deaths"}:
        return "death_event"
    if cleaned in {"survival", "death_event"}:
        return cleaned
    return cleaned


def _infer_direction(
    finding_text: str,
    effect_size: float | None = None,
    effect_size_label: str | None = None,
) -> str:
    lowered = finding_text.lower()
    for direction, phrases in _DIRECTION_HINTS.items():
        if any(phrase in lowered for phrase in phrases):
            return direction
    if effect_size is not None:
        if effect_size_label in {"odds_ratio", "hazard_ratio", "risk_ratio"}:
            if effect_size > 1:
                return "positive"
            if effect_size < 1:
                return "negative"
            return "none"
        if effect_size > 0:
            return "positive"
        if effect_size < 0:
            return "negative"
        return "none"
    return "unknown"


def _significance_label(significant: bool | None, fallback: str) -> str:
    if significant is True:
        return "significant"
    if significant is False:
        return "not_significant"
    return fallback if fallback in {"significant", "not_significant"} else "unclear"


def _matched_by_label(qfinding: ComparableFinding, hfinding: ComparableFinding) -> str:
    same_variable = bool(qfinding.variable and hfinding.variable and qfinding.variable == hfinding.variable)
    same_endpoint = _endpoints_compatible(qfinding.endpoint, hfinding.endpoint)
    if same_variable and same_endpoint:
        return "variable+endpoint"
    if same_variable:
        return "variable"
    return "lexical_fallback"


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _bools_match(left: bool | None, right: bool | None) -> bool:
    return left is not None and right is not None and left == right


def _directions_compatible(left: str, right: str) -> bool:
    if "unknown" in {left, right}:
        return True
    return left == right


def _endpoints_compatible(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    return {left, right} == {"mortality", "survival"}


def _agreement_rationale(qfinding: ComparableFinding, hfinding: ComparableFinding) -> str:
    variable = _display_variable(qfinding, hfinding)
    endpoint = _display_endpoint(qfinding, hfinding)
    if endpoint:
        return f"Both reports identify {variable} as significantly associated with {endpoint}."
    return f"Both reports identify {variable} as significantly associated with the outcome."


def _contradiction_rationale(qfinding: ComparableFinding, hfinding: ComparableFinding) -> str:
    variable = _display_variable(qfinding, hfinding)
    endpoint = _display_endpoint(qfinding, hfinding) or "the outcome"
    if hfinding.significant is True and qfinding.significant is False:
        return f"The human report states that {variable} predicts {endpoint}, but Q-Trial did not identify {variable} as a significant predictor."
    if hfinding.significant is False and qfinding.significant is True:
        return f"The human report says {variable} was not significantly associated with {endpoint}, but Q-Trial identified a significant association."
    return f"Both reports discuss {variable}, but they disagree on whether it is significantly associated with {endpoint}."


def _partial_agreement_rationale(qfinding: ComparableFinding, hfinding: ComparableFinding) -> str:
    variable = _display_variable(qfinding, hfinding)
    q_endpoint = _display_endpoint(qfinding, None)
    h_endpoint = _display_endpoint(None, hfinding)
    if q_endpoint and h_endpoint and q_endpoint != h_endpoint:
        return f"Both reports discuss {variable}, but they link it to different endpoints."
    if q_endpoint or h_endpoint:
        endpoint = q_endpoint or h_endpoint
        return f"Both reports discuss {variable}, but only one report explicitly links it to {endpoint}."
    return f"Both reports discuss {variable}, but the endpoint or significance framing is incomplete."


def _display_variable(qfinding: ComparableFinding | None, hfinding: ComparableFinding | None) -> str:
    variable = (
        (qfinding.variable if qfinding and qfinding.variable else None)
        or (hfinding.variable if hfinding and hfinding.variable else None)
        or "this finding"
    )
    return variable.replace("_", " ")


def _display_endpoint(qfinding: ComparableFinding | None, hfinding: ComparableFinding | None) -> str | None:
    endpoint = (
        (qfinding.endpoint if qfinding and qfinding.endpoint else None)
        or (hfinding.endpoint if hfinding and hfinding.endpoint else None)
    )
    if not endpoint:
        return None
    return endpoint.replace("_", " ")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _suffix(filename: str) -> str:
    return "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def _add_unique_finding(
    findings: list[ComparableFinding],
    seen_texts: set[str],
    finding: ComparableFinding,
) -> None:
    key = finding.normalized_text or finding.finding_text.lower()
    if key in seen_texts:
        return
    seen_texts.add(key)
    findings.append(finding)


@dataclass(frozen=True)
class _CandidatePair:
    qi: int
    hi: int
    score: float
