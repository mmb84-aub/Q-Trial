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
from math import log, log10
from dataclasses import dataclass
from typing import Any

from qtrial_backend.agentic.finding_categories import (
    classify_claim_type,
    classify_finding_category,
    is_endpoint_like_variable,
    is_analytical_category,
    is_comparison_claim_type,
    is_followup_time_variable,
    is_raw_stat_artifact_finding,
    is_raw_statistical_artifact_text,
    is_user_facing_clinical_finding_eligible,
    is_user_facing_nonfinding_artifact,
)
from qtrial_backend.agentic.schemas import (
    ComparableFinding,
    ComparisonMetrics,
    ComparisonReport,
    FinalReportSchema,
    FindingMatch,
    HumanReportParseResult,
    StatisticalEvidence,
    StatisticalEvidenceComparison,
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
_P_VALUE_RE = re.compile(
    r"\bp(?:\s*[- ]?\s*value)?\s*(<=|>=|≤|≥|=|<|>)\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)
_NUM_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
_EFFECT_PATTERNS: list[tuple[str, str]] = [
    (rf"\b(?:HR|hazard ratio)\s*[:=]?\s*{_NUM_RE}", "hazard_ratio"),
    (rf"\b(?:OR|odds ratio)\s*[:=]?\s*{_NUM_RE}", "odds_ratio"),
    (rf"\b(?:RR|risk ratio)\s*[:=]?\s*{_NUM_RE}", "risk_ratio"),
    (rf"\bCram[ée]r'?s?\s*V\s*[:=]?\s*{_NUM_RE}", "cramers_v"),
    (rf"\bCohen'?s?\s*d\s*[:=]?\s*{_NUM_RE}", "cohen_d"),
    (rf"\b(?:Spearman(?:'s)?\s*(?:rho|ρ)|rho|ρ)\s*[:=]?\s*{_NUM_RE}", "spearman_r"),
    (rf"\b(?:Pearson(?:'s)?\s*r|correlation(?: coefficient)?|r)\s*[:=]\s*{_NUM_RE}", "correlation_r"),
    (rf"\bAUC\s*[:=]?\s*{_NUM_RE}", "auc"),
    (rf"\bSMD\s*[:=]?\s*{_NUM_RE}", "smd"),
    (rf"\bmean difference\s*[:=]?\s*{_NUM_RE}", "mean_difference"),
]
_CI_RE = re.compile(
    rf"(?:95\s*%\s*)?(?:ci|confidence interval)\s*[:=]?\s*[\[(]?\s*{_NUM_RE}\s*(?:,|to|[-–—])\s*{_NUM_RE}",
    re.IGNORECASE,
)
_STATISTIC_PATTERNS: list[tuple[str, str]] = [
    (rf"\b(?:chi[- ]?square|χ²|chi²|x2)\s*[:=]?\s*{_NUM_RE}", "chi_square"),
    (rf"\bt\s*[:=]\s*{_NUM_RE}", "t_statistic"),
    (rf"\bz\s*[:=]\s*{_NUM_RE}", "z_statistic"),
    (rf"\bF\s*[:=]\s*{_NUM_RE}", "f_statistic"),
]
_SAMPLE_SIZE_RE = re.compile(r"\b(?:n|sample size)\s*[:=]\s*(\d+)\b", re.IGNORECASE)
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
    "clinical analysis",
    "clinical analysis report",
    "analytical findings",
    "statistical notes",
    "data quality notes",
]
_GENERIC_VARIABLE_SUBJECT_RE = re.compile(
    r"^\s*(?:these|those|this|that|the)\s+"
    r"(?:variables?|factors?|predictors?|features?|covariates?)\s+"
    r"(?:was|were|is|are|did|does|had|has|showed|shows|predicted|predicts|"
    r"associated|correlated|increased|decreased|remained)\b",
    re.IGNORECASE,
)
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
    "serum_creatinine": {
        "serum creatinine",
        "serum creatinine level",
        "serum creatinine levels",
        "creatinine",
        "creatinine level",
        "creatinine levels",
    },
    "ejection_fraction": {"ejection fraction"},
    "serum_sodium": {"serum sodium", "serum sodium level", "serum sodium levels", "sodium"},
    "creatinine_phosphokinase": {"cpk", "creatinine phosphokinase"},
    "platelets": {"platelet", "platelets", "platelet level", "platelet levels"},
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
_GENERIC_VARIABLE_ARTIFACTS = {
    "these_variables",
    "those_variables",
    "these_factors",
    "those_factors",
    "this_factor",
    "that_factor",
    "this_variable",
    "that_variable",
    "this_predictor",
    "that_predictor",
    "these_predictors",
    "those_predictors",
    "the_variable",
    "the_factor",
    "the_predictor",
    "variables",
    "factors",
    "predictors",
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
        if is_user_facing_nonfinding_artifact(cleaned):
            continue
        if not is_user_facing_clinical_finding_eligible(cleaned):
            continue
        if _is_generic_variable_subject_claim(cleaned):
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
    qtrial_findings = [
        finding for finding in normalize_qtrial_findings(final_report)
        if not is_user_facing_nonfinding_artifact(finding)
        and is_user_facing_clinical_finding_eligible(finding)
    ]
    known_variables = {finding.variable for finding in qtrial_findings if finding.variable}
    human_parse = parse_human_report_text(
        analyst_report_name,
        analyst_report_text,
        known_variables=known_variables,
    )
    human_comparison_findings = [
        finding for finding in human_parse.findings
        if is_comparison_claim_type(finding.claim_type)
        and not is_user_facing_nonfinding_artifact(finding)
        and is_user_facing_clinical_finding_eligible(finding)
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
            source_text = _best_qtrial_text(
                comparison_claim_text=getattr(finding, "comparison_claim_text", None),
                plain_text=finding.finding_text_plain,
                finding_text=finding.finding_text,
                raw_text=finding.finding_text_raw,
            )
            if is_user_facing_nonfinding_artifact(finding) or is_raw_statistical_artifact_text(source_text):
                continue
            finding_category = getattr(finding, "finding_category", "analytical")
            inferred_category = classify_finding_category(
                source_text,
                variable=getattr(finding, "variable", None),
                endpoint=getattr(finding, "endpoint", None),
            )
            if is_analytical_category(finding_category) and not is_analytical_category(inferred_category):
                finding_category = inferred_category
            claim_type = getattr(
                finding,
                "claim_type",
                classify_claim_type(
                    source_text,
                    finding_category=finding_category,
                ),
            )
            if not is_analytical_category(finding_category) or not is_comparison_claim_type(claim_type):
                continue
            if not is_user_facing_clinical_finding_eligible(finding):
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
                finding_text=source_text,
                section="grounded_findings",
                finding_category=finding_category,
                claim_type=claim_type,
                variable=getattr(finding, "variable", None),
                endpoint=getattr(finding, "endpoint", None),
                direction=getattr(finding, "direction", "unknown") or "unknown",
                significance=getattr(finding, "significance", None) or _significance_from_grounding_status(finding.grounding_status),
                significant=getattr(finding, "significant", None),
                p_value=getattr(finding, "p_value", None),
                effect_size=getattr(finding, "effect_size", None),
                effect_size_label=getattr(finding, "effect_size_label", None),
                evidence_score=evidence_score,
                evidence_label=evidence_label,
                citations_present=bool(finding.citations),
                metadata={
                    "grounding_status": finding.grounding_status,
                    "confidence_warning": finding.confidence_warning,
                    "comparison_claim_text": getattr(finding, "comparison_claim_text", None),
                    "finding_text_raw": finding.finding_text_raw,
                    "finding_text_plain": finding.finding_text_plain,
                    "test_type": getattr(finding, "test_type", None),
                    **(getattr(finding, "metadata", {}) or {}),
                },
            )
            _add_unique_finding(findings, seen_texts, comparable)

    clinical_analysis = final_report.clinical_analysis or {}
    corrected = (
        clinical_analysis.get("stage_3_corrections", {}) if isinstance(clinical_analysis, dict) else {}
    ).get("corrected_findings", [])
    if isinstance(corrected, list):
        for idx, finding in enumerate(corrected, start=1):
            candidate_text = str(
                finding.get("comparison_claim_text")
                or finding.get("finding_text_plain")
                or finding.get("finding_text_raw")
                or finding.get("finding_text")
                or ""
            )
            structured_payload = _has_structured_association_payload(finding)
            if (
                (is_user_facing_nonfinding_artifact(finding) or is_raw_statistical_artifact_text(candidate_text))
                and not structured_payload
            ):
                continue
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
            if structured_payload and finding_category == "artifact_excluded":
                finding_category = "analytical"
            inferred_category = classify_finding_category(
                candidate_text,
                variable=variable,
                endpoint=endpoint_hint,
                analysis_type=str(finding.get("analysis_type") or "association"),
            )
            if is_analytical_category(finding_category) and not is_analytical_category(inferred_category):
                finding_category = "analytical" if structured_payload and inferred_category == "artifact_excluded" else inferred_category
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
            if not is_analytical_category(finding_category):
                claim_type = classify_claim_type(candidate_text, finding_category=finding_category)
            if not is_analytical_category(finding_category) or not is_comparison_claim_type(claim_type):
                continue
            if variable and (is_endpoint_like_variable(variable, endpoint_hint) or is_followup_time_variable(variable)):
                continue
            if finding_category in {"endpoint_result", "survival_result"} and variable in {"death_event", "mortality", "survival", "survival_primary"}:
                variable = None
            summary_parts = [variable or endpoint_hint or f"finding {idx}"]
            raw_p = finding.get("raw_p_value")
            adj_p = finding.get("adjusted_p_value")
            odds_ratio = _safe_float(finding.get("odds_ratio"))
            provided_effect_label = str(finding.get("effect_size_label") or "").strip() or None
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
            structured_sentence = _structured_finding_sentence(
                variable=variable,
                endpoint=endpoint_hint,
                significant=_bool_or_none(finding.get("significant_after_correction")),
            )
            direction = _infer_direction(
                finding_text=raw_summary,
                effect_size=odds_ratio if odds_ratio is not None else _safe_float(effect),
                effect_size_label="odds_ratio" if odds_ratio is not None else provided_effect_label or "effect_size",
                variable=variable,
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
                    raw_text=structured_sentence or raw_text,
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
                effect_size_label="odds_ratio" if odds_ratio is not None else provided_effect_label or "effect_size",
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
                    "test_type": finding.get("test_type"),
                    **(finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {}),
                },
            )
            if is_user_facing_nonfinding_artifact(comparable):
                continue
            if not is_user_facing_clinical_finding_eligible(comparable):
                continue
            _add_unique_finding(findings, seen_texts, comparable)

    return findings


def _match_findings(
    qtrial_findings: list[ComparableFinding],
    human_findings: list[ComparableFinding],
    provider: ProviderName,
) -> tuple[list[FindingMatch], list[ComparableFinding], list[ComparableFinding]]:
    qtrial_findings = [
        finding for finding in qtrial_findings
        if not is_user_facing_nonfinding_artifact(finding)
        and is_user_facing_clinical_finding_eligible(finding)
    ]
    human_findings = [
        finding for finding in human_findings
        if not is_user_facing_nonfinding_artifact(finding)
        and is_user_facing_clinical_finding_eligible(finding)
    ]
    if not qtrial_findings or not human_findings:
        return [], list(qtrial_findings), list(human_findings)

    candidate_pairs: list[_CandidatePair] = []
    for qi, qfinding in enumerate(qtrial_findings):
        for hi, hfinding in enumerate(human_findings):
            pairing_confidence = _candidate_match_score(qfinding, hfinding)
            if pairing_confidence >= 0.28:
                candidate_pairs.append(_CandidatePair(qi=qi, hi=hi, pairing_confidence=pairing_confidence))

    candidate_pairs.sort(key=lambda item: item.pairing_confidence, reverse=True)
    used_q: set[int] = set()
    used_h: set[int] = set()
    matches: list[FindingMatch] = []

    for pair in candidate_pairs:
        if pair.qi in used_q or pair.hi in used_h:
            continue
        qfinding = qtrial_findings[pair.qi]
        hfinding = human_findings[pair.hi]
        relation, rationale = _classify_relation(qfinding, hfinding, provider)
        statistical_comparison = _compare_statistical_evidence(qfinding, hfinding)
        relation, rationale = _apply_statistical_comparison_to_relation(
            relation,
            rationale,
            statistical_comparison,
        )
        matched_by = _matched_by_label(qfinding, hfinding)
        matches.append(
            FindingMatch(
                qtrial_finding=qfinding,
                human_finding=hfinding,
                relation=relation,
                pairing_confidence=round(pair.pairing_confidence, 4),
                match_score=round(pair.pairing_confidence, 4),
                rationale=rationale,
                qtrial_evidence_stronger=qfinding.evidence_score > hfinding.evidence_score,
                matched_by=matched_by,
                variable_detected=bool(qfinding.variable and hfinding.variable),
                endpoint_detected=bool(qfinding.endpoint and hfinding.endpoint),
                text_used_for_matching={
                    "qtrial": qfinding.finding_text,
                    "human": hfinding.finding_text,
                },
                statistical_comparison=statistical_comparison,
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
        if (
            qfinding.significant is True
            and hfinding.significant is True
            and _finding_direction_effect(qfinding) in {"increases_endpoint_risk", "decreases_endpoint_risk"}
            and _finding_direction_effect(hfinding) in {"increases_endpoint_risk", "decreases_endpoint_risk"}
            and _finding_direction_effect(qfinding) != _finding_direction_effect(hfinding)
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


def _finding_direction_effect(finding: ComparableFinding) -> str:
    evidence = finding.statistical_evidence
    if evidence and evidence.direction_effect_on_endpoint != "unknown":
        return evidence.direction_effect_on_endpoint
    if finding.direction == "positive":
        return "increases_endpoint_risk"
    if finding.direction == "negative":
        return "decreases_endpoint_risk"
    if finding.direction == "none":
        return "no_direction"
    return "unknown"


def _compare_statistical_evidence(
    qfinding: ComparableFinding,
    hfinding: ComparableFinding,
) -> StatisticalEvidenceComparison:
    # Statistical agreement is a transparent composite for paired findings, not
    # a universal published metric. It follows evidence-comparison principles:
    # prioritize effect sizes over p-values, compare ratio effects on the log
    # scale, use CI/null-side behavior when available, and let missing evidence
    # reduce coverage instead of pretending to agree.
    qev = qfinding.statistical_evidence
    hev = hfinding.statistical_evidence
    notes: list[str] = []
    warnings: list[str] = []
    if qev is None or hev is None:
        return StatisticalEvidenceComparison(
            available=False,
            reason_if_unavailable="statistical evidence was not extracted for both findings.",
            agreement_label="not_assessed",
            qtrial_evidence=qev,
            human_evidence=hev,
            notes=notes,
            warnings=warnings,
        )

    human_has_quant = _has_quantitative_evidence(hev)
    qtrial_has_quant = _has_quantitative_evidence(qev)
    if not human_has_quant:
        return StatisticalEvidenceComparison(
            available=False,
            reason_if_unavailable="insufficient reported statistics: human report did not provide numeric p-values, effect sizes, confidence intervals, test statistics, or ranks.",
            agreement_label="not_assessed",
            significance_agreement=_component_significance_agreement(qev, hev)[0],
            direction_agreement=_component_direction_agreement(qev, hev)[0],
            statistical_agreement_coverage=0.0,
            coverage_score=0.0,
            qtrial_evidence=qev,
            human_evidence=hev,
            notes=notes,
            warnings=warnings,
        )
    if not qtrial_has_quant:
        return StatisticalEvidenceComparison(
            available=False,
            reason_if_unavailable="insufficient reported statistics: Q-Trial finding did not provide numeric statistical evidence.",
            agreement_label="not_assessed",
            significance_agreement=_component_significance_agreement(qev, hev)[0],
            direction_agreement=_component_direction_agreement(qev, hev)[0],
            statistical_agreement_coverage=0.0,
            coverage_score=0.0,
            qtrial_evidence=qev,
            human_evidence=hev,
            notes=notes,
            warnings=warnings,
        )

    components: list[tuple[str, float, float]] = []

    sig_label, sig_score, sig_available = _component_significance_agreement(qev, hev)
    if sig_available:
        components.append(("significance", sig_score, 0.25))

    dir_label, dir_score, dir_available = _component_direction_agreement(qev, hev)
    if dir_available:
        components.append(("direction", dir_score, 0.20))

    effect_label, effect_score, effect_available, effect_delta, effect_relative_delta, effect_note = _component_effect_agreement(qev, hev)
    if effect_available:
        components.append(("effect_size", effect_score, 0.25))
    if effect_note:
        notes.append(effect_note)

    p_label, p_score, p_available, p_delta, p_log_delta, p_note = _component_p_value_agreement(qev, hev)
    if p_available:
        components.append(("p_value", p_score, 0.10))
    if p_note:
        notes.append(p_note)

    ci_label, ci_score, ci_available, ci_overlap, ci_note = _component_ci_agreement(qev, hev)
    if ci_available:
        components.append(("ci", ci_score, 0.15))
    if ci_note:
        notes.append(ci_note)

    test_label, test_score, test_available = _component_test_type_agreement(qev, hev)
    if test_available:
        components.append(("test_type", test_score, 0.05))

    rank_label, rank_score, rank_available = _component_rank_agreement(qev, hev)
    possible_weight = 1.05 if rank_available else 1.0
    if rank_available:
        components.append(("rank", rank_score, 0.05))

    weight = sum(item[2] for item in components)
    score = round(sum(score * item_weight for _, score, item_weight in components) / weight, 4) if weight else None
    coverage = round(weight / possible_weight, 4)
    contradiction = any(
        label in {"conflict", "opposite"}
        for label in (sig_label, dir_label, effect_label, p_label)
    )
    agreement_label = _statistical_agreement_label(score, contradiction)
    if score is None:
        notes.append("No paired statistical components were comparable after extraction.")
    return StatisticalEvidenceComparison(
        available=score is not None,
        reason_if_unavailable=None if score is not None else "no paired statistical components were comparable.",
        statistical_agreement_score=score,
        statistical_agreement_coverage=coverage,
        overall_statistical_agreement_score=score,
        agreement_label=agreement_label,  # type: ignore[arg-type]
        significance_agreement=sig_label,
        direction_agreement=dir_label,
        effect_size_agreement=effect_label,
        p_value_agreement=p_label,
        ci_agreement=ci_label,
        test_type_agreement=test_label,
        rank_agreement=rank_label,
        effect_size_delta=effect_delta,
        effect_size_relative_delta=effect_relative_delta,
        p_value_delta=p_delta,
        p_value_log_delta=p_log_delta,
        ci_overlap=ci_overlap,
        coverage_score=coverage,
        qtrial_evidence=qev,
        human_evidence=hev,
        notes=notes,
        warnings=warnings,
    )


def _apply_statistical_comparison_to_relation(
    relation: str,
    rationale: str,
    comparison: StatisticalEvidenceComparison,
) -> tuple[str, str]:
    if not comparison.available:
        return relation, _append_stat_note(rationale, comparison.reason_if_unavailable)
    if comparison.agreement_label == "contradiction":
        return "contradict", _append_stat_note(rationale, "reported statistical evidence contradicts on significance, direction, or effect side of null.")
    if relation == "agree" and comparison.agreement_label in {"weak", "moderate"}:
        if comparison.effect_size_agreement in {"far", "opposite"} or comparison.p_value_agreement == "different_strength":
            return "partial_agree", _append_stat_note(rationale, "semantic conclusion matches, but reported statistical evidence differs materially.")
    return relation, _append_stat_note(rationale, _summarize_statistical_comparison(comparison))


def _append_stat_note(rationale: str, note: str | None) -> str:
    if not note:
        return rationale
    if not rationale:
        return f"Statistical comparison: {note}"
    return f"{rationale} Statistical comparison: {note}"


def _summarize_statistical_comparison(comparison: StatisticalEvidenceComparison) -> str:
    if comparison.overall_statistical_agreement_score is None:
        return comparison.reason_if_unavailable or "not assessed."
    return (
        f"{comparison.agreement_label} statistical evidence agreement "
        f"(score={comparison.overall_statistical_agreement_score:.2f}, "
        f"coverage={comparison.coverage_score:.2f})."
    )


def _has_quantitative_evidence(evidence: StatisticalEvidence) -> bool:
    return any(
        value is not None
        for value in (
            evidence.p_value,
            evidence.raw_p_value,
            evidence.adjusted_p_value,
            evidence.effect_size,
            evidence.ci_lower,
            evidence.ci_upper,
            evidence.statistic_value,
            evidence.rank,
            evidence.importance_score,
        )
    )


def _component_significance_agreement(qev: StatisticalEvidence, hev: StatisticalEvidence) -> tuple[str, float, bool]:
    q_sig = qev.significant if qev.significant is not None else _significant_from_evidence_p(qev)
    h_sig = hev.significant if hev.significant is not None else _significant_from_evidence_p(hev)
    if q_sig is None or h_sig is None:
        return "not_assessed", 0.0, False
    return ("agree", 1.0, True) if q_sig == h_sig else ("conflict", 0.0, True)


def _component_direction_agreement(qev: StatisticalEvidence, hev: StatisticalEvidence) -> tuple[str, float, bool]:
    q_dir = qev.direction_effect_on_endpoint
    h_dir = hev.direction_effect_on_endpoint
    if q_dir == "unknown" or h_dir == "unknown":
        return "not_assessed", 0.0, False
    if "no_direction" in {q_dir, h_dir}:
        return ("agree", 1.0, True) if q_dir == h_dir else ("partial", 0.5, True)
    if q_dir == h_dir:
        return "agree", 1.0, True
    q_sig = qev.significant if qev.significant is not None else _significant_from_evidence_p(qev)
    h_sig = hev.significant if hev.significant is not None else _significant_from_evidence_p(hev)
    if q_sig is True and h_sig is True:
        return "conflict", 0.0, True
    return "partial", 0.4, True


def _component_effect_agreement(
    qev: StatisticalEvidence,
    hev: StatisticalEvidence,
) -> tuple[str, float, bool, float | None, float | None, str | None]:
    q = qev.effect_size
    h = hev.effect_size
    if q is None or h is None:
        return "not_assessed", 0.0, False, None, None, None
    q_label = _canonical_effect_label(qev.effect_size_label)
    h_label = _canonical_effect_label(hev.effect_size_label)
    if q_label is None or h_label is None:
        return "not_comparable", 0.0, False, None, None, "Effect sizes were present but effect labels were missing."
    if not _effect_labels_compatible(q_label, h_label):
        return "not_comparable", 0.0, False, None, None, f"Effect labels are not comparable ({q_label} vs {h_label})."
    delta = abs(q - h)
    relative_delta = delta / max(abs(h), 1e-12)
    if _opposite_effect_side(q, h, q_label):
        return "opposite", 0.0, True, delta, relative_delta, "Effect sizes are on opposite sides of the null."
    if q_label in {"odds_ratio", "hazard_ratio", "risk_ratio"}:
        if q <= 0 or h <= 0:
            return "not_comparable", 0.0, False, delta, relative_delta, "Ratio effect sizes must be positive for log-scale comparison."
        log_delta = abs(log(q) - log(h))
        if log_delta <= 0.25:
            return "close", 1.0, True, delta, relative_delta, None
        if log_delta <= 0.70:
            return "moderate", 0.65, True, delta, relative_delta, "Ratio effect sizes differ moderately on the log scale."
        return "far", 0.2, True, delta, relative_delta, "Ratio effect sizes differ materially on the log scale."
    if q_label in {"correlation", "cohen_d", "smd", "cramers_v", "mean_difference", "auc"}:
        if delta <= 0.10:
            return "close", 1.0, True, delta, relative_delta, None
        if delta <= 0.30:
            return "moderate", 0.65, True, delta, relative_delta, "Effect sizes differ moderately."
        return "far", 0.2, True, delta, relative_delta, "Effect sizes differ materially."
    return "not_comparable", 0.0, False, delta, relative_delta, f"Unsupported effect label for paired comparison: {q_label}."


def _component_p_value_agreement(
    qev: StatisticalEvidence,
    hev: StatisticalEvidence,
) -> tuple[str, float, bool, float | None, float | None, str | None]:
    q = _evidence_p_value(qev)
    h = _evidence_p_value(hev)
    if q is None or h is None:
        return "not_assessed", 0.0, False, None, None, None
    q_sig = _significant_from_evidence_p(qev)
    h_sig = _significant_from_evidence_p(hev)
    delta = abs(q - h)
    bounded_q = max(q, 1e-300)
    bounded_h = max(h, 1e-300)
    log_delta = abs(log10(bounded_q) - log10(bounded_h))
    note = None
    if qev.p_operator in {"<", "<="} or hev.p_operator in {"<", "<="}:
        note = "At least one p-value was reported as a bound; log-scale closeness treats the bound as the reported threshold."
    if q_sig is not None and h_sig is not None and q_sig != h_sig:
        return "conflict", 0.0, True, delta, log_delta, note or "P-values fall on opposite sides of the significance threshold."
    if log_delta <= 0.5:
        return "close", 1.0, True, delta, log_delta, note
    if log_delta <= 1.0:
        return "moderate", 0.7, True, delta, log_delta, note or "P-values have the same significance side but differ moderately in strength."
    return "different_strength", 0.4, True, delta, log_delta, note or "P-values have the same significance side but materially different evidence strength."


def _component_ci_agreement(
    qev: StatisticalEvidence,
    hev: StatisticalEvidence,
) -> tuple[str, float, bool, bool | None, str | None]:
    if None in (qev.ci_lower, qev.ci_upper, hev.ci_lower, hev.ci_upper):
        return "not_assessed", 0.0, False, None, None
    q_low = float(qev.ci_lower)  # type: ignore[arg-type]
    q_high = float(qev.ci_upper)  # type: ignore[arg-type]
    h_low = float(hev.ci_lower)  # type: ignore[arg-type]
    h_high = float(hev.ci_upper)  # type: ignore[arg-type]
    overlap = max(q_low, h_low) <= min(q_high, h_high)
    null = _null_value_for_effect(qev.effect_size_label or hev.effect_size_label)
    q_excludes = q_high < null or q_low > null
    h_excludes = h_high < null or h_low > null
    if overlap and q_excludes == h_excludes:
        return "agree", 1.0, True, overlap, None
    if overlap:
        return "partial", 0.65, True, overlap, "Confidence intervals overlap, but null-exclusion differs."
    if q_excludes != h_excludes:
        return "partial", 0.35, True, overlap, "Confidence intervals do not overlap and differ on null exclusion."
    return "weak", 0.45, True, overlap, "Confidence intervals do not overlap."


def _component_test_type_agreement(qev: StatisticalEvidence, hev: StatisticalEvidence) -> tuple[str, float, bool]:
    if not qev.test_family and not hev.test_family and not qev.test_type and not hev.test_type:
        return "not_assessed", 0.0, False
    if qev.test_family and hev.test_family:
        if qev.test_family == hev.test_family:
            return "agree", 1.0, True
        if _test_families_compatible(qev.test_family, hev.test_family):
            return "compatible", 0.7, True
        return "not_comparable", 0.3, True
    return "partial", 0.5, True


def _component_rank_agreement(qev: StatisticalEvidence, hev: StatisticalEvidence) -> tuple[str, float, bool]:
    if qev.rank is None or hev.rank is None:
        return "not_assessed", 0.0, False
    delta = abs(qev.rank - hev.rank)
    if delta == 0:
        return "agree", 1.0, True
    if delta == 1:
        return "partial", 0.6, True
    return "weak", 0.25, True


def _statistical_agreement_label(score: float | None, contradiction: bool) -> str:
    if contradiction:
        return "contradiction"
    if score is None:
        return "not_assessed"
    if score >= 0.85:
        return "strong"
    if score >= 0.60:
        return "moderate"
    return "weak"


def _evidence_p_value(evidence: StatisticalEvidence) -> float | None:
    return evidence.p_value if evidence.p_value is not None else evidence.adjusted_p_value or evidence.raw_p_value


def _significant_from_evidence_p(evidence: StatisticalEvidence, alpha: float = 0.05) -> bool | None:
    p_value = _evidence_p_value(evidence)
    if p_value is None:
        return evidence.significant
    op = evidence.p_operator or "="
    if op == "=":
        return p_value < alpha
    if op in {"<", "<="}:
        return True if p_value <= alpha else None
    if op in {">", ">="}:
        return False if p_value >= alpha else None
    return p_value < alpha


def _effect_labels_compatible(left: str | None, right: str | None) -> bool:
    left_c = _canonical_effect_label(left)
    right_c = _canonical_effect_label(right)
    if left_c == right_c:
        return True
    return {left_c, right_c} <= {"correlation", "spearman_r", "pearson_r"}


def _opposite_effect_side(left: float, right: float, effect_label: str | None) -> bool:
    canonical = _canonical_effect_label(effect_label)
    if canonical in {"odds_ratio", "hazard_ratio", "risk_ratio"}:
        return (left - 1.0) * (right - 1.0) < 0
    if canonical in {"correlation", "cohen_d", "smd", "mean_difference"}:
        return left * right < 0
    return False


def _null_value_for_effect(effect_label: str | None) -> float:
    return 1.0 if _canonical_effect_label(effect_label) in {"odds_ratio", "hazard_ratio", "risk_ratio"} else 0.0


def _test_families_compatible(left: str, right: str) -> bool:
    compatible_sets = (
        {"logistic_or_binary_association", "categorical_association"},
        {"survival_regression", "survival_group_comparison"},
        {"group_difference", "categorical_association"},
    )
    return any({left, right} <= group for group in compatible_sets)


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
    assessed_statistical_scores = [
        match.statistical_comparison.statistical_agreement_score
        for match in matches
        if match.statistical_comparison is not None
        and match.statistical_comparison.statistical_agreement_score is not None
    ]
    statistical_coverages = [
        match.statistical_comparison.statistical_agreement_coverage
        for match in matches
        if match.statistical_comparison is not None
    ]

    total_qtrial = len(qtrial_findings)
    total_human = len(human_findings)
    precision = _ratio(matched_pairs, total_qtrial)
    recall = _ratio(matched_pairs, total_human)
    f1 = round((2 * precision * recall) / (precision + recall), 4) if precision + recall > 0 else 0.0
    return ComparisonMetrics(
        total_qtrial_findings=total_qtrial,
        total_human_findings=total_human,
        matched_pairs=matched_pairs,
        qtrial_only_count=len(qtrial_only),
        human_only_count=len(human_only),
        recall_against_human=recall,
        precision_against_human=precision,
        f1_against_human=f1,
        novel_rate=_ratio(len(qtrial_only), total_qtrial),
        agreement_count=agreement_count,
        partial_agreement_count=partial_count,
        contradiction_count=contradiction_count,
        agreement_rate_over_matched=_ratio(agreement_count, matched_pairs),
        contradiction_rate_over_matched=_ratio(contradiction_count, matched_pairs),
        evidence_upgrade_rate=_ratio(evidence_upgrades, matched_pairs),
        average_statistical_agreement_score=round(sum(assessed_statistical_scores) / len(assessed_statistical_scores), 4)
        if assessed_statistical_scores
        else None,
        average_statistical_evidence_coverage=round(sum(statistical_coverages) / len(statistical_coverages), 4)
        if statistical_coverages
        else 0.0,
    )


def _build_summary(metrics: ComparisonMetrics) -> str:
    summary = (
        f"Matched {metrics.matched_pairs} of {metrics.total_human_findings} human findings. "
        f"Q-Trial surfaced {metrics.qtrial_only_count} additional findings, "
        f"agreed on {metrics.agreement_count}, partially agreed on {metrics.partial_agreement_count}, "
        f"and contradicted {metrics.contradiction_count} matched findings."
    )
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
    """Return claim-pairing confidence, not agreement.

    This score estimates whether the Q-Trial and human findings discuss the
    same clinical claim. It intentionally excludes p-values, effect sizes, CIs,
    and other statistical-evidence values; those are assessed only after pairing
    in `_compare_statistical_evidence`.

    Formula:
      0.50 * variable_match
    + 0.25 * endpoint_match
    + 0.10 * claim_type_or_category_match
    + 0.15 * normalized_text_token_overlap
    """
    token_overlap = _token_overlap_ratio(qfinding.normalized_text, hfinding.normalized_text)
    same_variable = bool(qfinding.variable and hfinding.variable and qfinding.variable == hfinding.variable)
    if qfinding.variable and hfinding.variable and not same_variable:
        return 0.0
    if (
        is_comparison_claim_type(qfinding.claim_type)
        and is_comparison_claim_type(hfinding.claim_type)
        and not same_variable
        and (qfinding.variable or hfinding.variable)
    ):
        return 0.0
    if not qfinding.variable and not hfinding.variable and not _endpoints_compatible(qfinding.endpoint, hfinding.endpoint):
        return 0.0
    same_endpoint = _endpoints_compatible(qfinding.endpoint, hfinding.endpoint)
    if token_overlap == 0.0 and not same_variable and not same_endpoint:
        return 0.0
    if token_overlap < 0.08 and not same_variable and not same_endpoint:
        return 0.0

    variable_component = 1.0 if same_variable else 0.0
    endpoint_component = 1.0 if same_endpoint else 0.0
    claim_component = _claim_pairing_component(qfinding, hfinding)
    lexical_component = token_overlap

    score = (
        0.50 * variable_component
        + 0.25 * endpoint_component
        + 0.10 * claim_component
        + 0.15 * lexical_component
    )

    return min(score, 1.0)


def _claim_pairing_component(qfinding: ComparableFinding, hfinding: ComparableFinding) -> float:
    if qfinding.claim_type and hfinding.claim_type:
        if qfinding.claim_type == hfinding.claim_type:
            return 1.0
        if is_comparison_claim_type(qfinding.claim_type) and is_comparison_claim_type(hfinding.claim_type):
            return 0.7
        return 0.0
    if qfinding.finding_category and hfinding.finding_category:
        if qfinding.finding_category == hfinding.finding_category:
            return 0.8
        if is_analytical_category(qfinding.finding_category) and is_analytical_category(hfinding.finding_category):
            return 0.5
        return 0.0
    return 0.5


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
    metadata = metadata or {}
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
        variable=variable_value,
    )
    parsed_claim_type = claim_type or classify_claim_type(
        finding_text,
        finding_category=finding_category,
        variable=variable_value,
        endpoint=endpoint_value,
        significant=parsed_significant,
        p_value=parsed_p_value,
    )
    parsed_category = finding_category
    if variable_value and is_endpoint_like_variable(variable_value, endpoint_value):
        parsed_claim_type = "artifact"
        parsed_category = "artifact_excluded"
    statistical_evidence = _build_statistical_evidence(
        source=source,
        finding_text=finding_text,
        variable=variable_value,
        endpoint=endpoint_value,
        p_value=parsed_p_value,
        significant=parsed_significant,
        direction=parsed_direction,
        effect_size=final_effect_size,
        effect_size_label=final_effect_label,
        metadata=metadata,
    )
    return ComparableFinding(
        finding_id=finding_id,
        source=source,  # type: ignore[arg-type]
        source_label=source_label,
        finding_text=finding_text.strip(),
        normalized_text=normalized_text,
        section=section,
        finding_category=parsed_category,
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
        metadata=metadata,
        statistical_evidence=statistical_evidence,
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
    return _WHITESPACE_RE.sub(" ", text).strip(" #:-•\t")


def _is_finding_candidate(text: str) -> bool:
    if is_user_facing_nonfinding_artifact(text):
        return False
    if _is_generic_variable_subject_claim(text):
        return False
    if len(text) < 24:
        return False
    if len(text.split()) < 4:
        return False
    if _looks_like_non_claim_heading(text):
        return False
    if _looks_like_section_header(text):
        return False
    lowered = text.lower()
    if any(token in lowered for token in ("dataset comprises", "dataset included", "study included", "cohort included")):
        return True
    analytical_markers = (
        "p=", "p <", "p<", "significant", "not significant", "hazard ratio",
        "odds ratio", "associated", "correlated", "predictor", "predicts",
        "confidence interval", "effect size", "higher probability", "lower probability",
        "higher risk", "lower risk", "increased risk", "reduced risk",
        "no relationship", "no association", "not associated", "relationship with",
        "cohen", "cramer", "spearman", "pearson", "auc", "differed",
        "increased mortality risk", "lowered mortality risk", "decreased mortality risk",
        "increased endpoint risk", "lowered endpoint risk", "decreased endpoint risk",
        "lowers", "lowered", "increases", "increased",
    )
    if any(marker in lowered for marker in analytical_markers) or _extract_p_value(text) is not None:
        return True
    keywords = (
        "baseline imbalance",
        "median survival",
        "median follow-up",
        "median follow up",
        "event rate",
        "mortality rate",
        "survival differed",
        "difference between",
    )
    return any(keyword in lowered for keyword in keywords)


def _is_generic_variable_subject_claim(text: str) -> bool:
    return bool(_GENERIC_VARIABLE_SUBJECT_RE.match(text))


def _looks_like_section_header(text: str) -> bool:
    cleaned = text.strip().rstrip(":").lower()
    if cleaned in _SECTION_HEADERS:
        return True
    if len(cleaned.split()) <= 4 and cleaned.isalpha():
        return cleaned in _SECTION_HEADERS
    return False


def _looks_like_non_claim_heading(text: str) -> bool:
    cleaned = text.strip().strip("#").strip().rstrip(":")
    lowered = cleaned.lower()
    if any(
        phrase in lowered
        for phrase in (
            "clinical analysis report",
            "mortality analysis report",
            "survival analysis report",
            "statistical analysis report",
            "analyst report",
        )
    ):
        return True
    if len(cleaned.split()) <= 8 and not re.search(
        r"\b(was|were|is|are|had|has|show|shows|showed|associated|correlated|predicts|predicted|significant|differed|difference|increased|decreased|reduced|lowered|lowers|p\s*[=<>])\b",
        lowered,
    ):
        if any(token in lowered for token in ("mortality", "survival", "endpoint", "ejection fraction", "analysis", "results")):
            return True
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
        operator, reported = _extract_p_operator_value(text)
        if operator in {"<", "<="}:
            return True if reported is not None and reported <= 0.05 else None
        if operator in {">", ">="}:
            return False if reported is not None and reported >= 0.05 else None
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
            "increased risk",
            "reduced risk",
            "lowered risk",
            "decreased risk",
            "increased endpoint risk",
            "lowered endpoint risk",
        )
    ):
        return True
    if re.search(
        r"\b(?:increased|increases|decreased|decreases|lowered|lowers|reduced|reduces)\b[^.]{0,60}\b(?:risk|odds|hazard|probability|endpoint|outcome|mortality|event)\b",
        lowered,
    ):
        return True
    return None


def _extract_p_value(text: str) -> float | None:
    _, parsed = _extract_p_operator_value(text)
    return parsed


def _extract_p_operator_value(text: str) -> tuple[str | None, float | None]:
    match = _P_VALUE_RE.search(text)
    if not match:
        return None, None
    comparator, value = match.groups()
    comparator = {"≤": "<=", "≥": ">="}.get(comparator, comparator)
    parsed = _safe_float(value)
    if parsed is None:
        return None, None
    return comparator, parsed


def _extract_effect_size(text: str) -> tuple[float | None, str | None]:
    for pattern, label in _EFFECT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _safe_float(match.group(1)), label
    return None, None


def _build_statistical_evidence(
    *,
    source: str,
    finding_text: str,
    variable: str | None,
    endpoint: str | None,
    p_value: float | None,
    significant: bool | None,
    direction: str,
    effect_size: float | None,
    effect_size_label: str | None,
    metadata: dict[str, Any],
) -> StatisticalEvidence:
    p_operator, text_p_value = _extract_p_operator_value(finding_text)
    raw_p_value = _first_float(
        metadata,
        "raw_p_value",
        "raw_p",
        "unadjusted_p_value",
        "p_value_raw",
    )
    adjusted_p_value = _first_float(
        metadata,
        "adjusted_p_value",
        "adjusted_p",
        "p_adj",
        "q_value",
        "fdr_p_value",
    )
    final_p = _safe_float(p_value)
    if final_p is None:
        final_p = text_p_value
    if raw_p_value is None and adjusted_p_value is None:
        raw_p_value = final_p

    metadata_effect = _metadata_effect(metadata)
    final_effect = metadata_effect[0] if metadata_effect[0] is not None else effect_size
    final_effect_label = metadata_effect[1] or effect_size_label
    final_effect_label = _canonical_effect_label(final_effect_label) if final_effect_label else None

    ci_lower, ci_upper = _extract_ci_from_metadata(metadata)
    if ci_lower is None or ci_upper is None:
        ci_lower, ci_upper = _extract_ci(finding_text)
    confidence_level = _extract_confidence_level(finding_text) if (ci_lower is not None and ci_upper is not None) else None

    statistic_label, statistic_value = _extract_statistic(finding_text, metadata)
    test_type = str(metadata.get("test_type") or metadata.get("test") or "").strip() or _extract_test_type(finding_text)
    test_family = _canonical_test_family(test_type, final_effect_label, finding_text)
    rank = _extract_rank(finding_text)
    importance_score = _extract_importance_score(metadata)
    sample_size = _extract_sample_size(finding_text, metadata)
    covariates = _extract_covariates(finding_text, metadata)
    effect_direction = _direction_from_effect(final_effect, final_effect_label)
    normalized_direction = direction if direction in {"positive", "negative", "none"} else effect_direction
    direction_effect = _direction_effect_on_endpoint(
        finding_text=finding_text,
        variable=variable,
        endpoint=endpoint,
        direction=normalized_direction,
        effect_size=final_effect,
        effect_size_label=final_effect_label,
    )
    evidence_confidence = _evidence_extraction_confidence(
        p_value=final_p,
        effect_size=final_effect,
        ci_lower=ci_lower,
        test_type=test_type,
        significant=significant,
        source=source,
    )
    return StatisticalEvidence(
        variable=variable,
        endpoint=endpoint,
        test_type=test_type or None,
        test_family=test_family,
        p_value=adjusted_p_value if adjusted_p_value is not None else final_p,
        p_operator=p_operator,
        adjusted_p_value=adjusted_p_value,
        raw_p_value=raw_p_value,
        effect_size=final_effect,
        effect_size_label=final_effect_label,
        effect_direction=effect_direction,  # type: ignore[arg-type]
        direction_effect_on_endpoint=direction_effect,  # type: ignore[arg-type]
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        statistic_value=statistic_value,
        statistic_label=statistic_label,
        significant=significant,
        direction=normalized_direction,  # type: ignore[arg-type]
        sample_size=sample_size,
        covariates=covariates,
        rank=rank,
        importance_score=importance_score,
        extraction_confidence=evidence_confidence,
        source_text=finding_text.strip(),
    )


def _first_float(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _safe_float(metadata.get(key))
        if value is not None:
            return value
    return None


def _metadata_effect(metadata: dict[str, Any]) -> tuple[float | None, str | None]:
    labeled_keys = (
        ("odds_ratio", "odds_ratio"),
        ("or", "odds_ratio"),
        ("hazard_ratio", "hazard_ratio"),
        ("hr", "hazard_ratio"),
        ("risk_ratio", "risk_ratio"),
        ("rr", "risk_ratio"),
        ("correlation", "correlation_r"),
        ("correlation_r", "correlation_r"),
        ("cohen_d", "cohen_d"),
        ("cramers_v", "cramers_v"),
        ("auc", "auc"),
    )
    for key, label in labeled_keys:
        value = _safe_float(metadata.get(key))
        if value is not None:
            return value, label
    value = _safe_float(metadata.get("effect_size"))
    label = str(metadata.get("effect_size_label") or "").strip() or None
    return value, label


def _extract_ci_from_metadata(metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    lower = _first_float(metadata, "ci_lower", "or_ci_lower", "hr_ci_lower", "lower_ci", "confidence_interval_lower")
    upper = _first_float(metadata, "ci_upper", "or_ci_upper", "hr_ci_upper", "upper_ci", "confidence_interval_upper")
    ci = metadata.get("ci_95") or metadata.get("confidence_interval") or metadata.get("ci")
    if (lower is None or upper is None) and isinstance(ci, list | tuple) and len(ci) >= 2:
        lower = _safe_float(ci[0])
        upper = _safe_float(ci[1])
    if lower is None or upper is None:
        return None, None
    return min(lower, upper), max(lower, upper)


def _extract_ci(text: str) -> tuple[float | None, float | None]:
    match = _CI_RE.search(text)
    if not match:
        return None, None
    left = _safe_float(match.group(1))
    right = _safe_float(match.group(2))
    if left is None or right is None:
        return None, None
    return min(left, right), max(left, right)


def _extract_confidence_level(text: str) -> float | None:
    match = re.search(r"\b(\d{2,3})\s*%\s*(?:ci|confidence interval)", text, re.IGNORECASE)
    if not match:
        return 0.95
    value = _safe_float(match.group(1))
    if value is None:
        return 0.95
    return value / 100.0 if value > 1 else value


def _extract_statistic(text: str, metadata: dict[str, Any]) -> tuple[str | None, float | None]:
    metadata_value = _safe_float(metadata.get("statistic_value"))
    metadata_label = str(metadata.get("statistic_label") or "").strip() or None
    if metadata_value is not None:
        return metadata_label, metadata_value
    for pattern, label in _STATISTIC_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = _safe_float(match.group(1))
            if value is not None:
                return label, value
    return None, None


def _extract_test_type(text: str) -> str | None:
    lowered = text.lower()
    tests = (
        ("logistic regression", ("logistic regression", "odds ratio")),
        ("cox_regression", ("cox", "hazard ratio")),
        ("log_rank", ("log-rank", "log rank")),
        ("kaplan_meier", ("kaplan", "kaplan-meier")),
        ("chi_square", ("chi-square", "chi square", "χ²", "chi²")),
        ("t_test", ("t-test", "t test")),
        ("mann_whitney", ("mann-whitney", "mann whitney", "wilcoxon rank-sum")),
        ("spearman_correlation", ("spearman", "ρ")),
        ("pearson_correlation", ("pearson", "correlation", " r=")),
        ("anova", ("anova",)),
        ("auc", ("auc",)),
    )
    for test_type, markers in tests:
        if any(marker in lowered for marker in markers):
            return test_type
    return None


def _canonical_test_family(test_type: str | None, effect_label: str | None, text: str) -> str | None:
    value = _canonicalize_variable_name(test_type) or ""
    effect = _canonical_effect_label(effect_label)
    lowered = text.lower()
    if value in {"logistic_regression"} or effect == "odds_ratio":
        return "logistic_or_binary_association"
    if value in {"cox_regression"} or effect == "hazard_ratio":
        return "survival_regression"
    if value in {"log_rank", "kaplan_meier"}:
        return "survival_group_comparison"
    if value in {"chi_square", "fisher_exact"} or "χ²" in lowered:
        return "categorical_association"
    if value in {"t_test", "mann_whitney", "anova"} or effect in {"cohen_d", "mean_difference", "smd"}:
        return "group_difference"
    if "correlation" in value or effect == "correlation":
        return "correlation"
    if value == "auc" or effect == "auc":
        return "discrimination"
    return value or None


def _canonical_effect_label(label: str | None) -> str | None:
    if not label:
        return None
    normalized = _canonicalize_variable_name(label)
    if normalized in {"odds_ratio", "or"}:
        return "odds_ratio"
    if normalized in {"hazard_ratio", "hr"}:
        return "hazard_ratio"
    if normalized in {"risk_ratio", "rr"}:
        return "risk_ratio"
    if normalized in {"correlation_r", "pearson_r", "spearman_r", "kendall_r", "r", "rho", "spearman_rho", "correlation"}:
        return "correlation"
    if normalized in {"cohen_d", "cohens_d", "d"}:
        return "cohen_d"
    if normalized in {"cramers_v", "cramer_v", "cramers"}:
        return "cramers_v"
    if normalized in {"smd", "standardized_mean_difference", "standardised_mean_difference"}:
        return "smd"
    if normalized in {"mean_difference", "mean_diff"}:
        return "mean_difference"
    if normalized == "auc":
        return "auc"
    return normalized


def _direction_from_effect(effect: float | None, effect_label: str | None) -> str:
    if effect is None:
        return "unknown"
    canonical = _canonical_effect_label(effect_label)
    if canonical in {"odds_ratio", "hazard_ratio", "risk_ratio"}:
        if effect > 1:
            return "positive"
        if effect < 1:
            return "negative"
        return "none"
    if effect > 0:
        return "positive"
    if effect < 0:
        return "negative"
    return "none"


def _extract_rank(text: str) -> int | None:
    lowered = text.lower()
    if re.search(r"\b(?:strongest|top|most important|primary)\s+(?:predictor|factor|variable|finding)\b", lowered):
        return 1
    ordinal_map = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
    }
    for word, rank in ordinal_map.items():
        if re.search(rf"\b{word}\s+(?:strongest|most important|ranked|predictor|factor)", lowered):
            return rank
    match = re.search(r"\brank(?:ed)?\s*(?:#|number|no\.?)?\s*(\d+)\b", lowered)
    if match:
        return int(match.group(1))
    return None


def _extract_importance_score(metadata: dict[str, Any]) -> float | None:
    return _first_float(metadata, "importance_score", "feature_importance", "importance")


def _extract_sample_size(text: str, metadata: dict[str, Any]) -> int | None:
    for key in ("sample_size", "n", "n_obs", "n_observations"):
        value = metadata.get(key)
        if isinstance(value, int):
            return value
        parsed = _safe_float(value)
        if parsed is not None:
            return int(parsed)
    match = _SAMPLE_SIZE_RE.search(text)
    return int(match.group(1)) if match else None


def _extract_covariates(text: str, metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("covariates")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    lowered = text.lower()
    match = re.search(r"\b(?:adjusted for|controlling for|controlled for)\s+([^.;)]+)", lowered)
    if not match:
        return []
    return [
        _canonicalize_variable_name(item.strip()) or item.strip()
        for item in re.split(r",|\band\b", match.group(1))
        if item.strip()
    ]


def _evidence_extraction_confidence(
    *,
    p_value: float | None,
    effect_size: float | None,
    ci_lower: float | None,
    test_type: str | None,
    significant: bool | None,
    source: str,
) -> float:
    score = 0.15 if significant is not None else 0.0
    if p_value is not None:
        score += 0.25
    if effect_size is not None:
        score += 0.25
    if ci_lower is not None:
        score += 0.15
    if test_type:
        score += 0.1
    if source == "qtrial":
        score += 0.1
    return min(score, 1.0)


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

    associated_with = re.search(r"\bassociated\s+with\s+([a-z][a-z0-9_ ]{2,40}?)(?:[.;,)]|$)", lowered)
    if associated_with:
        candidate = _canonicalize_variable_name(associated_with.group(1))
        if candidate and not is_endpoint_like_variable(candidate):
            return candidate

    no_association_phrase = re.match(
        r"^\s*([a-z][a-z0-9_ ]{2,40}?)\s+(?:did|does)\s+not\s+(?:show|demonstrate|have)\b",
        lowered,
    )
    if no_association_phrase:
        return _canonicalize_variable_name(no_association_phrase.group(1))

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


def _column_aliases(column: str) -> set[str]:
    canonical = _canonicalize_variable_name(column) or column.lower()
    aliases = {
        column.lower(),
        canonical,
        canonical.replace("_", " "),
        canonical.replace("_", ""),
    }
    aliases.update(_VARIABLE_SYNONYMS.get(canonical, set()))
    return {alias for alias in aliases if alias}


def _canonicalize_variable_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^a-z0-9_ ]", " ", value.lower())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    if not cleaned:
        return None
    cleaned = re.sub(r"^the_", "", cleaned)
    if cleaned in _GENERIC_VARIABLE_ARTIFACTS:
        return None
    for suffix in (
        "_did_not_show",
        "_did_not_demonstrate",
        "_did_not_have",
        "_does_not_show",
        "_does_not_demonstrate",
        "_does_not_have",
        "_did_not",
        "_does_not",
        "_showed",
        "_shows",
        "_show",
        "_were",
        "_was",
        "_are",
        "_is",
        "_levels",
        "_level",
    ):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip("_")
            break
    if not cleaned:
        return None
    if cleaned in _GENERIC_VARIABLE_ARTIFACTS:
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
    variable: str | None = None,
) -> str:
    ratio_direction = _ratio_direction(effect_size, effect_size_label)
    if ratio_direction != "unknown":
        return ratio_direction
    textual_direction = _textual_effect_direction(finding_text, variable=variable)
    if textual_direction != "unknown":
        return textual_direction
    lowered = finding_text.lower()
    for direction, phrases in _DIRECTION_HINTS.items():
        if any(phrase in lowered for phrase in phrases):
            return direction
    if effect_size is not None:
        if effect_size > 0:
            return "positive"
        if effect_size < 0:
            return "negative"
        return "none"
    return "unknown"


def _direction_effect_on_endpoint(
    *,
    finding_text: str,
    variable: str | None,
    endpoint: str | None,
    direction: str,
    effect_size: float | None,
    effect_size_label: str | None,
) -> str:
    ratio_direction = _ratio_direction(effect_size, effect_size_label)
    if ratio_direction == "positive":
        return "increases_endpoint_risk"
    if ratio_direction == "negative":
        return "decreases_endpoint_risk"
    if ratio_direction == "none":
        return "no_direction"
    textual = _textual_effect_direction(finding_text, variable=variable, endpoint=endpoint)
    if textual == "positive":
        return "increases_endpoint_risk"
    if textual == "negative":
        return "decreases_endpoint_risk"
    if textual == "none":
        return "no_direction"
    effect_direction = _direction_from_effect(effect_size, effect_size_label)
    source = direction if direction != "unknown" else effect_direction
    if source == "positive":
        return "increases_endpoint_risk"
    if source == "negative":
        return "decreases_endpoint_risk"
    if source == "none":
        return "no_direction"
    return "unknown"


def _ratio_direction(effect_size: float | None, effect_size_label: str | None) -> str:
    if effect_size is None:
        return "unknown"
    if _canonical_effect_label(effect_size_label) not in {"odds_ratio", "hazard_ratio", "risk_ratio"}:
        return "unknown"
    if effect_size > 1:
        return "positive"
    if effect_size < 1:
        return "negative"
    return "none"


def _textual_effect_direction(
    finding_text: str,
    *,
    variable: str | None,
    endpoint: str | None = None,
) -> str:
    lowered = f" {finding_text.lower()} "
    if any(phrase in lowered for phrase in (" no association ", " no relationship ", " not associated ", "not significantly associated")):
        return "none"
    if "protective" in lowered:
        return "negative"
    if any(phrase in lowered for phrase in ("risk-increasing", "risk increasing", "risk factor")):
        return "positive"
    if not variable:
        return "unknown"

    endpoint_terms = _endpoint_direction_terms(endpoint)
    endpoint_up = r"(?:higher|increased|increases|increase|elevated|greater|more|worse|raises|raised|higher\s+(?:risk|odds|hazard|probability|rate))"
    endpoint_down = r"(?:lower|decreased|decreases|decrease|reduced|reduces|reduce|less|fewer|better|lowers|lowered|lower\s+(?:risk|odds|hazard|probability|rate))"
    endpoint_object = rf"(?:{'|'.join(re.escape(term) for term in sorted(endpoint_terms, key=len, reverse=True))})"

    for alias in sorted(_column_aliases(variable), key=len, reverse=True):
        escaped = re.escape(alias.lower())
        high_pred = rf"(?:higher|increased|elevated|older|greater|more)\s+{escaped}|{escaped}\s+(?:is|was|were|are)?\s*(?:higher|increased|elevated)"
        low_pred = rf"(?:lower|decreased|reduced|younger|less|fewer)\s+{escaped}|{escaped}\s+(?:is|was|were|are)?\s*(?:lower|decreased|reduced)"
        alias_subject = rf"{escaped}\s+(?:is|was|were|are)?\s*"

        if re.search(rf"\b(?:{high_pred})\b[^.;&]{{0,120}}\b(?:{endpoint_down})\b(?:[^.;&]{{0,60}}\b{endpoint_object}\b)?", lowered):
            return "negative"
        if re.search(rf"\b(?:{low_pred})\b[^.;&]{{0,120}}\b(?:{endpoint_up})\b(?:[^.;&]{{0,60}}\b{endpoint_object}\b)?", lowered):
            return "negative"
        if re.search(rf"\b(?:{high_pred})\b[^.;&]{{0,120}}\b(?:{endpoint_up})\b(?:[^.;&]{{0,60}}\b{endpoint_object}\b)?", lowered):
            return "positive"
        if re.search(rf"\b(?:{low_pred})\b[^.;&]{{0,120}}\b(?:{endpoint_down})\b(?:[^.;&]{{0,60}}\b{endpoint_object}\b)?", lowered):
            return "positive"
        if re.search(rf"\b{alias_subject}(?:lowers|lowered|reduces|reduced|decreases|decreased)\b[^.;&]{{0,60}}\b{endpoint_object}\b", lowered):
            return "negative"
        if re.search(rf"\b{alias_subject}(?:raises|raised|increases|increased|elevates|elevated)\b[^.;&]{{0,60}}\b{endpoint_object}\b", lowered):
            return "positive"

    return "unknown"


def _endpoint_direction_terms(endpoint: str | None) -> set[str]:
    terms = {
        "risk",
        "odds",
        "hazard",
        "probability",
        "rate",
        "event",
        "events",
        "endpoint",
        "outcome",
    }
    if endpoint:
        cleaned = endpoint.replace("_", " ").lower()
        terms.update({cleaned, endpoint.lower()})
    for aliases in _ENDPOINT_ALIASES.values():
        terms.update(alias.lower() for alias in aliases)
    return {term for term in terms if term}


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


def _has_structured_association_payload(finding: dict[str, Any]) -> bool:
    if not isinstance(finding, dict):
        return False
    has_stat = any(
        finding.get(key) is not None
        for key in (
            "adjusted_p_value",
            "raw_p_value",
            "p_value",
            "effect_size",
            "odds_ratio",
            "significant_after_correction",
            "significant",
        )
    )
    return has_stat and bool(str(finding.get("finding_id") or finding.get("variable") or "").strip())


def _structured_finding_sentence(
    *,
    variable: str | None,
    endpoint: str | None,
    significant: bool | None,
) -> str | None:
    if not variable:
        return None
    subject = variable.replace("_", " ").capitalize()
    outcome = (endpoint or "the outcome").replace("_", " ")
    if significant is False:
        return f"{subject} was not significantly associated with {outcome}."
    if significant is True:
        return f"{subject} was significantly associated with {outcome}."
    return f"{subject} was evaluated for association with {outcome}."


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
    significant = qfinding.significant if qfinding.significant is not None else hfinding.significant
    association = "a statistically significant association" if significant is not False else "no statistically significant association"
    if endpoint:
        return f"Both reports identify {association} between {variable} and {endpoint}."
    return f"Both reports identify {association} for {variable}."


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
    pairing_confidence: float
