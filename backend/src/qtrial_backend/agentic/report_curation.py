from __future__ import annotations

import re
from typing import Any

from qtrial_backend.agentic.schemas import FinalReportSchema, GroundedFinding, InsightSynthesisOutput


_BANNED_EXACT_PHRASES = (
    "Now I have all the necessary information.",
    "Let me compile the comprehensive final report.",
    "Next step: run analyze",
    "Generated ... Fully deterministic, no LLM",
)

_PROCESS_NARRATION_RE = re.compile(
    r"\b(?:now\s+i\s+have|let\s+me\s+(?:compile|summarize|prepare)|next\s+step:|"
    r"i\s+will\s+(?:now\s+)?(?:run|analy(?:z|s)e|compile|generate)|"
    r"i'm\s+going\s+to\s+(?:run|analy(?:z|s)e|compile|generate))\b",
    re.IGNORECASE,
)

_DETERMINISTIC_HEADER_RE = re.compile(r"^\s*>\s*generated\b.*\bdeterministic\b.*\bno\s+llm\b", re.IGNORECASE)

_OVERCLAIM_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwill\s+improve\s+survival\b", re.IGNORECASE), "could improve survival"),
    (re.compile(r"\bwill\s+reduce\s+mortality\b", re.IGNORECASE), "could reduce mortality"),
)


def _strip_internal_lines(text: str) -> str:
    if not text:
        return text
    # Fast exact removals.
    if text.strip() in _BANNED_EXACT_PHRASES:
        return ""
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in _BANNED_EXACT_PHRASES:
            continue
        if _DETERMINISTIC_HEADER_RE.match(line):
            continue
        if _PROCESS_NARRATION_RE.search(line):
            continue
        lines.append(raw)
    return "\n".join(lines).strip()


def _cautious_text(text: str, *, study_context: str | None) -> str:
    cleaned = _strip_internal_lines(text)
    if not cleaned:
        return cleaned

    updated = cleaned
    for pattern, replacement in _OVERCLAIM_REPLACEMENTS:
        updated = pattern.sub(replacement, updated)

    # If the study context suggests observational / retrospective data, enforce a cautious framing.
    ctx = (study_context or "").lower()
    observational = any(token in ctx for token in ("observational", "retrospective", "registry", "cohort"))
    if observational and "hypothesis-generating" not in updated.lower():
        # Only append when the string already reads as a recommendation/hypothesis sentence.
        if re.search(r"\b(?:recommend|should|could|future\s+trial|prospective|test)\b", updated, re.IGNORECASE):
            updated = updated.rstrip(".") + " (hypothesis-generating)."
    return updated.strip()


def _importance_score(text: str) -> int:
    lowered = (text or "").lower()
    score = 0
    if any(token in lowered for token in ("missing", "imput", "complete case")):
        score += 3
    if any(token in lowered for token in ("duplicate", "integrity", "inconsistent", "invalid")):
        score += 3
    if any(token in lowered for token in ("outlier", "extreme", "winsor", "skew", "heavy-tailed")):
        score += 2
    if any(token in lowered for token in ("digit preference", "rounding", "heaping")):
        score += 2
    if any(token in lowered for token in ("censor", "follow-up", "followup", "time-to-event", "log-rank", "kaplan")):
        score += 3
    if any(token in lowered for token in ("small sample", "limited power", "underpowered")):
        score += 1
    return score


def _curate_qc_findings(findings: list[GroundedFinding], *, cap: int = 12) -> list[GroundedFinding]:
    # De-dupe by normalized text, drop narration, keep most important notes.
    seen: set[str] = set()
    curated: list[GroundedFinding] = []
    scored: list[tuple[int, str, GroundedFinding]] = []
    for f in findings:
        text = (f.finding_text_plain or f.finding_text or "").strip()
        cleaned = _strip_internal_lines(text)
        if not cleaned:
            continue
        key = " ".join(cleaned.lower().split())
        if key in seen:
            continue
        seen.add(key)
        lowered = cleaned.lower()
        bucket = "other"
        if any(t in lowered for t in ("missing", "imput", "complete case")):
            bucket = "missingness"
        elif any(t in lowered for t in ("duplicate", "integrity", "inconsistent", "invalid")):
            bucket = "integrity"
        elif any(t in lowered for t in ("outlier", "extreme", "winsor", "skew", "heavy-tailed")):
            bucket = "outliers"
        elif any(t in lowered for t in ("digit preference", "rounding", "heaping")):
            bucket = "digit_pref"
        elif any(t in lowered for t in ("censor", "follow-up", "followup", "time-to-event", "log-rank", "kaplan")):
            bucket = "survival"
        scored.append(
            (_importance_score(cleaned), bucket, f.model_copy(update={"finding_text": cleaned, "finding_text_plain": cleaned}))
        )

    # Guarantee at least one note from key buckets when present, then fill by score.
    bucket_order = ("missingness", "integrity", "outliers", "digit_pref", "survival")
    for bucket in bucket_order:
        candidates = [t for t in scored if t[1] == bucket]
        if not candidates:
            continue
        best = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
        curated.append(best[2])

    scored.sort(key=lambda x: x[0], reverse=True)
    for _, _, f in scored:
        if len(curated) >= cap:
            break
        if any((f.finding_text_plain or "").strip() == (c.finding_text_plain or "").strip() for c in curated):
            continue
        curated.append(f)
    return curated


def _curate_final_insights(final_insights: InsightSynthesisOutput, *, study_context: str | None) -> InsightSynthesisOutput:
    return final_insights.model_copy(
        update={
            "key_findings": [_strip_internal_lines(t) for t in final_insights.key_findings if _strip_internal_lines(t)],
            "risks_and_bias_signals": [
                _strip_internal_lines(t) for t in final_insights.risks_and_bias_signals if _strip_internal_lines(t)
            ],
            "required_metadata_or_questions": [
                _strip_internal_lines(t)
                for t in final_insights.required_metadata_or_questions
                if _strip_internal_lines(t)
            ],
            "recommended_next_analyses": [
                rec.model_copy(
                    update={
                        "analysis": _strip_internal_lines(rec.analysis),
                        "rationale": _cautious_text(rec.rationale, study_context=study_context),
                    }
                )
                for rec in final_insights.recommended_next_analyses
                if _strip_internal_lines(rec.analysis)
            ],
        }
    )


def curate_user_facing_report_sections(report: FinalReportSchema) -> FinalReportSchema:
    """
    Final user-facing report curation layer.

    This is a last-chance safety gate: remove internal/agent narration text,
    enforce cautious recommendation phrasing, and cap verbose QC notes.
    """
    study_context = report.study_context

    updated = report
    if updated.prior_analysis_report:
        updated = updated.model_copy(update={"prior_analysis_report": _strip_internal_lines(updated.prior_analysis_report)})

    if updated.final_insights:
        updated = updated.model_copy(update={"final_insights": _curate_final_insights(updated.final_insights, study_context=study_context)})

    if updated.grounded_findings and updated.grounded_findings.findings:
        analytical: list[GroundedFinding] = []
        qc: list[GroundedFinding] = []
        for f in updated.grounded_findings.findings:
            category = getattr(f, "finding_category", "analytical") or "analytical"
            if category in {"data_quality", "data_quality_note", "qc_note", "preprocessing", "statistical_note", "pipeline_warning"}:
                qc.append(f)
            else:
                # Still strip narration from analytical.
                text = (f.finding_text_plain or f.finding_text or "").strip()
                cleaned = _strip_internal_lines(text)
                if not cleaned:
                    continue
                analytical.append(f.model_copy(update={"finding_text": cleaned, "finding_text_plain": cleaned}))

        curated_qc = _curate_qc_findings(qc, cap=12)
        updated = updated.model_copy(
            update={
                "grounded_findings": updated.grounded_findings.model_copy(update={"findings": analytical + curated_qc})
            }
        )

    # Apply phrase-level curation to comparison strings (summary only; findings are curated earlier).
    if updated.comparison_report:
        summary = _strip_internal_lines(updated.comparison_report.summary)
        updated = updated.model_copy(
            update={"comparison_report": updated.comparison_report.model_copy(update={"summary": summary})}
        )

    return updated


def report_contains_banned_user_facing_text(report: FinalReportSchema) -> list[str]:
    """
    Helper for tests: returns matched banned phrase snippets found in user-facing payload.
    """
    dumped: dict[str, Any] = report.model_dump(mode="json")  # type: ignore[assignment]
    matches: list[str] = []

    def _walk(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            for phrase in _BANNED_EXACT_PHRASES:
                if phrase.lower() in obj.lower():
                    matches.append(phrase)
            if _PROCESS_NARRATION_RE.search(obj):
                matches.append("process_narration")
            if "treatment = death_event" in obj.lower():
                matches.append("treatment = DEATH_EVENT")
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
            return
        if isinstance(obj, list):
            for v in obj:
                _walk(v)
            return

    _walk(dumped)
    return sorted(set(matches))
