from __future__ import annotations

import json
import re
from typing import Any

from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName

_SYSTEM_PROMPT = """\
You verbalize structured statistical findings for clinicians.

Rules:
- Use only the structured evidence provided.
- Do not analyze the dataset.
- Do not invent direction, endpoint meaning, prognosis, or causal language.
- If direction is unknown, use neutral wording and do not use words like higher, lower, increased, decreased, worse, better, or risk.
- If endpoint is null or unclear, say "the outcome" and nothing more specific.
- Output exactly one short sentence per finding.
- No bullets, no markdown, no explanations, no recommendations, no qualifiers after the sentence.
- If significant is false, say it did not show a statistically significant association.
- If significant is true and direction is unknown, say it was significantly associated with the endpoint or outcome.
- Use directional wording only when direction is explicitly provided as positive or negative.
- Never use causal words such as cause, causes, caused, leads to, results in, drives, prevents, protects, or improves outcomes.
- Never use recommendation language such as should, recommend, consider, need to, or important risk factor.
- If endpoint is unclear, say "the outcome".

Return JSON only:
{"findings":[{"finding_id":"...","sentence":"..."}]}
"""

_RECOMMENDATION_PATTERNS = (
    r"\bshould\b",
    r"\brecommend\b",
    r"\bconsider\b",
    r"\bneed to\b",
    r"\bimportant risk factor\b",
)
_CAUSAL_PATTERNS = (
    r"\bcause(?:s|d)?\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\bdrives?\b",
    r"\bprevents?\b",
    r"\bprotects?\b",
    r"\bimproves?\b",
)
_ENDPOINT_WORDS = ("mortality", "death", "survival", "follow-up")
_UNKNOWN_DIRECTION_TERMS = ("higher", "lower", "increased", "decreased", "worse", "better", "risk")


def verbalize_statistical_findings(
    findings: list[dict[str, Any]],
    provider: ProviderName,
) -> list[dict[str, Any]]:
    if not findings:
        return findings

    client = get_client(provider)
    payload = [
        {
            "finding_id": str(f.get("finding_id", "")),
            "variable": f.get("variable"),
            "endpoint": f.get("endpoint"),
            "significant": f.get("significant_after_correction", f.get("significant")),
            "p_value": f.get("adjusted_p_value", f.get("raw_p_value")),
            "direction": f.get("direction", "unknown"),
            "direction_label": f.get("direction_label"),
            "analysis_type": f.get("analysis_type", "association"),
            "test_type": f.get("test_type"),
            "effect_size": f.get("effect_size"),
            "effect_size_label": f.get("effect_size_label"),
            "odds_ratio": f.get("odds_ratio"),
            "metadata": f.get("metadata"),
            "raw_statistical_text": f.get("finding_text_raw"),
        }
        for f in findings
    ]
    req = LLMRequest(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f"Structured findings:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
        payload={"temperature": 0},
    )

    try:
        resp = client.generate(req)
        raw = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        sentence_map: dict[str, str] = {}
        for item in data.get("findings", []):
            finding_id = str(item.get("finding_id", "")).strip()
            sentence = str(item.get("sentence", "")).strip()
            if not finding_id or finding_id in sentence_map:
                continue
            sentence_map[finding_id] = sentence
    except Exception:
        sentence_map = {}

    updated: list[dict[str, Any]] = []
    for finding in findings:
        clone = dict(finding)
        sentence = sentence_map.get(str(finding.get("finding_id", "")).strip())
        validated = _validate_sentence(sentence, finding) if sentence else None
        clone["finding_text_plain"] = _prefer_evidence_sentence(validated, finding)
        updated.append(clone)
    return updated


def _validate_sentence(sentence: str, finding: dict[str, Any]) -> str | None:
    cleaned = re.sub(r"\s+", " ", sentence.replace("\n", " ")).strip().strip('"')
    if not cleaned:
        return None
    if len(cleaned) > 220:
        return None
    if _sentence_count(cleaned) != 1:
        return None

    lowered = cleaned.lower()
    if any(re.search(pattern, lowered) for pattern in _RECOMMENDATION_PATTERNS):
        return None
    if any(re.search(pattern, lowered) for pattern in _CAUSAL_PATTERNS):
        return None

    direction = str(finding.get("direction") or "unknown")
    if direction == "unknown" and any(term in lowered for term in _UNKNOWN_DIRECTION_TERMS):
        return None

    endpoint = finding.get("endpoint")
    if not endpoint and any(word in lowered for word in _ENDPOINT_WORDS):
        return None

    significant = finding.get("significant_after_correction", finding.get("significant"))
    if significant is False and ("significant" in lowered and "not" not in lowered and "did not" not in lowered):
        return None
    if significant is True and ("did not show" in lowered or "not significant" in lowered):
        return None

    return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."


def _sentence_count(text: str) -> int:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return len(parts)


def _fallback_sentence(finding: dict[str, Any]) -> str:
    subject = _humanize_variable(str(finding.get("variable") or "This variable"))
    outcome = _endpoint_phrase(finding.get("endpoint"))
    significant = finding.get("significant_after_correction", finding.get("significant"))
    category = str(finding.get("finding_category") or "")
    variable = str(finding.get("variable") or finding.get("finding_id") or "")
    if category == "artifact_excluded":
        return f"{subject} was excluded from analytical findings because it is an endpoint or derived endpoint-like variable."
    if category == "statistical_note":
        p_text = _stat_parenthetical(finding)
        if variable.lower() == "survival_primary":
            return f"The primary survival result was retained as a statistical note{p_text}, not as a baseline clinical predictor."
        return f"{subject} was retained as a statistical note{p_text}, not as a primary clinical predictor."

    if significant is False:
        return f"{subject} was not significantly associated with {outcome}{_stat_parenthetical(finding)}."
    if significant is True:
        effect_label = _effect_label(finding)
        direction = _ratio_direction(finding, effect_label) or str(finding.get("direction") or "unknown")
        directional_subject = _directional_subject(str(finding.get("variable") or ""), direction)
        if direction == "positive":
            if effect_label == "odds_ratio" and outcome == "mortality":
                return f"{directional_subject or subject} was associated with higher odds of mortality{_stat_parenthetical(finding)}."
            if effect_label == "hazard_ratio" and outcome == "mortality":
                return f"{directional_subject or subject} was associated with higher mortality hazard{_stat_parenthetical(finding)}."
            return f"{directional_subject or subject} was associated with higher {outcome}{_stat_parenthetical(finding)}."
        if direction == "negative":
            if effect_label in {"odds_ratio", "hazard_ratio", "risk_ratio"} and outcome == "mortality":
                return (
                    f"Higher {subject.lower()} was associated with lower odds of mortality"
                    f"{_stat_parenthetical(finding)}, consistent with lower {subject.lower()} indicating higher mortality risk."
                )
            high_subject = _directional_subject(str(finding.get("variable") or ""), "positive") or f"Higher {subject.lower()}"
            return f"{high_subject} was associated with lower {outcome}{_stat_parenthetical(finding)}."
        return f"{subject} was significantly associated with {outcome}{_stat_parenthetical(finding)}."
    return f"{subject} was evaluated for association with {outcome}{_stat_parenthetical(finding)}."


def _prefer_evidence_sentence(validated: str | None, finding: dict[str, Any]) -> str:
    if _has_structured_evidence(finding):
        return _fallback_sentence(finding)
    return validated or _fallback_sentence(finding)


def _humanize_variable(variable: str) -> str:
    cleaned = variable.replace("_", " ").strip()
    return cleaned.capitalize() if cleaned else "This variable"


def _directional_subject(variable: str, direction: str) -> str | None:
    cleaned = variable.replace("_", " ").strip().lower()
    if not cleaned:
        return None
    if direction == "positive":
        if cleaned == "age":
            return "Older age"
        return f"Higher {cleaned}"
    if direction == "negative":
        if cleaned == "age":
            return "Younger age"
        return f"Lower {cleaned}"
    return None


def _endpoint_phrase(endpoint: Any) -> str:
    if endpoint == "mortality":
        return "mortality"
    if endpoint == "survival":
        return "survival"
    if endpoint == "primary_outcome":
        return "the primary outcome"
    return "the outcome"


def _has_structured_evidence(finding: dict[str, Any]) -> bool:
    return any(
        finding.get(key) is not None
        for key in ("adjusted_p_value", "raw_p_value", "p_value", "effect_size", "odds_ratio", "test_type")
    )


def _stat_parenthetical(finding: dict[str, Any]) -> str:
    parts: list[str] = []
    effect_label = _effect_label(finding)
    effect_value = finding.get("odds_ratio") if effect_label == "odds_ratio" else finding.get("effect_size")
    if effect_label and effect_value is not None:
        parts.append(f"{_display_effect_label(effect_label)} {_format_number(effect_value)}")
    p_value = finding.get("adjusted_p_value", finding.get("p_value", finding.get("raw_p_value")))
    test_type = str(finding.get("test_type") or "").strip()
    if p_value is not None:
        p_text = _format_p_value(p_value)
        p_display = f"p {p_text[0]} {p_text[1:]}" if p_text.startswith(("<", ">")) else f"p={p_text}"
        if test_type:
            parts.append(f"{test_type} {p_display}")
        else:
            parts.append(p_display)
    if not parts:
        return ""
    return f" ({', '.join(parts)})"


def _effect_label(finding: dict[str, Any]) -> str | None:
    if finding.get("odds_ratio") is not None:
        return "odds_ratio"
    label = finding.get("effect_size_label")
    return str(label) if label else None


def _ratio_direction(finding: dict[str, Any], effect_label: str | None) -> str | None:
    if effect_label not in {"odds_ratio", "hazard_ratio", "risk_ratio"}:
        return None
    effect_value = finding.get("odds_ratio") if effect_label == "odds_ratio" else finding.get("effect_size")
    try:
        ratio = float(effect_value)
    except (TypeError, ValueError):
        return None
    if ratio > 1:
        return "positive"
    if ratio < 1:
        return "negative"
    return "none"


def _display_effect_label(label: str) -> str:
    normalized = label.lower()
    return {
        "odds_ratio": "OR",
        "hazard_ratio": "HR",
        "risk_ratio": "RR",
        "cramers_v": "Cramer's V",
        "cramer_v": "Cramer's V",
        "cohen_d": "Cohen's d",
        "mean_difference": "mean difference",
        "correlation": "correlation",
        "correlation_coefficient": "correlation",
    }.get(normalized, label.replace("_", " "))


def _format_p_value(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return str(value)
    if parsed == 0:
        return "<1e-12"
    if parsed >= 0.9995:
        return ">0.99"
    if abs(parsed) < 0.001:
        return f"{parsed:.2e}"
    return f"{parsed:.4f}".rstrip("0").rstrip(".")


def _format_number(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return str(value)
    if parsed != 0 and abs(parsed) < 0.001:
        return f"{parsed:.2e}"
    return f"{parsed:.4f}".rstrip("0").rstrip(".")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
