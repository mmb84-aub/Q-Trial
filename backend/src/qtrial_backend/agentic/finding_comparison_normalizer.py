from __future__ import annotations

import json
import re
from typing import Any

from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName

_SYSTEM_PROMPT = """\
You rewrite structured analytical findings into comparison-ready clinical claims.

Rules:
- Use only the structured evidence provided.
- Do not analyze the dataset.
- Output exactly one short sentence per finding.
- No bullets, markdown, explanations, recommendations, or follow-up commentary.
- No causal language.
- Do not invent endpoint wording.
- Do not invent direction if direction is unknown.
- If significant is false, say it did not show a statistically significant association with the endpoint or outcome.
- If significant is true and direction is unknown, use neutral wording.
- Use directional wording only when direction is explicitly provided as positive or negative.
- Prefer analyst-style phrasing such as "Older age..." or "Lower ejection fraction..." when the variable semantics make that wording natural.
- If endpoint is mortality and direction is positive, prefer wording like "Higher X was associated with higher mortality risk."
- If endpoint is mortality and direction is negative, prefer wording like "Lower X was associated with higher mortality risk."
- If the variable is age and direction is positive, "Older age..." is preferred over "Higher age...".

Return JSON only:
{"findings":[{"finding_id":"...","sentence":"..."}]}
"""

_RECOMMENDATION_PATTERNS = (
    r"\bshould\b",
    r"\brecommend\b",
    r"\bconsider\b",
    r"\bneed to\b",
)
_CAUSAL_PATTERNS = (
    r"\bcause(?:s|d)?\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\bdrives?\b",
    r"\bprevents?\b",
    r"\bprotects?\b",
)
_UNKNOWN_DIRECTION_TERMS = ("higher", "lower", "increased", "decreased", "worse", "better", "risk")
_ENDPOINT_WORDS = ("mortality", "death", "survival", "follow-up", "follow up")
_POSITIVE_DIRECTION_TERMS = ("higher", "increased", "elevated", "older")
_NEGATIVE_DIRECTION_TERMS = ("lower", "decreased", "reduced", "younger")


def normalize_comparison_claims(
    findings: list[dict[str, Any]],
    provider: ProviderName,
) -> list[dict[str, Any]]:
    if not findings:
        return findings

    eligible = [f for f in findings if str(f.get("claim_type") or "") == "association_claim"]
    if not eligible:
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
            "analysis_type": f.get("analysis_type", "association"),
            "effect_size": f.get("effect_size"),
            "odds_ratio": f.get("odds_ratio"),
            "raw_statistical_text": f.get("finding_text_raw"),
        }
        for f in eligible
    ]
    req = LLMRequest(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f"Structured analytical findings:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
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
        if str(finding.get("claim_type") or "") != "association_claim":
            clone.setdefault("comparison_claim_text", None)
            updated.append(clone)
            continue
        sentence = sentence_map.get(str(finding.get("finding_id", "")).strip())
        validated = _validate_sentence(sentence, finding) if sentence else None
        clone["comparison_claim_text"] = validated or _fallback_sentence(finding)
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
    if direction == "positive" and not any(term in lowered for term in _POSITIVE_DIRECTION_TERMS):
        return None
    if direction == "negative" and not any(term in lowered for term in _NEGATIVE_DIRECTION_TERMS):
        return None

    endpoint = finding.get("endpoint")
    if not endpoint and any(word in lowered for word in _ENDPOINT_WORDS):
        return None
    if endpoint == "mortality" and "survival" in lowered and "mortality" not in lowered and "death" not in lowered:
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
    direction = str(finding.get("direction") or "unknown")
    if significant is False:
        return f"{subject} did not show a statistically significant association with {outcome}."
    directional = _directional_subject(str(finding.get("variable") or ""), direction)
    if significant is True and directional and finding.get("endpoint") == "mortality":
        return f"{directional} was associated with higher mortality risk."
    if significant is True and directional:
        return f"{directional} was associated with {outcome}."
    return f"{subject} was significantly associated with {outcome}."


def _humanize_variable(variable: str) -> str:
    cleaned = variable.replace("_", " ").strip()
    return cleaned.capitalize() if cleaned else "This variable"


def _directional_subject(variable: str, direction: str) -> str | None:
    cleaned = variable.replace("_", " ").strip().lower()
    if direction == "positive":
        if cleaned == "age":
            return "Older age"
        return f"Higher {cleaned}" if cleaned else None
    if direction == "negative":
        return f"Lower {cleaned}" if cleaned else None
    return None


def _endpoint_phrase(endpoint: Any) -> str:
    if endpoint == "mortality":
        return "mortality"
    if endpoint == "survival":
        return "survival"
    if endpoint == "primary_outcome":
        return "the primary outcome"
    return "the outcome"
