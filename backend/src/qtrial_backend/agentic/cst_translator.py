"""
Clinical Search Term (CST) Translator.

Translates statistical findings into clinically phrased query strings suitable
for PubMed, Cochrane Library, and ClinicalTrials.gov searches.

Rules enforced by the LLM prompt:
  - No raw statistical values (r=, p=, p<, chi-square, t-test, F-statistic,
    z-score, Pearson) as primary search content.
  - Study_Context is incorporated to ensure domain relevance.
  - Output is a PubMed-compatible search string.
"""
from __future__ import annotations

import json
from typing import Any

from qtrial_backend.agentic.schemas import ClinicalSearchTerm
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName

_SYSTEM_PROMPT = """\
You are a clinical research librarian. Your task is to convert a statistical \
finding into a clinically phrased PubMed search string.

Rules:
1. Do NOT include raw statistical values such as correlation coefficients \
(r=, r<, r>), p-values (p=, p<, p>), test statistic names (chi-square, \
t-test, F-statistic, z-score, Pearson, Spearman), or numeric thresholds as \
the primary search content.
2. Use the study context to ensure the query is relevant to the clinical domain.
3. Output a concise, PubMed-compatible search string of 3–8 clinical terms.
4. Return ONLY a JSON object with a single key "term" whose value is the \
search string. No markdown, no explanation.

Example:
  Finding: "Pearson r=0.72 (p<0.001) between bilirubin and prothrombin time"
  Study context: "Phase III RCT of D-penicillamine in primary biliary cirrhosis"
  Output: {"term": "bilirubin prothrombin time primary biliary cirrhosis liver function"}
"""


def translate_findings_to_cst(
    findings: list[str | dict[str, Any]],
    study_context: str,
    provider: ProviderName = "gemini",
    max_terms: int = 10,
) -> list[ClinicalSearchTerm]:
    """
    Translate each statistical finding into a ClinicalSearchTerm.

    Capped at *max_terms* findings (most informative first).
    Translations run in parallel (up to 8 threads) to reduce wall time.

    On LLM failure for any individual finding, sets translation_failed=True
    and attaches a failure_note; processing continues for remaining findings.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = get_client(provider)
    # Deduplicate and cap
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for f in findings:
        payload = _coerce_finding_payload(f)
        key = (payload["plain"] or payload["raw"]).strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(payload)
        if len(deduped) >= max_terms:
            break

    def _base_cst_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_finding_raw": payload["raw"],
            "source_finding_plain": payload["plain"] or payload["raw"],
            "comparison_claim_text": payload.get("comparison_claim_text") or None,
            "finding_category": payload["finding_category"],
            "claim_type": payload["claim_type"],
            "variable": payload.get("variable") or None,
            "endpoint": payload.get("endpoint") or None,
            "direction": payload.get("direction") or "unknown",
            "direction_label": payload.get("direction_label") or None,
            "significant": payload.get("significant"),
            "significance": payload.get("significance") or "unclear",
            "p_value": payload.get("p_value"),
            "effect_size": payload.get("effect_size"),
            "effect_size_label": payload.get("effect_size_label") or None,
            "test_type": payload.get("test_type") or None,
            "confidence_warning": payload.get("confidence_warning") or None,
            "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        }

    def _translate_one(payload: dict[str, Any]) -> ClinicalSearchTerm:
        finding = payload["plain"] or payload["raw"]
        user_prompt = (
            f"Study context: {study_context}\n\n"
            f"Statistical finding: {finding}\n\n"
            "Return the clinical search term as JSON: {{\"term\": \"...\"}}"
        )
        req = LLMRequest(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            payload={"temperature": 0},
        )
        try:
            resp = client.generate(req)
            raw = resp.text.strip().strip("```json").strip("```").strip()
            data = json.loads(raw)
            term = data.get("term", "").strip()
            if not term:
                raise ValueError("Empty term returned")
            return ClinicalSearchTerm(
                source_finding=finding,
                **_base_cst_kwargs(payload),
                term=term,
                study_context_used=study_context,
            )
        except Exception as exc:
            return ClinicalSearchTerm(
                source_finding=finding,
                **_base_cst_kwargs(payload),
                term="",
                study_context_used=study_context,
                translation_failed=True,
                failure_note=(
                    f"Literature grounding was not performed for this finding "
                    f"because the search term translation step encountered an error: {exc}"
                ),
            )

    # Preserve input order in results
    results: list[ClinicalSearchTerm | None] = [None] * len(deduped)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_translate_one, f): i for i, f in enumerate(deduped)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    return [r for r in results if r is not None]


def _coerce_finding_payload(finding: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(finding, str):
        return {
            "raw": finding,
            "plain": finding,
            "finding_category": "analytical",
            "claim_type": "association_claim",
            "direction": "unknown",
            "significance": "unclear",
        }
    raw = str(finding.get("finding_text_raw") or finding.get("raw") or finding.get("finding_text") or "").strip()
    plain = str(finding.get("finding_text_plain") or finding.get("plain") or finding.get("finding_text") or raw).strip()
    comparison_claim = str(
        finding.get("comparison_claim_text") or finding.get("comparison_claim") or ""
    ).strip()
    category = str(finding.get("finding_category") or "analytical").strip() or "analytical"
    claim_type = str(finding.get("claim_type") or "association_claim").strip() or "association_claim"
    p_value = finding.get("p_value", finding.get("adjusted_p_value", finding.get("raw_p_value")))
    effect_size_label = finding.get("effect_size_label")
    effect_size = finding.get("effect_size")
    if finding.get("odds_ratio") is not None:
        effect_size = finding.get("odds_ratio")
        effect_size_label = "odds_ratio"
    significant = finding.get("significant_after_correction", finding.get("significant"))
    return {
        "raw": raw or plain,
        "plain": plain or raw,
        "comparison_claim_text": comparison_claim,
        "finding_category": category,
        "claim_type": claim_type,
        "variable": finding.get("variable"),
        "endpoint": finding.get("endpoint"),
        "direction": finding.get("direction") or "unknown",
        "direction_label": finding.get("direction_label"),
        "significant": significant if isinstance(significant, bool) else None,
        "significance": (
            "significant"
            if significant is True
            else "not_significant"
            if significant is False
            else str(finding.get("significance") or "unclear")
        ),
        "p_value": p_value,
        "effect_size": effect_size,
        "effect_size_label": effect_size_label,
        "test_type": finding.get("test_type") or finding.get("test_used"),
        "confidence_warning": finding.get("confidence_warning"),
        "metadata": finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {},
    }
