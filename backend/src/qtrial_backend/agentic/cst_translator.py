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
    findings: list[str],
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
    deduped: list[str] = []
    for f in findings:
        key = f.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(f)
        if len(deduped) >= max_terms:
            break

    def _translate_one(finding: str) -> ClinicalSearchTerm:
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
                term=term,
                study_context_used=study_context,
            )
        except Exception as exc:
            return ClinicalSearchTerm(
                source_finding=finding,
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
