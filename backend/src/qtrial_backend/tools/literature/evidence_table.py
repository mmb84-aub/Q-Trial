from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool

# ── Regex patterns for extracting common clinical metrics from abstracts ──────
_HR_PATTERN = re.compile(r"HR[:\s=]*([0-9]+\.[0-9]+)", re.IGNORECASE)
_OR_PATTERN = re.compile(r"\bOR[:\s=]*([0-9]+\.[0-9]+)", re.IGNORECASE)
_RR_PATTERN = re.compile(r"\bRR[:\s=]*([0-9]+\.[0-9]+)", re.IGNORECASE)
_CI_PATTERN = re.compile(r"95%\s*CI[:\s]*([0-9]+\.[0-9]+)[–\-,\s]+([0-9]+\.[0-9]+)", re.IGNORECASE)
_P_PATTERN = re.compile(r"p[\s=<>]+([0-9]+\.[0-9]+)", re.IGNORECASE)
_N_PATTERN = re.compile(r"\bn\s*=\s*([0-9,]+)", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _extract_metrics(text: str) -> dict:
    """Best-effort extraction of quantitative metrics from unstructured text."""
    if not text:
        return {}
    metrics: dict = {}

    m = _HR_PATTERN.search(text)
    if m:
        metrics["hazard_ratio"] = float(m.group(1))

    m = _OR_PATTERN.search(text)
    if m:
        metrics["odds_ratio"] = float(m.group(1))

    m = _RR_PATTERN.search(text)
    if m:
        metrics["risk_ratio"] = float(m.group(1))

    ci_matches = _CI_PATTERN.findall(text)
    if ci_matches:
        lo, hi = ci_matches[0]
        metrics["ci_95"] = [float(lo), float(hi)]

    p_matches = _P_PATTERN.findall(text)
    if p_matches:
        metrics["p_values_found"] = [float(v) for v in p_matches[:3]]

    n_matches = _N_PATTERN.findall(text)
    if n_matches:
        # Take largest n found as total sample size
        ns = [int(v.replace(",", "")) for v in n_matches]
        metrics["sample_size"] = max(ns)

    return metrics


class EvidenceTableParams(BaseModel):
    papers: list[dict] = Field(
        description=(
            "List of paper dictionaries, each containing at minimum a 'title' and one of "
            "'abstract', 'summary', or 'snippet' field. Additional fields like 'pmid', "
            "'year', 'authors' are included if available from search tool results."
        )
    )
    outcome_keywords: Optional[list[str]] = Field(
        default=None,
        description=(
            "Keywords to look for in abstracts when labelling primary outcomes "
            "(e.g. ['survival', 'mortality', 'transplant']). Defaults to common clinical terms."
        ),
    )


_DEFAULT_OUTCOME_KEYWORDS = [
    "survival", "mortality", "death", "transplant", "relapse",
    "response", "remission", "progression", "event-free", "recurrence",
]


@tool(
    name="evidence_table_builder",
    description=(
        "Structure a list of paper search results into a comparative evidence table. "
        "Extracts: sample size, year, effect measures (HR, OR, RR), 95% CI, p-values, "
        "and detected outcome keywords from abstracts. "
        "Call this after collecting papers via search_pubmed / search_semantic_scholar "
        "to enable direct comparison with your dataset findings."
    ),
    params_model=EvidenceTableParams,
    category="literature",
)
def evidence_table_builder(params: EvidenceTableParams, ctx: AgentContext) -> dict:
    keywords = params.outcome_keywords or _DEFAULT_OUTCOME_KEYWORDS
    rows: list[dict] = []

    for paper in params.papers:
        # Collect text for extraction
        text_parts = [
            paper.get("abstract", ""),
            paper.get("summary", ""),
            paper.get("snippet", ""),
            paper.get("title", ""),
        ]
        full_text = " ".join(str(p) for p in text_parts if p)

        metrics = _extract_metrics(full_text)

        # Detect outcome keywords
        found_outcomes = [
            kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", full_text, re.IGNORECASE)
        ]

        # Year extraction
        year = paper.get("year") or paper.get("publicationDate", "")
        if not year:
            year_match = _YEAR_PATTERN.search(full_text)
            year = int(year_match.group()) if year_match else None
        else:
            try:
                year = int(str(year)[:4])
            except (ValueError, TypeError):
                year = None

        row: dict = {
            "title": paper.get("title", "Unknown title"),
            "year": year,
            "source": paper.get("source", paper.get("journal", "Unknown source")),
            "pmid": paper.get("pmid", paper.get("paperId", None)),
            "authors": paper.get("authors", None),
            "sample_size": metrics.get("sample_size"),
            "primary_outcomes_detected": found_outcomes,
            "extracted_metrics": metrics,
            "has_hr": "hazard_ratio" in metrics,
            "has_or": "odds_ratio" in metrics,
            "has_rr": "risk_ratio" in metrics,
        }
        rows.append(row)

    # Summary statistics
    n_with_hr = sum(r["has_hr"] for r in rows)
    n_with_n = sum(r["sample_size"] is not None for r in rows)
    years = [r["year"] for r in rows if r["year"]]

    return {
        "n_papers": len(rows),
        "n_with_hazard_ratio": n_with_hr,
        "n_with_sample_size": n_with_n,
        "year_range": [min(years), max(years)] if years else None,
        "table": rows,
        "instructions": (
            "Use this table to compare your dataset findings against published benchmarks. "
            "Register papers you cite via citation_manager before including them in your report."
        ),
    }
