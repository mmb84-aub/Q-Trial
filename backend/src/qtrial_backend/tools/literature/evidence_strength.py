"""
Evidence strength scorer.

Scores a LiteratureArticle on a 0–100 scale derived from:
  - Recency (publication year):        0–40 pts
  - Study type hierarchy (OCEBM):      0–40 pts
  - Sample size (from abstract):       0–20 pts

Labels:
  >= 70  → Strong
  >= 45  → Moderate
  >= 20  → Weak
  <  20  → Insufficient
"""
from __future__ import annotations

import re
from datetime import datetime

from qtrial_backend.tools.literature.rag import LiteratureArticle
from qtrial_backend.agentic.schemas import EvidenceStrengthScore

_CURRENT_YEAR = datetime.now().year

# Study type patterns ordered from highest to lowest evidence level
_STUDY_TYPE_PATTERNS: list[tuple[str, int, str]] = [
    (r"meta.?analysis|systematic.?review", 40, "meta-analysis"),
    (r"randomized.?controlled|randomised.?controlled|rct\b", 32, "RCT"),
    (r"randomized|randomised|random.?assignment", 28, "RCT"),
    (r"cohort|prospective|longitudinal", 20, "cohort"),
    (r"case.?control|observational|cross.?sectional", 12, "observational"),
    (r"case.?report|case.?series|case.?study", 4, "case study"),
]

_SAMPLE_SIZE_RE = re.compile(
    r"\b(?:n\s*=\s*|sample\s+(?:size\s+)?(?:of\s+)?|enrolled\s+|included\s+|participants?\s*[=:]\s*)(\d[\d,]*)",
    re.IGNORECASE,
)


def _year_score(year_str: str | None) -> int:
    if not year_str:
        return 0
    try:
        year = int(str(year_str)[:4])
    except (ValueError, TypeError):
        return 0
    age = _CURRENT_YEAR - year
    if age <= 2:
        return 40
    if age <= 5:
        return 30
    if age <= 10:
        return 20
    if age <= 20:
        return 10
    return 0


def _study_type_score(text: str) -> tuple[int, str]:
    for pattern, score, label in _STUDY_TYPE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return score, label
    return 8, "unknown"


def _sample_size_score(text: str) -> tuple[int, int | None]:
    match = _SAMPLE_SIZE_RE.search(text)
    if not match:
        return 5, None
    try:
        n = int(match.group(1).replace(",", ""))
    except ValueError:
        return 5, None
    if n >= 10_000:
        return 20, n
    if n >= 1_000:
        return 15, n
    if n >= 100:
        return 10, n
    if n >= 20:
        return 5, n
    return 0, n


def score_evidence_strength(article: LiteratureArticle) -> EvidenceStrengthScore:
    """
    Compute an EvidenceStrengthScore for a single LiteratureArticle.

    Deterministic: same input always produces the same output.
    """
    search_text = f"{article.title} {article.abstract_snippet}"

    y_score = _year_score(article.year)
    st_score, study_type = _study_type_score(search_text)
    ss_score, sample_size = _sample_size_score(search_text)

    total = y_score + st_score + ss_score

    if total >= 70:
        label = "Strong"
    elif total >= 45:
        label = "Moderate"
    elif total >= 20:
        label = "Weak"
    else:
        label = "Insufficient"

    year_int: int | None = None
    if article.year:
        try:
            year_int = int(str(article.year)[:4])
        except (ValueError, TypeError):
            pass

    n_str = f"{sample_size:,}" if sample_size else "an unknown number of"
    plain = f"{label} — based on a {year_int or 'undated'} {study_type} of {n_str} patients"

    return EvidenceStrengthScore(
        score=total,
        label=label,
        plain_language=plain,
        year=year_int,
        study_type=study_type,
        sample_size=sample_size,
    )
