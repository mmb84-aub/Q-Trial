"""
Literature Grounder.

Wraps PubMed (existing), Cochrane Library (new), and ClinicalTrials.gov (new)
into a single pipeline that:
  1. Accepts Clinical Search Terms (CSTs) produced by cst_translator.
  2. Queries each source with per-source rate limiting and exponential backoff.
  3. Caches results within a session so identical queries never hit the API twice.
  4. Assigns Grounding_Status (Supported / Contradicted / Novel) via LLM call.
  5. Scores Evidence_Strength for each retrieved article.
  6. Returns a list of GroundedFinding objects.
"""
from __future__ import annotations

import json
import time
from typing import Any

from qtrial_backend.agentic.schemas import (
    ClinicalSearchTerm,
    EvidenceStrengthScore,
    GroundedFinding,
    LiteratureQueryRecord,
)
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName
from qtrial_backend.tools.literature.cochrane import cochrane_fetch
from qtrial_backend.tools.literature.clinicaltrials import clinicaltrials_fetch
from qtrial_backend.tools.literature.evidence_strength import score_evidence_strength
from qtrial_backend.tools.literature.rag import LiteratureArticle

# Default minimum delays between successive calls per source (seconds)
_DEFAULT_DELAYS: dict[str, float] = {
    "pubmed": 0.35,
    "cochrane": 1.0,
    "clinicaltrials": 0.5,
    "semantic_scholar": 0.5,
}

_GROUNDING_SYSTEM = """\
You are a clinical evidence reviewer. Given a statistical finding and a list of \
literature abstracts, decide whether the finding is:
  - "Supported": at least one abstract corroborates the direction and magnitude.
  - "Contradicted": at least one abstract directly contradicts the finding.
  - "Novel": no abstract addresses the finding.

Return ONLY a JSON object: {"status": "Supported"|"Contradicted"|"Novel", "rationale": "..."}
No markdown, no extra keys.
"""


def _pubmed_fetch_raw(query: str, max_results: int = 5) -> list[dict]:
    """Thin wrapper around the existing PubMed fetcher returning raw dicts."""
    from qtrial_backend.tools.literature.rag import _pubmed_fetch  # type: ignore[attr-defined]
    try:
        return _pubmed_fetch(query, max_results=max_results)
    except Exception:
        return []


def _to_literature_article(raw: dict, source: str, alias: str) -> LiteratureArticle:
    abstract = raw.get("abstract", "")
    snippet = abstract[:300]
    if len(abstract) > 300:
        snippet += "…"
    return LiteratureArticle(
        source=source if source in ("pubmed", "semantic_scholar") else "pubmed",
        paper_id=raw.get("pmid", raw.get("nct_id", "")),
        title=raw.get("title", ""),
        authors=raw.get("authors", []),
        year=raw.get("year"),
        abstract_snippet=snippet,
        citation_alias=alias,
        search_query=raw.get("search_query", ""),
    )


class LiteratureGrounder:
    """
    Session-scoped pipeline for multi-source literature validation.

    Instantiate once per analysis run. The session cache persists for the
    lifetime of the instance.
    """

    def __init__(
        self,
        provider: ProviderName = "gemini",
        min_delays: dict[str, float] | None = None,
        max_results_per_source: int = 5,
    ) -> None:
        self.provider = provider
        self.min_delays = {**_DEFAULT_DELAYS, **(min_delays or {})}
        self.max_results = max_results_per_source
        # Cache keyed by (source, query_string) → list[dict]
        self._cache: dict[tuple[str, str], list[dict]] = {}
        self._last_call: dict[str, float] = {}
        self.query_records: list[LiteratureQueryRecord] = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rate_limit(self, source: str) -> None:
        delay = self.min_delays.get(source, 0.35)
        last = self._last_call.get(source, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_call[source] = time.monotonic()

    def _fetch_with_backoff(
        self, source: str, query: str
    ) -> tuple[list[dict], bool, str | None]:
        """
        Fetch from *source* with exponential backoff (1s/2s/4s, max 3 retries).
        Returns (results, cached, error_message).
        """
        cache_key = (source, query)
        if cache_key in self._cache:
            return self._cache[cache_key], True, None

        fetcher = {
            "pubmed": lambda q: _pubmed_fetch_raw(q, self.max_results),
            "cochrane": lambda q: cochrane_fetch(q, self.max_results),
            "clinicaltrials": lambda q: clinicaltrials_fetch(q, self.max_results),
        }.get(source)

        if fetcher is None:
            return [], False, f"Unknown source: {source}"

        last_error: str | None = None
        for attempt, wait in enumerate([1, 2, 4]):
            try:
                self._rate_limit(source)
                results = fetcher(query)
                self._cache[cache_key] = results
                return results, False, None
            except Exception as exc:
                last_error = str(exc)
                if attempt < 2:
                    time.sleep(wait)

        return [], False, last_error

    def _assign_grounding_status(
        self, finding: str, articles: list[LiteratureArticle]
    ) -> tuple[str, str]:
        """LLM call to assign Supported / Contradicted / Novel."""
        if not articles:
            return "Novel", "No literature retrieved for this finding."

        abstracts_block = "\n\n".join(
            f"[{a.citation_alias}] {a.title} ({a.year})\n{a.abstract_snippet}"
            for a in articles[:8]
        )
        user = (
            f"Finding: {finding}\n\n"
            f"Literature abstracts:\n{abstracts_block}\n\n"
            'Return JSON: {"status": "...", "rationale": "..."}'
        )
        client = get_client(self.provider)
        req = LLMRequest(
            system_prompt=_GROUNDING_SYSTEM,
            user_prompt=user,
            payload={"temperature": 0},
        )
        try:
            resp = client.generate(req)
            raw = resp.text.strip().strip("```json").strip("```").strip()
            data = json.loads(raw)
            status = data.get("status", "Novel")
            if status not in ("Supported", "Contradicted", "Novel"):
                status = "Novel"
            return status, data.get("rationale", "")
        except Exception:
            return "Novel", "Grounding status could not be determined."

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self, csts: list[ClinicalSearchTerm]
    ) -> list[GroundedFinding]:
        """
        Run the full validation pipeline for a list of ClinicalSearchTerms.
        Returns one GroundedFinding per CST.
        """
        grounded: list[GroundedFinding] = []
        article_counter = 0

        for cst in csts:
            # Skip findings where CST translation failed
            if cst.translation_failed:
                grounded.append(
                    GroundedFinding(
                        finding_text=cst.source_finding,
                        grounding_status="Novel",
                        literature_skipped=True,
                        literature_skip_note=cst.failure_note,
                    )
                )
                continue

            all_articles: list[LiteratureArticle] = []

            for source in ("pubmed", "cochrane", "clinicaltrials"):
                raw_results, cached, error = self._fetch_with_backoff(source, cst.term)

                self.query_records.append(
                    LiteratureQueryRecord(
                        source=source,  # type: ignore[arg-type]
                        query_string=cst.term,
                        results_count=len(raw_results),
                        cached=cached,
                        error=error,
                    )
                )

                for raw in raw_results:
                    alias = f"lit[{article_counter}]"
                    article_counter += 1
                    article = _to_literature_article(raw, source, alias)
                    all_articles.append(article)

            # Assign grounding status
            status, _rationale = self._assign_grounding_status(
                cst.source_finding, all_articles
            )

            # Score evidence strength per article
            scored_articles = all_articles[:5]  # top 5 for the finding card
            best_strength: EvidenceStrengthScore | None = None
            if scored_articles:
                scores = [score_evidence_strength(a) for a in scored_articles]
                best_strength = max(scores, key=lambda s: s.score)

            novel_statement: str | None = None
            if status == "Novel":
                novel_statement = (
                    "No precedent for this finding was identified in PubMed, "
                    "Cochrane Library, or ClinicalTrials.gov. This finding may "
                    "be a candidate for publication or further investigation."
                )

            grounded.append(
                GroundedFinding(
                    finding_text=cst.source_finding,
                    grounding_status=status,  # type: ignore[arg-type]
                    citations=scored_articles,
                    evidence_strength=best_strength if status != "Novel" else None,
                    novel_statement=novel_statement,
                )
            )

        return grounded
