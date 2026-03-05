"""
Task 6 — Hypothesis-driven literature RAG.

Workflow:
  1. Take candidate hypotheses from the reasoning state.
  2. Auto-generate PubMed search queries from hypothesis statements +
     dataset context (disease area, key columns).
  3. Retrieve abstracts from PubMed (primary; uses NCBI API key when set).
  4. Optionally enrich with Semantic Scholar TLDR + citation counts.
  5. Return a LiteratureRAGReport with stable lit[i] citation aliases.
  6. format_literature_for_agents() renders a compact block that can be
     appended to prior_analysis_report before the final InsightSynthesisAgent
     call, enabling the LLM to cite real papers.

No embeddings, no vector DB — this is query-time retrieval grounding.
"""
from __future__ import annotations

import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field
from qtrial_backend.config import settings

# ── Pydantic models defined here (not in agentic.schemas) to avoid circular import ──

class LiteratureArticle(BaseModel):
    """A single retrieved paper from PubMed or Semantic Scholar."""
    source: Literal["pubmed", "semantic_scholar"]
    paper_id: str = Field(description="PMID or Semantic Scholar paper ID.")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: str | None = None
    abstract_snippet: str = Field(
        default="",
        description="First ≤300 chars of abstract.",
    )
    citation_count: int | None = None
    tldr: str | None = Field(
        default=None,
        description="AI-generated one-sentence summary (Semantic Scholar only).",
    )
    citation_alias: str = Field(
        default="",
        description="Stable alias e.g. 'lit[0]' for use in agent prompts.",
    )
    search_query: str = Field(
        default="",
        description="The query that retrieved this article.",
    )


class LiteratureRAGReport(BaseModel):
    """
    Aggregated results from hypothesis-driven literature retrieval (Task 6).
    Stored in FinalReportSchema.literature_report.
    """
    articles: list[LiteratureArticle] = Field(default_factory=list)
    queries_used: list[str] = Field(default_factory=list)
    total_retrieved: int = 0
    sources_used: list[str] = Field(default_factory=list)
    summary: str = ""
from qtrial_backend.tools.literature._http import get_http_client

# ── constants ─────────────────────────────────────────────────────────────────

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
S2_URL      = "https://api.semanticscholar.org/graph/v1/paper/search"

_MAX_RESULTS_PER_QUERY = 3   # papers per hypothesis query
_MAX_TOTAL             = 10  # hard cap across all queries
_ABSTRACT_SNIPPET_LEN  = 300
_QUERY_MAX_CHARS       = 120

# Stopwords to strip from hypothesis text before building a PubMed query
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "and", "or", "but", "in", "on",
    "at", "to", "for", "of", "with", "as", "by", "not", "no", "so",
    "if", "it", "its", "we", "our", "there", "between", "across", "will",
    "would", "could", "may", "might", "should", "can", "do", "does",
    "did", "has", "have", "had", "more", "less", "highly", "significantly",
    "statistically", "strong", "strongly", "significant", "association",
    "associated", "correlation", "correlated", "difference", "differences",
    "variable", "variables", "column", "columns", "dataset", "data",
    "value", "values", "result", "results", "analysis", "based", "evidence",
}


# ── query generation ──────────────────────────────────────────────────────────

def _clean_query(text: str) -> str:
    """
    Strip hypothesis boilerplate and produce a concise PubMed-friendly query.
    """
    # Remove citations like (dispatched[0]) or (top_correlations[1])
    text = re.sub(r"\([^)]*\)", "", text)
    # Remove special characters
    text = re.sub(r"[\"'(){}\[\]]", " ", text)
    # Lowercase, tokenise
    tokens = re.split(r"[\s,;:.!?/\\]+", text.lower())
    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t and len(t) > 2 and t not in _STOPWORDS]
    # Take first 12 meaningful tokens
    query = " ".join(tokens[:12])
    return query[:_QUERY_MAX_CHARS].strip()


def _queries_from_hypotheses(
    hypotheses: list[dict[str, Any]],
    preview: dict[str, Any],
    evidence: dict[str, Any],
) -> list[str]:
    """
    Generate one focused PubMed search query per hypothesis.
    Injects dataset context (disease area from column names) when available.
    """
    queries: list[str] = []

    # Try to extract a disease/context hint from column names + preview
    col_names: list[str] = list(preview.get("schema", {}).keys())
    ds_summary: str = preview.get("dataset_summary", "")

    for hypo in hypotheses[:4]:  # cap at 4 queries to save quota
        statement: str = hypo.get("statement", "")
        if not statement:
            continue

        query = _clean_query(statement)
        if not query:
            continue

        # Append disease context if we can detect it
        if any(k in " ".join(col_names).lower() for k in
               ["bili", "albumin", "cirrhosis", "hepat", "liver"]):
            if "biliary" not in query and "cholangitis" not in query:
                query = query + " biliary liver"

        # Append "clinical trial" for survival/treatment hypotheses
        for kw in ["survival", "treatment", "trt", "arm", "randomiz"]:
            if kw in statement.lower() and "clinical trial" not in query:
                query = query + " clinical trial"
                break

        queries.append(query[:_QUERY_MAX_CHARS])

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


# ── PubMed retrieval ──────────────────────────────────────────────────────────

import xml.etree.ElementTree as ET


def _pubmed_fetch(query: str, max_results: int = 3) -> list[dict]:
    """
    Search PubMed for *query* and return article dicts.
    Returns [] on any error (graceful degradation).
    """
    client = get_http_client()
    api_key = settings.ncbi_api_key

    try:
        search_p: dict = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 10),
            "retmode": "json",
        }
        if api_key:
            search_p["api_key"] = api_key

        resp = client.get(ESEARCH_URL, params=search_p)
        resp.raise_for_status()
        pmids: list[str] = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        # Small delay to respect NCBI polite rate (3 req/s without key)
        if not api_key:
            time.sleep(0.35)

        fetch_p: dict = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if api_key:
            fetch_p["api_key"] = api_key

        fetch_resp = client.get(EFETCH_URL, params=fetch_p)
        fetch_resp.raise_for_status()
        return _parse_xml(fetch_resp.text, query)

    except Exception:
        return []


def _parse_xml(xml_text: str, query: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    articles: list[dict] = []
    for art_el in root.findall(".//PubmedArticle"):
        medline = art_el.find(".//MedlineCitation")
        if medline is None:
            continue
        pmid = (medline.findtext("PMID") or "").strip()
        art = medline.find(".//Article")
        if art is None:
            continue
        title = (art.findtext("ArticleTitle") or "").strip()

        authors: list[str] = []
        for a in art.findall(".//Author"):
            last = a.findtext("LastName", "")
            ini  = a.findtext("Initials", "")
            if last:
                authors.append(f"{last} {ini}".strip())

        abstract_parts: list[str] = []
        for ab in art.findall(".//AbstractText"):
            if ab.text:
                abstract_parts.append(ab.text)
        abstract = " ".join(abstract_parts)

        year = ""
        pub_date = art.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.findtext("Year", "")

        articles.append({
            "source": "pubmed",
            "paper_id": pmid,
            "title": title,
            "authors": authors[:5],
            "year": year,
            "abstract": abstract,
            "search_query": query,
        })
    return articles


# ── Semantic Scholar enrichment (optional) ────────────────────────────────────

def _s2_fetch(query: str, max_results: int = 2) -> list[dict]:
    """
    Fetch from Semantic Scholar for *query*. Returns [] on any error.
    Used as a supplementary source when PubMed returns < expected results.
    """
    client = get_http_client()
    headers: dict = {}
    if settings.s2_api_key:
        headers["x-api-key"] = settings.s2_api_key

    try:
        resp = client.get(
            S2_URL,
            params={
                "query": query,
                "limit": min(max_results, 5),
                "fields": "title,authors,abstract,year,citationCount,tldr",
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        results: list[dict] = []
        for p in data.get("data", []):
            results.append({
                "source": "semantic_scholar",
                "paper_id": p.get("paperId", ""),
                "title": p.get("title", ""),
                "authors": [a.get("name", "") for a in p.get("authors", [])][:5],
                "year": str(p.get("year", "") or ""),
                "abstract": (p.get("abstract") or ""),
                "citation_count": p.get("citationCount"),
                "tldr": (p.get("tldr") or {}).get("text"),
                "search_query": query,
            })
        return results
    except Exception:
        return []


# ── public entry point ────────────────────────────────────────────────────────

def run_literature_rag(
    hypotheses: list[dict[str, Any]],
    preview: dict[str, Any],
    evidence: dict[str, Any],
    use_semantic_scholar: bool = True,
    max_per_query: int = _MAX_RESULTS_PER_QUERY,
    max_total: int = _MAX_TOTAL,
) -> LiteratureRAGReport:
    """
    Run hypothesis-driven literature retrieval.

    Parameters
    ----------
    hypotheses : list of dicts with 'statement' key (from reasoning_state)
    preview    : dataset preview dict (used for disease-context injection)
    evidence   : evidence dict (not used yet but kept for future query enrichment)
    use_semantic_scholar : also query S2 when PubMed returns < expected results
    max_per_query : max papers per hypothesis query
    max_total     : hard cap on total articles in report

    Returns
    -------
    LiteratureRAGReport with stable lit[i] citation aliases.
    """
    if not hypotheses:
        return LiteratureRAGReport(summary="No hypotheses — literature RAG skipped.")

    queries = _queries_from_hypotheses(hypotheses, preview, evidence)
    if not queries:
        return LiteratureRAGReport(summary="Could not generate search queries.")

    all_raw: list[dict] = []
    seen_ids: set[str] = set()

    for query in queries:
        if len(all_raw) >= max_total:
            break

        raw = _pubmed_fetch(query, max_results=max_per_query)

        # Supplement with Semantic Scholar if PubMed returned nothing
        if not raw and use_semantic_scholar:
            raw = _s2_fetch(query, max_results=max_per_query)

        for r in raw:
            pid = r.get("paper_id", "")
            if pid and pid in seen_ids:
                continue
            if pid:
                seen_ids.add(pid)
            all_raw.append(r)
            if len(all_raw) >= max_total:
                break

    # Convert to typed LiteratureArticle with stable aliases
    articles: list[LiteratureArticle] = []
    sources_used: set[str] = set()
    for i, raw in enumerate(all_raw):
        abstract = raw.get("abstract", "")
        snippet = abstract[:_ABSTRACT_SNIPPET_LEN]
        if len(abstract) > _ABSTRACT_SNIPPET_LEN:
            snippet += "…"
        article = LiteratureArticle(
            source=raw["source"],
            paper_id=raw.get("paper_id", ""),
            title=raw.get("title", ""),
            authors=raw.get("authors", []),
            year=raw.get("year"),
            abstract_snippet=snippet,
            citation_count=raw.get("citation_count"),
            tldr=raw.get("tldr"),
            citation_alias=f"lit[{i}]",
            search_query=raw.get("search_query", ""),
        )
        articles.append(article)
        sources_used.add(raw["source"])

    if not articles:
        return LiteratureRAGReport(
            queries_used=queries,
            summary="Literature search returned no results.",
        )

    summary = (
        f"{len(articles)} article(s) retrieved from "
        f"{', '.join(sorted(sources_used))} "
        f"across {len(queries)} query/queries."
    )

    return LiteratureRAGReport(
        articles=articles,
        queries_used=queries,
        total_retrieved=len(articles),
        sources_used=sorted(sources_used),
        summary=summary,
    )


# ── formatting for agent injection ───────────────────────────────────────────

def format_literature_for_agents(report: LiteratureRAGReport) -> str:
    """
    Render the literature report as a compact text block to inject into
    the InsightSynthesisAgent prompt as ``LITERATURE CONTEXT``.

    Each article is given a stable ``lit[i]`` alias that the LLM must use
    when citing it in key_findings or recommended_next_analyses.
    """
    if not report.articles:
        return ""

    lines: list[str] = [
        "═══════════════════════════════════════════════════════════",
        "LITERATURE CONTEXT (Task 6 — hypothesis-driven RAG retrieval)",
        "═══════════════════════════════════════════════════════════",
        "The following papers were retrieved to ground your synthesis.",
        "When citing a finding supported by one of these papers, append",
        "its citation alias (e.g. lit[0]) to the evidence_citation field.",
        "",
    ]

    for art in report.articles:
        author_str = ", ".join(art.authors[:3])
        if len(art.authors) > 3:
            author_str += " et al."
        year_str = f" ({art.year})" if art.year else ""
        pmid_str = f"PMID {art.paper_id}" if art.source == "pubmed" else f"S2:{art.paper_id[:8]}"
        lines.append(f"[{art.citation_alias}] {pmid_str}{year_str}")
        lines.append(f"  Title   : {art.title}")
        if author_str:
            lines.append(f"  Authors : {author_str}")
        if art.tldr:
            lines.append(f"  Summary : {art.tldr}")
        elif art.abstract_snippet:
            lines.append(f"  Abstract: {art.abstract_snippet}")
        if art.citation_count is not None:
            lines.append(f"  Citations: {art.citation_count}")
        lines.append("")

    lines.append(
        "CITATION RULES: Only cite papers listed above. "
        "Do NOT invent PMIDs or paper titles. "
        "Use the exact alias (lit[0], lit[1], …)."
    )
    lines.append(
        "═══════════════════════════════════════════════════════════"
    )
    return "\n".join(lines)
