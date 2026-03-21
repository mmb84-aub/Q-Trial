"""
Cochrane Library retriever.

Queries the Cochrane CDSR (Cochrane Database of Systematic Reviews) REST API
and returns results in the same dict shape as _parse_pubmed_xml output so the
LiteratureGrounder can treat all sources uniformly.
"""
from __future__ import annotations

from qtrial_backend.tools.literature._http import get_http_client

# Cochrane REST API base — searches CDSR titles/abstracts
_COCHRANE_SEARCH_URL = "https://www.cochranelibrary.com/api/search/results"


def cochrane_fetch(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the Cochrane Library for systematic reviews matching *query*.

    Returns a list of dicts with keys:
      pmid, title, authors, year, abstract

    Falls back to an empty list on any HTTP or parse error so the pipeline
    can continue with remaining sources.
    """
    client = get_http_client()
    try:
        resp = client.get(
            _COCHRANE_SEARCH_URL,
            params={
                "q": query,
                "rows": min(max_results, 10),
                "start": 0,
                "type": "review",
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results: list[dict] = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            {
                "pmid": item.get("doi", item.get("id", "")),
                "title": item.get("title", ""),
                "authors": [a.get("name", "") for a in item.get("authors", [])][:5],
                "year": str(item.get("publicationDate", ""))[:4],
                "abstract": item.get("abstract", "")[:800],
            }
        )
    return results
