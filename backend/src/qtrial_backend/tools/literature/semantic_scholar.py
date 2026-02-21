from __future__ import annotations

from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.literature._http import get_http_client
from qtrial_backend.tools.registry import tool

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarSearchParams(BaseModel):
    query: str = Field(description="Search query for academic papers")
    max_results: int = Field(
        default=5, description="Number of results (max 10)"
    )
    year_range: str | None = Field(
        default=None, description="Year range filter, e.g. '2020-2025'"
    )


@tool(
    name="search_semantic_scholar",
    description=(
        "Search Semantic Scholar for academic papers. Returns titles, "
        "authors, abstracts, citation counts, and TLDRs."
    ),
    params_model=SemanticScholarSearchParams,
    category="literature",
)
def search_semantic_scholar(
    params: SemanticScholarSearchParams, ctx: AgentContext
) -> dict:
    client = get_http_client()

    query_params: dict = {
        "query": params.query,
        "limit": min(params.max_results, 10),
        "fields": "title,authors,abstract,year,citationCount,tldr",
    }
    if params.year_range:
        query_params["year"] = params.year_range

    resp = client.get(S2_SEARCH_URL, params=query_params)
    resp.raise_for_status()
    data = resp.json()

    results: list[dict] = []
    for paper in data.get("data", []):
        results.append(
            {
                "title": paper.get("title"),
                "authors": [
                    a.get("name") for a in paper.get("authors", [])
                ][:5],
                "year": paper.get("year"),
                "abstract": (paper.get("abstract") or "")[:500],
                "citation_count": paper.get("citationCount"),
                "tldr": (
                    paper.get("tldr", {}).get("text")
                    if paper.get("tldr")
                    else None
                ),
            }
        )

    return {
        "query": params.query,
        "total_found": data.get("total", 0),
        "results": results,
    }
