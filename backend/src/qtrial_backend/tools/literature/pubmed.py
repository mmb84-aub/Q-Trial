from __future__ import annotations

import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.config import settings
from qtrial_backend.tools.literature._http import get_http_client
from qtrial_backend.tools.registry import tool

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedSearchParams(BaseModel):
    query: str = Field(
        description=(
            "PubMed search query, e.g. "
            "'primary biliary cholangitis treatment outcomes'"
        )
    )
    max_results: int = Field(
        default=5, description="Number of results to return (max 10)"
    )


@tool(
    name="search_pubmed",
    description=(
        "Search PubMed for biomedical literature. Returns titles, authors, "
        "abstracts, and PMIDs for relevant papers."
    ),
    params_model=PubMedSearchParams,
    category="literature",
)
def search_pubmed(params: PubMedSearchParams, ctx: AgentContext) -> dict:
    client = get_http_client()
    api_key = settings.ncbi_api_key

    # Step 1: search for PMIDs
    search_params: dict = {
        "db": "pubmed",
        "term": params.query,
        "retmax": min(params.max_results, 10),
        "retmode": "json",
    }
    if api_key:
        search_params["api_key"] = api_key

    search_resp = client.get(ESEARCH_URL, params=search_params)
    search_resp.raise_for_status()
    search_data = search_resp.json()
    pmids = search_data.get("esearchresult", {}).get("idlist", [])

    if not pmids:
        return {"query": params.query, "results": [], "total_found": 0}

    # Step 2: fetch abstracts
    fetch_params: dict = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if api_key:
        fetch_params["api_key"] = api_key

    fetch_resp = client.get(EFETCH_URL, params=fetch_params)
    fetch_resp.raise_for_status()
    articles = _parse_pubmed_xml(fetch_resp.text)

    return {
        "query": params.query,
        "total_found": int(search_data["esearchresult"].get("count", 0)),
        "results": articles,
    }


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed efetch XML into article dicts."""
    root = ET.fromstring(xml_text)
    articles: list[dict] = []

    for article_el in root.findall(".//PubmedArticle"):
        medline = article_el.find(".//MedlineCitation")
        if medline is None:
            continue

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        art = medline.find(".//Article")
        if art is None:
            continue

        title_el = art.find("ArticleTitle")
        title = title_el.text if title_el is not None else ""

        # Authors
        authors: list[str] = []
        for author_el in art.findall(".//Author"):
            last = author_el.findtext("LastName", "")
            first = author_el.findtext("Initials", "")
            if last:
                authors.append(f"{last} {first}".strip())

        # Abstract
        abstract_parts: list[str] = []
        for abs_text in art.findall(".//AbstractText"):
            if abs_text.text:
                abstract_parts.append(abs_text.text)
        abstract = " ".join(abstract_parts)[:800]  # Truncate

        # Year
        pub_date = art.find(".//PubDate")
        year = pub_date.findtext("Year", "") if pub_date is not None else ""

        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "authors": authors[:5],  # Top 5
                "year": year,
                "abstract": abstract,
            }
        )

    return articles
