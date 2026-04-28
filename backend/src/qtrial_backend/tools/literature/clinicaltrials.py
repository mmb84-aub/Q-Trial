"""
ClinicalTrials.gov retriever.

Queries the ClinicalTrials.gov API v2 (/api/v2/studies) and returns results
in the same dict shape as _parse_pubmed_xml output.
"""
from __future__ import annotations

from qtrial_backend.tools.literature._http import get_http_client

_CT_SEARCH_URL = "https://clinicaltrials.gov/api/v2/studies"


def clinicaltrials_fetch(query: str, max_results: int = 5) -> list[dict]:
    """
    Search ClinicalTrials.gov for studies matching *query*.

    Returns a list of dicts with keys:
      pmid, title, authors, year, abstract

    Falls back to an empty list on any HTTP or parse error.
    """
    client = get_http_client()
    try:
        resp = client.get(
            _CT_SEARCH_URL,
            params={
                "query.term": query,
                "pageSize": min(max_results, 10),
                "format": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results: list[dict] = []
    for study in data.get("studies", [])[:max_results]:
        proto = study.get("protocolSection", {})
        id_module = proto.get("identificationModule", {})
        desc_module = proto.get("descriptionModule", {})
        status_module = proto.get("statusModule", {})
        contacts_module = proto.get("contactsLocationsModule", {})

        nct_id = id_module.get("nctId", "")
        title = id_module.get("briefTitle", "")
        abstract = desc_module.get("briefSummary", "")[:800]

        # Start year from primary completion or start date
        start_date = status_module.get("startDateStruct", {}).get("date", "")
        year = start_date[:4] if start_date else ""

        # Investigators as "authors"
        investigators = contacts_module.get("overallOfficials", [])
        authors = [
            inv.get("name", "") for inv in investigators[:5] if inv.get("name")
        ]

        results.append(
            {
                "pmid": nct_id,
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
            }
        )
    return results
