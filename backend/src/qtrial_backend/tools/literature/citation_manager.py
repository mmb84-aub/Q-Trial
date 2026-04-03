from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class CitationManagerParams(BaseModel):
    action: str = Field(
        description=(
            "Action to perform: "
            "'register' — store a paper before citing it; "
            "'list' — show all registered citations; "
            "'check' — verify a paper ID is registered (returns bool)."
        )
    )
    paper_id: Optional[str] = Field(
        default=None,
        description=(
            "Unique identifier for the paper: PubMed PMID, Semantic Scholar paper ID, "
            "or DOI. Required for 'register' and 'check' actions."
        ),
    )
    title: Optional[str] = Field(
        default=None,
        description="Paper title. Required for 'register'.",
    )
    authors: Optional[str] = Field(
        default=None,
        description="Authors string (e.g. 'Smith J, Jones A'). Used for 'register'.",
    )
    year: Optional[int] = Field(
        default=None,
        description="Publication year. Used for 'register'.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Journal or source name. Used for 'register'.",
    )
    key_finding: Optional[str] = Field(
        default=None,
        description=(
            "One-sentence summary of the finding you intend to cite. "
            "Store this to keep your evidence traceable."
        ),
    )


@tool(
    name="citation_manager",
    description=(
        "Manage the citation registry to prevent hallucinated references. "
        "Before citing any paper in your report, you MUST register it here using "
        "the paper ID returned by search_pubmed or search_semantic_scholar. "
        "Only papers registered via this tool may be cited — this enforces that every "
        "reference has a verified, traceable source. "
        "Actions: register (store a paper), list (show all stored), check (verify ID is registered)."
    ),
    params_model=CitationManagerParams,
    category="literature",
)
def citation_manager(params: CitationManagerParams, ctx: AgentContext) -> dict:
    store = ctx.citation_store
    action = params.action.lower()

    if action == "register":
        if not params.paper_id:
            raise ValueError("paper_id is required for 'register' action.")
        if not params.title:
            raise ValueError("title is required for 'register' action.")

        key = str(params.paper_id).strip()
        if key in store:
            return {
                "action": "register",
                "status": "already_registered",
                "paper_id": key,
                "message": f"Paper '{key}' was already registered.",
                "entry": store[key],
            }

        entry = {
            "paper_id": key,
            "title": params.title,
            "authors": params.authors,
            "year": params.year,
            "source": params.source,
            "key_finding": params.key_finding,
        }
        store[key] = entry
        return {
            "action": "register",
            "status": "registered",
            "paper_id": key,
            "message": (
                f"Paper registered. You may now cite it in the report as: "
                f"{params.authors or 'Authors'} ({params.year or 'n.d.'}). {params.title}."
            ),
            "n_total_registered": len(store),
        }

    elif action == "list":
        return {
            "action": "list",
            "n_registered": len(store),
            "citations": list(store.values()),
            "reminder": (
                "Only cite papers present in this list. "
                "Do not fabricate references or cite papers not registered here."
            ),
        }

    elif action == "check":
        if not params.paper_id:
            raise ValueError("paper_id is required for 'check' action.")
        key = str(params.paper_id).strip()
        is_registered = key in store
        return {
            "action": "check",
            "paper_id": key,
            "is_registered": is_registered,
            "message": (
                f"Paper '{key}' IS registered — safe to cite."
                if is_registered
                else f"Paper '{key}' is NOT registered — register it before citing."
            ),
        }

    else:
        raise ValueError(
            f"action must be 'register', 'list', or 'check'. Got: '{params.action}'"
        )
