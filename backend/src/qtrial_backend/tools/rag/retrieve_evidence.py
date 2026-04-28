from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.rag.bm25_retriever import tokenize_query
from qtrial_backend.rag.models import SourceType
from qtrial_backend.tools.registry import tool


class RetrieveEvidenceParams(BaseModel):
    query: str = Field(description="Natural-language evidence retrieval query")
    top_k: int = Field(default=5, description="Maximum number of evidence chunks to return")
    min_score: float = Field(default=0.0, description="Minimum relevance score threshold")
    source_types: list[SourceType] | None = Field(
        default=None,
        description=(
            "Optional source type filter. Allowed: file, literature, tool_result, "
            "data_dictionary, user_input"
        ),
    )
    metadata_filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata key-value equality filters",
    )
    snippet_chars: int = Field(
        default=300,
        description="Max characters of chunk text to return per hit",
    )
    include_chunk_text: bool = Field(
        default=False,
        description="When true, include full chunk text for each result",
    )


@tool(
    name="retrieve_evidence",
    description=(
        "Retrieve ranked evidence chunks from the runtime evidence index. "
        "Use this to ground conclusions in previously indexed sources: "
        "data dictionaries, analyst notes, external evidence files, and prior tool outputs. "
        "Returns chunk text snippets, scores, and full provenance metadata."
    ),
    params_model=RetrieveEvidenceParams,
    category="retrieval",
)
def retrieve_evidence(params: RetrieveEvidenceParams, ctx: AgentContext) -> dict:
    if not params.query.strip():
        raise ValueError("query must be a non-empty string")

    if ctx.evidence_store.chunk_count == 0:
        return {
            "query": params.query,
            "results": [],
            "index": {
                "is_indexed": ctx.retriever.is_indexed,
                "index_size": ctx.retriever.index_size,
                "documents": ctx.evidence_store.document_count,
                "chunks": ctx.evidence_store.chunk_count,
            },
            "message": "No evidence chunks are currently indexed.",
        }

    # Ensure retriever reflects current store state.
    if ctx.retriever.index_size != ctx.evidence_store.chunk_count:
        ctx.reindex_retriever()

    hits = ctx.retrieve(
        query=params.query,
        top_k=max(1, min(params.top_k, 20)),
        min_score=params.min_score,
        source_types=params.source_types,
        metadata_filters=params.metadata_filters,
    )

    query_terms = set(tokenize_query(params.query))

    rows: list[dict[str, Any]] = []
    max_chars = max(80, params.snippet_chars)
    max_score = max([h.score for h in hits], default=0.0)
    for hit in hits:
        chunk_text = hit.chunk.text
        snippet = chunk_text if len(chunk_text) <= max_chars else chunk_text[:max_chars] + "..."
        chunk_terms = set(tokenize_query(chunk_text))
        matched_terms = sorted(list(query_terms.intersection(chunk_terms)))
        normalized_score = (hit.score / max_score) if max_score > 0 else 0.0

        evidence_card: dict[str, Any] = {
            "rank": hit.rank,
            "score": hit.score,
            "score_norm": round(normalized_score, 4),
            "matched_terms": matched_terms[:20],
            "match_count": len(matched_terms),
            "snippet": snippet,
            "source": {
                "source_id": hit.provenance.source_id,
                "source_name": hit.provenance.source_name,
                "source_type": hit.provenance.source_type,
            },
            "provenance": {
                "chunk_id": hit.provenance.chunk_id,
                "chunk_index": hit.provenance.chunk_index,
                "char_start": hit.provenance.char_start,
                "char_end": hit.provenance.char_end,
            },
            "metadata": hit.chunk.metadata,
            "citation_hint": _citation_hint(hit.chunk.metadata),
        }
        if params.include_chunk_text:
            evidence_card["chunk_text"] = chunk_text

        rows.append(
            evidence_card
        )

    return {
        "query": params.query,
        "query_terms": sorted(list(query_terms)),
        "applied_filters": {
            "source_types": params.source_types,
            "metadata_filters": params.metadata_filters,
            "min_score": params.min_score,
        },
        "index": {
            "is_indexed": ctx.retriever.is_indexed,
            "index_size": ctx.retriever.index_size,
            "documents": ctx.evidence_store.document_count,
            "chunks": ctx.evidence_store.chunk_count,
            "index_version": ctx.retrieval_metadata.get("index_version", 0),
        },
        "n_results": len(rows),
        "results": rows,
    }


def _citation_hint(metadata: dict[str, Any]) -> str | None:
    paper_id = metadata.get("paper_id")
    title = metadata.get("title")
    year = metadata.get("year")
    if paper_id and title:
        return f"{title} ({year}) [ID: {paper_id}]" if year else f"{title} [ID: {paper_id}]"
    return None
