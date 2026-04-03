from __future__ import annotations

"""
Core data models for the Q-Trial RAG subsystem.

These models represent the evidence pipeline:
    SourceDocument  →  Chunk  →  RetrievalResult (with Provenance)

All models are Pydantic BaseModels, consistent with the rest of the repo
(core/types.py, tool param models).
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Source type taxonomy
# ---------------------------------------------------------------------------

SourceType = Literal[
    "file",              # plain-text / markdown / CSV companion files
    "literature",        # papers from PubMed / Semantic Scholar
    "tool_result",       # output of a statistical or literature tool
    "data_dictionary",   # column-name → description JSON mapping
    "user_input",        # free-text provided by the analyst
]


# ---------------------------------------------------------------------------
# Source document
# ---------------------------------------------------------------------------

class SourceDocument(BaseModel):
    """An ingested evidence source before chunking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Human-readable name or filename")
    source_type: SourceType = Field(description="Category of the source")
    content: str = Field(description="Raw text content of the document")
    metadata: dict[str, Any] = Field(default_factory=dict)
    file_path: str | None = Field(
        default=None, description="Original file path, if file-based"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def char_count(self) -> int:
        return len(self.content)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A segment of a SourceDocument, sized for retrieval."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = Field(description="ID of the parent SourceDocument")
    text: str = Field(description="Chunk text content")
    index: int = Field(
        description="Zero-based position of this chunk within the source"
    )
    char_start: int = Field(
        description="Start character offset in the source content"
    )
    char_end: int = Field(
        description="End character offset in the source content"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Inherited source metadata merged with chunk-specific metadata "
            "(source_name, source_type, chunking_strategy, etc.)"
        ),
    )
    token_estimate: int | None = Field(
        default=None,
        description="Rough token count estimate (chars // 4)",
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    """Tracks the exact origin of a piece of retrieved evidence.

    Sufficient to trace any retrieved chunk back to its source document
    and character range, enabling future claim-verification workflows.
    """

    source_id: str
    source_name: str
    source_type: SourceType
    chunk_id: str
    chunk_index: int
    char_start: int
    char_end: int


# ---------------------------------------------------------------------------
# Retrieval query / result
# ---------------------------------------------------------------------------

class RetrievalQuery(BaseModel):
    """Encapsulates a retrieval request with optional filters."""

    text: str = Field(description="Query text")
    top_k: int = Field(default=5, description="Maximum results to return")
    min_score: float = Field(
        default=0.0, description="Minimum score threshold (inclusive)"
    )
    source_types: list[SourceType] | None = Field(
        default=None,
        description="Restrict results to these source types only",
    )
    metadata_filters: dict[str, Any] | None = Field(
        default=None,
        description="Key-value equality filters applied to chunk metadata",
    )


class RetrievalResult(BaseModel):
    """A single retrieval hit: a scored chunk with full provenance."""

    chunk: Chunk
    score: float = Field(description="Retrieval score (higher = more relevant)")
    rank: int = Field(description="1-based rank in the result set")
    provenance: Provenance
