from __future__ import annotations

"""
Retriever interface (abstract base class).

Every retrieval strategy — sparse (BM25), dense (embedding-based),
hybrid, reranked — implements this ABC so the rest of the system
can call retrieval without caring about the underlying algorithm.

The contract is intentionally small:
    index()     — build or rebuild the search index from chunks
    retrieve()  — answer a RetrievalQuery and return scored results
    is_indexed  — whether the index is ready for queries

Provenance construction is handled by a shared helper because the
chunk metadata already carries everything needed (set by the Chunker).
"""

from abc import ABC, abstractmethod

from qtrial_backend.rag.models import (
    Chunk,
    Provenance,
    RetrievalQuery,
    RetrievalResult,
)


class Retriever(ABC):
    """Abstract retrieval interface.

    Subclasses must implement ``index`` and ``retrieve``.
    Provenance building is provided for free via ``_build_provenance``.
    """

    # ── Core interface ────────────────────────────────────────────────

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Build (or fully replace) the search index from *chunks*.

        After this call, ``is_indexed`` must return ``True`` and
        ``retrieve`` must be functional.
        """

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Execute a retrieval query and return ranked results.

        Implementations must respect:
        - ``query.top_k``          — maximum number of results
        - ``query.min_score``      — minimum score threshold
        - ``query.source_types``   — restrict to matching source types
        - ``query.metadata_filters`` — key-value equality on chunk metadata
        """

    @property
    @abstractmethod
    def is_indexed(self) -> bool:
        """True if the index has been built and is ready for queries."""

    @property
    @abstractmethod
    def index_size(self) -> int:
        """Number of chunks currently in the index."""

    # ── Shared helper ─────────────────────────────────────────────────

    @staticmethod
    def _build_provenance(chunk: Chunk) -> Provenance:
        """Construct a Provenance record from chunk metadata.

        The Chunker propagates ``source_name`` and ``source_type`` into
        every chunk's metadata dict, so this works for any retriever
        without needing a reference back to the EvidenceStore.
        """
        return Provenance(
            source_id=chunk.source_id,
            source_name=chunk.metadata.get("source_name", "unknown"),
            source_type=chunk.metadata.get("source_type", "file"),
            chunk_id=chunk.id,
            chunk_index=chunk.index,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )

    @staticmethod
    def _apply_filters(
        chunks: list[Chunk],
        query: RetrievalQuery,
    ) -> set[str]:
        """Return the set of chunk IDs that pass the query's filters.

        Shared across retriever implementations so filtering logic is
        consistent regardless of scoring strategy.
        """
        passing: set[str] = set()
        for chunk in chunks:
            if query.source_types:
                chunk_type = chunk.metadata.get("source_type")
                if chunk_type not in query.source_types:
                    continue
            if query.metadata_filters:
                if not all(
                    chunk.metadata.get(k) == v
                    for k, v in query.metadata_filters.items()
                ):
                    continue
            passing.add(chunk.id)
        return passing
