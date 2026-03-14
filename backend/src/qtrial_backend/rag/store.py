from __future__ import annotations

"""
In-memory evidence store.

Holds the corpus of SourceDocuments and their Chunks, provides lookup,
filtering, and automatic chunking on ingestion.  Decoupled from any
specific retrieval strategy — a Retriever reads chunks from the store
and maintains its own index.

Typical usage::

    store   = EvidenceStore()
    doc     = ingest_file("notes.md")
    chunks  = store.add_document(doc)          # auto-chunked
    retriever.index(store.all_chunks())        # hand off to retriever
"""

from typing import Any

from qtrial_backend.rag.chunking import Chunker, ChunkingConfig
from qtrial_backend.rag.models import Chunk, SourceDocument, SourceType


class EvidenceStore:
    """In-memory corpus of SourceDocuments and their derived Chunks."""

    def __init__(self, chunking_config: ChunkingConfig | None = None) -> None:
        self._documents: dict[str, SourceDocument] = {}
        self._chunks: dict[str, Chunk] = {}
        self._chunks_by_source: dict[str, list[str]] = {}
        self._chunker = Chunker(chunking_config)

    # ── Ingestion ─────────────────────────────────────────────────────

    def add_document(
        self,
        document: SourceDocument,
        auto_chunk: bool = True,
    ) -> list[Chunk]:
        """Store a document and (optionally) chunk it.

        Returns the list of chunks produced.  If a document with the same
        ID already exists it is silently replaced.
        """
        self._documents[document.id] = document

        if auto_chunk:
            chunks = self._chunker.chunk(document)
        else:
            chunks = []

        chunk_ids: list[str] = []
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            chunk_ids.append(chunk.id)

        self._chunks_by_source[document.id] = chunk_ids
        return chunks

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Insert pre-built chunks (e.g. from an external chunking step)."""
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._chunks_by_source.setdefault(chunk.source_id, []).append(
                chunk.id
            )

    # ── Lookup ────────────────────────────────────────────────────────

    def get_document(self, doc_id: str) -> SourceDocument | None:
        return self._documents.get(doc_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_chunks_for_source(self, source_id: str) -> list[Chunk]:
        ids = self._chunks_by_source.get(source_id, [])
        return [self._chunks[cid] for cid in ids if cid in self._chunks]

    # ── Bulk access ───────────────────────────────────────────────────

    def all_documents(self) -> list[SourceDocument]:
        return list(self._documents.values())

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ── Filtering ─────────────────────────────────────────────────────

    def filter_chunks(
        self,
        source_types: list[SourceType] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Return chunks matching the given source-type and metadata criteria."""
        result: list[Chunk] = []
        for chunk in self._chunks.values():
            if source_types:
                chunk_type = chunk.metadata.get("source_type")
                if chunk_type not in source_types:
                    continue
            if metadata_filters:
                if not all(
                    chunk.metadata.get(k) == v
                    for k, v in metadata_filters.items()
                ):
                    continue
            result.append(chunk)
        return result

    # ── Housekeeping ──────────────────────────────────────────────────

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and all its chunks.  Returns True if found."""
        if doc_id not in self._documents:
            return False
        del self._documents[doc_id]
        chunk_ids = self._chunks_by_source.pop(doc_id, [])
        for cid in chunk_ids:
            self._chunks.pop(cid, None)
        return True

    def clear(self) -> None:
        """Drop all documents and chunks."""
        self._documents.clear()
        self._chunks.clear()
        self._chunks_by_source.clear()

    def stats(self) -> dict[str, Any]:
        """Summary statistics about the current corpus."""
        type_counts: dict[str, int] = {}
        for doc in self._documents.values():
            type_counts[doc.source_type] = (
                type_counts.get(doc.source_type, 0) + 1
            )

        total_chars = sum(c.char_end - c.char_start for c in self._chunks.values())

        return {
            "documents": len(self._documents),
            "chunks": len(self._chunks),
            "documents_by_type": type_counts,
            "total_chunk_chars": total_chars,
            "avg_chunk_chars": (
                round(total_chars / len(self._chunks))
                if self._chunks
                else 0
            ),
        }
