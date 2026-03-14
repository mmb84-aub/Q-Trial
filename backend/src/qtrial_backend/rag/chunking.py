from __future__ import annotations

"""
Document chunking module.

Splits SourceDocuments into Chunks using configurable strategies:
    - fixed    : character-window with overlap
    - sentence : group sentences up to a size limit
    - paragraph: group paragraphs up to a size limit

All strategies preserve character offsets and propagate source metadata
into every chunk, ensuring full provenance traceability.
"""

import re
from dataclasses import dataclass
from typing import Literal

from qtrial_backend.rag.models import Chunk, SourceDocument


ChunkStrategy = Literal["fixed", "sentence", "paragraph"]


@dataclass(frozen=True)
class ChunkingConfig:
    """Tunable parameters for the chunking module."""

    strategy: ChunkStrategy = "sentence"
    chunk_size: int = 512
    """Target chunk size in characters."""
    chunk_overlap: int = 64
    """Character overlap between consecutive chunks."""
    min_chunk_size: int = 50
    """Discard chunks shorter than this."""
    sentence_delimiters: str = r"(?<=[.!?])\s+"
    """Regex pattern used to detect sentence boundaries."""
    paragraph_delimiter: str = "\n\n"
    """Literal string that separates paragraphs."""


class Chunker:
    """Splits SourceDocuments into Chunks with a configurable strategy.

    Usage::

        chunker = Chunker(ChunkingConfig(strategy="sentence", chunk_size=512))
        chunks  = chunker.chunk(document)
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    # ── Public API ────────────────────────────────────────────────────

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Split a single document into chunks."""
        text = document.content
        if not text.strip():
            return []

        strategy = self.config.strategy
        if strategy == "fixed":
            spans = self._fixed_spans(text)
        elif strategy == "sentence":
            spans = self._sentence_spans(text)
        elif strategy == "paragraph":
            spans = self._paragraph_spans(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy!r}")

        chunks: list[Chunk] = []
        for idx, (start, end) in enumerate(spans):
            chunk_text = text[start:end].strip()
            if len(chunk_text) < self.config.min_chunk_size:
                continue

            chunks.append(
                Chunk(
                    source_id=document.id,
                    text=chunk_text,
                    index=idx,
                    char_start=start,
                    char_end=end,
                    metadata={
                        **document.metadata,
                        "source_name": document.name,
                        "source_type": document.source_type,
                        "chunking_strategy": self.config.strategy,
                    },
                    token_estimate=len(chunk_text) // 4,
                )
            )

        return chunks

    def chunk_batch(self, documents: list[SourceDocument]) -> list[Chunk]:
        """Chunk multiple documents, preserving document order."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    # ── Strategy: fixed-size windows ──────────────────────────────────

    def _fixed_spans(self, text: str) -> list[tuple[int, int]]:
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = max(size - overlap, 1)
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            spans.append((start, end))
            if end >= len(text):
                break
            start += step
        return spans

    # ── Strategy: sentence grouping ───────────────────────────────────

    def _sentence_spans(self, text: str) -> list[tuple[int, int]]:
        """Split on sentence boundaries, then group into sized chunks."""
        boundaries = list(
            re.finditer(self.config.sentence_delimiters, text)
        )

        unit_spans: list[tuple[int, int]] = []
        prev = 0
        for m in boundaries:
            unit_spans.append((prev, m.end()))
            prev = m.end()
        if prev < len(text):
            unit_spans.append((prev, len(text)))

        if not unit_spans:
            return [(0, len(text))]

        return self._group_spans(unit_spans, text)

    # ── Strategy: paragraph grouping ──────────────────────────────────

    def _paragraph_spans(self, text: str) -> list[tuple[int, int]]:
        """Split on paragraph boundaries, then group into sized chunks."""
        delim = self.config.paragraph_delimiter
        parts = text.split(delim)

        unit_spans: list[tuple[int, int]] = []
        offset = 0
        for i, part in enumerate(parts):
            start = offset
            end = offset + len(part)
            unit_spans.append((start, end))
            offset = end + len(delim) if i < len(parts) - 1 else end

        if not unit_spans:
            return [(0, len(text))]

        return self._group_spans(unit_spans, text)

    # ── Shared: group atomic spans into sized chunks with overlap ─────

    def _group_spans(
        self,
        unit_spans: list[tuple[int, int]],
        text: str,
    ) -> list[tuple[int, int]]:
        """Greedily group atomic spans (sentences / paragraphs) into chunks
        that respect ``chunk_size``, inserting overlap by rewinding to an
        earlier unit boundary when starting a new chunk.
        """
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        chunks: list[tuple[int, int]] = []
        current_start = unit_spans[0][0]
        current_end = unit_spans[0][1]

        for span_start, span_end in unit_spans[1:]:
            prospective_len = span_end - current_start
            if prospective_len > size and current_end > current_start:
                # Emit the current chunk
                chunks.append((current_start, current_end))
                # Start the next chunk with overlap
                overlap_start = self._find_overlap_start(
                    unit_spans, current_end, overlap
                )
                current_start = overlap_start
            current_end = span_end

        # Emit the final chunk
        if current_end > current_start:
            chunks.append((current_start, current_end))

        return chunks

    @staticmethod
    def _find_overlap_start(
        unit_spans: list[tuple[int, int]],
        current_end: int,
        overlap: int,
    ) -> int:
        """Find the best unit-aligned start position for a new chunk so that
        approximately ``overlap`` characters of the previous chunk are
        re-included.
        """
        target = current_end - overlap
        best = current_end
        for start, _end in unit_spans:
            if start >= current_end:
                break
            if start <= target:
                best = start
        # Fallback: if no good boundary found, just rewind by overlap chars
        return best if best < current_end else max(0, current_end - overlap)
