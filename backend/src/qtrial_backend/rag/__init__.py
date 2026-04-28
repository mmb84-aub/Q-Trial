from __future__ import annotations

"""
Q-Trial RAG subsystem — public API.

Package layout::

    rag/
        models.py          Data models: SourceDocument, Chunk, Provenance, etc.
        ingestion.py       Convert raw sources → SourceDocuments
        chunking.py        Split SourceDocuments → Chunks
        store.py           In-memory corpus (documents + chunks)
        retriever.py       Retriever ABC
        bm25_retriever.py  First concrete retriever (Okapi BM25)
"""

# Models
from qtrial_backend.rag.models import (
    Chunk as Chunk,
    Provenance as Provenance,
    RetrievalQuery as RetrievalQuery,
    RetrievalResult as RetrievalResult,
    SourceDocument as SourceDocument,
    SourceType as SourceType,
)

# Ingestion
from qtrial_backend.rag.ingestion import (
    ingest_data_dictionary as ingest_data_dictionary,
    ingest_file as ingest_file,
    ingest_literature_results as ingest_literature_results,
    ingest_text as ingest_text,
    ingest_tool_result as ingest_tool_result,
)

# Chunking
from qtrial_backend.rag.chunking import (
    Chunker as Chunker,
    ChunkingConfig as ChunkingConfig,
)

# Store
from qtrial_backend.rag.store import EvidenceStore as EvidenceStore

# Retrieval
from qtrial_backend.rag.retriever import Retriever as Retriever
from qtrial_backend.rag.bm25_retriever import BM25Retriever as BM25Retriever
