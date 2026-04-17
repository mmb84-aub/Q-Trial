"""
AgentContext — shared state passed into AgentLoop for Stage 4.

Input:  DataFrame, dataset_preview dict, evidence dict, optional RAG stores.
Output: dataclass instance accessed by AgentLoop and all statistical tools.
Does:   bundles the dataset and pre-computed evidence so tools can query the
        DataFrame without reloading it, and the LLM has structured context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from qtrial_backend.rag import BM25Retriever, EvidenceStore
from qtrial_backend.rag.ingestion import (
    ingest_data_dictionary,
    ingest_file,
    ingest_text,
    ingest_tool_result,
)
from qtrial_backend.rag.models import SourceDocument, SourceType
from qtrial_backend.rag.models import RetrievalResult


@dataclass
class AgentContext:
    """Shared state passed to every tool invocation."""

    dataframe: pd.DataFrame
    dataset_name: str
    column_names: list[str] = field(default_factory=list)
    shape: tuple[int, int] = (0, 0)

    # Citation registry — populated by citation_manager tool
    citation_store: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Step-by-step analysis audit trail — populated by log_analysis_step tool
    analysis_log: list[dict[str, Any]] = field(default_factory=list)

    # Deduplication cache: "tool_name::sorted_args_json" -> result_str
    _call_cache: dict[str, str] = field(default_factory=dict)

    # RAG runtime state
    evidence_store: EvidenceStore = field(default_factory=EvidenceStore)
    retriever: BM25Retriever = field(default_factory=BM25Retriever)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.column_names = list(self.dataframe.columns)
        self.shape = (self.dataframe.shape[0], self.dataframe.shape[1])
        self.retrieval_metadata = {
            "index_version": 0,
            "indexed_documents": 0,
            "indexed_chunks": 0,
            "tool_results_indexed": 0,
            "tool_results_skipped": 0,
        }
        self.reindex_retriever()

    def reindex_retriever(self) -> None:
        """Rebuild the retriever index from all chunks currently in store."""
        chunks = self.evidence_store.all_chunks()
        self.retriever.index(chunks)
        self.retrieval_metadata["index_version"] = (
            int(self.retrieval_metadata.get("index_version", 0)) + 1
        )
        self.retrieval_metadata["indexed_documents"] = self.evidence_store.document_count
        self.retrieval_metadata["indexed_chunks"] = self.evidence_store.chunk_count

    def add_evidence_document(
        self,
        document: SourceDocument,
        reindex: bool = True,
    ) -> None:
        self.evidence_store.add_document(document)
        if reindex:
            self.reindex_retriever()

    def ingest_evidence_file(
        self,
        path: str,
        source_type: SourceType = "file",
        metadata: dict[str, Any] | None = None,
        reindex: bool = True,
    ) -> None:
        doc = ingest_file(path=path, source_type=source_type, metadata=metadata)
        self.add_evidence_document(doc, reindex=reindex)

    def ingest_evidence_text(
        self,
        text: str,
        name: str,
        source_type: SourceType = "user_input",
        metadata: dict[str, Any] | None = None,
        reindex: bool = True,
    ) -> None:
        doc = ingest_text(
            text=text,
            name=name,
            source_type=source_type,
            metadata=metadata,
        )
        self.add_evidence_document(doc, reindex=reindex)

    def ingest_data_dictionary(
        self,
        path: str,
        reindex: bool = True,
    ) -> None:
        doc = ingest_data_dictionary(path)
        self.add_evidence_document(doc, reindex=reindex)

    def index_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_text: str,
        is_error: bool = False,
    ) -> None:
        """Add a tool output as evidence and refresh retrieval index.

        Error outputs are skipped by default to keep corpus quality high.
        """
        if is_error:
            self.retrieval_metadata["tool_results_skipped"] = (
                int(self.retrieval_metadata.get("tool_results_skipped", 0)) + 1
            )
            return

        if not self._should_index_tool_result(tool_name, result_text):
            self.retrieval_metadata["tool_results_skipped"] = (
                int(self.retrieval_metadata.get("tool_results_skipped", 0)) + 1
            )
            return

        doc = ingest_tool_result(
            tool_name=tool_name,
            arguments=arguments,
            result_text=result_text,
            metadata={"dataset_name": self.dataset_name},
        )
        self.add_evidence_document(doc, reindex=True)
        self.retrieval_metadata["tool_results_indexed"] = (
            int(self.retrieval_metadata.get("tool_results_indexed", 0)) + 1
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_types: list[SourceType] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Thin runtime hook for future grounding injection.

        Keeps retrieval access local to AgentContext so future prompt-injection
        logic can call one stable method.
        """
        from qtrial_backend.rag.models import RetrievalQuery

        if self.retriever.index_size != self.evidence_store.chunk_count:
            self.reindex_retriever()

        rq = RetrievalQuery(
            text=query,
            top_k=top_k,
            min_score=min_score,
            source_types=source_types,
            metadata_filters=metadata_filters,
        )
        return self.retriever.retrieve(rq)

    @staticmethod
    def _should_index_tool_result(tool_name: str, result_text: str) -> bool:
        """Skip low-value/noisy outputs to keep retrieval corpus clean."""
        if tool_name in {"retrieve_evidence"}:
            return False

        text = result_text.strip()
        if not text:
            return False
        if text in {"{}", "[]", "null", "None"}:
            return False

        lowered = text.lower()
        if "\"error\"" in lowered and len(text) < 300:
            return False
        if len(text) < 60:
            return False

        return True
