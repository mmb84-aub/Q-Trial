from __future__ import annotations

"""
Okapi BM25 sparse retriever — pure Python, zero extra dependencies.

This is the first concrete Retriever implementation for the Q-Trial RAG
subsystem.  It provides strong lexical retrieval with:

* Inverted-index construction from Chunks
* Okapi BM25 scoring (Robertson & Zaragoza, 2009)
* Source-type and metadata filtering (via the Retriever ABC helpers)
* min_score thresholding
* Full Provenance on every result

The implementation is intentionally dependency-free so the repo's
footprint stays unchanged.  It is designed as the **first stage** of a
real retrieval pipeline — later phases will add dense retrieval, hybrid
fusion, and reranking behind the same Retriever ABC.

BM25 parameter defaults (k1=1.5, b=0.75) follow the original Okapi
recommendations and work well across biomedical and general text.
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from qtrial_backend.rag.models import Chunk, RetrievalQuery, RetrievalResult
from qtrial_backend.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

# Minimal clinical-safe stopword set.  Kept small to avoid discarding
# meaningful biomedical terms (e.g. "patient", "group").
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "into", "through", "during", "it", "its",
    "this", "that", "these", "those", "not", "no", "nor",
    "do", "does", "did", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "have", "has", "had",
    "if", "then", "than", "so", "very", "just", "about",
})

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", re.IGNORECASE)

_QUERY_EXPANSIONS: dict[str, list[str]] = {
    "hr": ["hazard", "ratio", "hazard_ratio"],
    "or": ["odds", "ratio", "odds_ratio"],
    "rr": ["risk", "ratio", "risk_ratio"],
    "ci": ["confidence", "interval"],
    "km": ["kaplan", "meier", "survival"],
    "p": ["pvalue", "significance"],
    "smd": ["standardized", "mean", "difference"],
}

_SOURCE_TYPE_BOOSTS: dict[str, float] = {
    "tool_result": 1.20,
    "data_dictionary": 1.15,
    "user_input": 1.10,
    "literature": 1.00,
    "file": 1.00,
}


def tokenize(text: str) -> list[str]:
    """Lowercase tokenisation with stopword removal.

    Splits on non-alphanumeric boundaries, preserves hyphenated words
    and apostrophes (e.g. "Kaplan-Meier", "D'Agostino").  Tokens shorter
    than 2 characters after lowering are dropped.
    """
    normalised = unicodedata.normalize("NFKD", text).encode(
        "ascii", "ignore"
    ).decode("ascii")

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalised.lower()):
        tok = match.group()
        tok = _light_stem(tok)
        if len(tok) >= 2 and tok not in _STOPWORDS:
            tokens.append(tok)
    return tokens


def _light_stem(token: str) -> str:
    """Very lightweight suffix normalisation for sparse retrieval quality.

    Keeps behaviour deterministic and dependency-free while improving matches
    across common morphological variants (e.g. "survived" vs "survival").
    """
    if len(token) <= 4:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def tokenize_query(text: str) -> list[str]:
    """Tokenise query text and expand common clinical/statistical shorthands."""
    base = tokenize(text)
    expanded: list[str] = list(base)
    for token in base:
        expanded.extend(_QUERY_EXPANSIONS.get(token, []))
    return expanded


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _PostingEntry:
    """One entry in a term's posting list."""
    chunk_idx: int          # index into BM25Retriever._chunks
    term_freq: int          # how many times the term appears in this chunk


@dataclass
class _BM25Index:
    """Precomputed BM25 index data."""
    chunks: list[Chunk] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    avg_dl: float = 0.0
    n_docs: int = 0
    postings: dict[str, list[_PostingEntry]] = field(default_factory=dict)
    doc_freqs: dict[str, int] = field(default_factory=dict)
    doc_token_sets: list[set[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

class BM25Retriever(Retriever):
    """Pure-Python Okapi BM25 retriever.

    Usage::

        retriever = BM25Retriever()
        retriever.index(store.all_chunks())
        results = retriever.retrieve(RetrievalQuery(text="survival analysis"))
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._idx: _BM25Index | None = None

    # ── Retriever interface ───────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        idx = _BM25Index()
        idx.chunks = list(chunks)
        idx.n_docs = len(chunks)

        if idx.n_docs == 0:
            self._idx = idx
            return

        # Tokenise every chunk and build posting lists
        for i, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            idx.doc_lens.append(len(tokens))
            idx.doc_token_sets.append(set(tokens))

            term_counts = Counter(tokens)
            for term, freq in term_counts.items():
                idx.postings.setdefault(term, []).append(
                    _PostingEntry(chunk_idx=i, term_freq=freq)
                )
                # Count each doc once per term
                idx.doc_freqs[term] = idx.doc_freqs.get(term, 0) + 1

        idx.avg_dl = sum(idx.doc_lens) / idx.n_docs
        self._idx = idx

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        if self._idx is None or self._idx.n_docs == 0:
            return []

        idx = self._idx
        query_tokens = tokenize_query(query.text)
        if not query_tokens:
            return []

        # Pre-compute the set of chunk IDs that pass filters
        allowed: set[str] | None = None
        if query.source_types or query.metadata_filters:
            allowed = self._apply_filters(idx.chunks, query)

        # Score every document (accumulate per-term contributions)
        scores: dict[int, float] = {}
        for term in query_tokens:
            postings = idx.postings.get(term)
            if postings is None:
                continue
            idf = self._idf(idx.n_docs, idx.doc_freqs[term])
            for entry in postings:
                # Skip chunks excluded by filters
                if allowed is not None:
                    if idx.chunks[entry.chunk_idx].id not in allowed:
                        continue
                tf_component = self._tf_score(
                    entry.term_freq,
                    idx.doc_lens[entry.chunk_idx],
                    idx.avg_dl,
                )
                base_score = scores.get(entry.chunk_idx, 0.0) + idf * tf_component
                chunk_type = str(
                    idx.chunks[entry.chunk_idx].metadata.get("source_type", "file")
                )
                source_boost = _SOURCE_TYPE_BOOSTS.get(chunk_type, 1.0)
                scores[entry.chunk_idx] = base_score * source_boost

        # Filter by min_score, sort, truncate to top_k
        scored = [
            (doc_idx, score)
            for doc_idx, score in scores.items()
            if score >= query.min_score
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = self._diversified_top_k(scored, idx, query_tokens, query.top_k)

        # Build results with provenance
        results: list[RetrievalResult] = []
        for rank_0, (doc_idx, score) in enumerate(scored):
            chunk = idx.chunks[doc_idx]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=round(score, 6),
                    rank=rank_0 + 1,
                    provenance=self._build_provenance(chunk),
                )
            )

        return results

    def _diversified_top_k(
        self,
        scored: list[tuple[int, float]],
        idx: _BM25Index,
        query_tokens: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Apply light novelty-aware reranking to reduce near-duplicate chunks."""
        if len(scored) <= 1:
            return scored[:top_k]

        query_set = set(query_tokens)
        selected: list[tuple[int, float]] = []
        remaining = list(scored)
        lambda_div = 0.15

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = float("-inf")

            for i, (doc_idx, base_score) in enumerate(remaining):
                novelty_penalty = 0.0
                doc_tokens = idx.doc_token_sets[doc_idx]
                for sel_doc_idx, _ in selected:
                    sel_tokens = idx.doc_token_sets[sel_doc_idx]
                    novelty_penalty = max(
                        novelty_penalty,
                        self._jaccard_similarity(doc_tokens, sel_tokens),
                    )

                query_overlap = self._jaccard_similarity(doc_tokens, query_set)
                adjusted = base_score * (1.0 + 0.05 * query_overlap) - (
                    lambda_div * novelty_penalty
                )

                if adjusted > best_score:
                    best_score = adjusted
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    @staticmethod
    def _jaccard_similarity(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a.intersection(b))
        union = len(a.union(b))
        return inter / union if union > 0 else 0.0

    @property
    def is_indexed(self) -> bool:
        return self._idx is not None

    @property
    def index_size(self) -> int:
        return self._idx.n_docs if self._idx else 0

    # ── BM25 math ─────────────────────────────────────────────────────

    @staticmethod
    def _idf(n_docs: int, doc_freq: int) -> float:
        """Inverse document frequency (Robertson-Sparck Jones flavour)."""
        return math.log(
            (n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
        )

    def _tf_score(
        self,
        term_freq: int,
        doc_len: int,
        avg_dl: float,
    ) -> float:
        """Term-frequency saturation component of Okapi BM25."""
        numerator = term_freq * (self._k1 + 1.0)
        denominator = term_freq + self._k1 * (
            1.0 - self._b + self._b * doc_len / avg_dl
        )
        return numerator / denominator
