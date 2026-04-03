from __future__ import annotations

"""
Evidence ingestion layer.

Converts raw sources (files, text, data-dictionaries, literature results)
into SourceDocument objects ready for chunking and indexing.

Ingestion is deliberately simple: read → normalise to plain text → wrap in
a SourceDocument with metadata.  Heavy parsing (PDF, DOCX) is deferred to
a later phase; the interfaces are ready for it.
"""

import json
from pathlib import Path
from typing import Any

from qtrial_backend.rag.models import SourceDocument, SourceType


# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".rst", ".text"})
_JSON_EXTENSIONS = frozenset({".json", ".jsonl"})


# ---------------------------------------------------------------------------
# Public ingestion functions
# ---------------------------------------------------------------------------


def ingest_file(
    path: str | Path,
    source_type: SourceType = "file",
    metadata: dict[str, Any] | None = None,
) -> SourceDocument:
    """Ingest a single file from disk into a SourceDocument.

    Supported formats:
        .txt / .md / .rst   → read as-is
        .json                → pretty-printed or dict-rendered text
        .csv                 → read as raw text (companion files, not datasets)
        anything else        → attempt UTF-8 read; raise on binary
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()

    if suffix in _TEXT_EXTENSIONS:
        content = p.read_text(encoding="utf-8")

    elif suffix in _JSON_EXTENSIONS:
        raw = p.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                content = _render_dict_as_text(parsed, label=p.name)
            elif isinstance(parsed, list):
                content = json.dumps(parsed, indent=2, ensure_ascii=False)
            else:
                content = str(parsed)
        except json.JSONDecodeError:
            content = raw

    elif suffix == ".csv":
        content = p.read_text(encoding="utf-8")

    else:
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                f"Cannot ingest binary file: {path}. "
                "Supported: .txt, .md, .json, .csv, and other UTF-8 text files. "
                "PDF support is planned for a future phase."
            )

    return SourceDocument(
        name=p.name,
        source_type=source_type,
        content=content,
        file_path=str(p.resolve()),
        metadata={**(metadata or {}), "file_extension": suffix},
    )


def ingest_text(
    text: str,
    name: str,
    source_type: SourceType = "user_input",
    metadata: dict[str, Any] | None = None,
) -> SourceDocument:
    """Ingest raw text (tool output, literature abstract, user notes)."""
    if not text.strip():
        raise ValueError("Cannot ingest empty text.")

    return SourceDocument(
        name=name,
        source_type=source_type,
        content=text,
        metadata=metadata or {},
    )


def ingest_data_dictionary(path: str | Path) -> SourceDocument:
    """Ingest a JSON data dictionary (column name → description mapping).

    The JSON is rendered as readable text so the chunker can segment it
    the same way it handles any other document.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data dictionary not found: {path}")

    with open(p, "r", encoding="utf-8") as fh:
        col_descs: dict[str, str] = json.load(fh)

    content = _render_dict_as_text(col_descs, label=p.name)

    return SourceDocument(
        name=p.name,
        source_type="data_dictionary",
        content=content,
        file_path=str(p.resolve()),
        metadata={
            "file_extension": p.suffix.lower(),
            "n_columns": len(col_descs),
            "columns": list(col_descs.keys()),
        },
    )


def ingest_literature_results(
    results: list[dict[str, Any]],
    query: str,
    source_label: str = "literature_search",
) -> list[SourceDocument]:
    """Ingest literature search results from PubMed / Semantic Scholar.

    Each paper becomes a separate SourceDocument so that chunks maintain
    per-paper provenance — essential for citation traceability.
    """
    documents: list[SourceDocument] = []

    for paper in results:
        title = paper.get("title", "Unknown title")
        authors = paper.get("authors", [])
        year = paper.get("year", "")
        abstract = paper.get("abstract", "")
        pmid = paper.get("pmid", paper.get("paperId", ""))
        tldr = paper.get("tldr", "")

        parts = [f"Title: {title}"]
        if authors:
            author_str = (
                ", ".join(authors[:5])
                if isinstance(authors, list)
                else str(authors)
            )
            parts.append(f"Authors: {author_str}")
        if year:
            parts.append(f"Year: {year}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if tldr:
            parts.append(f"TLDR: {tldr}")

        content = "\n".join(parts)

        meta: dict[str, Any] = {
            "search_query": query,
            "title": title,
            "year": year,
        }
        if pmid:
            meta["paper_id"] = str(pmid)
        if authors:
            meta["authors"] = (
                authors[:5] if isinstance(authors, list) else [str(authors)]
            )

        doc = SourceDocument(
            name=f"paper:{pmid or title[:60]}",
            source_type="literature",
            content=content,
            metadata=meta,
        )
        documents.append(doc)

    return documents


def ingest_tool_result(
    tool_name: str,
    arguments: dict[str, Any],
    result_text: str,
    metadata: dict[str, Any] | None = None,
) -> SourceDocument:
    """Ingest the output of a statistical or literature tool.

    This allows tool results to be indexed and later retrieved as
    supporting evidence during report generation or claim verification.
    """
    if not result_text.strip():
        raise ValueError("Cannot ingest empty tool result.")

    result_text_norm, quality = _normalise_tool_result_text(result_text)

    header = (
        f"Tool: {tool_name}\n"
        f"Arguments: {json.dumps(arguments, default=str)}\n"
        f"Quality Score: {quality:.3f}\n\n"
    )
    content = header + result_text_norm

    parsed_keys: list[str] = []
    is_json = False
    try:
        parsed = json.loads(result_text)
        if isinstance(parsed, dict):
            parsed_keys = sorted([str(k) for k in parsed.keys()])
            is_json = True
    except Exception:
        pass

    return SourceDocument(
        name=f"tool:{tool_name}",
        source_type="tool_result",
        content=content,
        metadata={
            **(metadata or {}),
            "tool_name": tool_name,
            "tool_arguments": arguments,
            "result_quality": quality,
            "result_is_json": is_json,
            "result_top_level_keys": parsed_keys,
            "result_char_count": len(result_text_norm),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_dict_as_text(d: dict[str, Any], label: str = "") -> str:
    """Render a dict as readable lines suitable for chunking."""
    lines: list[str] = []
    if label:
        lines.append(f"Data Dictionary: {label}")
        lines.append("")
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k2, v2 in value.items():
                lines.append(f"  {k2}: {v2}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _normalise_tool_result_text(text: str) -> tuple[str, float]:
    """Normalise tool output and compute a lightweight quality score.

    Score is a bounded heuristic [0,1] based on:
    - non-trivial length,
    - presence of structured keys/stat terms,
    - avoidance of obvious error-only payloads.
    """
    stripped = text.strip()
    lowered = stripped.lower()

    if not stripped:
        return stripped, 0.0

    # Common low-value/error payload patterns
    if lowered in {"{}", "[]", "null", "none"}:
        return stripped, 0.0
    if "\"error\"" in lowered and len(stripped) < 250:
        return stripped, 0.05

    score = 0.2
    if len(stripped) >= 80:
        score += 0.25
    if len(stripped) >= 250:
        score += 0.15

    signal_terms = (
        "p_value",
        "confidence",
        "hazard",
        "odds",
        "risk",
        "median",
        "significant",
        "effect",
        "correlation",
    )
    if any(term in lowered for term in signal_terms):
        score += 0.25

    if stripped.startswith("{") and stripped.endswith("}"):
        score += 0.15

    score = max(0.0, min(1.0, score))
    return stripped, round(score, 3)
