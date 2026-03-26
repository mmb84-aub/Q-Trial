from __future__ import annotations

import json
import re
from pathlib import Path

from qtrial_backend.agentic.schemas import PriorReportNormalized, PriorReportSection, PriorReportClaim


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


# ── Claim type detection via section title heuristics ──────────────────────────

def _infer_claim_type_from_section_title(title: str) -> str:
    """
    Map section title to claim type using simple heuristic matching.
    Returns one of: key_finding, risk_signal, bias_signal, grounded_finding,
                   recommendation, assumption, methodology, limitation, unknown.
    """
    title_lower = title.lower().strip()
    
    # Key findings
    if any(kw in title_lower for kw in ["key finding", "main result", "primary result", "finding"]):
        if "risk" not in title_lower and "bias" not in title_lower:
            return "key_finding"
    
    # Risk signals
    if any(kw in title_lower for kw in ["risk", "adverse", "safety", "harm"]):
        return "risk_signal"
    
    # Bias/limitation signals
    if any(kw in title_lower for kw in ["bias", "limitation", "limitation", "weakness", "constraint"]):
        if "risk" not in title_lower:
            return "bias_signal"
    
    # Grounded findings (from literature)
    if any(kw in title_lower for kw in ["literature", "grounded", "supported", "evidence", "validation"]):
        return "grounded_finding"
    
    # Recommendations
    if any(kw in title_lower for kw in ["recommendation", "suggest", "future", "next step", "improvement"]):
        return "recommendation"
    
    # Assumptions
    if any(kw in title_lower for kw in ["assumption", "assum", "postulate", "presume"]):
        return "assumption"
    
    # Methodology
    if any(kw in title_lower for kw in ["method", "design", "approach", "protocol"]):
        return "methodology"
    
    # Default: unknown
    return "unknown"


def _extract_atomic_claims_from_sections(
    sections: list[PriorReportSection],
) -> list[PriorReportClaim]:
    """
    Parse sections into atomic structured claims using deterministic heuristics.
    
    For each section:
    1. Infer claim type from section title
    2. Split content into sentences and bullet points
    3. Create PriorReportClaim for each non-trivial text unit
    4. Assign parser confidence based on extraction method
    
    Returns a list of PriorReportClaim objects.
    """
    claims: list[PriorReportClaim] = []
    claim_counter = 0
    
    for section in sections:
        if not section.content or not section.content.strip():
            continue
        
        claim_type = _infer_claim_type_from_section_title(section.title)
        
        # Split content into text units (sentences and bullet points)
        text_units = _split_into_text_units(section.content)
        
        for unit in text_units:
            if len(unit.strip()) < 15:  # Skip very short fragments
                continue
            
            claim_counter += 1
            claim = PriorReportClaim(
                claim_id=f"claim_{claim_counter}",
                section_id=section.section_id,
                section_title=section.title,
                claim_text=unit.strip(),
                claim_type=claim_type,
                source_excerpt=unit.strip()[:200],  # First 200 chars as excerpt
                parser_confidence=0.85 if claim_type != "unknown" else 0.7,
            )
            claims.append(claim)
    
    return claims


def _split_into_text_units(text: str) -> list[str]:
    """
    Split text into sentences and bullet points for claim extraction.
    
    Handles:
    - Bullet points (-, *, •)
    - Numbered lists (1., 2., etc.)
    - Sentence splitting (. ! ?)
    """
    units: list[str] = []
    
    # First, split by bullet points
    bullet_pattern = re.compile(r"^\s*[-*•]\s+(.+?)$", re.MULTILINE)
    bullets = bullet_pattern.findall(text)
    if bullets:
        units.extend(b.strip() for b in bullets if b.strip())
    
    # Then, split by numbered lists
    numbered_pattern = re.compile(r"^\s*\d+\.\s+(.+?)$", re.MULTILINE)
    numbered = numbered_pattern.findall(text)
    if numbered:
        units.extend(n.strip() for n in numbered if n.strip())
    
    # If no bullets/numbered, split by sentences
    if not units:
        # Simple sentence splitter: split on . ! ? followed by space and capital letter
        sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_pattern.split(text)
        units.extend(s.strip() for s in sentences if s.strip())
    
    # If still no units, return entire text as one unit
    if not units:
        units = [text.strip()]
    
    return units


def parse_prior_report_file(path: str | Path) -> PriorReportNormalized:
    """
    Load an uploaded prior report and normalize it into a canonical schema.

    Supported scaffolding formats:
    - .json: expects optional keys like raw_text/sections/claims/metadata
    - .md/.markdown/.txt: parsed as plain text with markdown heading splitting
    - other: best-effort text normalization
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()

    if suffix == ".json":
        return _normalize_from_json(raw, source_name=p.name)

    if suffix in {".md", ".markdown", ".txt"}:
        fmt = "markdown" if suffix in {".md", ".markdown"} else "text"
        return _normalize_from_text(raw, source_name=p.name, source_format=fmt)

    return _normalize_from_text(raw, source_name=p.name, source_format="unknown")


def normalize_prior_report_text(
    text: str,
    *,
    source_name: str = "inline",
    source_format: str = "text",
) -> PriorReportNormalized:
    """Normalize in-memory prior-report text payloads."""
    fmt = source_format if source_format in {"markdown", "text", "unknown"} else "unknown"
    return _normalize_from_text(text, source_name=source_name, source_format=fmt)


def _normalize_from_json(raw_json: str, *, source_name: str) -> PriorReportNormalized:
    try:
        payload = json.loads(raw_json)
    except Exception:
        # Keep parsing resilient: fallback to raw text normalization.
        return _normalize_from_text(raw_json, source_name=source_name, source_format="unknown")

    raw_text = str(payload.get("raw_text") or "").strip()
    sections: list[PriorReportSection] = []

    src_sections = payload.get("sections") or []
    if isinstance(src_sections, list):
        for i, item in enumerate(src_sections, 1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            span = item.get("source_span")
            if not content and not title:
                continue
            sections.append(
                PriorReportSection(
                    section_id=f"sec_{i}",
                    title=title,
                    content=content,
                    source_span=str(span) if span is not None else None,
                )
            )

    claims = payload.get("claims") or payload.get("extracted_claims") or []
    extracted_claims = [str(c).strip() for c in claims if str(c).strip()] if isinstance(claims, list) else []

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    if not raw_text and sections:
        raw_text = "\n\n".join(
            [f"{s.title}\n{s.content}".strip() for s in sections if s.title or s.content]
        ).strip()

    if not sections and raw_text:
        sections = _split_into_sections(raw_text)

    # Extract atomic claims from sections
    atomic_claims = _extract_atomic_claims_from_sections(sections)

    # Backward-compatible fallback: claims-only JSON payloads.
    # If top-level claims exist but section parsing produced no atomic claims,
    # synthesize deterministic atomic claim entries directly from those strings.
    if not atomic_claims and extracted_claims:
        section_id = sections[0].section_id if sections else "sec_1"
        section_title = sections[0].title if sections else "Claims"
        inferred_type = _infer_claim_type_from_section_title(section_title)
        atomic_claims = []
        for i, claim_text in enumerate(extracted_claims, 1):
            text = claim_text.strip()
            if not text:
                continue
            atomic_claims.append(
                PriorReportClaim(
                    claim_id=f"claim_{i}",
                    section_id=section_id,
                    section_title=section_title,
                    claim_text=text,
                    claim_type=inferred_type if inferred_type else "unknown",
                    source_excerpt=text[:200],
                    parser_confidence=0.75,
                )
            )

    return PriorReportNormalized(
        source_name=source_name,
        source_format="json",
        raw_text=raw_text,
        sections=sections,
        extracted_claims=extracted_claims,
        extracted_atomic_claims=atomic_claims,
        metadata=metadata,
    )


def _normalize_from_text(
    text: str,
    *,
    source_name: str,
    source_format: str,
) -> PriorReportNormalized:
    cleaned = _clean_text(text)
    sections = _split_into_sections(cleaned)
    atomic_claims = _extract_atomic_claims_from_sections(sections)
    return PriorReportNormalized(
        source_name=source_name,
        source_format=source_format if source_format in {"markdown", "text", "unknown"} else "unknown",
        raw_text=cleaned,
        sections=sections,
        extracted_claims=[],
        extracted_atomic_claims=atomic_claims,
        metadata={},
    )


def _clean_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_into_sections(text: str) -> list[PriorReportSection]:
    if not text:
        return []

    lines = text.split("\n")
    sections: list[PriorReportSection] = []

    current_title = "Overview"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines, current_title
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                PriorReportSection(
                    section_id=f"sec_{len(sections) + 1}",
                    title=current_title,
                    content=content,
                )
            )
        current_lines = []

    for ln in lines:
        m = _HEADING_RE.match(ln)
        if m:
            flush()
            current_title = m.group(2).strip() or "Untitled"
            continue
        current_lines.append(ln)

    flush()

    if not sections:
        sections = [
            PriorReportSection(section_id="sec_1", title="Overview", content=text)
        ]

    return sections
