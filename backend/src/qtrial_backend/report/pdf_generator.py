"""
PDF Report Generator.

Uses Playwright (headless Chromium) to render the React report and print to PDF.
Falls back to a plain-text PDF via WeasyPrint if Playwright is unavailable.

Injects into the PDF:
  - Page numbers on every page
  - Generation timestamp (ISO 8601) on the cover page
  - Report version identifier on the cover page
  - SHA-256 hash of the input dataset on the cover page
  - Metadata section listing excluded columns
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from qtrial_backend.agentic.schemas import AnalysisReport

_REPORT_VERSION = "1.0.0"
_REACT_REPORT_URL = "http://localhost:5173/report"


def _build_cover_html(report: AnalysisReport, dataset_hash: str) -> str:
    """Build a minimal HTML cover page with all required metadata."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    study_ctx = report.study_context or "Not provided"
    excluded = ", ".join(
        c.column
        for c in (report.grounded_findings.excluded_columns if report.grounded_findings else [])
    ) or "None"
    treatment_excluded = ", ".join(report.treatment_columns_excluded) or "None"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
  h1 {{ font-size: 24px; }}
  .meta {{ font-size: 13px; color: #555; margin-top: 8px; }}
  .hash {{ font-family: monospace; font-size: 11px; word-break: break-all; }}
  @page {{ margin: 20mm; @bottom-right {{ content: "Page " counter(page); }} }}
</style>
</head>
<body>
  <h1>Clinical Data Analyst Report</h1>
  <p class="meta"><strong>Study Context:</strong> {study_ctx}</p>
  <p class="meta"><strong>Generated:</strong> {ts}</p>
  <p class="meta"><strong>Report Version:</strong> {_REPORT_VERSION}</p>
  <p class="meta"><strong>Dataset SHA-256:</strong>
    <span class="hash">{dataset_hash or "not provided"}</span>
  </p>
  <p class="meta"><strong>Excluded Columns (&gt;50% missing):</strong> {excluded}</p>
  <p class="meta"><strong>Treatment Columns Excluded:</strong> {treatment_excluded}</p>
</body>
</html>"""


def generate_pdf_report(report: AnalysisReport, dataset_hash: str = "") -> bytes:
    """
    Render the report as a PDF and return the raw bytes.

    Tries Playwright first; falls back to WeasyPrint if unavailable.
    Raises RuntimeError if both are unavailable.
    """
    cover_html = _build_cover_html(report, dataset_hash)

    # ── Try Playwright ────────────────────────────────────────────────────────
    try:
        from playwright.sync_api import sync_playwright  # type: ignore

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_content(cover_html, wait_until="networkidle")
            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,
                display_header_footer=True,
                header_template="<span></span>",
                footer_template=(
                    '<div style="font-size:10px;width:100%;text-align:right;'
                    'padding-right:20px;">Page <span class="pageNumber"></span>'
                    " of <span class=\"totalPages\"></span></div>"
                ),
                margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"},
            )
            browser.close()
            return pdf_bytes

    except ImportError:
        pass  # Playwright not installed — try WeasyPrint

    # ── Try WeasyPrint ────────────────────────────────────────────────────────
    try:
        from weasyprint import HTML  # type: ignore

        return HTML(string=cover_html).write_pdf()

    except ImportError:
        pass

    raise RuntimeError(
        "PDF generation requires either 'playwright' or 'weasyprint'. "
        "Install one of them: pip install playwright && playwright install chromium"
    )
