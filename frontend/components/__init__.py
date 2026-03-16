"""
Q-Trial frontend components — Python helpers to load HTML/CSS/JS
components into Streamlit via st.components.v1.html().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit.components.v1 as st_components

_DIR = Path(__file__).parent


def _load(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8")


# ── Pipeline Tracker ─────────────────────────────────────────────────────────

# Canonical ordered list of pipeline stages
PIPELINE_STAGES = [
    {"key": "StaticAnalysis",   "label": "Static Analysis"},
    {"key": "dataset",          "label": "Evidence & Guardrails"},
    {"key": "plan",             "label": "Planner"},
    {"key": "DataQualityAgent", "label": "Data Quality"},
    {"key": "ClinicalSemanticsAgent", "label": "Clinical Semantics"},
    {"key": "UnknownsAgent",    "label": "Unknowns"},
    {"key": "InsightSynthesisAgent", "label": "Insight Synthesis"},
    {"key": "judge",            "label": "Judge"},
    {"key": "reasoning",        "label": "Reasoning Engine"},
    {"key": "hypotheses",       "label": "Hypotheses"},
    {"key": "dispatch",         "label": "Tool Dispatch"},
    {"key": "literature",       "label": "Literature RAG"},
]


def _build_stage_states(completed_stages: list[str]) -> list[dict[str, str]]:
    """Map completed stage keys → state (done/active/pending) for each stage."""
    completed_set = set(completed_stages)
    stages = []
    found_first_pending = False
    for s in PIPELINE_STAGES:
        if s["key"] in completed_set:
            stages.append({"label": s["label"], "state": "done"})
        elif not found_first_pending:
            stages.append({"label": s["label"], "state": "active"})
            found_first_pending = True
        else:
            stages.append({"label": s["label"], "state": "pending"})
    return stages


def render_pipeline_tracker(completed_stages: list[str], height: int = 100) -> None:
    """Render the pipeline progress tracker as an HTML component."""
    css = _load("styles.css")
    js = _load("pipeline.js")
    stages_json = json.dumps(_build_stage_states(completed_stages))

    html = f"""
    <html><head><meta charset="utf-8">
    <style>{css}</style></head>
    <body>
    <div class="pipeline-tracker" id="pipeline-tracker">
      <div class="pipeline-title">Pipeline Progress</div>
      <div class="pipeline-stages" id="pipeline-stages"></div>
    </div>
    <script>{js}</script>
    <script>renderPipeline({stages_json});</script>
    </body></html>
    """
    st_components.html(html, height=height, scrolling=False)


def render_pipeline_tracker_from_report(report: dict, height: int = 100) -> None:
    """Infer completed stages from a finished report and render the tracker."""
    done: list[str] = []

    # Static analysis is done if the report has prior_analysis_report
    if report.get("prior_analysis_report"):
        done.append("StaticAnalysis")

    # Evidence/guardrails always runs if we have a report
    done.append("dataset")

    # Plan always runs
    if report.get("plan"):
        done.append("plan")

    # Check agent runs
    agent_names = {r.get("agent", "") for r in (report.get("agent_runs") or [])}
    for stage in PIPELINE_STAGES:
        if stage["key"] in agent_names:
            done.append(stage["key"])

    # Judge
    if report.get("judge"):
        done.append("judge")

    # Reasoning
    rs = report.get("reasoning_state") or {}
    if rs.get("claims"):
        done.append("reasoning")
    if rs.get("hypotheses"):
        done.append("hypotheses")
    if rs.get("dispatched_tool_results"):
        done.append("dispatch")

    # Literature
    if report.get("literature_report"):
        done.append("literature")

    render_pipeline_tracker(done, height=height)


# ── Static Report Viewer ─────────────────────────────────────────────────────

def render_static_report(markdown_text: str, height: int = 600) -> None:
    """Render the static analysis markdown report as an interactive HTML viewer."""
    css = _load("styles.css")
    js = _load("pipeline.js")
    md_json = json.dumps(markdown_text)

    html = f"""
    <html><head><meta charset="utf-8">
    <style>{css}</style></head>
    <body>
    <div class="static-report-viewer">
      <div class="report-header">
        <div style="font-size:14px;font-weight:700;color:#111;">Deterministic Statistical Report</div>
        <div class="report-badge"><span class="dot"></span> No LLM — pure data</div>
      </div>
      <div id="static-report-container"></div>
    </div>
    <script>{js}</script>
    <script>renderStaticReport({md_json}, 'static-report-container');</script>
    </body></html>
    """
    st_components.html(html, height=height, scrolling=True)
