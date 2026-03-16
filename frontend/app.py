"""
Q-Trial · Statistical Reasoning Engine
Streamlit dashboard — professional black/white interface

Setup (one-time):
    cd frontend
    pip install -r requirements.txt

Run (backend must be running first):
    # Terminal 1 — backend
    cd backend
    poetry run uvicorn qtrial_backend.api:app --port 8000

    # Terminal 2 — frontend
    cd frontend
    streamlit run app.py
"""
from __future__ import annotations

import html as _html
import io
import json
import os
import traceback
from typing import Any

import pandas as pd
import requests
import streamlit as st

from components import render_pipeline_tracker_from_report, render_static_report

API_BASE = os.getenv("QTRIAL_API_BASE_URL", "http://localhost:8000").rstrip("/")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Q-Trial",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background: #ffffff;
    color: #111111;
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}
[data-testid="stSidebar"] {
    background: #f7f7f7;
    border-right: 1.5px solid #e4e4e4;
}
[data-testid="stSidebar"] h2 {
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555;
    border-bottom: 1px solid #ddd;
    padding-bottom: 4px;
    margin-top: 16px;
}
h1, h2, h3 { color: #000; font-weight: 700; letter-spacing: -0.02em; }
h2 {
    font-size: 1.1rem;
    border-bottom: 2px solid #000;
    padding-bottom: 4px;
    margin-top: 24px;
}
[data-baseweb="tab-list"] {
    border-bottom: 2px solid #000;
    gap: 0;
    background: transparent;
}
[data-baseweb="tab"] {
    color: #777;
    font-size: 13px;
    font-weight: 500;
    padding: 10px 16px;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: transparent;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #000;
    font-weight: 700;
    border-bottom: 2px solid #000;
    background: transparent;
}
.stButton > button {
    background: #000 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    letter-spacing: 0.03em;
    width: 100%;
    transition: background 0.15s;
}
.stButton > button:hover { background: #333 !important; }
details[data-testid="stExpander"] {
    border: 1px solid #e0e0e0 !important;
    border-radius: 4px !important;
    margin-bottom: 8px;
}
details[data-testid="stExpander"] summary { font-size: 13px; font-weight: 600; padding: 10px 14px; }
[data-testid="stMetric"] { background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px 12px; }
[data-testid="stMetricLabel"] { font-size: 10px !important; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #666 !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 800; color: #000 !important; }
[data-testid="stFileUploader"] { border: 1.5px dashed #bbb; border-radius: 6px; padding: 8px; }
[data-testid="stAlert"] { border-radius: 4px; }
/* custom */
.qtrial-header {
    display: flex; align-items: center; gap: 14px;
    background: #000; color: #fff;
    padding: 22px 28px; border-radius: 8px; margin-bottom: 28px;
}
.qtrial-header .title  { font-size: 1.5rem; font-weight: 800; margin: 0; line-height: 1; }
.qtrial-header .sub    { color: #aaa; font-size: 12px; margin: 4px 0 0; letter-spacing: 0.04em; }
.card { background:#fff; border:1px solid #e0e0e0; border-radius:6px; padding:14px 18px; margin-bottom:10px; }
.card-dark { background:#f5f5f5; border:1px solid #d0d0d0; border-radius:6px; padding:14px 18px; margin-bottom:10px; }
.section-label { font-size:10px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#888; margin-bottom:6px; }
.badge { display:inline-block; padding:2px 9px; border-radius:3px; font-size:11px; font-weight:700; margin-right:4px; line-height:1.6; }
.badge-high    { background:#111; color:#fff; }
.badge-medium  { background:#555; color:#fff; }
.badge-low     { background:#aaa; color:#fff; }
.badge-ok      { background:#e8f5e9; color:#1b5e20; border:1px solid #c8e6c9; }
.badge-warn    { background:#fff8e1; color:#7b5800; border:1px solid #ffe082; }
.badge-error   { background:#fce4ec; color:#880e4f; border:1px solid #f8bbd0; }
.badge-neutral { background:#f0f0f0; color:#444; border:1px solid #ddd; }
.badge-blue    { background:#e3f2fd; color:#0d47a1; border:1px solid #bbdefb; }
.badge-purple  { background:#f3e5f5; color:#4a148c; border:1px solid #e1bee7; }
.finding-item { border-left:3px solid #000; padding:8px 14px; margin-bottom:8px; background:#fafafa; border-radius:0 4px 4px 0; font-size:13px; }
.risk-item    { border-left:3px solid #888; padding:8px 14px; margin-bottom:8px; background:#fafafa; border-radius:0 4px 4px 0; font-size:13px; }
.score-row { margin-bottom:14px; }
.score-label { font-size:12px; color:#444; margin-bottom:4px; font-weight:600; }
.score-bar  { background:#f0f0f0; border-radius:3px; height:8px; }
.score-fill { background:#111; border-radius:3px; height:8px; }
.lit-card { border:1px solid #e0e0e0; border-radius:6px; padding:14px 18px; margin-bottom:10px; background:#fafafa; }
.lit-alias { display:inline-block; font-size:11px; font-weight:700; color:#4a148c; background:#f3e5f5; border:1px solid #e1bee7; padding:2px 8px; border-radius:3px; margin-bottom:6px; }
.qa-q { background:#f5f5f5; border-left:3px solid #000; padding:10px 14px; margin-bottom:8px; font-size:13px; border-radius:0 4px 4px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _e(v: Any) -> str:
    return _html.escape(str(v) if v is not None else "")

def _badge(text: str, kind: str = "medium") -> str:
    return f'<span class="badge badge-{kind}">{_e(text)}</span>'

def _sev_badge(sev: str) -> str:
    kind = {"high": "high", "medium": "medium", "low": "low"}.get(sev, "neutral")
    return _badge(sev.upper(), kind)

def _conf_kind(c: str) -> str:
    return {"high": "ok", "medium": "warn", "low": "error"}.get(c, "neutral")

def _score_bar(label: str, score: int) -> str:
    pct = max(0, min(100, score))
    return (
        f'<div class="score-row"><div class="score-label">{_e(label)} — {pct}/100</div>'
        f'<div class="score-bar"><div class="score-fill" style="width:{pct}%"></div></div></div>'
    )

def _g(d: dict, *keys: str, default: Any = None) -> Any:
    """Safe nested dict getter."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


# ── API client ────────────────────────────────────────────────────────────────

def _api_run(
    file_bytes: bytes,
    file_name: str,
    provider: str,
    run_judge: bool,
    max_rows: int,
    metadata_json: str | None = None,
) -> dict | None:
    """POST to /api/run, return parsed JSON dict or None on failure."""
    form_data = {
        "provider": (None, provider),
        "run_judge": (None, str(run_judge).lower()),
        "max_rows": (None, str(max_rows)),
    }
    if metadata_json:
        form_data["metadata_json"] = (None, metadata_json)

    files = {"file": (file_name, io.BytesIO(file_bytes), "application/octet-stream")}

    try:
        resp = requests.post(
            f"{API_BASE}/api/run",
            data={k: v[1] for k, v in form_data.items()},
            files=files,
            timeout=300,  # 5-minute timeout for pipeline runs
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to backend. Start it with:\n\n"
            "```\ncd backend\npoetry run uvicorn qtrial_backend.api:app --port 8000\n```"
        )
        return None
    except requests.exceptions.Timeout:
        st.error("Pipeline timed out (>5 min). Try with fewer rows or a smaller dataset.")
        return None
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"Backend error {exc.response.status_code}: {detail or exc}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


def _api_run_stream(
    file_bytes: bytes,
    file_name: str,
    provider: str,
    run_judge: bool,
    max_rows: int,
    metadata_json: str | None = None,
) -> dict | None:
    """POST to /api/run/stream, consuming SSE events with live progress display.
    Must be called while a st.status() context is active.
    Returns the final report dict or None on failure."""
    form_data: dict = {
        "provider": provider,
        "run_judge": str(run_judge).lower(),
        "max_rows": str(max_rows),
    }
    if metadata_json:
        form_data["metadata_json"] = metadata_json

    try:
        resp = requests.post(
            f"{API_BASE}/api/run/stream",
            data=form_data,
            files={"file": (file_name, io.BytesIO(file_bytes), "application/octet-stream")},
            stream=True,
            timeout=420,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to backend. Start it with:\n\n"
            "```\ncd backend\npoetry run uvicorn qtrial_backend.api:app --port 8000\n```"
        )
        return None
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"Backend error {exc.response.status_code}: {detail or exc}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error starting stream: {exc}")
        return None

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8", errors="replace")
        if not raw_line.startswith("data: "):
            continue
        try:
            event = json.loads(raw_line[6:])
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")
        stage = event.get("stage", "")
        msg   = event.get("message", "")

        if etype == "stage_complete":
            st.write(f"✓ **{stage}** — {msg}")
            # Capture static report from the StaticAnalysis stage event
            if stage == "StaticAnalysis" and "static_report" in event:
                st.session_state["static_report"] = event["static_report"]
        elif etype == "progress":
            st.write(f"⟳ {msg}")
        elif etype == "complete":
            return event.get("data")
        elif etype == "error":
            st.error(f"Pipeline error: {msg}")
            return None

    st.error("Stream ended without a result — check backend logs.")
    return None


def _check_backend_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Header ────────────────────────────────────────────────────────────────────

def _render_header() -> None:
    st.markdown(
        '<div class="qtrial-header">'
        '<div style="font-size:2rem;">⚕</div>'
        '<div>'
        '<div class="title">Q-Trial</div>'
        '<div class="sub">Statistical Reasoning Engine · Clinical Trial Dataset Analysis</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Landing ───────────────────────────────────────────────────────────────────

def _render_landing() -> None:
    st.markdown("### Upload a dataset to begin")
    st.markdown(
        "Q-Trial uses a multi-agent LLM pipeline to analyse clinical trial datasets. "
        "Upload a CSV or XLSX, choose a provider, then click **Run Analysis**."
    )
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Tasks covered**")
        st.markdown(
            "- Task 3 — Interactive Q&A closed-loop\n"
            "- Task 4B — Hypothesis-driven tool dispatch\n"
            "- Task 5 — Robustness guardrails\n"
            "- Task 6 — Literature RAG (PubMed)\n"
        )
    with c2:
        st.markdown("**Agents**")
        st.markdown(
            "- DataQualityAgent\n"
            "- ClinicalSemanticsAgent\n"
            "- UnknownsAgent\n"
            "- InsightSynthesisAgent\n"
            "- JudgeAgent *(optional)*\n"
        )
    with c3:
        st.markdown("**Providers**")
        st.markdown(
            "- Google Gemini *(recommended)*\n"
            "- OpenAI GPT\n"
            "- Anthropic Claude\n"
        )


# ── Tab: Overview ─────────────────────────────────────────────────────────────

def _tab_overview(r: dict, file_name: str) -> None:
    rs     = r.get("reasoning_state") or {}
    cs     = rs.get("confidence_summary") or {}
    judge  = r.get("judge_after") or r.get("judge") or {}
    lr     = r.get("literature_report") or {}
    gr     = r.get("guardrail_report") or {}

    confidence  = (cs.get("overall") or "—").upper()
    judge_score = judge.get("overall_score")
    n_lit       = len(lr.get("articles", []))
    n_flags     = len(gr.get("flags", []))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("File",        (file_name or "—")[:20])
    m2.metric("Provider",    r.get("provider", "—"))
    m3.metric("Confidence",  confidence)
    m4.metric("Judge Score", f"{judge_score}/100" if judge_score is not None else "Not run")
    m5.metric("Literature",  f"{n_lit} article(s)")

    st.markdown("---")
    plan = r.get("plan") or {}
    st.markdown("## Execution Plan")
    st.markdown(f"*{_e(plan.get('dataset_summary', ''))}*")
    for step in plan.get("steps", []):
        st.markdown(
            f"**{step['step_number']}.** {_e(step['name'])} "
            f"<span style='color:#888;font-size:12px;'>→ {_e(step['agent_to_call'])}</span>",
            unsafe_allow_html=True,
        )
        st.caption(step.get("goal", ""))

    st.markdown("---")
    st.markdown("## Reasoning Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Hypotheses:** {len(rs.get('hypotheses', []))}")
        st.markdown(f"**Claims:** {len(rs.get('claims', []))}")
        st.markdown(f"**Dispatched tools:** {len(rs.get('dispatched_tool_results', []))}")
        st.markdown(f"**Guardrail flags:** {n_flags}")
        if cs:
            st.markdown(f"**Supported claims:** {cs.get('num_supported_claims', 0)}")
            st.markdown(f"**Flagged claims:** {cs.get('num_flagged_claims', 0)}")
    with col2:
        sc = rs.get("stop_condition") or {}
        if sc:
            met = sc.get("met", False)
            st.markdown(
                f'<div class="card-dark">'
                f'<div class="section-label">Stop Condition</div>'
                f'{_badge("COMPLETE" if met else "INCOMPLETE", "ok" if met else "warn")}'
                f'<div style="font-size:13px;margin-top:6px;">{_e(sc.get("reason",""))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        limiting = cs.get("limiting_factors", [])
        if limiting:
            st.markdown("**Limiting factors:**")
            for lf in limiting:
                st.markdown(f"- {lf}")

    unresolved = (r.get("unknowns") or {}).get("unresolved_high_impact", [])
    if unresolved:
        st.warning(
            f"⚠ {len(unresolved)} high-impact unknown(s) unresolved — "
            "use the **Interactive Q&A** tab to answer and re-run."
        )


# ── Tab: Guardrails (Task 5) ──────────────────────────────────────────────────

def _tab_guardrails(r: dict) -> None:
    gr = r.get("guardrail_report") or {}
    if not gr:
        st.info("No guardrail report in this run.")
        return

    st.markdown(f"**{_e(gr.get('summary', ''))}**")
    rm = gr.get("repeated_measures")
    if rm:
        st.markdown(
            f'<div class="card">'
            f'{_badge("REPEATED MEASURES DETECTED", "blue")}'
            f'<div style="margin-top:8px;font-size:13px;">'
            f'ID col: <code>{_e(rm["id_column"])}</code> · '
            f'{rm["n_subjects"]} subjects · {rm["total_rows"]} rows · '
            f'max {rm["max_repeats_per_subject"]} repeats · '
            f'longitudinal: <strong>{"Yes" if rm.get("likely_longitudinal") else "No"}</strong>'
            f'</div>'
            f'<div style="font-size:12px;color:#666;margin-top:4px;">{_e(rm.get("detail",""))}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    flags = gr.get("flags", [])
    st.markdown("---")
    if not flags:
        st.success("✓ No guardrail flags raised.")
        return

    st.markdown(f"## {len(flags)} Flag(s)")
    for i, flag in enumerate(flags):
        with st.expander(
            f"[guardrails[{i}]]  {flag['check_type'].replace('_',' ').title()} · "
            f"col: {flag.get('column') or '—'} · {flag['severity'].upper()}"
        ):
            st.markdown(
                f'{_badge(flag["check_type"].replace("_"," ").title(), "neutral")}'
                f'{_sev_badge(flag["severity"])}',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Detail:** {flag.get('detail', '')}")
            st.markdown(f"**Suggested action:** {flag.get('suggested_action', '')}")
            ev = flag.get("evidence")
            if ev:
                st.json(ev)


# ── Tab: Hypotheses (Task 4A / 4C) ───────────────────────────────────────────

def _tab_hypotheses(r: dict) -> None:
    rs = r.get("reasoning_state") or {}
    hyps = rs.get("hypotheses", [])
    if not hyps:
        st.info("No hypotheses generated.")
        return

    st.markdown(f"## {len(hyps)} Hypothesis/Hypotheses")
    for h in hyps:
        status = h.get("status", "candidate")
        status_kind = {
            "supported": "ok", "contradicted": "error",
            "candidate": "neutral", "deferred": "warn", "dropped": "error",
        }.get(status, "neutral")
        stmt = h.get("statement", "")
        with st.expander(f"[{h['hypothesis_id'].upper()}] {stmt[:90]}{'...' if len(stmt)>90 else ''}"):
            conf = h.get("confidence", "low")
            st.markdown(
                f'{_badge(status.upper(), status_kind)}'
                f'{_badge("Confidence: " + conf.upper(), _conf_kind(conf))}'
                f'<span class="section-label" style="margin-left:6px;line-height:1.8;">'
                f'{_e(h.get("source_agent",""))}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Statement:** {stmt}")
            for ev in h.get("evidence_support", []):
                icon = "✓" if ev.get("supports_claim") else "✗"
                st.markdown(f"- {icon} `{ev['citation_key']}` — {ev.get('explanation','')}")
            for c_key in h.get("contradictions", []):
                st.markdown(f"- ✗ contradiction: `{c_key}`")

    fcs = rs.get("falsification_checks", [])
    if fcs:
        st.markdown("---")
        st.markdown("## Falsification Checks (Task 4C)")
        for fc in fcs:
            verdict = fc.get("verdict", "inconclusive")
            v_kind = {"supports": "ok", "contradicts": "error"}.get(verdict, "neutral")
            st.markdown(
                f'<div class="card">'
                f'{_badge(fc["hypothesis_id"].upper(), "neutral")}'
                f'{_badge(verdict.upper(), v_kind)}'
                f'<div style="font-weight:600;font-size:13px;margin-top:6px;">'
                f'{_e(fc.get("test_description",""))}</div>'
                f'<div style="font-size:12px;color:#555;margin-top:4px;">'
                f'✓ If true: {_e(fc.get("expected_if_true",""))}<br>'
                f'✗ If false: {_e(fc.get("expected_if_false",""))}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    hqs = rs.get("hidden_questions", [])
    if hqs:
        st.markdown("---")
        st.markdown("## Hidden Questions (Task 4C)")
        for q in hqs:
            imp = q.get("impact", "low")
            st.markdown(
                f'<div class="card">'
                f'{_badge(imp.upper() + " IMPACT", {"high":"high","medium":"warn","low":"low"}.get(imp,"neutral"))}'
                f'<span class="section-label" style="margin-left:6px;">{_e(q.get("category",""))}</span>'
                f'<div style="font-weight:600;font-size:13px;margin-top:6px;">{_e(q.get("question",""))}</div>'
                f'<div style="font-size:12px;color:#555;margin-top:3px;">{_e(q.get("rationale",""))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Tab: Tool Dispatch (Task 4B) ──────────────────────────────────────────────

def _tab_dispatch(r: dict) -> None:
    rs = r.get("reasoning_state") or {}
    results = rs.get("dispatched_tool_results", [])
    if not results:
        st.info("No tool dispatch results in this run.")
        return

    n_ok  = sum(1 for d in results if not d.get("error"))
    n_err = len(results) - n_ok
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Dispatched", len(results))
    c2.metric("Succeeded", n_ok)
    c3.metric("Errors", n_err)
    st.markdown("---")

    for d in results:
        req   = d.get("request") or {}
        alias = d.get("citation_alias", "—")
        tool  = d.get("tool_called", "—")
        hid   = req.get("hypothesis_id", "—")
        icon  = "✓" if not d.get("error") else "✗"
        with st.expander(f"[{alias}] {icon} {tool} → h={hid}"):
            st.markdown(
                f'{_badge(alias, "blue")}'
                f'{_badge(req.get("tool_type","").replace("_"," ").title(), "neutral")}'
                f'{_badge("h=" + hid.upper(), "neutral")}'
                f'{_badge(req.get("priority","").upper() + " PRIORITY", _conf_kind(req.get("priority","low")))}',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Tool called:** `{tool}`")
            st.markdown(f"**Rationale:** {req.get('rationale', '')}")
            cols_used = req.get("columns", [])
            if cols_used:
                st.markdown(f"**Columns:** {', '.join(f'`{c}`' for c in cols_used)}")
            gc = req.get("group_column")
            if gc:
                st.markdown(f"**Group column:** `{gc}`")
            if d.get("error"):
                st.error(f"Error: {d['error']}")
            elif d.get("result"):
                st.markdown("**Result:**")
                st.json(d["result"])


# ── Tab: Static Analysis (deterministic statistical report) ──────────────────

def _tab_static_analysis(r: dict) -> None:
    static_md = r.get("prior_analysis_report") or st.session_state.get("static_report")
    if not static_md:
        st.info(
            "No static analysis report available. "
            "This is generated automatically when the pipeline runs."
        )
        return

    st.markdown(
        '<div class="section-label">Deterministic Analysis — No LLM</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This report was generated purely from the data using statistical tools "
        "(normality tests, correlations, survival analysis, baseline balance, etc.) "
        "before any AI agent runs. It is fed as grounding context to the reasoning pipeline."
    )

    render_static_report(static_md, height=650)


# ── Tab: Statistical (dedicated statistical engine outputs) ─────────────────

def _tab_statistical(r: dict) -> None:
    rs = r.get("reasoning_state") or {}
    results = rs.get("dispatched_tool_results", [])
    if not results:
        st.info("No statistical tool results in this run.")
        return

    stat_tools = {
        "survival_analysis",
        "baseline_balance",
        "group_by_summary",
        "distribution_info",
        "correlation",
        "crosstab",
        "hypothesis_test",
        "normality_test",
        "pairwise_test",
        "regression",
        "effect_size",
        "column_stats",
    }

    stat_results = [
        d for d in results
        if (d.get("tool_called") in stat_tools) or (d.get("request", {}).get("tool_type") in stat_tools)
    ]

    if not stat_results:
        st.info("No recognized statistical dispatch entries were found in this report.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Statistical Runs", len(stat_results))
    c2.metric("Successful", sum(1 for d in stat_results if not d.get("error")))
    c3.metric("Errors", sum(1 for d in stat_results if d.get("error")))

    st.markdown("---")
    st.markdown("## Key Statistical Findings")

    survival = next((d for d in stat_results if d.get("tool_called") == "survival_analysis"), None)
    if survival and survival.get("result"):
        payload = next(iter((survival.get("result") or {}).values()), {})
        col1, col2, col3 = st.columns(3)
        col1.metric("N Total", payload.get("n_total", "-"))
        col2.metric("N Events", payload.get("n_events", "-"))
        event_rate = payload.get("event_rate_pct")
        col3.metric("Event Rate", f"{event_rate}%" if event_rate is not None else "-")

    balance = next((d for d in stat_results if d.get("tool_called") == "baseline_balance"), None)
    if balance and balance.get("result"):
        payload = next(iter((balance.get("result") or {}).values()), {})
        st.markdown("### Baseline Balance")
        left, right = st.columns(2)
        left.metric("Variables Checked", payload.get("n_variables", "-"))
        right.metric("Imbalanced", payload.get("n_imbalanced", "-"))
        imbalanced = payload.get("imbalanced_variables", [])
        if imbalanced:
            st.markdown("**Imbalanced variables:** " + ", ".join(f"`{v}`" for v in imbalanced))

    st.markdown("---")
    st.markdown("## Statistical Tool Outputs")
    for d in stat_results:
        req = d.get("request") or {}
        alias = d.get("citation_alias", "-")
        tool = d.get("tool_called", "-")
        tool_type = req.get("tool_type", "-")
        with st.expander(f"[{alias}] {tool} ({tool_type})"):
            rationale = req.get("rationale")
            if rationale:
                st.markdown(f"**Rationale:** {rationale}")
            cols = req.get("columns", [])
            if cols:
                st.markdown("**Columns:** " + ", ".join(f"`{c}`" for c in cols))
            grp = req.get("group_column")
            if grp:
                st.markdown(f"**Group column:** `{grp}`")
            if d.get("error"):
                st.error(d["error"])
            else:
                st.json(d.get("result") or {})


# ── Tab: Literature (Task 6) ──────────────────────────────────────────────────

def _lit_article_card(art: dict, verdict_html: str = "") -> str:
    """Return the HTML for a single literature card, with an optional verdict block."""
    src     = art.get("source", "pubmed")
    pid     = art.get("paper_id", "")
    link    = (
        f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
        if src == "pubmed"
        else f"https://www.semanticscholar.org/paper/{pid}"
    )
    authors = art.get("authors", [])
    authors_str = (
        ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        if authors else "—"
    )
    cite_c = art.get("citation_count")
    tldr   = art.get("tldr") or ""
    link_label = "View on PubMed ↗" if src == "pubmed" else "View on Semantic Scholar ↗"
    return (
        f'<div class="lit-card">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        f'<span class="lit-alias">{_e(art.get("citation_alias",""))}</span>'
        f'{_badge(src.upper(), "purple")}'
        f'<span style="font-size:11px;color:#888;">{_e(art.get("year","—"))}</span>'
        + (f'<span style="font-size:11px;color:#888;">{_e(str(cite_c))} citations</span>' if cite_c is not None else "")
        + f'</div>'
        f'<div style="font-weight:700;font-size:14px;margin-bottom:4px;">{_e(art.get("title",""))}</div>'
        f'<div style="font-size:12px;color:#555;margin-bottom:6px;">{_e(authors_str)}</div>'
        f'<div style="font-size:12px;color:#333;line-height:1.5;">{_e(art.get("abstract_snippet",""))}</div>'
        + (f'<div style="font-size:12px;font-style:italic;color:#444;margin-top:6px;border-top:1px solid #eee;padding-top:6px;">TLDR: {_e(tldr)}</div>' if tldr else "")
        + verdict_html
        + f'<div style="margin-top:10px;padding-top:6px;border-top:1px solid #eee;">'
        f'<a href="{_e(link)}" target="_blank" style="font-size:11px;color:#000;text-decoration:underline;font-weight:600;">'
        f'{link_label}'
        f'</a></div></div>'
    )


def _tab_literature(r: dict) -> None:
    lr  = r.get("literature_report") or {}
    rs  = r.get("reasoning_state") or {}
    arts = lr.get("articles", [])

    if not arts:
        st.info("No literature retrieved in this run.")
        return

    # ── Build lookups ─────────────────────────────────────────────────────────
    # group articles by the hypothesis query that generated them
    from collections import defaultdict
    by_query: dict[str, list] = defaultdict(list)
    for art in arts:
        by_query[art.get("search_query", "")].append(art)

    # map citation_alias → list of evidence verdicts (from reasoning engine)
    alias_verdicts: dict[str, list] = defaultdict(list)
    for hyp in rs.get("hypotheses", []):
        for ev in hyp.get("evidence_support", []):
            alias = ev.get("citation_key", "")
            if alias:
                alias_verdicts[alias].append({
                    "hypothesis_id": hyp.get("hypothesis_id", ""),
                    "statement": hyp.get("statement", ""),
                    "status": hyp.get("status", ""),
                    "supports": ev.get("supports_claim", False),
                    "explanation": ev.get("explanation", ""),
                })

    queries_used = lr.get("queries_used", [])
    hypotheses   = rs.get("hypotheses", [])

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown(f"**{_e(lr.get('summary', ''))}**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles", lr.get("total_retrieved", len(arts)))
    col2.metric("Hypotheses grounded", len(queries_used))
    col3.metric("Sources", ", ".join(lr.get("sources_used", [])) or "—")
    col4.metric("With verdicts", sum(1 for a in arts if a.get("citation_alias","") in alias_verdicts))

    st.markdown("---")

    # ── Per-hypothesis evidence view ──────────────────────────────────────────
    st.markdown("## Hypothesis Verification by Literature")
    st.caption(
        "Each hypothesis is shown with its reasoning verdict and the articles "
        "retrieved to ground or refute it."
    )

    matched_aliases: set[str] = set()

    for i, query in enumerate(queries_used):
        hyp      = hypotheses[i] if i < len(hypotheses) else None
        hyp_arts = by_query.get(query, [])

        status  = (hyp.get("status", "candidate") if hyp else "unknown")
        stmt    = (hyp.get("statement", query)     if hyp else query)
        conf    = (hyp.get("confidence", "low")    if hyp else "low")
        hid     = (hyp.get("hypothesis_id", f"h{i+1}") if hyp else f"h{i+1}")

        status_kind = {
            "supported": "ok", "contradicted": "error",
            "candidate": "neutral", "deferred": "warn", "dropped": "error",
        }.get(status, "neutral")

        # headline: coloured verdict + title
        n_arts_label = f"{len(hyp_arts)} article(s)" if hyp_arts else "no articles"
        with st.expander(
            f"[{hid.upper()}] {stmt[:75]}{'...' if len(stmt)>75 else ''}  ·  {n_arts_label}",
            expanded=True,
        ):
            # verdict banner
            st.markdown(
                f'<div style="background:#f8f8f8;border:1px solid #e0e0e0;border-radius:6px;'
                f'padding:12px 16px;margin-bottom:12px;">'
                f'<div class="section-label">Reasoning Engine Verdict</div>'
                f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px;">'
                f'{_badge(status.upper(), status_kind)}'
                f'{_badge("Confidence: " + conf.upper(), _conf_kind(conf))}'
                f'</div>'
                f'<div style="font-size:13px;font-weight:600;margin-top:8px;color:#111;">{_e(stmt)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if not hyp_arts:
                st.caption("No articles were retrieved for this hypothesis query.")
                continue

            for art in hyp_arts:
                alias = art.get("citation_alias", "")
                matched_aliases.add(alias)

                # build evidence verdict block from reasoning engine
                verdicts = alias_verdicts.get(alias, [])
                verdict_html = ""
                if verdicts:
                    blocks = []
                    for vd in verdicts:
                        colour_bg     = "#e8f5e9" if vd["supports"] else "#fce4ec"
                        colour_border = "#2e7d32" if vd["supports"] else "#c62828"
                        icon          = "✓ Supports hypothesis" if vd["supports"] else "✗ Contradicts hypothesis"
                        vkind         = "ok" if vd["supports"] else "error"
                        expl          = vd.get("explanation", "")
                        blocks.append(
                            f'<div style="margin-top:10px;padding:8px 12px;'
                            f'border-left:3px solid {colour_border};'
                            f'background:{colour_bg};border-radius:0 4px 4px 0;">'
                            f'<div style="font-size:11px;font-weight:700;">'
                            f'{_badge(icon, vkind)}</div>'
                            f'<div style="font-size:12px;color:#333;margin-top:4px;">{_e(expl)}</div>'
                            f'</div>'
                        )
                    verdict_html = "".join(blocks)
                else:
                    verdict_html = (
                        '<div style="margin-top:10px;padding:6px 10px;'
                        'border-left:3px solid #bbb;background:#f5f5f5;'
                        'border-radius:0 4px 4px 0;font-size:12px;color:#666;">'
                        'No explicit verdict citation from reasoning engine for this article.'
                        '</div>'
                    )

                st.markdown(_lit_article_card(art, verdict_html), unsafe_allow_html=True)

    # ── Articles not matched to a hypothesis ──────────────────────────────────
    unmatched = [a for a in arts if a.get("citation_alias", "") not in matched_aliases]
    if unmatched:
        st.markdown("---")
        st.markdown("## Other Retrieved Articles")
        st.caption("These articles were retrieved but not directly matched to a specific hypothesis query.")
        for art in unmatched:
            alias = art.get("citation_alias", "")
            verdicts = alias_verdicts.get(alias, [])
            verdict_html = ""
            if verdicts:
                for vd in verdicts:
                    colour_bg     = "#e8f5e9" if vd["supports"] else "#fce4ec"
                    colour_border = "#2e7d32" if vd["supports"] else "#c62828"
                    icon          = "✓ Supports hypothesis" if vd["supports"] else "✗ Contradicts hypothesis"
                    vkind         = "ok" if vd["supports"] else "error"
                    verdict_html += (
                        f'<div style="margin-top:10px;padding:8px 12px;'
                        f'border-left:3px solid {colour_border};background:{colour_bg};'
                        f'border-radius:0 4px 4px 0;">'
                        f'<div style="font-size:11px;font-weight:700;">{_badge(icon, vkind)}</div>'
                        f'<div style="font-size:12px;color:#333;margin-top:4px;">{_e(vd.get("explanation",""))}</div>'
                        f'</div>'
                    )
            st.markdown(_lit_article_card(art, verdict_html), unsafe_allow_html=True)

    # ── Search queries used ───────────────────────────────────────────────────
    with st.expander("Search queries sent to PubMed"):
        for qi, q in enumerate(queries_used):
            hyp_lbl = f" ← {hypotheses[qi]['hypothesis_id'].upper()}" if qi < len(hypotheses) else ""
            st.markdown(f"- `{q}`{hyp_lbl}")


# ── Tab: Insights ─────────────────────────────────────────────────────────────

def _tab_insights(r: dict) -> None:
    # prefer post-metadata re-run
    insights = r.get("final_insights_after") or r.get("final_insights") or {}
    if not insights:
        st.info("No insights generated.")
        return

    kf = insights.get("key_findings", [])
    if kf:
        st.markdown("## Key Findings")
        for f in kf:
            st.markdown(f'<div class="finding-item">{_e(f)}</div>', unsafe_allow_html=True)

    rb = insights.get("risks_and_bias_signals", [])
    if rb:
        st.markdown("## Risks & Bias Signals")
        for risk in rb:
            st.markdown(f'<div class="risk-item">{_e(risk)}</div>', unsafe_allow_html=True)

    ra = insights.get("recommended_next_analyses", [])
    if ra:
        st.markdown("## Recommended Next Analyses")
        for item in ra:
            analysis = item.get("analysis", "")
            with st.expander(f"#{item.get('rank','?')}. {analysis[:80]}{'...' if len(analysis)>80 else ''}"):
                st.markdown(f"**Analysis:** {analysis}")
                st.markdown(f"**Rationale:** {item.get('rationale', '')}")
                ec = item.get("evidence_citation")
                if ec:
                    st.markdown(f'{_badge(ec, "blue")}', unsafe_allow_html=True)

    rq = insights.get("required_metadata_or_questions", [])
    if rq:
        st.markdown("## Further Information Requested")
        for q in rq:
            st.markdown(f"- {q}")

    # Before/after comparison
    before = r.get("final_insights_before") or {}
    after  = r.get("final_insights_after") or {}
    if before and after:
        st.markdown("---")
        with st.expander("Compare: Before vs After metadata re-run"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Before (initial)**")
                for f in before.get("key_findings", []):
                    st.markdown(f"- {f}")
            with c2:
                st.markdown("**After (re-synthesised)**")
                for f in after.get("key_findings", []):
                    st.markdown(f"- {f}")


# ── Tab: Judge ────────────────────────────────────────────────────────────────

def _judge_grade(score: int) -> tuple[str, str]:
    """Return (letter_grade, interpretation) for a 0-100 score."""
    if score >= 90:
        return "A", "Excellent — strong evidence support, minimal overreach."
    if score >= 75:
        return "B", "Good — well-supported with minor gaps."
    if score >= 60:
        return "C", "Fair — notable issues in evidence or reasoning."
    if score >= 45:
        return "D", "Weak — significant unsupported claims or overreach."
    return "F", "Poor — pervasive issues; consider re-running with richer context."


def _tab_judge(r: dict) -> None:
    judge = r.get("judge_after") or r.get("judge") or {}
    if not judge:
        st.info("Judge was not run. Enable the **Run Judge Agent** toggle in the sidebar and re-run.")
        return

    score  = judge.get("overall_score", 0)
    rubric = judge.get("rubric") or {}

    grade, interpretation = _judge_grade(score)
    failed   = judge.get("failed_claims", [])
    rewrites = judge.get("rewrite_instructions", [])
    n_high   = sum(1 for fc in failed if fc.get("severity") == "high")
    n_med    = sum(1 for fc in failed if fc.get("severity") == "medium")

    # ── Score + grade panel ───────────────────────────────────────────────────
    col_score, col_grade, col_stats = st.columns([1, 1, 2])
    with col_score:
        color = "#1b5e20" if score >= 75 else "#7b5800" if score >= 50 else "#880e4f"
        st.markdown(
            f'<div style="text-align:center;padding:24px 16px;border:2px solid {color};'
            f'border-radius:8px;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:0.1em;color:#666;'
            f'text-transform:uppercase;margin-bottom:6px;">Overall Score</div>'
            f'<div style="font-size:3.5rem;font-weight:900;color:{color};line-height:1;">{score}</div>'
            f'<div style="font-size:13px;color:#888;">/100</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_grade:
        st.markdown(
            f'<div style="text-align:center;padding:24px 16px;border:1px solid #e0e0e0;'
            f'border-radius:8px;background:#f8f8f8;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:0.1em;color:#666;'
            f'text-transform:uppercase;margin-bottom:6px;">Grade</div>'
            f'<div style="font-size:3.5rem;font-weight:900;color:#111;line-height:1;">{grade}</div>'
            f'<div style="font-size:11px;color:#666;margin-top:4px;">{interpretation}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_stats:
        st.markdown("**Claim audit**")
        st.markdown(
            f'<div class="card">'
            f'<div style="display:flex;gap:24px;align-items:center;">'
            f'<div style="text-align:center;">'
            f'<div style="font-size:1.8rem;font-weight:900;color:#880e4f;">{n_high}</div>'
            f'<div style="font-size:10px;text-transform:uppercase;color:#666;">High-sev failures</div>'
            f'</div>'
            f'<div style="text-align:center;">'
            f'<div style="font-size:1.8rem;font-weight:900;color:#7b5800;">{n_med}</div>'
            f'<div style="font-size:10px;text-transform:uppercase;color:#666;">Medium-sev failures</div>'
            f'</div>'
            f'<div style="text-align:center;">'
            f'<div style="font-size:1.8rem;font-weight:900;color:#1b5e20;">{len(rewrites)}</div>'
            f'<div style="font-size:10px;text-transform:uppercase;color:#666;">Rewrite instructions</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Rubric breakdown ──────────────────────────────────────────────────────
    st.markdown("## Rubric Breakdown")
    rubric_items = [
        ("Evidence Support",
         "How well are claims backed by tool dispatch results and literature.",
         rubric.get("evidence_support", 0), False),
        ("Uncertainty Handling",
         "Are limitations and gaps acknowledged appropriately.",
         rubric.get("uncertainty_handling", 0), False),
        ("Internal Consistency",
         "Do hypotheses, findings, and conclusions align without contradiction.",
         rubric.get("internal_consistency", 0), False),
        ("Clinical Overreach",
         "Lower is better — overconfident causal claims are penalised.",
         rubric.get("clinical_overreach", 0), True),
    ]
    for label, desc, val, lower_is_better in rubric_items:
        pct   = max(0, min(100, val))
        color = "#880e4f" if lower_is_better and pct > 40 else "#1b5e20" if pct >= 70 else "#7b5800" if pct >= 45 else "#880e4f"
        st.markdown(
            f'<div class="score-row">'
            f'<div class="score-label" style="display:flex;justify-content:space-between;">'
            f'<span>{_e(label)}</span>'
            f'<span style="color:{color};font-weight:800;">{pct}/100'
            + (" ↓ lower is better" if lower_is_better else "")
            + f'</span></div>'
              f'<div style="font-size:11px;color:#888;margin-bottom:4px;">{_e(desc)}</div>'
              f'<div class="score-bar"><div class="score-fill" style="width:{pct}%;background:{color};"></div></div>'
              f'</div>',
            unsafe_allow_html=True,
        )

    # ── Judge reasoning ───────────────────────────────────────────────────────
    reasoning = judge.get("judge_reasoning", "")
    if reasoning:
        st.markdown("---")
        st.markdown("## Judge's Analysis")
        st.caption(
            "The Judge reviewed all hypotheses, tool dispatch results, and literature "
            "citations to produce the assessment below."
        )
        st.markdown(
            f'<div style="background:#f8f8f8;border-left:4px solid #000;'
            f'padding:16px 20px;border-radius:0 6px 6px 0;font-size:13px;line-height:1.7;">'
            f'{_e(reasoning)}</div>',
            unsafe_allow_html=True,
        )

    # ── Failed claims ─────────────────────────────────────────────────────────
    if failed:
        st.markdown("---")
        st.markdown(f"## {len(failed)} Failed Claim(s)")
        st.caption("Claims that the Judge found unsupported, overstated, or inconsistent.")

        # group by severity
        for target_sev in ("high", "medium", "low"):
            tier = [fc for fc in failed if fc.get("severity") == target_sev]
            if not tier:
                continue
            sev_color = {"high": "#880e4f", "medium": "#7b5800", "low": "#555"}.get(target_sev, "#555")
            st.markdown(
                f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.08em;color:{sev_color};margin:14px 0 6px;">▸ {target_sev} severity</div>',
                unsafe_allow_html=True,
            )
            for fc in tier:
                text = fc.get("claim_text", "")
                reason = fc.get("reason", "")
                me     = fc.get("missing_evidence") or ""
                with st.expander(f"{text[:90]}{'...' if len(text)>90 else ''}"):
                    st.markdown(
                        f'<div class="card">'
                        f'<div class="section-label">Claim</div>'
                        f'<div style="font-size:13px;margin-bottom:10px;">{_e(text)}</div>'
                        f'<div class="section-label">Why it failed</div>'
                        f'<div style="font-size:13px;color:#333;margin-bottom:10px;">{_e(reason)}</div>'
                        + (
                            f'<div class="section-label">Missing evidence</div>'
                            f'<div style="font-size:13px;color:#555;">{_e(me)}</div>'
                            if me else ""
                        )
                        + f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Rewrite instructions ──────────────────────────────────────────────────
    if rewrites:
        st.markdown("---")
        st.markdown("## How to Improve the Analysis")
        st.caption(
            "Act on these to raise the score: provide more metadata, run additional analyses, "
            "or clarify clinical context in the Interactive Q&A tab."
        )
        for i, instr in enumerate(rewrites, 1):
            st.markdown(
                f'<div class="finding-item">'
                f'<span style="font-size:10px;font-weight:700;color:#888;margin-right:8px;">#{i}</span>'
                f'{_e(instr)}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Tab: Agent Runs (raw per-agent outputs) ─────────────────────────────────

def _tab_agent_runs(r: dict) -> None:
    runs = r.get("agent_runs") or []
    if not runs:
        st.info("No agent outputs captured in this run.")
        return

    st.markdown(f"## {len(runs)} Agent Run(s)")
    st.caption("Full outputs from each agent step in execution order.")

    for run in runs:
        step = run.get("step_number", "?")
        agent = run.get("agent", "UnknownAgent")
        goal = run.get("goal", "")
        with st.expander(f"Step {step} - {agent}", expanded=False):
            if goal:
                st.markdown(f"**Goal:** {goal}")
            output = run.get("output")
            if output is None:
                st.caption("No output payload for this step.")
            else:
                st.json(output)


# ── Tab: All In One (single-scroll report) ───────────────────────────────────

def _tab_all_in_one(r: dict, file_name: str) -> None:
    st.caption("Single-page report view: all major outputs in one place.")

    st.markdown("## Overview")
    _tab_overview(r, file_name)

    st.markdown("---")
    st.markdown("## Static Analysis")
    _tab_static_analysis(r)

    st.markdown("---")
    st.markdown("## Guardrails")
    _tab_guardrails(r)

    st.markdown("---")
    st.markdown("## Hypotheses")
    _tab_hypotheses(r)

    st.markdown("---")
    st.markdown("## Statistical")
    _tab_statistical(r)

    st.markdown("---")
    st.markdown("## Tool Dispatch")
    _tab_dispatch(r)

    st.markdown("---")
    st.markdown("## Literature")
    _tab_literature(r)

    st.markdown("---")
    st.markdown("## Agent Runs")
    _tab_agent_runs(r)

    st.markdown("---")
    st.markdown("## Insights")
    _tab_insights(r)

    st.markdown("---")
    st.markdown("## Judge")
    _tab_judge(r)


# ── Tab: Interactive Q&A (Task 3) ─────────────────────────────────────────────

def _tab_qa(r: dict, file_bytes: bytes, file_name: str, provider: str,
            run_judge: bool, max_rows: int) -> None:
    unknowns_obj = r.get("unknowns") or {}
    unresolved   = unknowns_obj.get("unresolved_high_impact", [])
    all_unknowns = unknowns_obj.get("ranked_unknowns", [])

    if not unresolved:
        st.success("✓ All high-impact unknowns are resolved.")
    else:
        st.warning(
            f"⚠ {len(unresolved)} high-impact question(s) unresolved. "
            "Answer them below and click **Re-run with Answers**."
        )

    if all_unknowns:
        st.markdown("## All Ranked Unknowns")
        for u in all_unknowns:
            imp = u.get("impact", "low")
            is_unres = u.get("question", "") in unresolved
            st.markdown(
                f'<div class="qa-q">'
                f'{_badge("Q" + str(u.get("rank","")), "neutral")}'
                f'{_badge(imp.upper() + " IMPACT", {"high":"high","medium":"warn","low":"low"}.get(imp,"neutral"))}'
                f'{_badge(u.get("category","").replace("_"," ").upper(), "neutral")}'
                + (_badge("UNRESOLVED", "error") if is_unres else _badge("OK", "ok"))
                + f'<div style="font-weight:600;margin-top:4px;">{_e(u.get("question",""))}</div>'
                f'<div style="font-size:12px;color:#666;">{_e(u.get("rationale",""))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("## Provide Answers")
    st.caption("Fill what you know. Blank fields are skipped.")

    with st.form("qa_form"):
        col1, col2 = st.columns(2)
        with col1:
            time_unit = st.selectbox(
                "Time unit for follow-up/event columns",
                ["", "days", "months", "years"],
            )
            study_design = st.text_input(
                "Study design",
                placeholder="e.g. double-blind RCT",
            )
            primary_endpoint = st.text_input(
                "Primary endpoint",
                placeholder="e.g. overall survival",
            )
        with col2:
            status_mapping_raw = st.text_area(
                "Status/event code mapping (JSON)",
                placeholder='{"0": "censored", "2": "death"}',
                height=80,
            )
            treatment_arms_raw = st.text_area(
                "Treatment arm mapping (JSON)",
                placeholder='{"1": "D-penicillamine", "2": "placebo"}',
                height=80,
            )

        qa_answers: dict[str, str] = {}
        if unresolved:
            st.markdown("**Answer the unresolved questions:**")
            for i, q in enumerate(unresolved):
                ans = st.text_area(
                    f"Q{i+1}: {q[:110]}{'...' if len(q)>110 else ''}",
                    key=f"qa_{i}",
                    height=60,
                    placeholder="Your answer…",
                )
                qa_answers[q] = ans

        submitted = st.form_submit_button("Re-run with Answers")

    if submitted:
        status_map: dict | None = None
        if status_mapping_raw.strip():
            try:
                status_map = json.loads(status_mapping_raw)
            except json.JSONDecodeError:
                st.error("Invalid JSON in status mapping.")
                return
        trt_map: dict | None = None
        if treatment_arms_raw.strip():
            try:
                trt_map = json.loads(treatment_arms_raw)
            except json.JSONDecodeError:
                st.error("Invalid JSON in treatment arm mapping.")
                return

        additional = {q: a for q, a in qa_answers.items() if a.strip()}
        meta_dict: dict = {}
        if time_unit:
            meta_dict["time_unit"] = time_unit
        if study_design:
            meta_dict["study_design"] = study_design
        if primary_endpoint:
            meta_dict["primary_endpoint"] = primary_endpoint
        if status_map:
            meta_dict["status_mapping"] = status_map
        if trt_map:
            meta_dict["treatment_arms"] = trt_map
        if additional:
            meta_dict["additional_answers"] = additional

        metadata_json = json.dumps(meta_dict) if meta_dict else None

        with st.status("Re-running pipeline with your answers…", expanded=True) as s:
            new_report = _api_run_stream(
                file_bytes, file_name, provider, run_judge, max_rows, metadata_json
            )
        if new_report:
            s.update(label="✓ Re-run complete", state="complete", expanded=False)
            st.session_state["report"] = new_report
            st.success("✓ Re-run complete. All tabs have been updated.")
            st.rerun()


# ── Pre-run metadata helpers ──────────────────────────────────────────────────

def _parse_simple_map(text: str) -> dict | None:
    """Parse 'key=value, key=value' or 'key:value; key:value' into a dict."""
    import re
    result: dict = {}
    for part in re.split(r"[,;]", text):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([^=:]+)[=:](.+)$", part)
        if m:
            k, v = m.group(1).strip(), m.group(2).strip()
            if k and v:
                result[k] = v
    return result if result else None


def _pre_run_context_panel() -> str | None:
    """Render an optional 'Dataset context' expander in the sidebar.
    Returns a MetadataInput-compatible JSON string, or None if nothing filled in."""
    with st.expander("📋 Dataset context (optional)"):
        st.caption(
            "Provide known facts **before** running so the pipeline's questions "
            "focus on genuinely unknown design decisions — not basic column codings."
        )
        status_raw = st.text_input(
            "Event / status code meanings",
            placeholder="0=censored, 1=transplant, 2=death",
            key="ctx_status",
        )
        time_unit = st.selectbox(
            "Follow-up time unit",
            ["", "days", "months", "years"],
            key="ctx_time_unit",
        )
        arms_raw = st.text_input(
            "Treatment arm codes",
            placeholder="1=D-penicillamine, 2=placebo",
            key="ctx_arms",
        )
        design = st.text_input(
            "Study design",
            placeholder="double-blind RCT / prospective cohort / …",
            key="ctx_design",
        )

    meta: dict = {}
    if status_raw.strip():
        parsed = _parse_simple_map(status_raw)
        if parsed:
            meta["status_mapping"] = parsed
    if time_unit:
        meta["time_unit"] = time_unit
    if arms_raw.strip():
        parsed = _parse_simple_map(arms_raw)
        if parsed:
            meta["treatment_arms"] = parsed
    if design.strip():
        meta["study_design"] = design.strip()

    return json.dumps(meta) if meta else None


def _infer_foundation_requirements(df: pd.DataFrame | None) -> dict[str, Any]:
    """Infer which foundational clarifications should be collected before analysis."""
    if df is None or df.empty:
        return {
            "need_status_mapping": False,
            "need_time_unit": False,
            "status_candidates": [],
            "time_candidates": [],
        }

    cols = list(df.columns)
    lowered = {c: str(c).strip().lower() for c in cols}

    # Detect likely time/follow-up columns.
    time_candidates = [
        c for c in cols
        if any(k in lowered[c] for k in ("time", "follow", "duration", "survival"))
    ]

    # Detect likely endpoint/status columns with compact numeric coding (e.g., 0/1/2).
    status_name_candidates = [
        c for c in cols
        if any(k in lowered[c] for k in ("event", "status", "death", "outcome"))
    ]
    status_candidates: list[str] = []
    for c in status_name_candidates:
        s = df[c]
        non_null = s.dropna()
        if non_null.empty:
            continue
        unique_vals = sorted(non_null.unique().tolist())
        if len(unique_vals) <= 4 and all(isinstance(v, (int, float)) for v in unique_vals):
            if all(float(v).is_integer() for v in unique_vals):
                status_candidates.append(c)

    return {
        "need_status_mapping": len(status_candidates) > 0,
        "need_time_unit": len(time_candidates) > 0,
        "status_candidates": status_candidates,
        "time_candidates": time_candidates,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        # Backend health check
        healthy = _check_backend_health()
        if healthy:
            st.markdown(
                '<div style="background:#e8f5e9;border:1px solid #c8e6c9;border-radius:4px;'
                'padding:6px 10px;font-size:12px;color:#1b5e20;font-weight:600;">'
                '● Backend connected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#fce4ec;border:1px solid #f8bbd0;border-radius:4px;'
                'padding:6px 10px;font-size:12px;color:#880e4f;font-weight:600;">'
                '● Backend offline</div>',
                unsafe_allow_html=True,
            )
            st.caption("Start with: `poetry run uvicorn qtrial_backend.api:app --port 8000`")

        st.markdown("## Dataset")
        uploaded = st.file_uploader(
            "Upload CSV or XLSX",
            type=["csv", "xlsx"],
        )

        st.markdown("## Configuration")
        provider = st.selectbox(
            "LLM Provider",
            ["gemini", "openai", "claude"],
            index=0,
        )
        run_judge = st.toggle("Run Judge Agent", value=True)
        with st.expander("⚙ Advanced settings"):
            max_rows = st.number_input(
                "LLM context rows",
                min_value=10,
                max_value=500,
                value=25,
                step=5,
                help=(
                    "Number of dataset rows included in the LLM context window. "
                    "Higher values give richer analysis but use more tokens."
                ),
            )

        pre_metadata_json = _pre_run_context_panel()

        st.markdown("## Run")
        run_clicked = st.button("▶  Run Analysis", use_container_width=True)

        if st.session_state.get("report"):
            st.markdown("---")
            st.success("✓ Report ready")
            if st.button("Clear results", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    return uploaded, provider, run_judge, max_rows, run_clicked, pre_metadata_json


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    _render_header()

    uploaded, provider, run_judge, max_rows, run_clicked, pre_metadata_json = _sidebar()

    # Read + store uploaded file
    file_bytes: bytes | None = None
    file_name  = ""
    df: pd.DataFrame | None = None

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_name  = uploaded.name
        try:
            if file_name.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(file_bytes))
            else:
                df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as exc:
            st.error(f"Failed to read file: {exc}")
            return
        st.session_state.update({
            "file_bytes": file_bytes, "file_name": file_name,
            "provider": provider, "run_judge": run_judge, "max_rows": max_rows,
        })

    # Restore from session
    if file_bytes is None:
        file_bytes = st.session_state.get("file_bytes")
        file_name  = st.session_state.get("file_name", "")
        provider   = st.session_state.get("provider", provider)
        run_judge  = st.session_state.get("run_judge", run_judge)
        max_rows   = st.session_state.get("max_rows", max_rows)
        if file_bytes:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes)) if not file_name.endswith(".xlsx") \
                     else pd.read_excel(io.BytesIO(file_bytes))
            except Exception:
                pass

    # Dataset preview
    if df is not None:
        with st.expander(
            f"Dataset preview — {file_name}  ({len(df)} rows × {len(df.columns)} cols)"
        ):
            st.dataframe(df.head(10), use_container_width=True)

    # Pre-analysis foundation gate (collect basic coding context first).
    foundation = _infer_foundation_requirements(df)
    pre_meta: dict[str, Any] = {}
    if pre_metadata_json:
        try:
            pre_meta = json.loads(pre_metadata_json)
        except Exception:
            pre_meta = {}

    inline_meta: dict[str, Any] = {}

    if df is not None and (foundation.get("need_status_mapping") or foundation.get("need_time_unit")):
        st.markdown("### Pre-analysis clarifications")
        if foundation.get("status_candidates"):
            cand = ", ".join(f"`{c}`" for c in foundation["status_candidates"])
            st.caption(f"Detected likely status/event columns: {cand}")
        if foundation.get("time_candidates"):
            cand = ", ".join(f"`{c}`" for c in foundation["time_candidates"])
            st.caption(f"Detected likely follow-up/time columns: {cand}")

        c1, c2 = st.columns(2)
        with c1:
            if foundation.get("need_status_mapping"):
                status_mapping_inline = st.text_input(
                    "Event/status code meanings",
                    value=st.session_state.get("pre_status_mapping_inline", ""),
                    placeholder="0=censored, 1=alive, 2=death",
                    key="pre_status_mapping_inline",
                    help="Accepted formats: 0=censored,1=death or JSON like {\"0\":\"censored\",\"2\":\"death\"}.",
                ).strip()
                if status_mapping_inline:
                    parsed_status: dict[str, Any] | None = None
                    try:
                        parsed_status = json.loads(status_mapping_inline)
                    except json.JSONDecodeError:
                        parsed_status = _parse_simple_map(status_mapping_inline)
                    if parsed_status:
                        inline_meta["status_mapping"] = parsed_status
                    else:
                        st.error("Could not parse event/status mapping. Use key=value pairs or valid JSON.")

        with c2:
            if foundation.get("need_time_unit"):
                default_time = ""
                if isinstance(pre_meta.get("time_unit"), str):
                    default_time = pre_meta["time_unit"]
                time_unit_inline = st.selectbox(
                    "Follow-up time unit",
                    ["", "days", "months", "years"],
                    index=["", "days", "months", "years"].index(default_time) if default_time in ("", "days", "months", "years") else 0,
                    key="pre_time_unit_inline",
                )
                if time_unit_inline:
                    inline_meta["time_unit"] = time_unit_inline

    effective_meta = dict(pre_meta)
    effective_meta.update(inline_meta)

    missing_foundations: list[str] = []
    if foundation.get("need_status_mapping") and not effective_meta.get("status_mapping"):
        missing_foundations.append("Provide event/status code meanings (e.g., 0/1/2 labels).")
    if foundation.get("need_time_unit") and not effective_meta.get("time_unit"):
        missing_foundations.append("Provide follow-up time unit (days/months/years).")

    effective_metadata_json = json.dumps(effective_meta) if effective_meta else None

    if df is not None and (foundation.get("need_status_mapping") or foundation.get("need_time_unit")):
        if missing_foundations:
            st.warning(
                "Please complete the foundation inputs below before running.\n\n"
                + "\n".join(f"- {m}" for m in missing_foundations)
            )
        else:
            st.success("✓ Foundation clarifications provided. Ready to run analysis.")

    # Trigger run
    if run_clicked:
        if file_bytes is None:
            st.warning("Please upload a dataset first.")
        elif missing_foundations:
            st.error("Run blocked until required pre-analysis clarifications are provided above.")
        else:
            with st.status("Running Q-Trial pipeline…", expanded=True) as status_box:
                st.write("▶ Starting multi-agent pipeline…")
                result = _api_run_stream(
                    file_bytes, file_name, provider, run_judge, max_rows,
                    effective_metadata_json,
                )
            if result:
                st.session_state["report"] = result
                status_box.update(label="✓ Analysis complete", state="complete", expanded=False)
                st.rerun()
            else:
                status_box.update(label="✗ Analysis failed", state="error", expanded=True)

    # Display results
    report: dict | None = st.session_state.get("report")
    if report is None:
        _render_landing()
        return

    # Pipeline progress tracker (shows stage completion visually)
    render_pipeline_tracker_from_report(report, height=100)

    tabs = st.tabs([
        "All In One",
        "Overview",
        "Static Analysis",
        "Guardrails",
        "Hypotheses",
        "Statistical",
        "Tool Dispatch",
        "Literature",
        "Agent Runs",
        "Insights",
        "Judge",
        "Interactive Q&A",
    ])

    fn = file_name or st.session_state.get("file_name", "")
    with tabs[0]: _tab_all_in_one(report, fn)
    with tabs[1]: _tab_overview(report, fn)
    with tabs[2]: _tab_static_analysis(report)
    with tabs[3]: _tab_guardrails(report)
    with tabs[4]: _tab_hypotheses(report)
    with tabs[5]: _tab_statistical(report)
    with tabs[6]: _tab_dispatch(report)
    with tabs[7]: _tab_literature(report)
    with tabs[8]: _tab_agent_runs(report)
    with tabs[9]: _tab_insights(report)
    with tabs[10]: _tab_judge(report)
    with tabs[11]:
        _tab_qa(
            report,
            file_bytes or st.session_state.get("file_bytes", b""),
            fn,
            provider,
            run_judge,
            max_rows,
        )


if __name__ == "__main__":
    main()
