"""
Q-Trial FastAPI server — exposes run_agentic_insights as an HTTP endpoint.

Run from backend/:
    poetry run uvicorn qtrial_backend.api:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
import re
import threading
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from rich.console import Console

from qtrial_backend.agentic.orchestrator import run_agentic_insights
from qtrial_backend.agentic.report_comparison import analyst_report_extension_supported
from qtrial_backend.agentic.schemas import MetadataInput
from qtrial_backend.agent.runner import run_statistical_agent_loop
from qtrial_backend.providers.gemini_client import set_thread_emit
from qtrial_backend.providers.openrouter_client import set_thread_model as set_openrouter_model
from qtrial_backend.providers.bedrock_client import set_thread_model as set_bedrock_model
from qtrial_backend.report.static import build_static_report
from qtrial_backend.dataset.treatment_detector import detect_treatment_columns
from qtrial_backend.dataset.load import classify_missingness
from qtrial_backend.report.adl import build_adl
from qtrial_backend.quantum import run_qubo_feature_selection

console = Console()

# ── Data dictionary sidecar loader ───────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data"


def _load_column_dict(dataset_name: str) -> dict[str, str] | None:
    """
    Look for a sidecar JSON data dictionary matching the dataset name.
    Tries <dataset_name>_dict.json and <dataset_name>.dict.json in the
    bundled data/ directory.  Returns None when not found.
    """
    candidates = [
        _DATA_DIR / f"{dataset_name}_dict.json",
        _DATA_DIR / f"{dataset_name}.dict.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _is_binary_like(series: pd.Series) -> bool:
    return series.dropna().nunique() <= 3


def _resolve_endpoint_column(df: pd.DataFrame, meta: MetadataInput | None) -> str | None:
    if df.empty:
        return None

    normalized_columns = {_normalize_column_name(col): col for col in df.columns}

    if meta and meta.primary_endpoint:
        if meta.primary_endpoint in df.columns:
            return meta.primary_endpoint
        normalized_meta = _normalize_column_name(meta.primary_endpoint)
        if normalized_meta in normalized_columns:
            return normalized_columns[normalized_meta]

    strong_endpoint_names = {
        "deathevent",
        "death",
        "mortality",
        "outcome",
        "survived",
        "response",
    }
    weak_endpoint_names = {"event", "status"}

    best_match: tuple[int, str] | None = None
    for col in df.columns:
        normalized = _normalize_column_name(col)
        binary_like = _is_binary_like(df[col])
        score = 0

        if normalized in strong_endpoint_names:
            score = 4
        elif normalized in weak_endpoint_names and binary_like:
            score = 3
        elif binary_like and any(token in normalized for token in ("death", "mortality", "surviv")):
            score = 2

        if score and (best_match is None or score > best_match[0]):
            best_match = (score, col)

    return best_match[1] if best_match else None


def _preserve_endpoint_missingness(
    missingness_disclosures: dict[str, Any],
    endpoint_column: str | None,
) -> list[str]:
    excluded_cols = [col for col, disclosure in missingness_disclosures.items() if disclosure.action == "excluded"]
    if endpoint_column and endpoint_column in missingness_disclosures:
        disclosure = missingness_disclosures[endpoint_column]
        if disclosure.action == "excluded":
            disclosure.action = "high_missingness_section"
        disclosure.excluded_from_primary_analysis = False
        excluded_cols = [col for col in excluded_cols if col != endpoint_column]
    return excluded_cols


def _ensure_endpoint_selected(
    df: pd.DataFrame,
    quantum_evidence: dict | None,
    endpoint_column: str | None,
    protected_columns: list[str] | None = None,
) -> pd.DataFrame:
    if quantum_evidence is None:
        return df

    selected_columns = list(quantum_evidence.get("selected_columns") or [])
    protected_columns = [c for c in (protected_columns or []) if c in df.columns]
    if endpoint_column and endpoint_column in df.columns and endpoint_column not in selected_columns:
        selected_columns.append(endpoint_column)
        quantum_evidence["selected_columns"] = selected_columns
        quantum_evidence["n_selected"] = len(selected_columns)
        excluded_columns = list(quantum_evidence.get("excluded_columns") or [])
        quantum_evidence["excluded_columns"] = [
            col for col in excluded_columns if col != endpoint_column
        ]

    added_protected = [c for c in protected_columns if c not in selected_columns]
    if added_protected:
        selected_columns.extend(added_protected)
        quantum_evidence["selected_columns"] = selected_columns
        quantum_evidence["n_selected"] = len(selected_columns)
        excluded_columns = list(quantum_evidence.get("excluded_columns") or [])
        quantum_evidence["excluded_columns"] = [col for col in excluded_columns if col not in added_protected]
        quantum_evidence["protected_columns"] = protected_columns
        quantum_evidence["protected_added_columns"] = added_protected

    return df[selected_columns] if selected_columns else df


_CLINICAL_PROTECTED_HINTS = re.compile(
    r"\b(age|sex|gender|sodium|na\b|creatinine|egfr|ejection|ef\b|blood\s+pressure|sbp|dbp|"
    r"diabet|anaemi|hypertens|smok|mortality|death|surviv|treat|dose|baseline|severity|"
    r"biomarker|lab|platelet|hemoglobin|cpk|creatinine\s+phosphokinase)\b",
    re.IGNORECASE,
)


def _compute_protected_columns(
    df: pd.DataFrame,
    *,
    endpoint_column: str | None,
    analyst_report_text: str | None,
    column_dict: dict[str, str] | None,
    meta: MetadataInput | None,
) -> list[str]:
    """
    Compute clinically important variables that should be protected from QUBO exclusion.
    
    Protects:
    1. Endpoint/outcome column
    2. Variables listed in metadata important_variables
    3. Variables matching clinical keywords (age, sodium, creatinine, etc.)
    4. Variables mentioned in data dictionary descriptions
    5. Variables mentioned in human analyst report
    
    Args:
        df: DataFrame with candidate columns
        endpoint_column: Name of endpoint/outcome column
        analyst_report_text: Uploaded human analyst report text
        column_dict: Data dictionary mapping column names to descriptions
        meta: Metadata input with important_variables list
    
    Returns:
        Sorted list of protected column names
    """
    protected: set[str] = set()
    
    # Rule 1: Always protect endpoint column
    if endpoint_column and endpoint_column in df.columns:
        protected.add(endpoint_column)
    
    # Rule 2: Protect variables from metadata important_variables list
    if meta and meta.important_variables:
        for v in meta.important_variables:
            if v in df.columns:
                protected.add(v)
            else:
                norm_v = _normalize_column_name(v)
                for col in df.columns:
                    if _normalize_column_name(col) == norm_v:
                        protected.add(col)
                        break
    
    # Rule 3: Protect clinically important variables by name or data dictionary semantics.
    for col in df.columns:
        if _CLINICAL_PROTECTED_HINTS.search(col):
            protected.add(col)
            continue
        if column_dict:
            desc = column_dict.get(col) or ""
            if desc and _CLINICAL_PROTECTED_HINTS.search(desc):
                protected.add(col)
    
    # Rule 4: If a human analyst report is provided, protect mentioned variables so they remain analyzable/comparable.
    if analyst_report_text:
        lowered = analyst_report_text.lower()
        for col in df.columns:
            token = col.lower()
            if len(token) >= 4 and token in lowered:
                protected.add(col)
                continue
            normalized = _normalize_column_name(col)
            if len(normalized) >= 4 and normalized in _normalize_column_name(lowered):
                protected.add(col)
    
    return sorted(protected)


async def _read_dataset_upload(file: UploadFile) -> tuple[pd.DataFrame, str]:
    content = await file.read()
    fname = file.filename or ""
    try:
        if fname.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")
    return df, fname


async def _read_optional_analyst_report(
    analyst_report_file: UploadFile | None,
) -> tuple[str | None, str | None]:
    if analyst_report_file is None:
        return None, None

    filename = analyst_report_file.filename or "analyst_report.txt"
    if not analyst_report_extension_supported(filename, analyst_report_file.content_type):
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported analyst report format. "
                "Supported formats: .txt, .md, .markdown, .text, .rst, .json, and UTF-8 text/plain uploads."
            ),
        )

    _MAX_ANALYST_REPORT_BYTES = 25 * 1024 * 1024  # 25 MB
    raw = await analyst_report_file.read(_MAX_ANALYST_REPORT_BYTES + 1)
    if len(raw) > _MAX_ANALYST_REPORT_BYTES:
        raise HTTPException(status_code=413, detail="Analyst report exceeds the 25 MB size limit.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="Analyst report must be a UTF-8 text file.",
        ) from exc

    if not text.strip():
        raise HTTPException(status_code=422, detail="Analyst report file is empty.")
    if filename.lower().endswith(".json"):
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid analyst report JSON: {exc}") from exc
    return filename, text

app = FastAPI(
    title="Q-Trial Statistical Reasoning Engine",
    version="0.1.0",
    description="Multi-agent LLM pipeline for clinical trial dataset analysis.",
)

# Allow Streamlit frontend (localhost:8501) and React dev server (localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", "http://127.0.0.1:8501",
        "http://localhost:5173", "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "qtrial-backend"}


@app.post("/api/run")
async def run_analysis(
    file: UploadFile = File(..., description="CSV or XLSX dataset"),
    analyst_report_file: UploadFile | None = File(
        None,
        description="Optional human analyst report (.txt, .md, .json, UTF-8 text)",
    ),
    provider: str = Form("gemini", description="openai | gemini | claude"),
    run_judge: bool = Form(False),
    max_rows: int = Form(25),
    study_context: str = Form(..., description="Plain-language study description (required)"),
    metadata_json: str | None = Form(
        None,
        description="Optional JSON string matching MetadataInput schema",
    ),
) -> JSONResponse:
    """
    Upload a dataset and run the full agentic pipeline.

    Returns the serialised FinalReportSchema as JSON.
    """
    # ── Read uploaded file ───────────────────────────────────────────────────
    df, fname = await _read_dataset_upload(file)
    analyst_report_name, analyst_report_text = await _read_optional_analyst_report(analyst_report_file)

    # ── Parse optional metadata ──────────────────────────────────────────────
    meta: MetadataInput | None = None
    if metadata_json:
        try:
            raw = json.loads(metadata_json)
            meta = MetadataInput.model_validate(raw)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid metadata JSON: {exc}"
            )

    # ── Run static statistical report first (deterministic, no LLM) ─────────
    dataset_name = fname.rsplit(".", 1)[0] if fname else "dataset"
    column_dict = _load_column_dict(dataset_name)
    endpoint_column = _resolve_endpoint_column(df, meta)

    # ── Classify missingness and drop >50% columns ───────────────────────────
    missingness_disclosures = classify_missingness(df)
    excluded_cols = _preserve_endpoint_missingness(missingness_disclosures, endpoint_column)
    if excluded_cols:
        df = df.drop(columns=excluded_cols)

    # ── QUBO Feature Selection (before static report) ────────────────────────
    quantum_evidence = None
    selected_df = df
    try:
        from qtrial_backend.dataset.evidence import build_dataset_evidence
        profile = build_dataset_evidence(df)
        quantum_evidence = await asyncio.to_thread(
            run_qubo_feature_selection, df, profile, endpoint_column, 0.5
        )
        protected_columns = _compute_protected_columns(
            df,
            endpoint_column=endpoint_column,
            analyst_report_text=analyst_report_text,
            column_dict=column_dict,
            meta=meta,
        )
        selected_df = _ensure_endpoint_selected(df, quantum_evidence, endpoint_column, protected_columns)
        console.print(
            f"[green]✓ Feature selection:[/green] "
            f"Selected {quantum_evidence['n_selected']} from {quantum_evidence['n_candidates']} columns "
            f"(redundancy reduced by {quantum_evidence['redundancy_reduction_pct']:.1f}%)"
        )
    except Exception as exc:
        console.print(f"[yellow]⚠ Feature selection warning: {exc}[/yellow]")
        quantum_evidence = None
        selected_df = df

    clinical_analysis_result: dict | None = None
    methodology_chapter: str | None = None
    try:
        static_report, methodology_chapter, clinical_analysis_result = await asyncio.to_thread(build_static_report, selected_df, dataset_name, None, quantum_evidence)
    except Exception:
        static_report = None

    # ── Run LLM-driven statistical agent loop ────────────────────────────────
    try:
        loop_report, tool_log = await asyncio.to_thread(
            run_statistical_agent_loop, selected_df, provider, dataset_name, None, None, column_dict, quantum_evidence
        )
    except Exception as exc:
        console.print(f"[red]⚠ Statistical agent loop FAILED: {exc}[/red]")
        console.print(traceback.format_exc())
        loop_report, tool_log = None, None

    parts = [p for p in [static_report, loop_report] if p]
    analysis_report = "\n\n---\n\n".join(parts) if parts else None

    # ── Run pipeline in thread pool (blocking → async) ───────────────────────
    try:
        report = await asyncio.to_thread(
            run_agentic_insights,
            selected_df,
            provider,
            max_rows,
            30,
            run_judge,
            meta,
            False,
            analysis_report,
            tool_log,
            None,
            study_context,
            column_dict,
            list(missingness_disclosures.values()),
            clinical_analysis_result,
            methodology_chapter,
            analyst_report_text,
            analyst_report_name,
            original_df=df,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Pipeline error: {exc}"
        )

    return JSONResponse(content=report.model_dump(mode="json"))


@app.post("/api/run/stream")
async def run_analysis_stream(
    file: UploadFile = File(..., description="CSV or XLSX dataset"),
    analyst_report_file: UploadFile | None = File(
        None,
        description="Optional human analyst report (.txt, .md, .json, UTF-8 text)",
    ),
    provider: str = Form("gemini", description="openai | gemini | claude | openrouter"),
    model: str | None = Form(None, description="Model override (used for openrouter)"),
    run_judge: bool = Form(False),
    max_rows: int = Form(25),
    study_context: str = Form(..., description="Plain-language study description (required)"),
    metadata_json: str | None = Form(
        None,
        description="Optional JSON string matching MetadataInput schema",
    ),
    confirmed_treatment_columns: list[str] = Form(default=[]),
    dict_file: UploadFile | None = File(None, description="Optional JSON column dictionary"),
    feature_selection_method: str = Form("mrmr", description="Feature selection method: univariate, mrmr, lasso, elastic_net, qubo"),
) -> StreamingResponse:
    """
    Same as /api/run but streams Server-Sent Events so the frontend can show
    live per-stage progress.  Each event is a JSON object on a `data: ...\\n\\n` line:

      {"type": "stage_complete", "stage": "DataQualityAgent", "message": "..."}
      {"type": "complete",       "data": {<FinalReportSchema>}}
      {"type": "error",          "message": "..."}
    """
    df, fname = await _read_dataset_upload(file)
    analyst_report_name, analyst_report_text = await _read_optional_analyst_report(analyst_report_file)

    meta: MetadataInput | None = None
    if metadata_json:
        try:
            raw = json.loads(metadata_json)
            meta = MetadataInput.model_validate(raw)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid metadata JSON: {exc}"
            )

    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()
    dataset_name = fname.rsplit(".", 1)[0] if fname else "dataset"
    endpoint_column = _resolve_endpoint_column(df, meta)

    # Classify missingness and drop >50% columns before pipeline
    missingness_disclosures = classify_missingness(df)
    excluded_cols = _preserve_endpoint_missingness(missingness_disclosures, endpoint_column)
    if excluded_cols:
        df = df.drop(columns=excluded_cols)

    # Uploaded dict takes priority over bundled sidecar
    column_dict: dict[str, str] | None = None
    if dict_file is not None:
        try:
            raw_dict = await dict_file.read()
            column_dict = json.loads(raw_dict.decode("utf-8"))
        except Exception:
            column_dict = None
    if column_dict is None:
        column_dict = _load_column_dict(dataset_name)

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(aq.put_nowait, event)

    def _run_pipeline() -> None:
        set_thread_emit(emit)
        set_openrouter_model(model if provider == "openrouter" else None)
        set_bedrock_model(model if provider == "bedrock" else None)
        try:
            # ── 0. Feature Selection ──────────────────────────────────────────
            quantum_evidence = None
            selected_df = df
            try:
                from qtrial_backend.dataset.evidence import build_dataset_evidence
                from qtrial_backend.feature_selection import select_features
                protected_columns = _compute_protected_columns(
                    df,
                    endpoint_column=endpoint_column,
                    analyst_report_text=analyst_report_text,
                    column_dict=column_dict,
                    meta=meta,
                )
                protected_columns = list(
                    dict.fromkeys(
                        protected_columns
                        + [col for col in confirmed_treatment_columns if col in df.columns]
                    )
                )
                method = feature_selection_method.lower()
                
                if method == 'qubo':
                    # Use QUBO feature selection
                    profile = build_dataset_evidence(df)
                    quantum_evidence = run_qubo_feature_selection(df, profile, endpoint_column, 0.5)
                    quantum_evidence["method"] = "qubo"
                    selected_df = _ensure_endpoint_selected(df, quantum_evidence, endpoint_column, protected_columns)
                else:
                    from qtrial_backend.feature_selection.utils import (
                        handle_mixed_types,
                        mean_pairwise_correlation,
                    )

                    candidate_pre = [c for c in df.columns if c != endpoint_column]
                    try:
                        prep_pre, _, _, _ = handle_mixed_types(df[candidate_pre])
                        X_pre = prep_pre.values.astype(float)
                        redundancy_before = float(mean_pairwise_correlation(X_pre)) if X_pre.shape[1] > 1 else 0.0
                    except Exception:
                        redundancy_before = 0.0

                    fs_result = select_features(df, outcome_column=endpoint_column, method=method)
                    selected_features = [
                        col for col in fs_result.get("selected_features", []) if col in df.columns
                    ]
                    must_include = [
                        c for c in protected_columns if c not in selected_features and c in df.columns
                    ]
                    selected_columns = list(dict.fromkeys(selected_features + must_include))
                    excluded_columns = [col for col in df.columns if col not in selected_columns]

                    redundancy_after = float(fs_result.get("redundancy_measure", 0.0))
                    redundancy_reduction_pct = (
                        (redundancy_before - redundancy_after) / redundancy_before * 100
                        if redundancy_before > 0 else 0.0
                    )
                    quantum_evidence = {
                        'n_candidates': max(len(df.columns) - (1 if endpoint_column else 0), 0),
                        'n_selected': len(selected_columns),
                        'selected_columns': selected_columns,
                        'excluded_columns': excluded_columns,
                        'protected_columns': protected_columns,
                        'protected_added_columns': must_include,
                        'redundancy_before': round(redundancy_before, 3),
                        'redundancy_after': round(redundancy_after, 3),
                        'redundancy_reduction_pct': round(redundancy_reduction_pct, 1),
                        'method': method,
                        'selection_method': method,
                    }
                    selected_df = df[selected_columns] if selected_columns else df
                
                console.print(
                    f"[green]✓ Feature selection:[/green] "
                    f"Method={method}, "
                    f"Selected {quantum_evidence['n_selected']} from {quantum_evidence['n_candidates']} features"
                )
            except Exception as exc:
                console.print(f"[yellow]⚠ Feature selection warning: {exc}[/yellow]")
                quantum_evidence = None
                selected_df = df

            # ── 1. Deterministic static report ───────────────────────────────
            clinical_analysis_result: dict | None = None
            methodology_chapter: str | None = None
            try:
                static_report, methodology_chapter, clinical_analysis_result = build_static_report(selected_df, dataset_name, emit=emit, quantum_evidence=quantum_evidence)
            except Exception:
                static_report = None

            if static_report is not None:
                # Include methodology chapter in the display-only SSE payload
                _display_static = static_report
                if methodology_chapter:
                    _display_static = static_report + "\n\n" + methodology_chapter
                loop.call_soon_threadsafe(
                    aq.put_nowait,
                    {
                        "type": "stage_complete",
                        "stage": "StaticAnalysis",
                        "message": "Static statistical report ready",
                        "static_report": _display_static,
                    },
                )

            # ── 2. LLM-driven statistical agent loop ─────────────────────────
            try:
                loop_report, tool_log = run_statistical_agent_loop(
                    selected_df, provider, dataset_name, emit=emit,
                    model=model if provider in ("openrouter", "bedrock") else None,
                    column_dict=column_dict,
                    quantum_evidence=quantum_evidence,
                )
            except Exception as exc:
                console.print(f"[red]⚠ Statistical agent loop FAILED (provider={provider}): {exc}[/red]")
                console.print(traceback.format_exc())
                loop_report, tool_log = None, None
                event_type = "error" if provider == "bedrock" else "warning"
                loop.call_soon_threadsafe(
                    aq.put_nowait,
                    {
                        "type": event_type,
                        "message": (
                            f"[{provider.upper()} agent loop failed] {type(exc).__name__}: {exc}"
                        ),
                    },
                )

            # ── 3. Combine static + loop report for the reasoning pipeline ───
            parts = [p for p in [static_report, loop_report] if p]
            analysis_report = "\n\n---\n\n".join(parts) if parts else None

            # ── 4. Full agentic + reasoning pipeline ─────────────────────────
            report = run_agentic_insights(
                selected_df, provider, max_rows, 30, run_judge, meta, False,
                analysis_report, tool_log, emit,
                study_context, column_dict,
                list(missingness_disclosures.values()),
                clinical_analysis_result,
                methodology_chapter,
                analyst_report_text,
                analyst_report_name,
                original_df=df,
            )
            loop.call_soon_threadsafe(
                aq.put_nowait,
                {"type": "complete", "data": report.model_dump(mode="json")},
            )
        except Exception as exc:
            loop.call_soon_threadsafe(
                aq.put_nowait,
                {"type": "error", "message": str(exc)},
            )

    def _sanitize_for_sse(obj: Any) -> Any:
        """
        Recursively walk obj and replace any string values that contain raw
        control characters (\\x00-\\x1f except \\t) with a sanitized version.
        This prevents 'Unterminated string' JSON parse errors on the frontend
        caused by LLM outputs that embed literal newlines inside string values.
        """
        if isinstance(obj, str):
            # Replace raw control characters that break JSON strings
            # (json.dumps with ensure_ascii handles \\n etc., but only when
            #  the string is a top-level value — nested dicts need this pass first)
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize_for_sse(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_sse(v) for v in obj]
        return obj

    def _safe_json(obj: Any) -> str:
        """
        Serialize obj to a JSON string that is safe to embed in an SSE line.

        Uses ensure_ascii=True so all non-ASCII characters are \\uXXXX-escaped,
        which also prevents unterminated-string errors caused by raw newlines or
        control characters inside LLM-generated text fields.
        """
        sanitized = _sanitize_for_sse(obj)
        return json.dumps(sanitized, default=str, ensure_ascii=True)

    async def event_generator():
        t = threading.Thread(target=_run_pipeline, daemon=True)
        t.start()
        # Poll the queue in short intervals so we can emit SSE keepalive
        # comments while the pipeline is running. This prevents proxies and
        # browsers from closing the connection during long-running stages.
        # Hard timeout: 30 minutes (1800s) — enough for 25 agent iterations
        # + parallel literature validation + synthesis.
        _KEEPALIVE_INTERVAL = 25.0   # seconds between keepalive pings
        _HARD_TIMEOUT = 1800.0       # 30 minutes total
        elapsed = 0.0
        while True:
            try:
                event = await asyncio.wait_for(aq.get(), timeout=_KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                elapsed += _KEEPALIVE_INTERVAL
                if elapsed >= _HARD_TIMEOUT:
                    yield (
                        f"data: {_safe_json({'type': 'error', 'message': 'Pipeline timed out after 30 minutes'})}\n\n"
                    )
                    break
                # SSE comment — keeps the connection alive, ignored by the parser
                yield ": keepalive\n\n"
                continue
            elapsed = 0.0  # reset on any real event
            try:
                serialized = _safe_json(event)
            except Exception as ser_exc:
                serialized = _safe_json({"type": "error", "message": f"Serialization error: {ser_exc}"})
            yield f"data: {serialized}\n\n"
            if event.get("type") in ("complete", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── New endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/detect-treatment")
async def detect_treatment(
    file: UploadFile = File(..., description="CSV or XLSX dataset"),
) -> JSONResponse:
    """
    Detect candidate treatment columns in the uploaded dataset using the
    name-pattern + cardinality heuristic.
    """
    content = await file.read()
    fname = file.filename or ""
    try:
        if fname.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    candidates = detect_treatment_columns(df)
    return JSONResponse(content={"candidate_columns": candidates})


@app.post("/api/report/pdf")
async def generate_pdf(
    report_json: str = Form(..., description="Serialised FinalReportSchema JSON"),
    dataset_hash: str = Form("", description="SHA-256 hash of the input dataset"),
) -> Any:
    """Generate a PDF report from the serialised FinalReportSchema."""
    try:
        from qtrial_backend.report.pdf_generator import generate_pdf_report
        from qtrial_backend.agentic.schemas import FinalReportSchema as _FRS
        import json as _json
        report_obj = _FRS.model_validate(_json.loads(report_json))
        pdf_bytes = await asyncio.to_thread(generate_pdf_report, report_obj, dataset_hash)
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=report.pdf"},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="PDF generation is temporarily unavailable. Your interactive report is still accessible.",
        )


@app.get("/api/report/reproducibility/{run_id}")
async def get_reproducibility_log(run_id: str) -> JSONResponse:
    """Return the reproducibility log for a completed analysis run."""
    import json as _json
    from pathlib import Path as _Path
    log_path = _Path("outputs") / f"{run_id}_reproducibility.json"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Reproducibility log not found.")
    try:
        data = _json.loads(log_path.read_text(encoding="utf-8"))
        return JSONResponse(content=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read log: {exc}")


@app.get("/api/report/adl")
async def get_adl() -> Any:
    """Return the Architecture Decision Log as Markdown."""
    from fastapi.responses import Response
    adl_text = build_adl()
    return Response(content=adl_text, media_type="text/markdown")
