"""
Q-Trial FastAPI server — exposes run_agentic_insights as an HTTP endpoint.

Run from backend/:
    poetry run uvicorn qtrial_backend.api:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
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
from qtrial_backend.agentic.schemas import MetadataInput
from qtrial_backend.agent.runner import run_statistical_agent_loop
from qtrial_backend.providers.gemini_client import set_thread_emit
from qtrial_backend.providers.openrouter_client import set_thread_model as set_openrouter_model
from qtrial_backend.providers.bedrock_client import set_thread_model as set_bedrock_model
from qtrial_backend.report.static import build_static_report
from qtrial_backend.dataset.treatment_detector import detect_treatment_columns
from qtrial_backend.dataset.load import classify_missingness
from qtrial_backend.report.adl import build_adl

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
    content = await file.read()
    fname = file.filename or ""
    try:
        if fname.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

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

    # ── Classify missingness and drop >50% columns ───────────────────────────
    missingness_disclosures = classify_missingness(df)
    excluded_cols = [col for col, d in missingness_disclosures.items() if d.action == "excluded"]
    if excluded_cols:
        df = df.drop(columns=excluded_cols)

    clinical_analysis_result: dict | None = None
    try:
        static_report, clinical_analysis_result = await asyncio.to_thread(build_static_report, df, dataset_name)
    except Exception:
        static_report = None

    # ── Run LLM-driven statistical agent loop ────────────────────────────────
    try:
        loop_report, tool_log = await asyncio.to_thread(
            run_statistical_agent_loop, df, provider, dataset_name, None, None, column_dict
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
            df,
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
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Pipeline error: {exc}"
        )

    return JSONResponse(content=report.model_dump(mode="json"))


@app.post("/api/run/stream")
async def run_analysis_stream(
    file: UploadFile = File(..., description="CSV or XLSX dataset"),
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
) -> StreamingResponse:
    """
    Same as /api/run but streams Server-Sent Events so the frontend can show
    live per-stage progress.  Each event is a JSON object on a `data: ...\\n\\n` line:

      {"type": "stage_complete", "stage": "DataQualityAgent", "message": "..."}
      {"type": "complete",       "data": {<FinalReportSchema>}}
      {"type": "error",          "message": "..."}
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

    # Classify missingness and drop >50% columns before pipeline
    missingness_disclosures = classify_missingness(df)
    excluded_cols = [col for col, d in missingness_disclosures.items() if d.action == "excluded"]
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
            # ── 1. Deterministic static report ───────────────────────────────
            clinical_analysis_result: dict | None = None
            try:
                static_report, clinical_analysis_result = build_static_report(df, dataset_name, emit=emit)
            except Exception:
                static_report = None

            if static_report is not None:
                loop.call_soon_threadsafe(
                    aq.put_nowait,
                    {
                        "type": "stage_complete",
                        "stage": "StaticAnalysis",
                        "message": "Static statistical report ready",
                        "static_report": static_report,
                    },
                )

            # ── 2. LLM-driven statistical agent loop ─────────────────────────
            try:
                loop_report, tool_log = run_statistical_agent_loop(
                    df, provider, dataset_name, emit=emit,
                    model=model if provider in ("openrouter", "bedrock") else None,
                    column_dict=column_dict,
                )
            except Exception as exc:
                console.print(f"[red]⚠ Statistical agent loop FAILED: {exc}[/red]")
                console.print(traceback.format_exc())
                loop_report, tool_log = None, None
                # Non-fatal — emit a warning so the frontend can show a toast
                # without blocking the rest of the pipeline
                loop.call_soon_threadsafe(
                    aq.put_nowait,
                    {"type": "warning", "message": str(exc)},
                )

            # ── 3. Combine static + loop report for the reasoning pipeline ───
            parts = [p for p in [static_report, loop_report] if p]
            analysis_report = "\n\n---\n\n".join(parts) if parts else None

            # ── 4. Full agentic + reasoning pipeline ─────────────────────────
            report = run_agentic_insights(
                df, provider, max_rows, 30, run_judge, meta, False,
                analysis_report, tool_log, emit,
                study_context, column_dict,
                list(missingness_disclosures.values()),
                clinical_analysis_result,
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
