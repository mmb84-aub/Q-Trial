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
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from qtrial_backend.agentic.orchestrator import run_agentic_insights
from qtrial_backend.agentic.schemas import MetadataInput
from qtrial_backend.providers.gemini_client import set_thread_emit
from qtrial_backend.report.static import build_static_report

app = FastAPI(
    title="Q-Trial Statistical Reasoning Engine",
    version="0.1.0",
    description="Multi-agent LLM pipeline for clinical trial dataset analysis.",
)

# Allow Streamlit frontend (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
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
    try:
        static_report = await asyncio.to_thread(build_static_report, df, dataset_name)
    except Exception:
        static_report = None

    # ── Run pipeline in thread pool (blocking → async) ───────────────────────
    try:
        report = await asyncio.to_thread(
            run_agentic_insights,
            df,
            provider,
            max_rows,
            30,           # max_cols
            run_judge,
            meta,
            False,        # interactive — handled by UI
            static_report,  # analysis_report — upstream statistical context
            None,           # tool_log
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Pipeline error: {exc}"
        )

    return JSONResponse(content=report.model_dump(mode="json"))


@app.post("/api/run/stream")
async def run_analysis_stream(
    file: UploadFile = File(..., description="CSV or XLSX dataset"),
    provider: str = Form("gemini", description="openai | gemini | claude"),
    run_judge: bool = Form(True),
    max_rows: int = Form(25),
    metadata_json: str | None = Form(
        None,
        description="Optional JSON string matching MetadataInput schema",
    ),
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

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(aq.put_nowait, event)

    def _run_pipeline() -> None:
        set_thread_emit(emit)
        try:
            # ── Run static report inside thread so emit works per-section ─────
            try:
                static_report = build_static_report(df, dataset_name, emit=emit)
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
            report = run_agentic_insights(
                df, provider, max_rows, 30, run_judge, meta, False,
                static_report, None, emit,
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

    async def event_generator():
        t = threading.Thread(target=_run_pipeline, daemon=True)
        t.start()
        while True:
            try:
                event = await asyncio.wait_for(aq.get(), timeout=420.0)
            except asyncio.TimeoutError:
                yield (
                    f"data: {json.dumps({'type': 'error', 'message': 'Pipeline timed out'})}\n\n"
                )
                break
            yield f"data: {json.dumps(event, default=str)}\n\n"
            if event.get("type") in ("complete", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
