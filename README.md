# Q-Trial

Q-Trial is an AI-powered clinical trial dataset analysis system. It combines deterministic statistical profiling, an iterative LLM-driven analysis agent, external literature validation, and structured synthesis into a single end-to-end pipeline.

---

## System Architecture

Q-Trial processes a clinical dataset through **8 sequential stages**. Three stages make LLM calls; five are pure code.

```text
  INPUT
  ┌─────────────────────────────────────────────────────────────┐
  │  study_context (str)   +   dataset (.csv / .xlsx)           │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │  Stage 1 — Clinical Context Input          [pure code]      │
  │  Caller supplies study_context string.                      │
  │  Out: context passed downstream to all LLM prompts.         │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │  Stage 2 — Dataset Upload + Blinding       [pure code]      │
  │  Load CSV/Excel, infer types, detect treatment columns.     │
  │  Out: sanitised DataFrame + column schema.                  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │  Stage 3 — Data Profiler + Data Quality    [pure code]      │
  │  build_dataset_preview() → structure, types, cardinality    │
  │  build_dataset_evidence() → missingness, duplicates,        │
  │    correlations, frequency tables                           │
  │  run_guardrails() → physiological bounds, unit checks,      │
  │    low-cardinality numerics, repeated-measure detection      │
  │  Out: dataset_preview dict + evidence dict + guardrail warns│
  └──────────────────────────────┬──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │  Stage 4 — Statistical Analysis Agent   [LLM agent loop]   │
  │  AgentLoop: LLM ↔ 30+ statistical tools (iterative).       │
  │  Tools: t-tests, survival, regression, MMRM, ANCOVA,        │
  │    MICE, power analysis, effect sizes, subgroup analysis…   │
  │  Post-loop: deterministic confidence-warning annotations    │
  │    (missingness > 30%, sample size below minimum).          │
  │  Out: analysis_report (Markdown) + tool_log (JSON)          │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │  Stage 5 — Literature Query Translation    [LLM mini-call]  │
  │  Extract key findings → Clinical Search Terms (CSTs).       │
  │  Out: list[ClinicalSearchTerm] — structured queries          │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │  Stage 6 — Literature Validator            [API calls]      │
  │  Query PubMed, Cochrane, ClinicalTrials.gov,                │
  │    Semantic Scholar for each CST.                           │
  │  Out: GroundedFinding[] — Supported / Contradicted / Novel  │
  │    with citations and evidence strength scores.             │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │  Stage 7 — Synthesis                       [LLM call]       │
  │  Single structured LLM call → FinalReportSchema:            │
  │    future_trial_hypothesis, endpoint_improvements,          │
  │    recommended_sample_size, control_variables,              │
  │    research_questions, narrative_summary.                   │
  │  Deterministic validation + 1-retry on schema failure.      │
  │  Out: FinalReportSchema (Pydantic, JSON-serialisable)        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │  Stage 8 — Report Generator               [React + Python]  │
  │  Frontend renders FinalReportSchema as interactive report.  │
  │  PDF export via pdf_generator.py.                           │
  │  Out: rendered report delivered to user.                    │
  └─────────────────────────────────────────────────────────────┘
```

**LLM call summary:**

| Stage                   | LLM?           | Provider     |
|-------------------------|----------------|--------------|
| 3 — Data Profiler       | No             | —            |
| 4 — Statistical Agent   | Yes (loop)     | configurable |
| 5 — CST Translation     | Yes (mini)     | configurable |
| 6 — Literature Validator| No (API calls) | —            |
| 7 — Synthesis           | Yes (single)   | configurable |

---

## Repository Structure

```text
Q-Trial/
├── backend/                          # Analysis engine + API
│   └── src/qtrial_backend/
│       ├── api.py                    # FastAPI: /api/run, /api/run/stream
│       ├── main.py                   # CLI entry point
│       ├── agent/
│       │   ├── loop.py               # AgentLoop — LLM ↔ tool while-loop
│       │   └── runner.py             # run_statistical_agent_loop() (Stage 4)
│       ├── agentic/
│       │   ├── orchestrator.py       # run_agentic_insights() (Stages 3, 5, 6, 7)
│       │   ├── schemas.py            # Pydantic models for all pipeline I/O
│       │   ├── cst_translator.py     # Stage 5: findings → search terms
│       │   ├── literature_validator.py # Stage 6: multi-source validation
│       │   ├── validation.py         # Deterministic synthesis validation
│       │   └── reproducibility.py   # Run metadata logging
│       ├── dataset/
│       │   ├── load.py               # CSV/Excel loading
│       │   ├── preview.py            # Dataset preview builder (Stage 3)
│       │   ├── evidence.py           # Deterministic evidence extraction (Stage 3)
│       │   ├── guardrails.py         # Risk detection (Stage 3)
│       │   └── treatment_detector.py # Treatment column identification
│       ├── report/
│       │   ├── static.py             # Deterministic Markdown report
│       │   └── pdf_generator.py      # PDF rendering (Stage 8)
│       ├── tools/
│       │   ├── stats/                # 30+ statistical tools (Stage 4)
│       │   └── literature/           # PubMed, Cochrane, CT.gov, S2 (Stage 6)
│       └── providers/                # OpenAI, Claude, Gemini, Bedrock, OpenRouter
└── frontend/                         # React + Vite interactive report viewer
```

---

## Quick Start

**Backend (API server):**

```bash
cd backend
poetry install
poetry run uvicorn qtrial_backend.api:app --reload
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

**CLI:**

```bash
cd backend
poetry run qtrial analyze --file data/trial.csv --provider openai
```

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/run` | POST | Full pipeline, returns JSON |
| `/api/run/stream` | POST | Full pipeline, streams SSE events |
| `/api/health` | GET | Health check |

**Request body:**

```json
{
  "file": "<base64-encoded CSV/XLSX>",
  "filename": "trial.csv",
  "provider": "openai",
  "study_context": "Phase III RCT comparing drug A vs placebo in NASH patients."
}
```

**SSE event types:** `progress`, `stage_complete`

---

## Supported LLM Providers

| Provider | Value |
|----------|-------|
| OpenAI | `openai` |
| Anthropic Claude | `claude` |
| Google Gemini | `gemini` |
| AWS Bedrock | `bedrock` |
| OpenRouter | `openrouter` |

Configure via environment variables (see `backend/src/qtrial_backend/config.py`).

---

## Tech Stack

- **Python 3.11+** — backend analysis engine
- **FastAPI** — REST + SSE API layer
- **Pandas / NumPy / SciPy / Statsmodels** — statistical tools
- **Pydantic v2** — schema validation throughout pipeline
- **React + Vite** — frontend report viewer
- **Poetry** — backend dependency management

---

## Design Rationale

The pipeline was designed to be **hybrid**: deterministic where ground truth is knowable (data profiling, guardrails, validation, literature search), and LLM-driven only where interpretation or synthesis is required (statistical agent loop, search-term extraction, final synthesis).

See [CHANGELOG.md](CHANGELOG.md) for a record of removed components and the reasoning behind each removal.
