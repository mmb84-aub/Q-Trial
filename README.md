# Q-Trial

Q-Trial is a clinical trial dataset analysis system with a FastAPI backend and a React/Vite frontend. It combines deterministic statistical reporting, feature selection, an LLM-guided statistical tool loop, literature grounding, report synthesis, statistical verification, and optional comparison against a human analyst report.

The main supported workflow is the web app: upload a CSV/XLSX dataset, confirm treatment columns, choose a feature-selection method, optionally attach a column dictionary and human analyst report, then stream progress until the interactive report is ready.

---

## Current Pipeline

```text
User input
  - Required: study context + CSV/XLSX dataset
  - Optional: primary outcome column, JSON column dictionary, human analyst report
  - Optional: LLM provider/model and feature-selection method

1. Treatment detection
   POST /api/detect-treatment scans uploaded columns and asks the user to confirm
   treatment/grouping columns in the frontend.

2. Dataset loading and missingness handling
   Backend accepts CSV and XLSX. Rows are capped at 3,000 for API uploads.
   Columns with >50% missingness are excluded from primary analysis, except the
   resolved endpoint column is preserved.

3. Feature selection
   The streaming API supports: none, univariate, mrmr, lasso, elastic_net, qubo.
   The frontend default is mrmr. The endpoint, explicitly important metadata
   variables, and confirmed treatment columns are protected.

4. Deterministic statistical report
   build_static_report() runs code-only clinical/statistical summaries and
   produces Markdown, methodology text, and structured clinical-analysis results.

5. LLM-guided statistical tool loop
   run_statistical_agent_loop() lets the selected provider call statistical tools
   such as clinical tests, survival analysis, regression, MMRM, ANCOVA, MICE,
   effect sizes, subgroup analysis, power analysis, and data-quality tools.

6. Agentic synthesis pipeline
   run_agentic_insights() builds dataset evidence, normalizes/statistically
   verbalizes findings, translates findings to clinical search terms, validates
   against literature sources, curates output quality, and returns a
   FinalReportSchema.

7. Optional analyst-report verification and comparison
   If a human analyst report is uploaded, Q-Trial extracts comparable claims,
   verifies analyst claims against the dataset where possible, semantically
   matches Q-Trial findings to human findings, compares statistical evidence, and
   returns precision/recall/F1/MCC-style metrics plus matched/missed/extra lists.

8. Interactive report
   React renders the final report, statistical verification, comparison section,
   missingness disclosures, finding cards, citations, PDF export, ADL viewer, and
   reproducibility-log download.
```

The streaming endpoint is the path used by the frontend and is the most complete path. The non-streaming `/api/run` endpoint still exists, but it has fewer request controls and always uses the older QUBO-first flow.

---

## Repository Structure

```text
Q-Trial/
|-- backend/
|   |-- pyproject.toml
|   |-- run_api.py
|   `-- src/qtrial_backend/
|       |-- api.py                         # FastAPI endpoints
|       |-- main.py                        # Typer CLI entry point
|       |-- agent/                         # LLM tool-loop runner
|       |-- agentic/
|       |   |-- orchestrator.py            # Final agentic pipeline
|       |   |-- report_comparison.py       # Human-vs-Q-Trial comparison
|       |   |-- statistical_verification.py
|       |   |-- finding_comparison_normalizer.py
|       |   |-- finding_verbalizer.py
|       |   |-- cst_translator.py
|       |   |-- literature_validator.py
|       |   |-- report_curation.py
|       |   |-- reproducibility.py
|       |   `-- schemas.py                 # Pydantic response schemas
|       |-- dataset/                       # loading, previews, evidence, guardrails
|       |-- feature_selection/             # none/univariate/mRMR/LASSO/Elastic Net
|       |-- quantum/feature_selector.py    # QUBO feature selection
|       |-- providers/                     # OpenAI, Gemini, Claude, OpenRouter, Bedrock
|       |-- report/                        # static report, PDF, ADL
|       `-- tools/                         # statistical and literature tools
`-- frontend/
    |-- package.json
    `-- src/
        |-- App.tsx
        |-- components/
        |   |-- UploadForm.tsx
        |   |-- FeatureSelectionMethodPicker.tsx
        |   |-- ProgressStream.tsx
        |   `-- report/
        |       |-- InteractiveReport.tsx
        |       |-- ComparisonSection.tsx
        |       |-- StatisticalVerificationSection.tsx
        |       `-- ReportActions.tsx
        `-- types.ts
```

---

## Quick Start

### Backend

```bash
cd backend
poetry install
poetry run uvicorn qtrial_backend.api:app --reload --port 8000
```

Alternative launcher:

```bash
poetry run python run_api.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server runs on Vite, usually `http://localhost:5173`, and calls the backend at `/api/...`.

### Environment

Create `backend/.env` with whichever providers you use:

```bash
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=...
CLAUDE_MODEL=claude-opus-4-6

OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o
OPENROUTER_MAX_TOKENS=4096

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_BEDROCK_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
AWS_BEDROCK_MAX_TOKENS=8192

NCBI_API_KEY=...
S2_API_KEY=...
MAX_AGENT_ITERATIONS=25
MAX_TOOL_RESULT_CHARS=4000
```

`GEMINI_API_KEY` may contain multiple comma-separated keys.

---

## Frontend Inputs

The web app currently supports:

- Dataset upload: `.csv`, `.xlsx`
- Human analyst report: `.txt`, `.md`, `.markdown`, `.text`, `.rst`, `.json`, or UTF-8 `text/plain`
- Column dictionary: JSON mapping column names to descriptions
- Optional primary outcome column
- Provider selection: Gemini, AWS Bedrock, OpenRouter, OpenAI, Claude
- Model override for OpenRouter and Bedrock
- Feature selection method: `none`, `univariate`, `mrmr`, `lasso`, `elastic_net`, `qubo`

Human analyst reports are limited to 25 MB and must be UTF-8 text. JSON analyst reports are parsed and pretty-printed before comparison.

---

## API

All analysis upload endpoints use `multipart/form-data`, not base64 JSON.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | Backend health check |
| `/api/detect-treatment` | POST | Upload dataset and return candidate treatment columns |
| `/api/run/stream` | POST | Primary full pipeline; streams Server-Sent Events |
| `/api/run` | POST | Synchronous full pipeline; older/narrower request surface |
| `/api/report/pdf` | POST | Generate PDF from a serialized `FinalReportSchema` |
| `/api/report/reproducibility/{run_id}` | GET | Download saved reproducibility JSON |
| `/api/report/adl` | GET | Return Architecture Decision Log Markdown |

### `/api/run/stream` form fields

| Field | Required | Notes |
| --- | --- | --- |
| `file` | yes | CSV or XLSX dataset |
| `study_context` | yes | Plain-language study description |
| `provider` | no | `gemini` default; also supports `openai`, `claude`, `openrouter`, `bedrock` |
| `model` | no | Used for OpenRouter and Bedrock model override |
| `feature_selection_method` | no | `mrmr` default; `none`, `univariate`, `mrmr`, `lasso`, `elastic_net`, `qubo` |
| `analyst_report_file` | no | Enables statistical verification and report comparison |
| `dict_file` | no | JSON column dictionary |
| `confirmed_treatment_columns` | no | Repeatable form field from frontend confirmation step |
| `metadata_json` | no | JSON matching `MetadataInput` |
| `run_judge` | no | Boolean, default false |
| `max_rows` | no | Preview/synthesis row limit, default 25 |

SSE event types include `stage_complete`, `warning`, `error`, and `complete`. The server also sends keepalive comments during long-running stages.

### `/api/run` form fields

`/api/run` accepts `file`, `analyst_report_file`, `provider`, `run_judge`, `max_rows`, `study_context`, and `metadata_json`. It does not expose the frontend's feature-selection picker, model override, uploaded column dictionary, or confirmed treatment columns.

---

## Feature Selection

Implemented methods:

| Method | Implementation |
| --- | --- |
| `none` | Passes all columns through selection metadata |
| `univariate` | `sklearn.feature_selection` F-tests/regression scoring |
| `mrmr` | Greedy minimum-redundancy maximum-relevance selection |
| `lasso` | LASSO/Elastic Net coefficient-based selection |
| `elastic_net` | Elastic Net mode of the LASSO selector |
| `qubo` | Simulated annealing QUBO selector using `dwave-neal` |

Default feature counts are adaptive: about 75% coverage for <=20 candidate features and about 55% for larger datasets, capped for interpretability. The API preserves the endpoint and explicitly important variables; the streaming frontend also preserves confirmed treatment columns.

For the streaming path, deterministic static statistics and the statistical tool loop run on the full post-missingness dataframe. Feature selection primarily constrains downstream agentic context and report metadata.

---

## Analyst Report Comparison

Uploading `analyst_report_file` enables two related outputs:

- `statistical_verification_report`: analyst claims are checked against available dataset evidence where possible.
- `comparison_report`: Q-Trial findings are compared with human analyst findings.

The comparison pipeline lives mainly in `backend/src/qtrial_backend/agentic/report_comparison.py` and uses:

- deterministic finding normalization and filtering to analytical/comparison claims
- LLM-assisted extraction/matching where configured
- deterministic fallback matching using normalized variables/endpoints and lexical similarity
- statistical evidence comparison using significance, p-values, adjusted p-values, direction, effect sizes, confidence intervals, and test metadata when available

Returned comparison metrics include matched count, Q-Trial-only count, human-only count, precision, recall, F1, MCC against human significance labels when calculable, agreement/contradiction counts, statistical evidence coverage, and evidence-upgrade rate.

The frontend displays these in `ComparisonSection.tsx` with matched findings, contradictions, Q-Trial-only findings, human-only findings, and per-match statistical evidence details.

---

## Outputs

The backend returns `FinalReportSchema`, defined in `backend/src/qtrial_backend/agentic/schemas.py`. Major frontend-visible sections include:

- static and agentic findings
- grounded findings with citations and evidence strength
- narrative summary and forward recommendations
- missingness disclosures and excluded columns
- statistical verification report, when an analyst report is uploaded
- comparison report, when an analyst report is uploaded
- reproducibility log reference

Reproducibility logs are written under `outputs/{run_id}_reproducibility.json` by `agentic/reproducibility.py` and can be downloaded through the report UI.

---

## CLI Status

`backend/src/qtrial_backend/main.py` still defines a Typer CLI entry point through the `qtrial` Poetry script. The current maintained path for full functionality is the FastAPI/React workflow. At the time of this README update, invoking the CLI help in the local environment exposes an import-cycle issue, so use the API server for complete feature selection, analyst-report comparison, PDF export, and reproducibility-log workflows.

---

## Tech Stack

- Python >=3.12
- FastAPI, Uvicorn, Pydantic v2, Typer, Rich
- Pandas, NumPy, SciPy, Statsmodels, Lifelines, scikit-learn
- D-Wave `neal`/samplers for QUBO-style feature selection
- OpenAI, Anthropic, Google Gemini, OpenRouter, AWS Bedrock clients
- React 18, Vite, TypeScript, Recharts, React Markdown
- Poetry for backend dependencies and npm for frontend dependencies

---

## Validation

Useful local checks:

```bash
cd backend
poetry run pytest
poetry run python -m py_compile src/qtrial_backend/api.py

cd ../frontend
npm run build
```
