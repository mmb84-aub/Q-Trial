# Q-Trial Frontend

Interactive interface for running and exploring Q-Trial clinical trial analyses.

The frontend allows users to upload datasets, launch the analysis pipeline, and inspect results produced by the backend in a structured and visual way.

It communicates with the backend through the FastAPI service and streams pipeline progress during execution.

---

# What the Frontend Does

The frontend provides a user-friendly interface for interacting with the Q-Trial analysis engine.

Through the UI users can:

- Upload clinical trial datasets
- Provide optional dataset metadata or data dictionaries
- Launch analysis runs
- Monitor pipeline progress in real time
- Inspect intermediate analysis stages
- Explore generated insights and statistical outputs

The interface is designed to make the backend analysis pipeline easier to interact with during experimentation and research.

---

# Key Features

### Dataset Upload

Users can upload datasets directly from the UI.

Supported formats:

- `.csv`
- `.xlsx`

Uploaded datasets are sent to the backend for analysis.

---

### Interactive Analysis Runs

Users can start a full analysis run from the interface.

The frontend sends the dataset and configuration options to the backend API and receives the results once the pipeline completes.

---

### Streaming Pipeline Progress

The frontend supports **streaming progress updates** while the backend pipeline runs.

This allows users to observe stages such as:

- dataset inspection
- agent reasoning
- statistical analysis
- literature retrieval
- report synthesis

---

### Results Exploration

Once the analysis completes, the interface allows users to explore different outputs produced by the pipeline.

These may include:

- dataset diagnostics
- guardrail warnings
- generated insights
- hypothesis checks
- literature comparisons
- evaluation results

---

# Backend Connection

The frontend communicates with the backend using HTTP requests to the FastAPI service.

Typical workflow:

1. User uploads dataset
2. Frontend sends request to backend API
3. Backend runs the analysis pipeline
4. Progress updates stream back to the UI
5. Final report is returned and rendered

---

# Requirements

- Python 3.10+
- Backend server running locally or remotely

---

# Installation

From inside the `frontend/` directory:

```
pip install -r requirements.txt
```

---

# Running the Frontend

Start the interface with:

```
streamlit run app.py
```

Once running, the interface will open in your browser.

The frontend expects the backend API to be running.

---

# Project Structure

```
frontend/
  app.py                  Main Streamlit application
  components/             UI components used by the interface
  requirements.txt        Frontend dependencies
```

---

# Development Notes

The frontend is intentionally lightweight.

Its primary role is to:

- provide an interactive interface for the backend analysis engine
- display pipeline progress
- visualize generated outputs

Most core logic lives in the backend.

---

# Current Status

The frontend is functional but still evolving alongside the backend.

Future improvements may include:

- richer visualizations
- improved result exploration
- enhanced metadata input
- better report rendering