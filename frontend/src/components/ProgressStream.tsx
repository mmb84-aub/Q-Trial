import { useEffect } from "react";
import { createParser } from "eventsource-parser";
import type { FinalReport, PipelineAction } from "../types";

// Maps backend stage_complete stage keys to what the user sees + a one-line explanation.
// Keys must match exactly what the backend emits in {"type":"stage_complete","stage":"<key>"}.
const STAGE_INFO: Record<string, { label: string; detail: string }> = {
  StaticAnalysis:       { label: "Deterministic profiling",    detail: "Running data quality checks, missingness analysis, and outlier detection — no AI involved." },
  StatisticalLoop:      { label: "Statistical analysis",       detail: "AI agent iteratively selects and runs statistical tests suited to your data." },
  dataset:              { label: "Evidence & guardrails",      detail: "Building the evidence profile and running robustness checks on the dataset." },
  cst_translation:      { label: "Clinical search terms",      detail: "Translating statistical findings into clinical search phrases for literature queries." },
  literature_validation:{ label: "Literature validation",      detail: "Checking findings against PubMed, Cochrane, and ClinicalTrials.gov." },
  synthesis:            { label: "Synthesis",                  detail: "Producing grounding status, evidence strength, recommendations, and narrative summary." },
  synthesis_scoring:    { label: "Quality check",              detail: "Self-scoring the synthesis — re-running if quality falls below threshold." },
};

interface Props {
  file: File;
  dictFile?: File | null;
  priorReportFile?: File | null;
  studyContext: string;
  confirmedTreatmentColumns: string[];
  provider?: string;
  model?: string;
  progressMessages: string[];
  dispatch: React.Dispatch<PipelineAction>;
}

export function ProgressStream({ file, dictFile, priorReportFile, studyContext, confirmedTreatmentColumns, provider = "gemini", model = "", progressMessages, dispatch }: Props) {
  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function stream() {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("study_context", studyContext);
      formData.append("provider", provider);
      if (model) formData.append("model", model);
      if (dictFile) formData.append("dict_file", dictFile);
      if (priorReportFile) formData.append("prior_report_file", priorReportFile);
      confirmedTreatmentColumns.forEach((col) => formData.append("confirmed_treatment_columns", col));

      let response: Response;
      try {
        response = await fetch("/api/run/stream", {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
      } catch (err) {
        if (cancelled) return;
        dispatch({ type: "ERROR", payload: "Unable to reach the analysis server. Please try again." });
        return;
      }

      if (!response.ok || !response.body) {
        if (cancelled) return;
        dispatch({ type: "ERROR", payload: "The analysis server returned an unexpected response." });
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const parser = createParser((event) => {
        if (event.type !== "event") return;
        try {
          const data = JSON.parse(event.data) as {
            type: string; stage?: string; message?: string; data?: FinalReport;
          };
          if (data.type === "stage_complete" && data.stage) {
            dispatch({ type: "PROGRESS", payload: data.stage });
          } else if (data.type === "warning" && data.message) {
            dispatch({ type: "WARNING", payload: data.message });
          } else if (data.type === "complete" && data.data) {
            dispatch({ type: "COMPLETE", payload: data.data });
          } else if (data.type === "error") {
            dispatch({ type: "ERROR", payload: data.message ?? "An unexpected error occurred during analysis." });
          }
        } catch { /* malformed SSE — ignore */ }
      });

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        parser.feed(decoder.decode(value, { stream: true }));
      }
    }

    stream();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []); // intentionally run once on mount

  const allStages = Object.keys(STAGE_INFO);
  const doneSet = new Set(progressMessages);
  // The "current" stage is the first one not yet done
  const currentIdx = allStages.findIndex((s) => !doneSet.has(s));

  return (
    <div style={{ maxWidth: 640, margin: "3rem auto", padding: "0 1.5rem" }}>
      <p style={{ fontSize: "0.8rem", color: "#6b7280", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Step 4 of 4
      </p>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0 0 0.5rem" }}>
        Analysis running
      </h1>
      <p style={{ color: "#4b5563", marginBottom: "1.75rem", lineHeight: 1.6 }}>
        The pipeline runs in stages. Each stage builds on the last — statistical tests
        first, then clinical interpretation, then literature grounding.
        This typically takes 2–5 minutes.
      </p>

      <ol style={{ listStyle: "none", padding: 0, margin: 0 }} aria-live="polite" aria-label="Analysis stages">
        {allStages.map((stageKey, i) => {
          const info = STAGE_INFO[stageKey];
          const done = doneSet.has(stageKey);
          const active = i === currentIdx;

          return (
            <li key={stageKey} style={{
              display: "flex", gap: "0.75rem", alignItems: "flex-start",
              padding: "0.6rem 0",
              borderBottom: i < allStages.length - 1 ? "1px solid #f3f4f6" : "none",
              opacity: done || active ? 1 : 0.4,
            }}>
              <span style={{
                width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: "0.75rem", fontWeight: 700, marginTop: 2,
                background: done ? "#16a34a" : active ? "#1d4ed8" : "#e5e7eb",
                color: done || active ? "#fff" : "#9ca3af",
              }}>
                {done ? "✓" : active ? "…" : i + 1}
              </span>
              <div>
                <div style={{ fontWeight: done || active ? 600 : 400, fontSize: "0.95rem" }}>
                  {info.label}
                  {active && (
                    <span style={{ marginLeft: "0.5rem", fontSize: "0.8rem", color: "#1d4ed8", fontWeight: 400 }}>
                      running…
                    </span>
                  )}
                </div>
                {(done || active) && (
                  <div style={{ fontSize: "0.8rem", color: "#6b7280", marginTop: "0.15rem" }}>
                    {info.detail}
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
