import { useEffect } from "react";
import { createParser } from "eventsource-parser";
import type { FinalReport, PipelineAction } from "../types";

// Maps backend stage names to what the user sees + a one-line explanation
const STAGE_INFO: Record<string, { label: string; detail: string }> = {
  StaticAnalysis:         { label: "Statistical analysis",       detail: "Running deterministic tests on your dataset — no AI involved yet." },
  DataQualityAgent:       { label: "Data quality check",         detail: "Flagging missing values, outliers, and structural issues." },
  ClinicalSemanticsAgent: { label: "Column interpretation",      detail: "Inferring what each column represents clinically." },
  UnknownsAgent:          { label: "Unknowns & assumptions",     detail: "Surfacing what the data can't tell us and what we're assuming." },
  InsightSynthesisAgent:  { label: "Synthesising insights",      detail: "Combining all findings into a coherent clinical narrative." },
  CSTTranslation:         { label: "Clinical search terms",      detail: "Translating statistical findings into literature search queries." },
  LiteratureValidation:   { label: "Literature validation",      detail: "Checking findings against PubMed, Cochrane, and ClinicalTrials.gov." },
  SynthesisScoring:       { label: "Quality check",              detail: "Self-scoring the synthesis — re-running if it falls below threshold." },
  ReproducibilityLog:     { label: "Reproducibility log",        detail: "Recording every LLM call and query so the run can be audited." },
};

interface Props {
  file: File;
  dictFile?: File | null;
  studyContext: string;
  confirmedTreatmentColumns: string[];
  provider?: string;
  model?: string;
  progressMessages: string[];
  dispatch: React.Dispatch<PipelineAction>;
}

export function ProgressStream({ file, dictFile, studyContext, confirmedTreatmentColumns, provider = "gemini", model = "", progressMessages, dispatch }: Props) {
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

  // When a column dictionary is provided the agent still runs internally,
  // but the user already knows what the columns mean — hide that step.
  const allStages = Object.keys(STAGE_INFO).filter(
    (s) => !(s === "ClinicalSemanticsAgent" && dictFile)
  );
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
