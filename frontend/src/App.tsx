import { useReducer } from "react";
import { reducer, initialState } from "./reducer";
import { StepTracker } from "./components/StepTracker";
import { StudyContextForm } from "./components/StudyContextForm";
import { UploadForm } from "./components/UploadForm";
import { TreatmentConfirmModal } from "./components/TreatmentConfirmModal";
import { ProgressStream } from "./components/ProgressStream";
import { InteractiveReport } from "./components/report/InteractiveReport";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ToastStack } from "./components/ToastStack";

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);

  async function handleDetect(file: File, outcomeColumn: string, provider: string, model: string) {
    dispatch({ type: "SET_FILE", payload: file });
    dispatch({ type: "SET_OUTCOME_COLUMN", payload: outcomeColumn });
    dispatch({ type: "SET_PROVIDER", payload: { provider, model } });
    dispatch({ type: "START_UPLOAD" });

    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/detect-treatment", { method: "POST", body: form });
      if (!res.ok) throw new Error("Detection failed");
      const data = (await res.json()) as { candidate_columns: string[] };
      dispatch({ type: "TREATMENT_DETECTED", payload: data.candidate_columns });
    } catch {
      dispatch({
        type: "ERROR",
        payload: "Unable to process the uploaded file. Please check the format and try again.",
      });
    }
  }

  return (
    <ErrorBoundary>
      <div style={{ minHeight: "100vh", background: "#fff", fontFamily: "system-ui, sans-serif" }}>
        {/* Persistent step tracker — hidden on complete/error to give report full width */}
        {state.stage !== "complete" && state.stage !== "error" && (
          <StepTracker stage={state.stage} />
        )}

        {state.stage === "idle" && (
          <StudyContextForm onSubmit={(ctx) => dispatch({ type: "SET_CONTEXT", payload: ctx })} />
        )}

        {state.stage === "context_entered" && (
          <UploadForm studyContext={state.studyContext} onDetect={handleDetect} />
        )}

        {(state.stage === "uploading" || state.stage === "detecting_treatment") && (
          <div style={{ maxWidth: 640, margin: "3rem auto", padding: "0 1.5rem" }}>
            <p style={{ fontSize: "0.8rem", color: "#6b7280", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Step 3 of 4
            </p>
            <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0 0 1rem" }}>Scanning dataset…</h1>
            <p style={{ color: "#4b5563" }}>
              Looking for columns that encode treatment group assignment.
              This takes a second.
            </p>
          </div>
        )}

        {state.stage === "awaiting_confirmation" && (
          <TreatmentConfirmModal
            detected={state.detectedTreatmentColumns}
            onConfirm={(cols) => {
              dispatch({ type: "CONFIRM_TREATMENT", payload: cols });
            }}
          />
        )}

        {(state.stage === "running" || state.stage === "error") && state.file && (
          <div style={{ position: "relative" }}>
            <ProgressStream
              key={state.retryCount}
              file={state.file}
              studyContext={state.studyContext}
              confirmedTreatmentColumns={state.confirmedTreatmentColumns}
              provider={state.provider}
              model={state.model}
              progressMessages={state.progressMessages}
              dispatch={dispatch}
            />

            {state.stage === "error" && (
              <div
                role="alert"
                style={{
                  position: "fixed", inset: 0,
                  background: "rgba(0,0,0,0.85)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  zIndex: 100,
                }}
              >
                <div style={{
                  background: "#111", border: "1px solid #333", borderRadius: 8,
                  padding: "2rem 2.5rem", maxWidth: 560, width: "90%",
                  fontFamily: "monospace",
                }}>
                  <p style={{ color: "#f87171", fontWeight: 700, fontSize: "0.85rem", margin: "0 0 0.75rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    Analysis failed
                  </p>
                  <pre style={{
                    color: "#e5e7eb", fontSize: "0.85rem", whiteSpace: "pre-wrap",
                    wordBreak: "break-word", margin: "0 0 1.5rem", lineHeight: 1.6,
                    background: "transparent", border: "none", padding: 0,
                  }}>
                    {state.errorMessage}
                  </pre>
                  <div style={{ display: "flex", gap: "0.75rem" }}>
                    <button
                      onClick={() => {
                        dispatch({ type: "RETRY" });
                      }}
                      style={{
                        padding: "0.5rem 1.25rem", background: "#1d4ed8", color: "#fff",
                        border: "none", borderRadius: 6, cursor: "pointer",
                        fontWeight: 600, fontSize: "0.85rem", fontFamily: "system-ui, sans-serif",
                      }}
                    >
                      Try again
                    </button>
                    <button
                      onClick={() => dispatch({ type: "RESET" })}
                      style={{
                        padding: "0.5rem 1.25rem", background: "none", color: "#9ca3af",
                        border: "1px solid #374151", borderRadius: 6, cursor: "pointer",
                        fontSize: "0.85rem", fontFamily: "system-ui, sans-serif",
                      }}
                    >
                      Start over
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {state.stage === "running" && !state.file && (
          <div style={{ maxWidth: 640, margin: "3rem auto", padding: "0 1.5rem" }}>
            <p style={{ color: "#dc2626" }}>Error: no file in state. Please start over.</p>
          </div>
        )}

        {state.stage === "complete" && state.report && (
          <InteractiveReport report={state.report} onReset={() => dispatch({ type: "RESET" })} />
        )}

        {/* Non-fatal warnings — toast stack, always on top, never blocks the pipeline */}
        <ToastStack warnings={state.warnings} dispatch={dispatch} />
      </div>
    </ErrorBoundary>
  );
}
