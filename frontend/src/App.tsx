import { useReducer } from "react";
import { reducer, initialState } from "./reducer";
import { StepTracker } from "./components/StepTracker";
import { StudyContextForm } from "./components/StudyContextForm";
import { UploadForm } from "./components/UploadForm";
import { TreatmentConfirmModal } from "./components/TreatmentConfirmModal";
import { ProgressStream } from "./components/ProgressStream";
import { InteractiveReport } from "./components/report/InteractiveReport";
import { ErrorBoundary } from "./components/ErrorBoundary";

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);

  async function handleDetect(file: File, outcomeColumn: string) {
    dispatch({ type: "SET_FILE", payload: file });
    dispatch({ type: "SET_OUTCOME_COLUMN", payload: outcomeColumn });
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
        {state.stage !== "complete" && (
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
            onConfirm={(cols) => dispatch({ type: "CONFIRM_TREATMENT", payload: cols })}
          />
        )}

        {state.stage === "running" && state.file && (
          <ProgressStream
            file={state.file}
            studyContext={state.studyContext}
            confirmedTreatmentColumns={state.confirmedTreatmentColumns}
            progressMessages={state.progressMessages}
            dispatch={dispatch}
          />
        )}

        {state.stage === "complete" && state.report && (
          <InteractiveReport report={state.report} onReset={() => dispatch({ type: "RESET" })} />
        )}

        {state.stage === "error" && (
          <div role="alert" style={{ maxWidth: 640, margin: "3rem auto", padding: "0 1.5rem" }}>
            <h1 style={{ fontSize: "1.5rem", fontWeight: 700, color: "#dc2626", margin: "0 0 0.75rem" }}>
              Analysis could not be completed
            </h1>
            <p style={{ color: "#4b5563", marginBottom: "1.5rem" }}>{state.errorMessage}</p>
            <button
              onClick={() => dispatch({ type: "RESET" })}
              style={{
                padding: "0.6rem 1.5rem", background: "#1d4ed8", color: "#fff",
                border: "none", borderRadius: 6, cursor: "pointer", fontWeight: 600,
              }}
            >
              Start over
            </button>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
}
