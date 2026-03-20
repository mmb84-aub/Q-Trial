import type { PipelineState, PipelineAction } from "./types";

export const initialState: PipelineState = {
  stage: "idle",
  studyContext: "",
  file: null,
  outcomeColumn: "",
  provider: "gemini",
  model: "",
  detectedTreatmentColumns: [],
  confirmedTreatmentColumns: [],
  progressMessages: [],
  warnings: [],
  report: null,
  errorMessage: null,
  retryCount: 0,
};

export function reducer(state: PipelineState, action: PipelineAction): PipelineState {
  switch (action.type) {
    case "SET_CONTEXT":
      return {
        ...state,
        studyContext: action.payload,
        stage: action.payload.trim() ? "context_entered" : "idle",
      };
    case "SET_FILE":
      return { ...state, file: action.payload };
    case "SET_OUTCOME_COLUMN":
      return { ...state, outcomeColumn: action.payload };
    case "SET_PROVIDER":
      return { ...state, provider: action.payload.provider, model: action.payload.model };
    case "START_UPLOAD":
      return { ...state, stage: "uploading" };
    case "TREATMENT_DETECTED":
      return {
        ...state,
        stage: "awaiting_confirmation",
        detectedTreatmentColumns: action.payload,
        confirmedTreatmentColumns: action.payload,
      };
    case "CONFIRM_TREATMENT":
      return { ...state, stage: "running", confirmedTreatmentColumns: action.payload };
    case "PROGRESS":
      return {
        ...state,
        progressMessages: [...state.progressMessages, action.payload],
      };
    case "WARNING":
      return { ...state, warnings: [...state.warnings, action.payload] };
    case "DISMISS_WARNING":
      return { ...state, warnings: state.warnings.filter((_, i) => i !== action.payload) };
    case "COMPLETE":
      return { ...state, stage: "complete", report: action.payload };
    case "ERROR":
      return { ...state, stage: "error", errorMessage: action.payload };
    case "RETRY":
      // Increment retryCount so ProgressStream gets a new key and remounts
      return { ...state, stage: "running", progressMessages: [], errorMessage: null, retryCount: state.retryCount + 1 };
    case "RESET":
      return { ...initialState };
    default:
      return state;
  }
}
