// ── Shared types mirroring backend Pydantic schemas ──────────────────────────

export interface EvidenceStrengthScore {
  score: number;
  label: "Strong" | "Moderate" | "Weak" | "Insufficient";
  plain_language: string;
  year: number | null;
  study_type: string;
  sample_size: number | null;
}

export interface LiteratureArticle {
  title: string;
  abstract: string;
  authors: string[];
  year: number | null;
  source: string;
  url: string | null;
  study_type: string;
  sample_size: number | null;
}

export interface GroundedFinding {
  finding_text: string;
  grounding_status: "Supported" | "Contradicted" | "Novel";
  citations: LiteratureArticle[];
  evidence_strength: EvidenceStrengthScore | null;
  novel_statement: string | null;
  literature_skipped: boolean;
  literature_skip_note: string | null;
  test_selection_rationale: string | null;
  missingness_disclosure: string | null;
}

export interface ResearchQuestion {
  question: string;
  source_finding: string;
}

export interface ControlVariable {
  variable: string;
  reason: string;
}

export interface SynthesisOutput {
  future_trial_hypothesis: string;
  endpoint_improvement_recommendations: string[];
  recommended_sample_size: string;
  variables_to_control: ControlVariable[];
  research_questions: ResearchQuestion[];
}

export interface ExcludedColumn {
  column: string;
  missingness_rate: number;
  reason: string;
}

export interface HighMissingnessColumn {
  column: string;
  missingness_rate: number;
}

export interface MissingnessDisclosure {
  column: string;
  missingness_rate: number;
  rows_dropped: number;
  action: "excluded" | "high_missingness_section" | "listwise_deletion";
}

export interface GroundedFindingsSchema {
  findings: GroundedFinding[];
  research_questions: ResearchQuestion[];
  synthesis: SynthesisOutput | null;
  excluded_columns: ExcludedColumn[];
  high_missingness_columns: HighMissingnessColumn[];
}

export interface SynthesisQualityScore {
  score: number;
  rationale: string;
  rerun_triggered: boolean;
}

export interface InsightSynthesisOutput {
  key_findings: string[];
  risks_and_bias_signals: string[];
  recommended_next_analyses: Array<{ rank: number; analysis: string; rationale: string; evidence_citation: string }>;
  required_metadata_or_questions: string[];
  /** Prose narrative summary produced by the synthesis call (preferred over key_findings list) */
  narrative_summary?: string;
}

export interface FinalReport {
  provider: string;
  model: string;
  study_context: string | null;
  grounded_findings: GroundedFindingsSchema | null;
  synthesis_quality_score: SynthesisQualityScore | null;
  treatment_columns_excluded: string[];
  final_insights: InsightSynthesisOutput;
  // run_id is derived from reproducibility_log if present
  reproducibility_log: { run_id: string } | null;
}

// ── Pipeline state machine ────────────────────────────────────────────────────

export type PipelineStage =
  | "idle"
  | "context_entered"
  | "uploading"
  | "detecting_treatment"
  | "awaiting_confirmation"
  | "running"
  | "complete"
  | "error";

export interface PipelineState {
  stage: PipelineStage;
  studyContext: string;
  file: File | null;
  outcomeColumn: string;
  provider: string;
  model: string;
  detectedTreatmentColumns: string[];
  confirmedTreatmentColumns: string[];
  progressMessages: string[];
  report: FinalReport | null;
  errorMessage: string | null;
  retryCount: number;
}

export type PipelineAction =
  | { type: "SET_CONTEXT"; payload: string }
  | { type: "SET_FILE"; payload: File }
  | { type: "SET_OUTCOME_COLUMN"; payload: string }
  | { type: "SET_PROVIDER"; payload: { provider: string; model: string } }
  | { type: "START_UPLOAD" }
  | { type: "TREATMENT_DETECTED"; payload: string[] }
  | { type: "CONFIRM_TREATMENT"; payload: string[] }
  | { type: "PROGRESS"; payload: string }
  | { type: "COMPLETE"; payload: FinalReport }
  | { type: "ERROR"; payload: string }
  | { type: "RETRY" }
  | { type: "RESET" };
