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
  finding_text_raw?: string | null;
  finding_text_plain?: string | null;
  comparison_claim_text?: string | null;
  grounding_status:
    | "Supported"
    | "Contradicted"
    | "Novel"
    | "Data Quality Note"
    | "Preprocessing Observation"
    | "Pipeline Warning"
    | "QC Observation";
  finding_category?:
    | "analytical"
    | "survival_result"
    | "endpoint_result"
    | "data_quality"
    | "preprocessing"
    | "pipeline_warning"
    | "qc_note";
  claim_type?:
    | "association_claim"
    | "descriptive_claim"
    | "data_quality_claim"
    | "setup_claim"
    | "metadata_claim";
  citations: LiteratureArticle[];
  evidence_strength: EvidenceStrengthScore | null;
  novel_statement: string | null;
  literature_skipped: boolean;
  literature_skip_note: string | null;
  test_selection_rationale: string | null;
  missingness_disclosure: string | null;
  confidence_warning: string | string[] | null;
}

export interface ComparableFinding {
  finding_id: string;
  source: "qtrial" | "human";
  source_label: string;
  finding_text: string;
  normalized_text: string;
  section: string | null;
  finding_category?:
    | "analytical"
    | "survival_result"
    | "endpoint_result"
    | "data_quality"
    | "preprocessing"
    | "pipeline_warning"
    | "qc_note"
    | null;
  claim_type?:
    | "association_claim"
    | "descriptive_claim"
    | "data_quality_claim"
    | "setup_claim"
    | "metadata_claim"
    | null;
  endpoint: string | null;
  significance: "significant" | "not_significant" | "unclear";
  p_value: number | null;
  effect_size: number | null;
  effect_size_label: string | null;
  evidence_score: number;
  evidence_label: string;
  citations_present: boolean;
  metadata: Record<string, unknown>;
}

export interface FindingMatch {
  qtrial_finding: ComparableFinding;
  human_finding: ComparableFinding;
  relation: "agree" | "partial_agree" | "contradict";
  match_score: number;
  rationale: string;
  qtrial_evidence_stronger: boolean;
  text_used_for_matching?: Record<string, string>;
}

export interface ComparisonMetrics {
  total_qtrial_findings: number;
  total_human_findings: number;
  matched_pairs: number;
  qtrial_only_count: number;
  human_only_count: number;
  recall_against_human: number;
  novel_rate: number;
  agreement_count: number;
  partial_agreement_count: number;
  contradiction_count: number;
  agreement_rate_over_matched: number;
  contradiction_rate_over_matched: number;
  evidence_upgrade_rate: number;
  mcc: number | null;
  mcc_interpretation: string | null;
}

export interface HumanReportParseResult {
  source_name: string;
  findings: ComparableFinding[];
  total_candidates: number;
  discarded_candidates: number;
}

export interface ComparisonReport {
  analyst_report_name: string;
  summary: string;
  metrics: ComparisonMetrics;
  matched_findings: FindingMatch[];
  contradictions: FindingMatch[];
  qtrial_only_findings: ComparableFinding[];
  human_only_findings: ComparableFinding[];
  human_report_parse: HumanReportParseResult | null;
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
  narrative_summary: string;
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
  comparison_report: ComparisonReport | null;
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
  dictFile: File | null;
  analystReportFile: File | null;
  outcomeColumn: string;
  provider: string;
  model: string;
  detectedTreatmentColumns: string[];
  confirmedTreatmentColumns: string[];
  progressMessages: string[];
  warnings: string[];
  report: FinalReport | null;
  errorMessage: string | null;
  retryCount: number;
}

export type PipelineAction =
  | { type: "SET_CONTEXT"; payload: string }
  | { type: "SET_FILE"; payload: File }
  | { type: "SET_DICT_FILE"; payload: File | null }
  | { type: "SET_ANALYST_REPORT_FILE"; payload: File | null }
  | { type: "SET_OUTCOME_COLUMN"; payload: string }
  | { type: "SET_PROVIDER"; payload: { provider: string; model: string } }
  | { type: "START_UPLOAD" }
  | { type: "TREATMENT_DETECTED"; payload: string[] }
  | { type: "CONFIRM_TREATMENT"; payload: string[] }
  | { type: "PROGRESS"; payload: string }
  | { type: "WARNING"; payload: string }
  | { type: "DISMISS_WARNING"; payload: number }
  | { type: "COMPLETE"; payload: FinalReport }
  | { type: "ERROR"; payload: string }
  | { type: "RETRY" }
  | { type: "RESET" };
