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
    | "clinical_association"
    | "negative_association"
    | "survival_result"
    | "endpoint_result"
    | "statistical_note"
    | "data_quality"
    | "data_quality_note"
    | "preprocessing"
    | "pipeline_warning"
    | "qc_note"
    | "artifact_excluded";
  claim_type?:
    | "association_claim"
    | "analytical_association"
    | "negative_association"
    | "descriptive_claim"
    | "descriptive_context"
    | "statistical_note"
    | "data_quality_claim"
    | "data_quality_note"
    | "setup_claim"
    | "metadata_claim"
    | "recommendation"
    | "artifact";
  variable?: string | null;
  endpoint?: string | null;
  direction?: "positive" | "negative" | "none" | "unknown";
  direction_label?: string | null;
  significant?: boolean | null;
  significance?: "significant" | "not_significant" | "unclear";
  p_value?: number | null;
  effect_size?: number | null;
  effect_size_label?: string | null;
  test_type?: string | null;
  citations: LiteratureArticle[];
  evidence_strength: EvidenceStrengthScore | null;
  novel_statement: string | null;
  literature_skipped: boolean;
  literature_skip_note: string | null;
  test_selection_rationale: string | null;
  missingness_disclosure: string | null;
  confidence_warning: string | string[] | null;
  metadata?: Record<string, unknown>;
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
    | "clinical_association"
    | "negative_association"
    | "survival_result"
    | "endpoint_result"
    | "statistical_note"
    | "data_quality"
    | "data_quality_note"
    | "preprocessing"
    | "pipeline_warning"
    | "qc_note"
    | "artifact_excluded"
    | null;
  claim_type?:
    | "association_claim"
    | "analytical_association"
    | "negative_association"
    | "descriptive_claim"
    | "descriptive_context"
    | "statistical_note"
    | "data_quality_claim"
    | "data_quality_note"
    | "setup_claim"
    | "metadata_claim"
    | "recommendation"
    | "artifact"
    | null;
  variable: string | null;
  endpoint: string | null;
  direction: "positive" | "negative" | "none" | "unknown";
  significant: boolean | null;
  significance: "significant" | "not_significant" | "unclear";
  p_value: number | null;
  effect_size: number | null;
  effect_size_label: string | null;
  evidence_score: number;
  evidence_label: string;
  citations_present: boolean;
  metadata: Record<string, unknown>;
  statistical_evidence: StatisticalEvidence | null;
}

export interface StatisticalEvidence {
  variable: string | null;
  endpoint: string | null;
  test_type: string | null;
  test_family: string | null;
  p_value: number | null;
  p_operator: string | null;
  adjusted_p_value: number | null;
  raw_p_value: number | null;
  effect_size: number | null;
  effect_size_label: string | null;
  effect_direction: "positive" | "negative" | "none" | "unknown";
  direction_effect_on_endpoint:
    | "increases_endpoint_risk"
    | "decreases_endpoint_risk"
    | "no_direction"
    | "unknown";
  ci_lower: number | null;
  ci_upper: number | null;
  confidence_level: number | null;
  statistic_value: number | null;
  statistic_label: string | null;
  significant: boolean | null;
  direction: "positive" | "negative" | "none" | "unknown";
  sample_size: number | null;
  covariates: string[];
  rank: number | null;
  importance_score: number | null;
  extraction_confidence: number;
  source_text: string;
}

export interface StatisticalEvidenceComparison {
  available: boolean;
  reason_if_unavailable: string | null;
  statistical_agreement_score: number | null;
  statistical_agreement_coverage: number;
  overall_statistical_agreement_score: number | null;
  agreement_label: "strong" | "moderate" | "weak" | "contradiction" | "not_assessed";
  significance_agreement: string;
  direction_agreement: string;
  effect_size_agreement: string;
  p_value_agreement: string;
  ci_agreement: string;
  test_type_agreement: string;
  rank_agreement: string;
  effect_size_delta: number | null;
  effect_size_relative_delta: number | null;
  p_value_delta: number | null;
  p_value_log_delta: number | null;
  ci_overlap: boolean | null;
  coverage_score: number;
  qtrial_evidence: StatisticalEvidence | null;
  human_evidence: StatisticalEvidence | null;
  notes: string[];
  warnings: string[];
}

export interface FindingMatch {
  qtrial_finding: ComparableFinding;
  human_finding: ComparableFinding;
  relation: "agree" | "partial_agree" | "contradict";
  pairing_confidence: number | null;
  match_score: number;
  rationale: string;
  qtrial_evidence_stronger: boolean;
  text_used_for_matching?: Record<string, string>;
  statistical_comparison: StatisticalEvidenceComparison | null;
}

export interface ComparisonMetrics {
  total_qtrial_findings: number;
  total_human_findings: number;
  matched_pairs: number;
  qtrial_only_count: number;
  human_only_count: number;
  recall_against_human: number;
  precision_against_human: number;
  f1_against_human: number;
  novel_rate: number;
  agreement_count: number;
  partial_agreement_count: number;
  contradiction_count: number;
  agreement_rate_over_matched: number;
  contradiction_rate_over_matched: number;
  evidence_upgrade_rate: number;
  average_statistical_agreement_score: number | null;
  average_statistical_evidence_coverage: number;
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

export interface StatisticalVerificationMetrics {
  total_claims: number;
  verified_count: number;
  contradicted_count: number;
  partial_count: number;
  unsupported_count: number;
  not_verifiable_count: number;
  verification_rate: number;
  contradiction_rate: number;
}

export interface VerifiedClaim {
  claim_id: string;
  source_text: string;
  label: "verified" | "contradicted" | "partial" | "unsupported" | "not_verifiable";
  variable: string | null;
  endpoint: string | null;
  test_used: string | null;
  recomputed_p_value: number | null;
  reported_p_value: number | null;
  effect_size: number | null;
  effect_size_label: string | null;
  ci_lower: number | null;
  ci_upper: number | null;
  reported_effect_size: number | null;
  reported_effect_size_label: string | null;
  reported_ci_lower: number | null;
  reported_ci_upper: number | null;
  effect_agreement: "agrees" | "conflicts" | "partial" | "not_assessed" | null;
  confidence_warnings: string[];
  rationale: string;
  metadata: Record<string, unknown>;
}

export interface StatisticalVerificationReport {
  summary: string;
  metrics: StatisticalVerificationMetrics;
  claims: VerifiedClaim[];
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
  statistical_verification_report?: StatisticalVerificationReport | null;
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
