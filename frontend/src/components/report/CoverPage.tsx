import type { FinalReport } from "../../types";

interface Props {
  report: FinalReport;
  generatedAt: string;
}

export function CoverPage({ report, generatedAt }: Props) {
  const runId = report.reproducibility_log?.run_id ?? "—";
  const treatmentBlinded =
    report.treatment_columns_excluded.length > 0
      ? `Treatment columns excluded from analysis: ${report.treatment_columns_excluded.join(", ")}.`
      : "No treatment columns were detected or excluded.";

  return (
    <section aria-labelledby="cover-title" style={{ marginBottom: "2rem" }}>
      <h1 id="cover-title">Clinical Data Analysis Report</h1>
      <dl>
        <dt>Study Context</dt>
        <dd>{report.study_context ?? "Not provided"}</dd>
        <dt>Generated</dt>
        <dd>{generatedAt}</dd>
        <dt>Report Version</dt>
        <dd>1.0.0</dd>
        <dt>Run ID</dt>
        <dd style={{ fontFamily: "monospace" }}>{runId}</dd>
        <dt>Treatment Blinding</dt>
        <dd>{treatmentBlinded}</dd>
      </dl>
    </section>
  );
}
