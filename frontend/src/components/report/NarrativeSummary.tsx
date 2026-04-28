import ReactMarkdown from "react-markdown";
import type { FinalReport } from "../../types";

interface Props {
  report: FinalReport;
}

export function NarrativeSummary({ report }: Props) {
  // Primary: prose narrative from synthesis (stored in grounded_findings.synthesis.narrative_summary)
  // Fallback: key_findings list from final_insights (backward compat)
  const prose =
    report.grounded_findings?.synthesis?.narrative_summary ||
    report.final_insights.narrative_summary;
  const findings = report.final_insights.key_findings;
  const risks = report.final_insights.risks_and_bias_signals;

  return (
    <section aria-labelledby="narrative-title" style={{ marginBottom: "2rem" }}>
      <h2 id="narrative-title" style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.75rem" }}>
        Summary
      </h2>

      {prose ? (
        <div style={{ lineHeight: 1.7, color: "#1f2937" }}>
          <ReactMarkdown>{prose}</ReactMarkdown>
        </div>
      ) : (
        findings.length > 0 && (
          <ul style={{ paddingLeft: "1.25rem", lineHeight: 1.7, color: "#1f2937" }}>
            {findings.map((f, i) => (
              <li key={i}>
                <ReactMarkdown>{f}</ReactMarkdown>
              </li>
            ))}
          </ul>
        )
      )}

      {risks.length > 0 && (
        <div style={{ marginTop: "1rem", background: "#fef9c3", border: "1px solid #fde047", borderRadius: 6, padding: "0.75rem 1rem" }}>
          <strong style={{ fontSize: "0.85rem", color: "#92400e" }}>Risks and bias signals</strong>
          <ul style={{ margin: "0.5rem 0 0 0", paddingLeft: "1.25rem", fontSize: "0.9rem", color: "#78350f" }}>
            {risks.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
