import type { ComparableFinding, ComparisonReport, FindingMatch } from "../../types";

interface Props {
  comparison: ComparisonReport;
}

const RELATION_LABELS: Record<FindingMatch["relation"], string> = {
  agree: "Agree",
  partial_agree: "Partial agreement",
  contradict: "Contradiction",
};

const RELATION_STYLES: Record<FindingMatch["relation"], { background: string; color: string; border: string }> = {
  agree: { background: "#ecfdf5", color: "#065f46", border: "1px solid #a7f3d0" },
  partial_agree: { background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe" },
  contradict: { background: "#fef2f2", color: "#b91c1c", border: "1px solid #fecaca" },
};

const METRIC_CARDS: Array<{ key: keyof ComparisonReport["metrics"]; label: string; format?: (value: number) => string }> = [
  { key: "matched_pairs", label: "Matched findings" },
  { key: "qtrial_only_count", label: "Q-Trial only" },
  { key: "human_only_count", label: "Human only" },
  { key: "agreement_count", label: "Agreements" },
  { key: "contradiction_count", label: "Contradictions" },
  { key: "evidence_upgrade_rate", label: "Evidence upgrade rate", format: formatPercent },
  { key: "mcc", label: "MCC", format: formatMcc },
];

export function ComparisonSection({ comparison }: Props) {
  return (
    <section aria-labelledby="comparison-title" style={{ marginBottom: "2rem" }}>
      <h2 id="comparison-title">Automated Report Comparison</h2>
      <p style={{ color: "#4b5563", lineHeight: 1.6, marginBottom: "1rem" }}>
        Comparing Q-Trial against uploaded analyst report: <strong>{comparison.analyst_report_name}</strong>
      </p>

      <p style={{
        background: "#f8fafc",
        border: "1px solid #e2e8f0",
        borderRadius: 8,
        padding: "0.85rem 1rem",
        lineHeight: 1.6,
        color: "#1f2937",
      }}>
        {comparison.summary}
      </p>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
        gap: "0.75rem",
        marginTop: "1rem",
        marginBottom: "1.5rem",
      }}>
        {METRIC_CARDS.map((card) => {
          const value = comparison.metrics[card.key];
          return (
            <div key={card.key} style={{
              border: "1px solid #e5e7eb",
              borderRadius: 8,
              padding: "0.9rem 1rem",
              background: "#fff",
            }}>
              <div style={{ fontSize: "0.78rem", textTransform: "uppercase", letterSpacing: "0.04em", color: "#6b7280" }}>
                {card.label}
              </div>
              <div style={{ fontSize: "1.35rem", fontWeight: 700, marginTop: "0.2rem", color: "#111827" }}>
                {typeof value === "number" ? (card.format ? card.format(value) : value) : "N/A"}
              </div>
            </div>
          );
        })}
      </div>
      {comparison.metrics.mcc_interpretation && (
        <p style={{ marginTop: "-0.75rem", marginBottom: "1.5rem", color: "#4b5563", fontSize: "0.9rem" }}>
          MCC is computed on matched pairs with explicit binary significance labels; partial agreements are excluded.
          Interpretation: <strong>{comparison.metrics.mcc_interpretation}</strong>.
        </p>
      )}

      <ComparisonMatchList title="Matched Findings" matches={comparison.matched_findings} />
      <ComparisonMatchList
        title="Contradictions"
        matches={comparison.contradictions}
        emptyText="No direct contradictions were detected in matched findings."
      />
      <ComparableFindingList
        title="Q-Trial Only Findings"
        findings={comparison.qtrial_only_findings}
        emptyText="Q-Trial did not surface unmatched additional findings."
      />
      <ComparableFindingList
        title="Human Only Findings"
        findings={comparison.human_only_findings}
        emptyText="All human findings were matched to Q-Trial findings."
      />
    </section>
  );
}

function ComparisonMatchList({
  title,
  matches,
  emptyText = "No matched findings available.",
}: {
  title: string;
  matches: FindingMatch[];
  emptyText?: string;
}) {
  return (
    <section style={{ marginBottom: "1.5rem" }}>
      <h3 style={{ marginBottom: "0.75rem" }}>{title}</h3>
      {matches.length === 0 ? (
        <p style={{ color: "#6b7280" }}>{emptyText}</p>
      ) : (
        matches.map((match, index) => (
          <article
            key={`${match.qtrial_finding.finding_id}-${match.human_finding.finding_id}-${index}`}
            style={{
              border: "1px solid #e5e7eb",
              borderRadius: 8,
              background: "#fafafa",
              padding: "1rem 1.1rem",
              marginBottom: "0.75rem",
            }}
          >
            <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap", marginBottom: "0.75rem" }}>
              <span style={{
                ...RELATION_STYLES[match.relation],
                borderRadius: 999,
                padding: "0.2rem 0.6rem",
                fontSize: "0.78rem",
                fontWeight: 700,
              }}>
                {RELATION_LABELS[match.relation]}
              </span>
              <span style={{ fontSize: "0.78rem", color: "#6b7280" }}>
                Match score {Math.round(match.match_score * 100)}%
              </span>
              {match.qtrial_evidence_stronger && (
                <span style={{
                  background: "#f0fdf4",
                  color: "#166534",
                  border: "1px solid #bbf7d0",
                  borderRadius: 999,
                  padding: "0.2rem 0.6rem",
                  fontSize: "0.78rem",
                  fontWeight: 600,
                }}>
                  Evidence upgraded by Q-Trial
                </span>
              )}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.75rem" }}>
              <FindingPanel title="Q-Trial" finding={match.qtrial_finding} />
              <FindingPanel title="Human report" finding={match.human_finding} />
            </div>

            {match.rationale && (
              <p style={{ marginTop: "0.75rem", fontSize: "0.9rem", color: "#4b5563", lineHeight: 1.6 }}>
                <strong>Why:</strong> {match.rationale}
              </p>
            )}
          </article>
        ))
      )}
    </section>
  );
}

function ComparableFindingList({
  title,
  findings,
  emptyText,
}: {
  title: string;
  findings: ComparableFinding[];
  emptyText: string;
}) {
  return (
    <section style={{ marginBottom: "1.5rem" }}>
      <h3 style={{ marginBottom: "0.75rem" }}>{title}</h3>
      {findings.length === 0 ? (
        <p style={{ color: "#6b7280" }}>{emptyText}</p>
      ) : (
        findings.map((finding) => (
          <article
            key={finding.finding_id}
            style={{
              border: "1px solid #e5e7eb",
              borderRadius: 8,
              background: "#fafafa",
              padding: "0.9rem 1rem",
              marginBottom: "0.75rem",
            }}
          >
            <FindingPanel title={finding.source === "qtrial" ? "Q-Trial" : "Human report"} finding={finding} />
          </article>
        ))
      )}
    </section>
  );
}

function FindingPanel({ title, finding }: { title: string; finding: ComparableFinding }) {
  return (
    <div>
      <div style={{ fontSize: "0.82rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#6b7280", marginBottom: "0.35rem" }}>
        {title}
      </div>
      <div style={{ fontSize: "0.95rem", color: "#111827", lineHeight: 1.6 }}>
        {finding.finding_text}
      </div>
      <div style={{ marginTop: "0.45rem", fontSize: "0.82rem", color: "#6b7280", lineHeight: 1.6 }}>
        {finding.endpoint && <span>Endpoint: {finding.endpoint}. </span>}
        {finding.significance !== "unclear" && <span>Significance: {finding.significance.replace("_", " ")}. </span>}
        {finding.p_value !== null && <span>p={finding.p_value}. </span>}
        <span>Evidence: {finding.evidence_label || "none"}.</span>
      </div>
    </div>
  );
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatMcc(value: number): string {
  return value.toFixed(2);
}
