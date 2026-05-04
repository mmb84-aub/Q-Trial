import type { ComparableFinding, ComparisonReport, FindingMatch, StatisticalEvidence, StatisticalEvidenceComparison } from "../../types";

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
  { key: "precision_against_qtrial", label: "Precision", format: formatPercent },
  { key: "recall_against_human", label: "Recall", format: formatPercent },
  { key: "f1_against_human", label: "F1", format: formatPercent },
  { key: "qtrial_only_count", label: "Q-Trial only" },
  { key: "human_only_count", label: "Human only" },
  { key: "precision_against_human", label: "Precision", format: formatPercent },
  { key: "recall_against_human", label: "Recall", format: formatPercent },
  { key: "f1_against_human", label: "F1", format: formatPercent },
  { key: "mcc_against_human", label: "MCC", format: formatMcc },
  { key: "agreement_count", label: "Agreements" },
  { key: "contradiction_count", label: "Contradictions" },
  { key: "average_statistical_agreement_score", label: "Avg stat agreement", format: formatPercent },
  { key: "average_statistical_evidence_coverage", label: "Stat evidence coverage", format: formatPercent },
  { key: "evidence_upgrade_rate", label: "Evidence upgrade rate", format: formatPercent },
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
              <span
                title="Pairing confidence estimates whether the two findings refer to the same claim. It is not an agreement score."
                style={{ fontSize: "0.78rem", color: "#6b7280" }}
              >
                Pairing confidence {Math.round(pairingConfidence(match) * 100)}%
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
                  Q-Trial provides quantitative evidence
                </span>
              )}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.75rem" }}>
              <FindingPanel title="Q-Trial" finding={match.qtrial_finding} />
              <FindingPanel title="Human report" finding={match.human_finding} />
            </div>

            <StatisticalEvidenceBlock comparison={match.statistical_comparison} />

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

function StatisticalEvidenceBlock({ comparison }: { comparison: StatisticalEvidenceComparison | null }) {
  if (!comparison) return null;
  const labelStyle = statisticalLabelStyle(comparison.agreement_label);
  const score = comparison.statistical_agreement_score ?? comparison.overall_statistical_agreement_score;
  const coverage = comparison.statistical_agreement_coverage ?? comparison.coverage_score;
  return (
    <div style={{
      marginTop: "0.85rem",
      border: "1px solid #e5e7eb",
      borderRadius: 8,
      background: "#fff",
      padding: "0.85rem",
    }}>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap", marginBottom: "0.65rem" }}>
        <strong style={{ fontSize: "0.9rem", color: "#111827" }}>Statistical Evidence Agreement</strong>
        <span style={{ ...labelStyle, borderRadius: 999, padding: "0.16rem 0.52rem", fontSize: "0.75rem", fontWeight: 700 }}>
          {displayToken(comparison.agreement_label)}
        </span>
        {score !== null && (
          <span style={{ fontSize: "0.78rem", color: "#6b7280" }}>
            Score {Math.round(score * 100)}%
          </span>
        )}
        <span style={{ fontSize: "0.78rem", color: "#6b7280" }}>
          Coverage {Math.round(coverage * 100)}%
        </span>
      </div>

      {!comparison.available && comparison.reason_if_unavailable ? (
        <p style={{ margin: 0, color: "#6b7280", fontSize: "0.86rem", lineHeight: 1.55 }}>
          Statistical evidence not assessed: {comparison.reason_if_unavailable}
        </p>
      ) : (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.65rem" }}>
            <EvidenceMiniPanel title="Q-Trial stats" evidence={comparison.qtrial_evidence} />
            <EvidenceMiniPanel title="Human stats" evidence={comparison.human_evidence} />
          </div>
          <div style={{ marginTop: "0.65rem", fontSize: "0.82rem", color: "#4b5563", lineHeight: 1.6 }}>
            <span>Significance: {displayToken(comparison.significance_agreement)}. </span>
            <span>Direction: {displayToken(comparison.direction_agreement)}. </span>
            <span>Effect: {displayToken(comparison.effect_size_agreement)}. </span>
            <span>P-value: {displayToken(comparison.p_value_agreement)}. </span>
            <span>CI: {displayToken(comparison.ci_agreement)}. </span>
            <span>Test: {displayToken(comparison.test_type_agreement)}. </span>
            {comparison.effect_size_delta !== null && <span>Effect delta {formatNumber(comparison.effect_size_delta)}. </span>}
            {comparison.effect_size_relative_delta !== null && <span>Relative effect delta {formatPercent(comparison.effect_size_relative_delta)}. </span>}
            {comparison.p_value_log_delta !== null && <span>p log10 delta {formatNumber(comparison.p_value_log_delta)}. </span>}
            {comparison.ci_overlap !== null && <span>CI overlap: {comparison.ci_overlap ? "yes" : "no"}. </span>}
          </div>
        </>
      )}

      {(comparison.notes.length > 0 || comparison.warnings.length > 0) && (
        <p style={{ margin: "0.55rem 0 0", color: "#6b7280", fontSize: "0.82rem", lineHeight: 1.55 }}>
          {[...comparison.notes, ...comparison.warnings].join(" ")}
        </p>
      )}
    </div>
  );
}

function pairingConfidence(match: FindingMatch) {
  return match.pairing_confidence ?? match.match_score;
}

function EvidenceMiniPanel({ title, evidence }: { title: string; evidence: StatisticalEvidence | null }) {
  if (!evidence) {
    return <div style={{ color: "#6b7280", fontSize: "0.84rem" }}>{title}: no extracted evidence.</div>;
  }
  return (
    <div style={{ fontSize: "0.82rem", color: "#4b5563", lineHeight: 1.65 }}>
      <div style={{ color: "#111827", fontWeight: 700, marginBottom: "0.2rem" }}>{title}</div>
      {evidence.p_value !== null && <span>{formatPLabel(evidence.p_value)}. </span>}
      {evidence.effect_size !== null && <span>{displayEffectLabel(evidence.effect_size_label)}={formatNumber(evidence.effect_size)}. </span>}
      {evidence.ci_lower !== null && evidence.ci_upper !== null && (
        <span>CI {formatNumber(evidence.ci_lower)} to {formatNumber(evidence.ci_upper)}. </span>
      )}
      {evidence.test_type && <span>Test: {displayToken(evidence.test_type)}. </span>}
      {evidence.direction !== "unknown" && <span>Direction: {displayDirection(evidence.direction)}. </span>}
      {evidence.rank !== null && <span>Rank: {evidence.rank}. </span>}
      {evidence.effect_size === null && evidence.p_value === null && evidence.ci_lower === null && (
        <span>No numeric p-value, effect size, or CI reported. </span>
      )}
    </div>
  );
}

function statisticalLabelStyle(label: StatisticalEvidenceComparison["agreement_label"]) {
  if (label === "strong") return { background: "#ecfdf5", color: "#065f46", border: "1px solid #a7f3d0" };
  if (label === "moderate") return { background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe" };
  if (label === "weak") return { background: "#fffbeb", color: "#92400e", border: "1px solid #fde68a" };
  if (label === "contradiction") return { background: "#fef2f2", color: "#b91c1c", border: "1px solid #fecaca" };
  return { background: "#f9fafb", color: "#4b5563", border: "1px solid #d1d5db" };
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
        {finding.variable && <span>Variable: {displayToken(finding.variable)}. </span>}
        {finding.endpoint && <span>Endpoint: {finding.endpoint}. </span>}
        {finding.significant !== false && finding.direction !== "unknown" && <span>Direction: {displayDirection(finding.direction)}. </span>}
        {finding.significance !== "unclear" && <span>Significance: {finding.significance.replace("_", " ")}. </span>}
        {finding.effect_size !== null && (
          <span>{displayEffectLabel(finding.effect_size_label)}={formatNumber(finding.effect_size)}. </span>
        )}
        {finding.p_value !== null && <span>{formatPLabel(finding.p_value)}. </span>}
        {finding.finding_category && <span>Category: {displayToken(finding.finding_category)}. </span>}
        <span>Evidence: {finding.evidence_label || "none"}.</span>
      </div>
    </div>
  );
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatMcc(value: number): string {
  return formatNumber(value);
}

function displayDirection(direction: string): string {
  if (direction === "positive") return "higher variable, higher endpoint";
  if (direction === "negative") return "higher variable, lower endpoint";
  if (direction === "none") return "no direction";
  return direction;
}

function displayEffectLabel(label: string | null): string {
  if (!label) return "Effect";
  const labels: Record<string, string> = {
    odds_ratio: "OR",
    hazard_ratio: "HR",
    risk_ratio: "RR",
    cramers_v: "Cramer's V",
    cohen_d: "Cohen's d",
  };
  return labels[label.toLowerCase()] ?? displayToken(label);
}

function displayToken(value: string): string {
  return value.replace(/_/g, " ");
}

function formatP(value: number): string {
  if (!Number.isFinite(value)) return "N/A";
  if (value === 0) return "<1e-12";
  if (value >= 0.9995) return ">0.99";
  if (Math.abs(value) < 0.001) return value.toExponential(2);
  return formatNumber(value);
}

function formatPLabel(value: number): string {
  const formatted = formatP(value);
  return formatted.startsWith("<") || formatted.startsWith(">") ? `p ${formatted[0]} ${formatted.slice(1)}` : `p=${formatted}`;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "N/A";
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}
