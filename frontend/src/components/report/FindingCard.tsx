import { useState } from "react";
import type { GroundedFinding } from "../../types";
import { GroundingBadge } from "./GroundingBadge";
import { ChartPanel } from "./ChartPanel";

interface Props {
  finding: GroundedFinding;
  index: number;
}

export function FindingCard({ finding, index }: Props) {
  const [expanded, setExpanded] = useState(false);
  const confidenceWarnings = Array.isArray(finding.confidence_warning)
    ? finding.confidence_warning
    : finding.confidence_warning
      ? [finding.confidence_warning]
      : [];

  return (
    <article
      aria-labelledby={`finding-${index}-title`}
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        padding: "1rem 1.25rem",
        marginBottom: "1rem",
        background: "#fafafa",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem", flexShrink: 0 }}>
          <GroundingBadge finding={finding} />
          {finding.finding_category && <CategoryBadge category={finding.finding_category} />}
        </div>
        <div style={{ flex: 1 }}>
          <h4
            id={`finding-${index}-title`}
            style={{ margin: 0, fontSize: "0.95rem", lineHeight: 1.5, color: "#111827" }}
          >
            {finding.finding_text}
          </h4>
          <FindingStats finding={finding} />
          {confidenceWarnings.length > 0 && (
            <div
              role="note"
              aria-label="Confidence warning"
              style={{
                marginTop: "0.5rem",
                background: "#fff7ed",
                border: "1px solid #fdba74",
                borderRadius: 6,
                padding: "0.5rem 0.75rem",
                fontSize: "0.82rem",
                color: "#9a3412",
              }}
            >
              <strong>Confidence warning:</strong>{" "}
              {confidenceWarnings.join(" ")}
            </div>
          )}
        </div>
        <button
          aria-expanded={expanded}
          aria-controls={`finding-${index}-details`}
          onClick={() => setExpanded((v) => !v)}
          style={{
            background: "none", border: "none", cursor: "pointer",
            fontSize: "0.8rem", color: "#6b7280", padding: "2px 6px",
            borderRadius: 4, flexShrink: 0,
          }}
        >
          {expanded ? "Hide ▲" : "Details ▼"}
        </button>
      </div>

      {expanded && (
        <div id={`finding-${index}-details`} style={{ marginTop: "1rem", borderTop: "1px solid #e5e7eb", paddingTop: "0.75rem" }}>
          {finding.test_selection_rationale && (
            <details style={{ marginBottom: "0.75rem" }}>
              <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: "0.85rem", color: "#374151" }}>
                Why this test was used
              </summary>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.85rem", color: "#4b5563" }}>
                {finding.test_selection_rationale}
              </p>
            </details>
          )}

          {finding.missingness_disclosure && (
            <p style={{ fontSize: "0.85rem", color: "#6b7280", marginBottom: "0.5rem" }}>
              <strong>Missingness:</strong> {finding.missingness_disclosure}
            </p>
          )}

          {finding.citations.length > 0 && (
            <div style={{ marginBottom: "0.5rem" }}>
              <strong style={{ fontSize: "0.85rem" }}>Citations:</strong>
              <ul style={{ margin: "0.25rem 0 0 0", paddingLeft: "1.25rem", fontSize: "0.85rem" }}>
                {finding.citations.map((c, i) => (
                  <li key={i}>
                    {c.url ? (
                      <a href={c.url} target="_blank" rel="noopener noreferrer" style={{ color: "#1d4ed8" }}>
                        {c.title}
                      </a>
                    ) : (
                      c.title
                    )}{" "}
                    ({c.year ?? "n.d."}, {c.source})
                  </li>
                ))}
              </ul>
            </div>
          )}

          {finding.novel_statement && (
            <p style={{ background: "#fffbeb", border: "1px solid #fcd34d", padding: "0.5rem 0.75rem", borderRadius: 4, fontSize: "0.85rem", margin: "0.5rem 0" }}>
              <strong>Novel finding:</strong> {finding.novel_statement}
            </p>
          )}

          {finding.literature_skipped && (
            <p style={{ color: "#92400e", fontSize: "0.8rem" }}>
              Literature validation skipped: {finding.literature_skip_note}
            </p>
          )}

          <ChartPanel finding={finding} />
        </div>
      )}
    </article>
  );
}

function FindingStats({ finding }: { finding: GroundedFinding }) {
  const stats = [
    finding.variable ? `Variable: ${displayToken(finding.variable)}` : null,
    finding.endpoint ? `Endpoint: ${displayToken(finding.endpoint)}` : null,
    finding.significant !== false && finding.direction && finding.direction !== "unknown"
      ? `Direction: ${displayDirection(finding.direction)}`
      : finding.significant !== false && finding.direction_label
        ? `Direction: ${finding.direction_label}`
        : null,
    finding.effect_size !== undefined && finding.effect_size !== null
      ? `${displayEffectLabel(finding.effect_size_label)}: ${formatNumber(finding.effect_size)}`
      : null,
    finding.p_value !== undefined && finding.p_value !== null ? formatPLabel(finding.p_value) : null,
    finding.test_type ? `Test: ${finding.test_type}` : null,
  ].filter((stat): stat is string => Boolean(stat));

  if (stats.length === 0) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", marginTop: "0.55rem" }}>
      {stats.map((stat) => (
        <span
          key={stat}
          style={{
            border: "1px solid #d1d5db",
            borderRadius: 4,
            padding: "0.18rem 0.45rem",
            fontSize: "0.78rem",
            color: "#374151",
            background: "#fff",
            lineHeight: 1.4,
          }}
        >
          {stat}
        </span>
      ))}
    </div>
  );
}

function CategoryBadge({ category }: { category: string }) {
  const note = ["statistical_note", "data_quality", "data_quality_note", "preprocessing", "pipeline_warning", "qc_note"].includes(category);
  return (
    <span
      aria-label={`Finding category: ${category}`}
      style={{
        border: note ? "1px solid #cbd5e1" : "1px solid #bfdbfe",
        color: note ? "#475569" : "#1d4ed8",
        background: note ? "#f8fafc" : "#eff6ff",
        padding: "2px 8px",
        borderRadius: 4,
        fontSize: "0.72rem",
        fontWeight: 700,
        whiteSpace: "nowrap",
      }}
    >
      {displayToken(category)}
    </span>
  );
}

function displayToken(value: string): string {
  return value.replace(/_/g, " ");
}

function displayDirection(direction: string): string {
  if (direction === "positive") return "higher variable, higher endpoint";
  if (direction === "negative") return "higher variable, lower endpoint";
  if (direction === "none") return "no direction";
  return direction;
}

function displayEffectLabel(label?: string | null): string {
  if (!label) return "Effect";
  const normalized = label.toLowerCase();
  const labels: Record<string, string> = {
    odds_ratio: "OR",
    hazard_ratio: "HR",
    risk_ratio: "RR",
    cramers_v: "Cramer's V",
    cohen_d: "Cohen's d",
    mean_difference: "Mean difference",
  };
  return labels[normalized] ?? label.replace(/_/g, " ");
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
