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
        <GroundingBadge finding={finding} />
        <h4
          id={`finding-${index}-title`}
          style={{ margin: 0, flex: 1, fontSize: "0.95rem", lineHeight: 1.5, color: "#111827" }}
        >
          {finding.finding_text}
        </h4>
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
