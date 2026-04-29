import type { GroundedFinding } from "../../types";
import { FindingCard } from "./FindingCard";

interface Props {
  findings: GroundedFinding[];
}

export function FindingsList({ findings }: Props) {
  if (findings.length === 0) return null;

  const analyticalFindings = findings.filter((finding) =>
    ["analytical", "clinical_association", "negative_association", undefined].includes(finding.finding_category),
  );
  const qcFindings = findings.filter((finding) =>
    !["analytical", "clinical_association", "negative_association", undefined].includes(finding.finding_category),
  );

  return (
    <section aria-labelledby="findings-title" style={{ marginBottom: "2rem" }}>
      <h2 id="findings-title">Findings</h2>
      {analyticalFindings.length > 0 && (
        <div style={{ marginTop: "1.5rem" }}>
          <h3 style={{ marginTop: "1.5rem", marginBottom: "1rem" }}>Analytical Findings</h3>
          {analyticalFindings.map((f, i) => (
            <FindingCard key={`analytical-${i}`} finding={f} index={i} />
          ))}
        </div>
      )}
      {qcFindings.length > 0 && (
        <div
          style={{
            marginTop: "1.5rem",
            borderTop: "1px solid #e5e7eb",
            paddingTop: "1rem",
            background: "#f8fafc",
            borderRadius: 8,
            padding: "1rem",
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>Data Quality / Statistical Notes</h3>
          {qcFindings.map((f, i) => (
            <FindingCard key={`qc-${i}`} finding={f} index={analyticalFindings.length + i} />
          ))}
        </div>
      )}
    </section>
  );
}
