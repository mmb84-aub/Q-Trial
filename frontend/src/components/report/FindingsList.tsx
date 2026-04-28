import type { GroundedFinding } from "../../types";
import { FindingCard } from "./FindingCard";

interface Props {
  findings: GroundedFinding[];
}

export function FindingsList({ findings }: Props) {
  if (findings.length === 0) return null;

  const analyticalFindings = findings.filter((finding) =>
    ["analytical", "survival_result", "endpoint_result", undefined].includes(finding.finding_category),
  );
  const qcFindings = findings.filter((finding) =>
    !["analytical", "survival_result", "endpoint_result", undefined].includes(finding.finding_category),
  );

  return (
    <section aria-labelledby="findings-title" style={{ marginBottom: "2rem" }}>
      <h2 id="findings-title">Findings</h2>
      {analyticalFindings.length > 0 && (
        <>
          <h3 style={{ marginTop: "1.5rem", marginBottom: "1rem" }}>Analytical Findings</h3>
          {analyticalFindings.map((f, i) => (
            <FindingCard key={`analytical-${i}`} finding={f} index={i} />
          ))}
        </>
      )}
      {qcFindings.length > 0 && (
        <>
          <h3 style={{ marginTop: "1.5rem", marginBottom: "1rem" }}>Data Quality / Statistical Notes</h3>
          {qcFindings.map((f, i) => (
            <FindingCard key={`qc-${i}`} finding={f} index={analyticalFindings.length + i} />
          ))}
        </>
      )}
    </section>
  );
}
