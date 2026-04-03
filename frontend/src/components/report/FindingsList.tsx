import type { GroundedFinding } from "../../types";
import { FindingCard } from "./FindingCard";

interface Props {
  findings: GroundedFinding[];
}

export function FindingsList({ findings }: Props) {
  if (findings.length === 0) return null;
  return (
    <section aria-labelledby="findings-title" style={{ marginBottom: "2rem" }}>
      <h2 id="findings-title">Grounded Findings</h2>
      {findings.map((f, i) => (
        <FindingCard key={i} finding={f} index={i} />
      ))}
    </section>
  );
}
