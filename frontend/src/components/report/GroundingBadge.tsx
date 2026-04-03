import type { GroundedFinding } from "../../types";

// Design-specified hex colours
const BADGE_STYLES: Record<string, { background: string; color: string }> = {
  Supported:    { background: "#0F6E56", color: "#ffffff" },
  Contradicted: { background: "#A32D2D", color: "#ffffff" },
  Novel:        { background: "#BA7517", color: "#ffffff" },
};

interface Props {
  finding: GroundedFinding;
}

export function GroundingBadge({ finding }: Props) {
  const style = BADGE_STYLES[finding.grounding_status] ?? { background: "#6b7280", color: "#ffffff" };
  return (
    <span
      aria-label={`Grounding status: ${finding.grounding_status}`}
      style={{
        ...style,
        padding: "2px 10px",
        borderRadius: 4,
        fontSize: "0.8rem",
        fontWeight: 700,
        letterSpacing: "0.03em",
        whiteSpace: "nowrap",
      }}
    >
      {finding.grounding_status}
    </span>
  );
}
