/**
 * ChartPanel renders a BarChart for categorical findings and a LineChart
 * for temporal findings, per the design spec.
 * When the backend supplies numeric series data the chart will be real;
 * until then a labelled placeholder is shown.
 */
import {
  BarChart, Bar,
  LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import type { GroundedFinding } from "../../types";

interface Props {
  finding: GroundedFinding;
}

const PLACEHOLDER_BAR = [
  { name: "Group A", value: 42 },
  { name: "Group B", value: 58 },
];

const PLACEHOLDER_LINE = [
  { name: "T0", value: 10 },
  { name: "T1", value: 18 },
  { name: "T2", value: 25 },
  { name: "T3", value: 22 },
];

function isTemporal(finding: GroundedFinding): boolean {
  const text = finding.finding_text.toLowerCase();
  return (
    text.includes("over time") ||
    text.includes("survival") ||
    text.includes("longitudinal") ||
    text.includes("follow-up") ||
    text.includes("months") ||
    text.includes("days") ||
    text.includes("trend")
  );
}

export function ChartPanel({ finding }: Props) {
  const temporal = isTemporal(finding);

  return (
    <div style={{ marginTop: "0.75rem" }} aria-label={`${temporal ? "Line" : "Bar"} chart for finding`}>
      <ResponsiveContainer width="100%" height={150}>
        {temporal ? (
          <LineChart data={PLACEHOLDER_LINE}>
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#0F6E56" strokeWidth={2} dot={false} />
          </LineChart>
        ) : (
          <BarChart data={PLACEHOLDER_BAR}>
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#1d4ed8" radius={[3, 3, 0, 0]} />
          </BarChart>
        )}
      </ResponsiveContainer>
      <p style={{ fontSize: "0.72rem", color: "#9ca3af", margin: "0.25rem 0 0 0" }}>
        Illustrative {temporal ? "line" : "bar"} chart — numeric series data required for real visualisation.
      </p>
    </div>
  );
}
