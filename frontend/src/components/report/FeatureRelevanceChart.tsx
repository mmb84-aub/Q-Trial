import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import type { QuantumEvidence } from "../../types";

interface Props {
  quantum: QuantumEvidence;
}

export function FeatureRelevanceChart({ quantum }: Props) {
  // Sort selected features by relevance score descending
  const selectedFeaturesData = quantum.selected_columns
    .map((col) => ({
      feature: col,
      relevance: quantum.relevance_scores[col] || 0,
    }))
    .sort((a, b) => b.relevance - a.relevance);

  if (selectedFeaturesData.length === 0) {
    return null;
  }

  // Color gradient: green (high) → yellow (medium) → gray (low)
  const getBarColor = (relevance: number) => {
    if (relevance >= 0.7) return "#10b981"; // Green
    if (relevance >= 0.4) return "#eab308"; // Yellow
    return "#9ca3af"; // Gray
  };

  return (
    <div
      style={{
        marginBottom: "2rem",
        background: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: 6,
        padding: "1.5rem",
      }}
    >
      <h3 style={{ fontSize: "1rem", fontWeight: 600, margin: "0 0 1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
        📊 Selected Features by Relevance
      </h3>
      <p style={{ fontSize: "0.85rem", color: "#6b7280", margin: "0 0 1rem" }}>
        Ranking of {selectedFeaturesData.length} selected features by their statistical relevance to the outcome column.
      </p>

      <ResponsiveContainer width="100%" height={Math.max(300, selectedFeaturesData.length * 25)}>
        <BarChart data={selectedFeaturesData} layout="vertical" margin={{ top: 5, right: 30, left: 150, bottom: 5 }}>
          <XAxis type="number" domain={[0, 1]} />
          <YAxis dataKey="feature" type="category" tick={{ fontSize: 12 }} width={140} />
          <Tooltip
            formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, "Relevance"]}
            contentStyle={{ background: "#1f2937", border: "none", borderRadius: 4, color: "#fff" }}
          />
          <Bar dataKey="relevance" radius={[0, 4, 4, 0]}>
            {selectedFeaturesData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.relevance)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div style={{ marginTop: "1rem", display: "flex", gap: "1.5rem", fontSize: "0.85rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <div style={{ width: 12, height: 12, background: "#10b981", borderRadius: 2 }} />
          <span>High (0.7–1.0)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <div style={{ width: 12, height: 12, background: "#eab308", borderRadius: 2 }} />
          <span>Medium (0.4–0.7)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <div style={{ width: 12, height: 12, background: "#9ca3af", borderRadius: 2 }} />
          <span>Low (0–0.4)</span>
        </div>
      </div>
    </div>
  );
}
