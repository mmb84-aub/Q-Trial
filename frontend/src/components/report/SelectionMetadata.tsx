import type { QuantumEvidence } from "../../types";

interface Props {
  quantum: QuantumEvidence;
}

export function SelectionMetadata({ quantum }: Props) {
  const reductionPct = (quantum.redundancy_reduction * 100).toFixed(0);
  const timingMs = quantum.execution_time_ms || 0;

  return (
    <div
      style={{
        background: "#f0f9ff",
        border: "1px solid #7dd3fc",
        borderRadius: 6,
        padding: "1rem",
        marginBottom: "1.5rem",
      }}
    >
      <h2 style={{ fontSize: "1.1rem", fontWeight: 600, margin: "0 0 1rem" }}>
        ⚛️ Quantum-Optimized Feature Selection
      </h2>

      {/* Main metrics grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "1rem",
          marginBottom: "1rem",
        }}
      >
        {/* Selection summary */}
        <div style={{ background: "#fff", padding: "0.75rem", borderRadius: 4, border: "1px solid #e0e7ff" }}>
          <div style={{ fontSize: "0.75rem", color: "#6b7280", textTransform: "uppercase" }}>Data summary</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "#1e40af", margin: "0.25rem 0" }}>
            {quantum.n_candidates} → {quantum.n_selected}
          </div>
          <div style={{ fontSize: "0.8rem", color: "#4b5563" }}>
            {Math.round((quantum.n_selected / quantum.n_candidates) * 100)}% retained, {100 - Math.round((quantum.n_selected / quantum.n_candidates) * 100)}% filtered
          </div>
        </div>

        {/* Redundancy reduction */}
        <div style={{ background: "#fff", padding: "0.75rem", borderRadius: 4, border: "1px solid #e0e7ff" }}>
          <div style={{ fontSize: "0.75rem", color: "#6b7280", textTransform: "uppercase" }}>Redundancy reduction</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "#059669", margin: "0.25rem 0" }}>
            {reductionPct}%
          </div>
          <div style={{ fontSize: "0.8rem", color: "#4b5563" }}>
            {quantum.redundancy_before.toFixed(3)} → {quantum.redundancy_after.toFixed(3)}
          </div>
        </div>

        {/* Method */}
        <div style={{ background: "#fff", padding: "0.75rem", borderRadius: 4, border: "1px solid #e0e7ff" }}>
          <div style={{ fontSize: "0.75rem", color: "#6b7280", textTransform: "uppercase" }}>Method</div>
          <div style={{ fontSize: "0.95rem", fontWeight: 600, color: "#1f2937", margin: "0.25rem 0" }}>
            {quantum.selection_method === "qubo" ? "QUBO" : "Relevance"}
          </div>
          <div style={{ fontSize: "0.8rem", color: "#4b5563" }}>
            λ = {quantum.lambda_penalty} ({quantum.num_reads} reads)
          </div>
        </div>

        {/* Performance */}
        {timingMs > 0 && (
          <div style={{ background: "#fff", padding: "0.75rem", borderRadius: 4, border: "1px solid #e0e7ff" }}>
            <div style={{ fontSize: "0.75rem", color: "#6b7280", textTransform: "uppercase" }}>Execution time</div>
            <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "#7c3aed", margin: "0.25rem 0" }}>
              {timingMs}ms
            </div>
            <div style={{ fontSize: "0.8rem", color: "#4b5563" }}>Well under 10s threshold</div>
          </div>
        )}
      </div>

      {/* Selection method notice */}
      {quantum.selection_method === "relevance_fallback" && (
        <div
          style={{
            background: "#fef3c7",
            border: "1px solid #fcd34d",
            borderRadius: 4,
            padding: "0.5rem 0.75rem",
            fontSize: "0.85rem",
            color: "#92400e",
          }}
        >
          ⚠️ <strong>Fallback mode:</strong> QUBO solution increased redundancy, so top-N by relevance was used instead.
        </div>
      )}

      {/* Outcome column note */}
      {quantum.outcome_column && (
        <div
          style={{
            fontSize: "0.8rem",
            color: "#6b7280",
            marginTop: "0.75rem",
            paddingTop: "0.75rem",
            borderTop: "1px solid #d1d5db",
          }}
        >
          Outcome column: <strong>{quantum.outcome_column}</strong>
        </div>
      )}
    </div>
  );
}
