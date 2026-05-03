const FEATURE_SELECTION_METHODS = [
  {
    id: "univariate",
    label: "Univariate Statistical",
    description: "Fast baseline using F-statistics. Good for initial screening.",
    speed: "0.02s",
    pros: ["Fastest", "Simple & transparent", "High significance rate (75%)"],
    cons: ["Lower redundancy reduction"],
  },
  {
    id: "mrmr",
    label: "mRMR (Recommended)",
    description: "Minimum Redundancy Maximum Relevance. Best redundancy reduction.",
    speed: "0.09s",
    pros: ["Best redundancy reduction", "Good stability", "Greedy approach"],
    cons: ["Medium speed"],
  },
  {
    id: "qubo",
    label: "QUBO",
    description: "Quantum-inspired feature selection using redundancy-aware optimization.",
    speed: "9.6s",
    pros: ["Redundancy-aware optimization", "Strong clinical-column preservation", "Greedy fallback when needed"],
    cons: ["Slower than baseline methods"],
  },
  {
    id: "lasso",
    label: "LASSO Regression",
    description: "Regularized regression with feature coefficients for interpretation.",
    speed: "0.12s",
    pros: ["Interpretable coefficients", "Good balance", "Handles multicollinearity"],
    cons: ["May select more features"],
  },
  {
    id: "elastic_net",
    label: "Elastic Net",
    description: "Strong alternative combining L1 and L2 regularization.",
    speed: "0.10s",
    pros: ["Handles multicollinearity well", "Good balance", "Fast"],
    cons: ["Similar to LASSO"],
  },
];

interface Props {
  selectedMethod: string;
  onChange: (method: string) => void;
}

export function FeatureSelectionMethodPicker({ selectedMethod, onChange }: Props) {
  const selected = FEATURE_SELECTION_METHODS.find((m) => m.id === selectedMethod);

  return (
    <div style={{ marginBottom: "1.5rem" }}>
      <label style={{ fontWeight: 600, display: "block", marginBottom: "0.6rem" }}>
        Feature Selection Method
      </label>
      
      <div style={{
        background: "#f9fafb",
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        padding: "1rem",
        marginBottom: "1rem",
      }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}>
          {FEATURE_SELECTION_METHODS.map((method) => (
            <button
              key={method.id}
              onClick={() => onChange(method.id)}
              style={{
                padding: "0.75rem 1rem",
                border: selectedMethod === method.id ? "2px solid #1d4ed8" : "1px solid #d1d5db",
                borderRadius: 6,
                background: selectedMethod === method.id ? "#eff6ff" : "#fff",
                cursor: "pointer",
                fontSize: "0.95rem",
                fontWeight: selectedMethod === method.id ? 600 : 500,
                color: selectedMethod === method.id ? "#1d4ed8" : "#374151",
                transition: "all 0.2s ease",
                textAlign: "left",
              }}
            >
              {method.label}
              {selectedMethod === method.id && <span style={{ marginLeft: "0.4rem" }}>✓</span>}
            </button>
          ))}
        </div>
      </div>

      {selected && (
        <div style={{
          background: "#f3f4f6",
          border: "1px solid #e5e7eb",
          borderRadius: 6,
          padding: "1rem",
          marginTop: "1rem",
        }}>
          <div style={{ marginBottom: "0.75rem" }}>
            <p style={{ margin: 0, fontWeight: 600, color: "#111827", marginBottom: "0.4rem" }}>
              {selected.label}
            </p>
            <p style={{ margin: 0, fontSize: "0.9rem", color: "#6b7280", lineHeight: 1.5 }}>
              {selected.description}
            </p>
            <p style={{ margin: "0.5rem 0 0", fontSize: "0.8rem", color: "#9ca3af" }}>
              Runtime: ~{selected.speed}
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", fontSize: "0.85rem" }}>
            <div>
              <p style={{ fontWeight: 600, color: "#059669", marginBottom: "0.3rem" }}>✓ Strengths:</p>
              <ul style={{ margin: 0, paddingLeft: "1.2rem", color: "#374151" }}>
                {selected.pros.map((pro, i) => (
                  <li key={i} style={{ marginBottom: "0.2rem" }}>{pro}</li>
                ))}
              </ul>
            </div>
            <div>
              <p style={{ fontWeight: 600, color: "#dc2626", marginBottom: "0.3rem" }}>⚠ Tradeoffs:</p>
              <ul style={{ margin: 0, paddingLeft: "1.2rem", color: "#374151" }}>
                {selected.cons.map((con, i) => (
                  <li key={i} style={{ marginBottom: "0.2rem" }}>{con}</li>
                ))}
              </ul>
            </div>
          </div>

          <div style={{
            background: "#fef3c7",
            border: "1px solid #fcd34d",
            borderRadius: 4,
            padding: "0.65rem 0.75rem",
            marginTop: "1rem",
            fontSize: "0.8rem",
            color: "#92400e",
            lineHeight: 1.5,
          }}>
            <strong>Recommendation:</strong> Use {selectedMethod === "mrmr" ? "mRMR" : selectedMethod === "univariate" ? "Univariate" : selectedMethod === "lasso" ? "LASSO" : selectedMethod === "qubo" ? "QUBO" : "Elastic Net"} for clinical trial analysis. All methods produce comparable results; choose based on your need for interpretability (LASSO/Elastic Net), minimal feature redundancy (mRMR), or optimization-based selection (QUBO).
          </div>
        </div>
      )}

      <p style={{ fontSize: "0.75rem", color: "#9ca3af", margin: "0.8rem 0 0" }}>
        All methods are benchmarked on the same dataset and compared for redundancy, speed, and statistical significance.
      </p>
    </div>
  );
}
