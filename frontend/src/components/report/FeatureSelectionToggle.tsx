import { useState } from "react";
import type { QuantumEvidence } from "../../types";

interface Props {
  quantum: QuantumEvidence;
}

export function FeatureSelectionToggle({ quantum }: Props) {
  const [activeTab, setActiveTab] = useState<"selected" | "excluded">("selected");
  const [searchTerm, setSearchTerm] = useState("");

  const selectedFeatures = quantum.selected_columns
    .filter((col) => col.toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) => (quantum.relevance_scores[b] || 0) - (quantum.relevance_scores[a] || 0));

  const excludedFeatures = quantum.excluded_columns.filter((col) =>
    col.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div
      style={{
        marginBottom: "2rem",
        background: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: 6,
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div style={{ padding: "1.5rem", borderBottom: "1px solid #e5e7eb" }}>
        <h3 style={{ fontSize: "1rem", fontWeight: 600, margin: "0 0 1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
          🎯 Feature Selection Details
        </h3>

        {/* Tab buttons */}
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
          <button
            onClick={() => setActiveTab("selected")}
            style={{
              padding: "0.5rem 1rem",
              fontSize: "0.9rem",
              fontWeight: activeTab === "selected" ? 600 : 400,
              background: activeTab === "selected" ? "#059669" : "#e5e7eb",
              color: activeTab === "selected" ? "#fff" : "#374151",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              transition: "all 0.2s",
            }}
          >
            ✓ Selected ({quantum.n_selected})
          </button>
          <button
            onClick={() => setActiveTab("excluded")}
            style={{
              padding: "0.5rem 1rem",
              fontSize: "0.9rem",
              fontWeight: activeTab === "excluded" ? 600 : 400,
              background: activeTab === "excluded" ? "#dc2626" : "#e5e7eb",
              color: activeTab === "excluded" ? "#fff" : "#374151",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              transition: "all 0.2s",
            }}
          >
            ✗ Excluded ({quantum.excluded_columns.length})
          </button>
        </div>

        {/* Search box */}
        <input
          type="text"
          placeholder="Search features..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{
            width: "100%",
            padding: "0.5rem 0.75rem",
            fontSize: "0.9rem",
            border: "1px solid #d1d5db",
            borderRadius: 4,
            fontFamily: "inherit",
          }}
        />
      </div>

      {/* Content */}
      <div style={{ padding: "1.5rem", maxHeight: 400, overflowY: "auto" }}>
        {activeTab === "selected" ? (
          <div>
            {selectedFeatures.length === 0 && searchTerm ? (
              <p style={{ color: "#9ca3af", fontSize: "0.9rem" }}>No selected features match "{searchTerm}"</p>
            ) : selectedFeatures.length === 0 ? (
              <p style={{ color: "#9ca3af", fontSize: "0.9rem" }}>No selected features</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                {selectedFeatures.map((col, idx) => {
                  const relevance = quantum.relevance_scores[col] || 0;
                  return (
                    <div
                      key={col}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "1rem",
                        padding: "0.75rem",
                        background: "#f9fafb",
                        borderRadius: 4,
                        border: "1px solid #e5e7eb",
                      }}
                    >
                      {/* Rank badge */}
                      <div
                        style={{
                          minWidth: 28,
                          width: 28,
                          height: 28,
                          background: "#059669",
                          color: "#fff",
                          borderRadius: "50%",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "0.8rem",
                          fontWeight: 700,
                        }}
                      >
                        {idx + 1}
                      </div>

                      {/* Feature name and relevance bar */}
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: "0.9rem", fontWeight: 500, color: "#1f2937", marginBottom: "0.25rem" }}>
                          {col}
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                          <div
                            style={{
                              flex: 1,
                              maxWidth: 100,
                              height: 4,
                              background: "#e5e7eb",
                              borderRadius: 2,
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                height: "100%",
                                width: `${relevance * 100}%`,
                                background: "#10b981",
                              }}
                            />
                          </div>
                          <span style={{ fontSize: "0.75rem", color: "#6b7280" }}>
                            {(relevance * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div>
            {excludedFeatures.length === 0 && searchTerm ? (
              <p style={{ color: "#9ca3af", fontSize: "0.9rem" }}>No excluded features match "{searchTerm}"</p>
            ) : excludedFeatures.length === 0 ? (
              <p style={{ color: "#9ca3af", fontSize: "0.9rem" }}>No excluded features</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                {excludedFeatures.map((col) => {
                  const relevance = quantum.relevance_scores[col] || 0;
                  return (
                    <div
                      key={col}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "1rem",
                        padding: "0.75rem",
                        background: "#fef2f2",
                        borderRadius: 4,
                        border: "1px solid #fee2e2",
                      }}
                    >
                      {/* Exclusion badge */}
                      <div
                        style={{
                          minWidth: 28,
                          width: 28,
                          height: 28,
                          background: "#dc2626",
                          color: "#fff",
                          borderRadius: "50%",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "0.9rem",
                        }}
                      >
                        ✕
                      </div>

                      {/* Feature name and reason */}
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: "0.9rem", fontWeight: 500, color: "#1f2937", marginBottom: "0.25rem" }}>
                          {col}
                        </div>
                        <div style={{ fontSize: "0.8rem", color: "#6b7280" }}>
                          {relevance < 0.1
                            ? "Low relevance to outcome"
                            : "High correlation with selected features"}
                        </div>
                      </div>

                      {/* Relevance score */}
                      <div style={{ fontSize: "0.85rem", color: "#9ca3af", minWidth: 40, textAlign: "right" }}>
                        {(relevance * 100).toFixed(0)}%
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
