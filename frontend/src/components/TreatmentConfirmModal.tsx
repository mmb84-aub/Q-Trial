import { useState } from "react";

interface Props {
  detected: string[];
  onConfirm: (confirmed: string[]) => void;
}

export function TreatmentConfirmModal({ detected, onConfirm }: Props) {
  const [selected, setSelected] = useState<Set<string>>(new Set(detected));
  const [custom, setCustom] = useState("");

  function toggle(col: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(col) ? next.delete(col) : next.add(col);
      return next;
    });
  }

  function addCustom() {
    const trimmed = custom.trim();
    if (trimmed) {
      setSelected((prev) => new Set([...prev, trimmed]));
      setCustom("");
    }
  }

  const noneDetected = detected.length === 0;

  return (
    <div style={{ maxWidth: 640, margin: "3rem auto", padding: "0 1.5rem" }}>
      <p style={{ fontSize: "0.8rem", color: "#6b7280", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Step 3 of 4
      </p>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0 0 0.5rem" }}>
        Confirm treatment columns
      </h1>
      <p style={{ color: "#4b5563", marginBottom: "0.75rem", lineHeight: 1.6 }}>
        Treatment assignment columns must be excluded from the statistical analysis.
        Including them would violate the intention-to-treat principle — the analysis
        would be measuring the treatment itself rather than its effect.
      </p>

      {noneDetected ? (
        <div style={{
          background: "#fefce8", border: "1px solid #fde68a", borderRadius: 6,
          padding: "0.75rem 1rem", marginBottom: "1rem", fontSize: "0.9rem", color: "#92400e",
        }}>
          No treatment columns were automatically detected. If your dataset has one,
          add it manually below.
        </div>
      ) : (
        <div style={{
          background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 6,
          padding: "0.75rem 1rem", marginBottom: "1rem", fontSize: "0.9rem", color: "#166534",
        }}>
          {detected.length} column{detected.length > 1 ? "s" : ""} detected.
          Uncheck any that are not treatment assignments.
        </div>
      )}

      <ul style={{ listStyle: "none", padding: 0, margin: "0 0 1rem" }}>
        {[...selected].map((col) => (
          <li key={col} style={{
            display: "flex", alignItems: "center", gap: "0.5rem",
            padding: "0.5rem 0", borderBottom: "1px solid #f3f4f6",
          }}>
            <input
              type="checkbox"
              id={`col-${col}`}
              checked={selected.has(col)}
              onChange={() => toggle(col)}
              style={{ width: 16, height: 16 }}
            />
            <label htmlFor={`col-${col}`} style={{ fontFamily: "monospace", fontSize: "0.95rem" }}>
              {col}
            </label>
            {detected.includes(col) && (
              <span style={{ fontSize: "0.75rem", color: "#6b7280", marginLeft: "auto" }}>auto-detected</span>
            )}
          </li>
        ))}
      </ul>

      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem" }}>
        <input
          type="text"
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          placeholder="Add a column manually"
          style={{
            flex: 1, padding: "0.5rem 0.75rem", fontSize: "0.95rem",
            border: "1px solid #d1d5db", borderRadius: 6,
          }}
          onKeyDown={(e) => e.key === "Enter" && addCustom()}
        />
        <button
          onClick={addCustom}
          style={{
            padding: "0.5rem 1rem", background: "#f3f4f6", border: "1px solid #d1d5db",
            borderRadius: 6, cursor: "pointer", fontWeight: 600,
          }}
        >
          Add
        </button>
      </div>

      <button
        onClick={() => onConfirm(Array.from(selected))}
        style={{
          padding: "0.6rem 1.75rem", fontSize: "1rem", fontWeight: 600,
          background: "#1d4ed8", color: "#fff",
          border: "none", borderRadius: 6, cursor: "pointer",
        }}
      >
        {selected.size === 0 ? "Continue with no exclusions →" : `Exclude ${selected.size} column${selected.size > 1 ? "s" : ""} and run →`}
      </button>
    </div>
  );
}
