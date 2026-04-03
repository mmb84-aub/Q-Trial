import { useState } from "react";

interface Props {
  onSubmit: (context: string) => void;
}

export function StudyContextForm({ onSubmit }: Props) {
  const [value, setValue] = useState("");

  return (
    <div style={{ maxWidth: 640, margin: "3rem auto", padding: "0 1.5rem" }}>
      <p style={{ fontSize: "0.8rem", color: "#6b7280", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Step 1 of 4
      </p>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0 0 0.5rem" }}>
        What is this trial about?
      </h1>
      <p style={{ color: "#4b5563", marginBottom: "1.5rem", lineHeight: 1.6 }}>
        Describe your study in plain language. The analysis engine uses this to frame every
        finding in clinical terms — so results read like a clinician wrote them, not a
        statistics package.
      </p>
      <textarea
        aria-label="Study context"
        rows={5}
        style={{
          width: "100%", padding: "0.75rem", fontSize: "1rem",
          border: "1px solid #d1d5db", borderRadius: 6, resize: "vertical",
          fontFamily: "inherit", lineHeight: 1.5, boxSizing: "border-box",
        }}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="e.g. A double-blind RCT comparing D-penicillamine vs placebo in 312 patients with primary biliary cirrhosis. Primary endpoint: time to death or liver transplant."
      />
      <p style={{ fontSize: "0.8rem", color: "#9ca3af", margin: "0.4rem 0 1rem" }}>
        One or two sentences is enough. You can be more specific if you have it.
      </p>
      <button
        disabled={!value.trim()}
        onClick={() => onSubmit(value.trim())}
        style={{
          padding: "0.6rem 1.75rem", fontSize: "1rem", fontWeight: 600,
          background: value.trim() ? "#1d4ed8" : "#e5e7eb",
          color: value.trim() ? "#fff" : "#9ca3af",
          border: "none", borderRadius: 6, cursor: value.trim() ? "pointer" : "not-allowed",
        }}
      >
        Continue →
      </button>
    </div>
  );
}
