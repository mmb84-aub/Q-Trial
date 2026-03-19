import { useRef, useState } from "react";

interface Props {
  studyContext: string;
  onDetect: (file: File, outcomeColumn: string) => void;
}

export function UploadForm({ studyContext, onDetect }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [outcomeColumn, setOutcomeColumn] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (file) onDetect(file, outcomeColumn);
  }

  return (
    <div style={{ maxWidth: 640, margin: "3rem auto", padding: "0 1.5rem" }}>
      <p style={{ fontSize: "0.8rem", color: "#6b7280", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Step 2 of 4
      </p>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0 0 0.5rem" }}>
        Upload your dataset
      </h1>

      {/* Echo back the study context so the user knows it was captured */}
      <div style={{
        background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 6,
        padding: "0.75rem 1rem", marginBottom: "1.5rem", fontSize: "0.9rem", color: "#1e40af",
      }}>
        <strong>Study:</strong> {studyContext}
      </div>

      <p style={{ color: "#4b5563", marginBottom: "1.5rem", lineHeight: 1.6 }}>
        Upload a CSV or Excel file. The system will scan column names to identify which
        column encodes treatment assignment — you'll confirm that before the analysis starts.
      </p>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "1.25rem" }}>
          <label htmlFor="dataset-file" style={{ fontWeight: 600, display: "block", marginBottom: "0.4rem" }}>
            Dataset file
          </label>
          <input
            id="dataset-file"
            type="file"
            accept=".csv,.xlsx"
            ref={inputRef}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            required
            style={{ fontSize: "0.95rem" }}
          />
          <p style={{ fontSize: "0.8rem", color: "#9ca3af", margin: "0.3rem 0 0" }}>
            CSV or XLSX, any size.
          </p>
        </div>

        <div style={{ marginBottom: "1.5rem" }}>
          <label htmlFor="outcome-col" style={{ fontWeight: 600, display: "block", marginBottom: "0.4rem" }}>
            Primary outcome column <span style={{ fontWeight: 400, color: "#6b7280" }}>(optional)</span>
          </label>
          <input
            id="outcome-col"
            type="text"
            value={outcomeColumn}
            onChange={(e) => setOutcomeColumn(e.target.value)}
            placeholder="e.g. status"
            style={{
              padding: "0.5rem 0.75rem", width: "100%", fontSize: "0.95rem",
              border: "1px solid #d1d5db", borderRadius: 6, boxSizing: "border-box",
            }}
          />
          <p style={{ fontSize: "0.8rem", color: "#9ca3af", margin: "0.3rem 0 0" }}>
            Helps the agent prioritise which column to focus on.
          </p>
        </div>

        <button
          type="submit"
          disabled={!file}
          style={{
            padding: "0.6rem 1.75rem", fontSize: "1rem", fontWeight: 600,
            background: file ? "#1d4ed8" : "#e5e7eb",
            color: file ? "#fff" : "#9ca3af",
            border: "none", borderRadius: 6, cursor: file ? "pointer" : "not-allowed",
          }}
        >
          Scan for treatment columns →
        </button>
      </form>
    </div>
  );
}
