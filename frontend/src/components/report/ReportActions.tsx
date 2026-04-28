import { useState } from "react";
import type { FinalReport } from "../../types";

interface Props {
  report: FinalReport;
}

export function PDFExportButton({ report }: Props) {
  const [loading, setLoading] = useState(false);

  async function handleExport() {
    setLoading(true);
    try {
      const form = new FormData();
      form.append("report_json", JSON.stringify(report));
      const res = await fetch("/api/report/pdf", { method: "POST", body: form });
      if (!res.ok) throw new Error("PDF generation failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "report.pdf";
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      alert("PDF export is temporarily unavailable. Your interactive report is still accessible.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <button onClick={handleExport} disabled={loading} style={{ marginRight: "0.5rem" }}>
      {loading ? "Generating PDF…" : "Export PDF"}
    </button>
  );
}

export function ReproducibilityLogDownload({ report }: Props) {
  const runId = report.reproducibility_log?.run_id;
  if (!runId) return null;

  async function handleDownload() {
    try {
      const res = await fetch(`/api/report/reproducibility/${runId}`);
      if (!res.ok) throw new Error("Log not found");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${runId}_reproducibility.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      alert("Reproducibility log is not available for this run.");
    }
  }

  return (
    <button onClick={handleDownload} style={{ marginRight: "0.5rem" }}>
      Download Reproducibility Log
    </button>
  );
}

export function ADLViewerButton() {
  const [adlContent, setAdlContent] = useState<string | null>(null);

  async function handleView() {
    try {
      const res = await fetch("/api/report/adl");
      if (!res.ok) throw new Error("ADL not available");
      const text = await res.text();
      setAdlContent(text);
    } catch {
      alert("Architecture Decision Log is temporarily unavailable.");
    }
  }

  return (
    <>
      <button onClick={handleView}>View Architecture Decision Log</button>
      {adlContent && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Architecture Decision Log"
          style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
            display: "flex", alignItems: "center", justifyContent: "center", zIndex: 200,
          }}
          onClick={() => setAdlContent(null)}
        >
          <div
            style={{
              background: "#fff", padding: "2rem", borderRadius: 8,
              maxWidth: 800, width: "90%", maxHeight: "80vh", overflowY: "auto",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setAdlContent(null)}
              style={{ float: "right", background: "none", border: "none", fontSize: "1.2rem", cursor: "pointer" }}
              aria-label="Close"
            >
              ✕
            </button>
            <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit" }}>{adlContent}</pre>
          </div>
        </div>
      )}
    </>
  );
}
