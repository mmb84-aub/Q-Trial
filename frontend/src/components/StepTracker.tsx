import type { PipelineStage } from "../types";

const STEPS: { id: PipelineStage | PipelineStage[]; label: string }[] = [
  { id: "idle",                  label: "Study context" },
  { id: "context_entered",       label: "Upload data" },
  { id: ["uploading", "detecting_treatment", "awaiting_confirmation"], label: "Review columns" },
  { id: "running",               label: "Analysis" },
  { id: "complete",              label: "Report" },
];

function matchesStep(stage: PipelineStage, id: PipelineStage | PipelineStage[]): boolean {
  return Array.isArray(id) ? id.includes(stage) : id === stage;
}

function stepIndex(stage: PipelineStage): number {
  return STEPS.findIndex((s) => matchesStep(stage, s.id));
}

interface Props { stage: PipelineStage }

export function StepTracker({ stage }: Props) {
  if (stage === "error") return null;
  const current = stepIndex(stage);

  return (
    <nav aria-label="Progress" style={{
      display: "flex", gap: 0, borderBottom: "1px solid #e0e0e0",
      padding: "0.75rem 2rem", background: "#fafafa",
    }}>
      {STEPS.map((step, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", flex: i < STEPS.length - 1 ? 1 : "none" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{
                width: 24, height: 24, borderRadius: "50%", display: "flex",
                alignItems: "center", justifyContent: "center", fontSize: "0.75rem",
                fontWeight: 700,
                background: done ? "#2563eb" : active ? "#1d4ed8" : "#e5e7eb",
                color: done || active ? "#fff" : "#6b7280",
                flexShrink: 0,
              }}>
                {done ? "✓" : i + 1}
              </span>
              <span style={{
                fontSize: "0.8rem",
                fontWeight: active ? 700 : 400,
                color: active ? "#1d4ed8" : done ? "#374151" : "#9ca3af",
                whiteSpace: "nowrap",
              }}>
                {step.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div style={{
                flex: 1, height: 2, margin: "0 0.5rem",
                background: done ? "#2563eb" : "#e5e7eb",
              }} />
            )}
          </div>
        );
      })}
    </nav>
  );
}
