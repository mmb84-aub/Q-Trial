import { useEffect } from "react";
import type { PipelineAction } from "../types";

interface Props {
  warnings: string[];
  dispatch: React.Dispatch<PipelineAction>;
}

// Auto-dismiss after 12 s, but user can also close manually
const AUTO_DISMISS_MS = 12_000;

export function ToastStack({ warnings, dispatch }: Props) {
  // Auto-dismiss the oldest toast after the timeout
  useEffect(() => {
    if (warnings.length === 0) return;
    const timer = setTimeout(() => {
      dispatch({ type: "DISMISS_WARNING", payload: 0 });
    }, AUTO_DISMISS_MS);
    return () => clearTimeout(timer);
  }, [warnings, dispatch]);

  if (warnings.length === 0) return null;

  return (
    <div
      aria-live="polite"
      style={{
        position: "fixed",
        bottom: "1.5rem",
        right: "1.5rem",
        zIndex: 200,
        display: "flex",
        flexDirection: "column",
        gap: "0.5rem",
        maxWidth: 420,
        width: "calc(100vw - 3rem)",
      }}
    >
      {warnings.map((msg, i) => (
        <div
          key={i}
          role="alert"
          style={{
            background: "#1c1917",
            border: "1px solid #44403c",
            borderLeft: "3px solid #f59e0b",
            borderRadius: 6,
            padding: "0.75rem 1rem",
            display: "flex",
            gap: "0.75rem",
            alignItems: "flex-start",
            boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
          }}
        >
          <span style={{ fontSize: "1rem", flexShrink: 0, marginTop: 1 }}>⚠</span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <p style={{
              margin: "0 0 0.25rem",
              fontSize: "0.8rem",
              fontWeight: 700,
              color: "#f59e0b",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}>
              Non-fatal warning
            </p>
            <p style={{
              margin: 0,
              fontSize: "0.8rem",
              color: "#d6d3d1",
              fontFamily: "monospace",
              wordBreak: "break-word",
              lineHeight: 1.5,
              // Truncate very long messages — user can expand
              maxHeight: 80,
              overflow: "hidden",
            }}>
              {msg}
            </p>
            <p style={{ margin: "0.35rem 0 0", fontSize: "0.75rem", color: "#78716c" }}>
              The pipeline is continuing. This warning will auto-dismiss.
            </p>
          </div>
          <button
            onClick={() => dispatch({ type: "DISMISS_WARNING", payload: i })}
            aria-label="Dismiss warning"
            style={{
              background: "none",
              border: "none",
              color: "#78716c",
              cursor: "pointer",
              fontSize: "1rem",
              padding: "0 0.25rem",
              flexShrink: 0,
              lineHeight: 1,
            }}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
