import type { SecondPassReviewOutput, SecondPassReviewIssue } from "../../types";

interface Props {
  review: SecondPassReviewOutput;
}

export function SecondPassReviewSection({ review }: Props) {
  const outcomeColor = {
    accept: "#dcfce7",
    reject: "#fee2e2",
    revise: "#fef3c7",
    needs_more_context: "#e0e7ff",
  };

  const severityColor = {
    critical: "#ef4444",
    high: "#fca5a5",
    medium: "#fcd34d",
    low: "#fed7aa",
  };

  return (
    <section aria-labelledby="second-pass-title" style={{ marginBottom: "2rem" }}>
      <h2
        id="second-pass-title"
        style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.75rem" }}
      >
        Prior Report Review
      </h2>

      {/* Deterministic Review Summary */}
      <div
        style={{
          background: outcomeColor[review.outcome],
          border: `1px solid ${outcomeColor[review.outcome]}`,
          borderRadius: 6,
          padding: "1rem",
          marginBottom: "1.5rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "0.75rem" }}>
          <strong style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.5px" }}>
            {review.outcome.replace("_", " ")}
          </strong>
          <span
            style={{
              fontSize: "0.8rem",
              color: "#666",
              fontStyle: "italic",
            }}
          >
            Deterministic review
          </span>
        </div>
        <p style={{ margin: "0.5rem 0 0 0", lineHeight: 1.6, color: "#1f2937" }}>
          {review.summary}
        </p>
      </div>

      {/* Deterministic Review Details */}
      {review.issues.length > 0 && (
        <div style={{ marginBottom: "1.5rem" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem", color: "#111827" }}>
            Issues Identified ({review.issues.length})
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            {review.issues.map((issue: SecondPassReviewIssue) => (
              <div
                key={issue.issue_id}
                style={{
                  border: `1px solid #e5e7eb`,
                  borderLeft: `3px solid ${severityColor[issue.severity]}`,
                  borderRadius: 4,
                  padding: "0.75rem 1rem",
                  background: "#fafafa",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    justifyContent: "space-between",
                    gap: "1rem",
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
                      <strong
                        style={{
                          fontSize: "0.85rem",
                          textTransform: "uppercase",
                          letterSpacing: "0.5px",
                          color: "#374151",
                        }}
                      >
                        {issue.severity}
                      </strong>
                      <span style={{ fontSize: "0.8rem", color: "#6b7280" }}>
                        {issue.category.replace(/_/g, " ")}
                      </span>
                    </div>
                    <p style={{ margin: "0.5rem 0", fontSize: "0.95rem", lineHeight: 1.5, color: "#111827" }}>
                      <strong>{issue.finding}</strong>
                    </p>
                    <p style={{ margin: "0.25rem 0", fontSize: "0.85rem", color: "#6b7280" }}>
                      <em>Prior report citation:</em> {issue.prior_report_citation}
                    </p>
                    {issue.expected_evidence_citation && (
                      <p style={{ margin: "0.25rem 0", fontSize: "0.85rem", color: "#6b7280" }}>
                        <em>Expected evidence:</em> {issue.expected_evidence_citation}
                      </p>
                    )}
                    <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.9rem", color: "#374151" }}>
                      <strong>Recommendation:</strong> {issue.recommendation}
                    </p>

                    {/* LLM Refinement Comment (optional, per-issue) */}
                    {issue.llm_refinement_comment && (
                      <div
                        style={{
                          marginTop: "0.75rem",
                          background: "#f3f4f6",
                          border: "1px solid #e5e7eb",
                          borderRadius: 4,
                          padding: "0.75rem",
                          fontSize: "0.85rem",
                          color: "#374151",
                          lineHeight: 1.6,
                        }}
                      >
                        <span style={{ fontStyle: "italic", color: "#6b7280" }}>LLM refinement: </span>
                        {issue.llm_refinement_comment}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Claim Summary (if present) */}
      {(review.accepted_claims.length > 0 ||
        review.revised_claims.length > 0 ||
        review.dropped_claims.length > 0) && (
        <div style={{ marginBottom: "1.5rem" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem", color: "#111827" }}>
            Claim Status
          </h3>
          {review.accepted_claims.length > 0 && (
            <div style={{ marginBottom: "0.75rem" }}>
              <strong style={{ fontSize: "0.9rem", color: "#065f46" }}>Accepted ({review.accepted_claims.length})</strong>
              <ul style={{ margin: "0.5rem 0 0 0", paddingLeft: "1.25rem", fontSize: "0.9rem", color: "#1f2937" }}>
                {review.accepted_claims.map((claim, i) => (
                  <li key={i}>{claim}</li>
                ))}
              </ul>
            </div>
          )}
          {review.revised_claims.length > 0 && (
            <div style={{ marginBottom: "0.75rem" }}>
              <strong style={{ fontSize: "0.9rem", color: "#92400e" }}>Revised ({review.revised_claims.length})</strong>
              <ul style={{ margin: "0.5rem 0 0 0", paddingLeft: "1.25rem", fontSize: "0.9rem", color: "#1f2937" }}>
                {review.revised_claims.map((claim, i) => (
                  <li key={i}>{claim}</li>
                ))}
              </ul>
            </div>
          )}
          {review.dropped_claims.length > 0 && (
            <div>
              <strong style={{ fontSize: "0.9rem", color: "#7c2d12" }}>Dropped ({review.dropped_claims.length})</strong>
              <ul style={{ margin: "0.5rem 0 0 0", paddingLeft: "1.25rem", fontSize: "0.9rem", color: "#1f2937" }}>
                {review.dropped_claims.map((claim, i) => (
                  <li key={i}>{claim}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Optional LLM Refinement Section */}
      {(review.delta_summary || review.refinement_notes || (review.follow_up_questions && review.follow_up_questions.length > 0)) && (
        <div
          style={{
            background: "#f9fafb",
            border: "1px solid #e5e7eb",
            borderRadius: 6,
            padding: "1rem",
          }}
        >
          <div style={{ marginBottom: "0.75rem" }}>
            <strong style={{ fontSize: "0.9rem", color: "#6b7280" }}>
              Bounded LLM Refinement
            </strong>
            <span style={{ fontSize: "0.8rem", color: "#9ca3af", marginLeft: "0.5rem" }}>
              (Optional enhancement applied)
            </span>
          </div>

          {review.delta_summary && (
            <div style={{ marginBottom: "0.75rem" }}>
              <strong style={{ fontSize: "0.85rem", color: "#374151" }}>Changes to Summary:</strong>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.9rem", color: "#1f2937", lineHeight: 1.6 }}>
                {review.delta_summary}
              </p>
            </div>
          )}

          {review.refinement_notes && (
            <div style={{ marginBottom: "0.75rem" }}>
              <strong style={{ fontSize: "0.85rem", color: "#374151" }}>Refinement Notes:</strong>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.9rem", color: "#1f2937", lineHeight: 1.6 }}>
                {review.refinement_notes}
              </p>
            </div>
          )}

          {review.follow_up_questions && review.follow_up_questions.length > 0 && (
            <div>
              <strong style={{ fontSize: "0.85rem", color: "#374151" }}>Follow-up Questions:</strong>
              <ul style={{ margin: "0.5rem 0 0 0", paddingLeft: "1.25rem", fontSize: "0.9rem", color: "#1f2937" }}>
                {review.follow_up_questions.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
