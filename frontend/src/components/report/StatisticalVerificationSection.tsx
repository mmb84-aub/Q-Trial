import type { StatisticalVerificationReport, VerifiedClaim } from "../../types";

interface Props {
  verification: StatisticalVerificationReport;
}

const CAVEAT_TEXT =
  "Dataset-grounded statistical verification recomputes a narrow set of supported statistical checks directly from the uploaded dataset. Claims marked unsupported or not verifiable are not necessarily false; they may require covariates, clearer endpoint metadata, survival event coding, literature evidence, or analyses outside the current MVP. Verification is conservative and should not be interpreted as full clinical validation.";

const LABEL_ORDER: VerifiedClaim["label"][] = [
  "contradicted",
  "partial",
  "unsupported",
  "not_verifiable",
  "verified",
];

const LABEL_LABELS: Record<VerifiedClaim["label"], string> = {
  verified: "Verified",
  contradicted: "Contradicted",
  partial: "Partial",
  unsupported: "Unsupported",
  not_verifiable: "Not verifiable",
};

const LABEL_STYLES: Record<VerifiedClaim["label"], { background: string; color: string; border: string }> = {
  verified: { background: "#ecfdf5", color: "#065f46", border: "1px solid #a7f3d0" },
  contradicted: { background: "#fef2f2", color: "#b91c1c", border: "1px solid #fecaca" },
  partial: { background: "#fffbeb", color: "#92400e", border: "1px solid #fde68a" },
  unsupported: { background: "#f8fafc", color: "#475569", border: "1px solid #cbd5e1" },
  not_verifiable: { background: "#f9fafb", color: "#4b5563", border: "1px solid #d1d5db" },
};

const METRIC_CARDS: Array<{ key: keyof StatisticalVerificationReport["metrics"]; label: string; format?: (value: number) => string }> = [
  { key: "total_claims", label: "Total claims" },
  { key: "verified_count", label: "Verified" },
  { key: "contradicted_count", label: "Contradicted" },
  { key: "partial_count", label: "Partial" },
  { key: "unsupported_count", label: "Unsupported" },
  { key: "not_verifiable_count", label: "Not verifiable" },
  { key: "verification_rate", label: "Verification rate", format: formatPercent },
  { key: "contradiction_rate", label: "Contradiction rate", format: formatPercent },
];

export function StatisticalVerificationSection({ verification }: Props) {
  const claimsByLabel = LABEL_ORDER.map((label) => ({
    label,
    claims: verification.claims.filter((claim) => claim.label === label),
  }));

  return (
    <section aria-labelledby="statistical-verification-title" style={{ marginBottom: "2rem" }}>
      <h2 id="statistical-verification-title">Dataset-Grounded Statistical Verification</h2>

      <p style={{
        background: "#f8fafc",
        border: "1px solid #e2e8f0",
        borderRadius: 8,
        padding: "0.85rem 1rem",
        lineHeight: 1.6,
        color: "#1f2937",
      }}>
        {verification.summary}
      </p>

      <p style={{
        background: "#fff7ed",
        border: "1px solid #fdba74",
        borderRadius: 8,
        padding: "0.85rem 1rem",
        lineHeight: 1.6,
        color: "#9a3412",
        fontSize: "0.9rem",
      }}>
        <strong>Scope note:</strong> {CAVEAT_TEXT}
      </p>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
        gap: "0.75rem",
        marginTop: "1rem",
        marginBottom: "1.5rem",
      }}>
        {METRIC_CARDS.map((card) => {
          const value = verification.metrics[card.key];
          return (
            <div key={card.key} style={{
              border: "1px solid #e5e7eb",
              borderRadius: 8,
              padding: "0.9rem 1rem",
              background: "#fff",
            }}>
              <div style={{ fontSize: "0.78rem", textTransform: "uppercase", letterSpacing: "0.04em", color: "#6b7280" }}>
                {card.label}
              </div>
              <div style={{ fontSize: "1.35rem", fontWeight: 700, marginTop: "0.2rem", color: "#111827" }}>
                {card.format ? card.format(value) : value}
              </div>
            </div>
          );
        })}
      </div>

      {claimsByLabel.map(({ label, claims }) => (
        <ClaimGroup key={label} label={label} claims={claims} />
      ))}
    </section>
  );
}

function ClaimGroup({ label, claims }: { label: VerifiedClaim["label"]; claims: VerifiedClaim[] }) {
  return (
    <section style={{ marginBottom: "1.5rem" }}>
      <h3 style={{ marginBottom: "0.75rem" }}>
        {LABEL_LABELS[label]} Claims
      </h3>
      {claims.length === 0 ? (
        <p style={{ color: "#6b7280" }}>No {LABEL_LABELS[label].toLowerCase()} claims.</p>
      ) : (
        claims.map((claim) => <ClaimCard key={claim.claim_id} claim={claim} />)
      )}
    </section>
  );
}

function ClaimCard({ claim }: { claim: VerifiedClaim }) {
  const warnings = claim.confidence_warnings ?? [];
  const showDirection =
    typeof claim.metadata?.direction === "string" &&
    (isSignificantPValue(claim.recomputed_p_value) || isSignificantPValue(claim.reported_p_value));
  return (
    <article style={{
      border: "1px solid #e5e7eb",
      borderRadius: 8,
      background: "#fafafa",
      padding: "1rem 1.1rem",
      marginBottom: "0.75rem",
    }}>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap", marginBottom: "0.75rem" }}>
        <span style={{
          ...LABEL_STYLES[claim.label],
          borderRadius: 999,
          padding: "0.2rem 0.6rem",
          fontSize: "0.78rem",
          fontWeight: 700,
        }}>
          {LABEL_LABELS[claim.label]}
        </span>
        {claim.test_used && (
          <span style={{ fontSize: "0.78rem", color: "#6b7280" }}>
            Test: {claim.test_used}
          </span>
        )}
      </div>

      <div style={{ fontSize: "0.95rem", color: "#111827", lineHeight: 1.6 }}>
        {claim.source_text}
      </div>

      <div style={{ marginTop: "0.55rem", fontSize: "0.84rem", color: "#6b7280", lineHeight: 1.7 }}>
        {claim.variable && <span>Variable: {claim.variable}. </span>}
        {claim.endpoint && <span>Endpoint: {claim.endpoint}. </span>}
        {claim.reported_p_value !== null && <span>Reported {formatPLabel(claim.reported_p_value)}. </span>}
        {claim.recomputed_p_value !== null && <span>Recomputed {formatPLabel(claim.recomputed_p_value)}. </span>}
        {showDirection && <span>Claimed direction: {String(claim.metadata.direction)}. </span>}
        {claim.reported_effect_size !== null && (
          <span>
            Reported {claim.reported_effect_size_label ?? "effect"}={formatNumber(claim.reported_effect_size)}.{" "}
          </span>
        )}
        {claim.effect_size !== null && (
          <span>
            Recomputed {claim.effect_size_label ?? "effect"}={formatNumber(claim.effect_size)}.{" "}
          </span>
        )}
        {claim.effect_agreement && claim.effect_agreement !== "not_assessed" && (
          <span>Effect agreement: {claim.effect_agreement}. </span>
        )}
        {claim.reported_ci_lower !== null && claim.reported_ci_upper !== null && (
          <span>Reported CI {formatNumber(claim.reported_ci_lower)} to {formatNumber(claim.reported_ci_upper)}. </span>
        )}
        {claim.ci_lower !== null && claim.ci_upper !== null && (
          <span>Recomputed CI {formatNumber(claim.ci_lower)} to {formatNumber(claim.ci_upper)}. </span>
        )}
      </div>

      {warnings.length > 0 && (
        <div style={{
          marginTop: "0.65rem",
          background: "#fff7ed",
          border: "1px solid #fdba74",
          borderRadius: 6,
          padding: "0.5rem 0.75rem",
          fontSize: "0.82rem",
          color: "#9a3412",
        }}>
          <strong>Confidence warning:</strong> {warnings.join(" ")}
        </div>
      )}

      {claim.rationale && (
        <p style={{ marginTop: "0.75rem", marginBottom: 0, fontSize: "0.9rem", color: "#4b5563", lineHeight: 1.6 }}>
          <strong>Why:</strong> {claim.rationale}
        </p>
      )}
    </article>
  );
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "N/A";
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatP(value: number): string {
  if (!Number.isFinite(value)) return "N/A";
  if (value === 0) return "<1e-12";
  if (value >= 0.9995) return ">0.99";
  if (Math.abs(value) < 0.001) return value.toExponential(2);
  return formatNumber(value);
}

function formatPLabel(value: number): string {
  const formatted = formatP(value);
  return formatted.startsWith("<") || formatted.startsWith(">") ? `p ${formatted[0]} ${formatted.slice(1)}` : `p=${formatted}`;
}

function isSignificantPValue(value: number | null): boolean {
  return typeof value === "number" && Number.isFinite(value) && value < 0.05;
}
