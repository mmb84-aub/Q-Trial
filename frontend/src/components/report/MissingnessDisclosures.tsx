import type { GroundedFindings } from "../../types";

interface Props {
  groundedFindings: GroundedFindings;
}

export function MissingnessDisclosures({ groundedFindings }: Props) {
  const listwise = groundedFindings.findings
    .filter((f) => f.missingness_disclosure)
    .map((f) => f.missingness_disclosure as string);

  if (listwise.length === 0 && groundedFindings.high_missingness_columns.length === 0) return null;

  return (
    <section aria-labelledby="missingness-title" style={{ marginBottom: "2rem" }}>
      <h2 id="missingness-title">Missingness Disclosures</h2>
      {listwise.length > 0 && (
        <>
          <h3>Listwise Deletion Applied</h3>
          <ul>
            {listwise.map((d, i) => <li key={i}>{d}</li>)}
          </ul>
        </>
      )}
      {groundedFindings.high_missingness_columns.length > 0 && (
        <>
          <h3>High Missingness Columns (20–50% — excluded from primary analysis)</h3>
          <ul>
            {groundedFindings.high_missingness_columns.map((c, i) => (
              <li key={i}>
                <strong>{c.column}</strong> — {(c.missingness_rate * 100).toFixed(1)}% missing
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
