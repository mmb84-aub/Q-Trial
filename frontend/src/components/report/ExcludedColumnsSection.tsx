import type { ExcludedColumn } from "../../types";

interface Props {
  columns: ExcludedColumn[];
}

export function ExcludedColumnsSection({ columns }: Props) {
  if (columns.length === 0) return null;
  return (
    <section aria-labelledby="excluded-title" style={{ marginBottom: "2rem" }}>
      <h2 id="excluded-title">Excluded Columns (&gt;50% Missing)</h2>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "0.4rem 0.8rem", borderBottom: "2px solid #dee2e6" }}>Column</th>
            <th style={{ textAlign: "left", padding: "0.4rem 0.8rem", borderBottom: "2px solid #dee2e6" }}>Missingness Rate</th>
            <th style={{ textAlign: "left", padding: "0.4rem 0.8rem", borderBottom: "2px solid #dee2e6" }}>Reason</th>
          </tr>
        </thead>
        <tbody>
          {columns.map((c, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #dee2e6" }}>
              <td style={{ padding: "0.4rem 0.8rem" }}>{c.column}</td>
              <td style={{ padding: "0.4rem 0.8rem" }}>{(c.missingness_rate * 100).toFixed(1)}%</td>
              <td style={{ padding: "0.4rem 0.8rem" }}>{c.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
