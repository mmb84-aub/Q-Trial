import type { FinalReport } from "../../types";
import { CoverPage } from "./CoverPage";
import { NarrativeSummary } from "./NarrativeSummary";
import { FindingsList } from "./FindingsList";
import { ExcludedColumnsSection } from "./ExcludedColumnsSection";
import { QuestionBank } from "./QuestionBank";
import { ForwardRecommendations } from "./ForwardRecommendations";
import { ComparisonSection } from "./ComparisonSection";
import { PDFExportButton, ReproducibilityLogDownload, ADLViewerButton } from "./ReportActions";

interface Props {
  report: FinalReport;
  onReset: () => void;
}

export function InteractiveReport({ report, onReset }: Props) {
  const generatedAt = new Date().toISOString();
  const gf = report.grounded_findings;
  const excluded = report.treatment_columns_excluded ?? [];

  return (
    <main style={{ maxWidth: 900, margin: "0 auto", padding: "2rem" }}>
      {/* Top bar */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: "1.5rem", flexWrap: "wrap", gap: "0.75rem",
      }}>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <PDFExportButton report={report} />
          <ReproducibilityLogDownload report={report} />
          <ADLViewerButton />
        </div>
        <button
          onClick={onReset}
          style={{
            padding: "0.4rem 1rem", fontSize: "0.85rem", background: "none",
            border: "1px solid #d1d5db", borderRadius: 6, cursor: "pointer", color: "#374151",
          }}
        >
          ← New analysis
        </button>
      </div>

      {/* Treatment blinding notice — prominent banner per design spec */}
      {excluded.length > 0 && (
        <div
          role="note"
          aria-label="Treatment blinding notice"
          style={{
            background: "#eff6ff",
            border: "1px solid #bfdbfe",
            borderRadius: 6,
            padding: "0.75rem 1rem",
            marginBottom: "1.5rem",
            fontSize: "0.875rem",
            color: "#1e40af",
          }}
        >
          <strong>Treatment group assignments were excluded from this analysis.</strong>
          {" "}Columns excluded: {excluded.join(", ")}.
          This preserves analytical integrity and prevents confounding.
        </div>
      )}

      <CoverPage report={report} generatedAt={generatedAt} />

      {/* Section order per design: narrative → findings → forward recs → research questions → excluded columns */}
      <NarrativeSummary report={report} />

      {gf && <FindingsList findings={gf.findings} />}

      {report.comparison_report && <ComparisonSection comparison={report.comparison_report} />}

      {gf?.synthesis && <ForwardRecommendations synthesis={gf.synthesis} />}

      {gf && (
        <QuestionBank
          questions={(() => {
            const seen = new Set<string>();
            return [
              ...gf.research_questions,
              ...(gf.synthesis?.research_questions ?? []),
            ].filter(q => {
              const key = q.question.trim().toLowerCase();
              if (seen.has(key)) return false;
              seen.add(key);
              return true;
            });
          })()}
        />
      )}

      {gf && <ExcludedColumnsSection columns={gf.excluded_columns} />}
    </main>
  );
}
