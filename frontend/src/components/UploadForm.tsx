import { useRef, useState } from "react";
import { FeatureSelectionMethodPicker } from "./FeatureSelectionMethodPicker";

// Well-known OpenRouter models the user can pick from, plus a free-text option
const OPENROUTER_MODELS = [
  { id: "openai/gpt-4o",                  label: "GPT-4o" },
  { id: "openai/gpt-4o-mini",             label: "GPT-4o Mini" },
  { id: "anthropic/claude-opus-4",        label: "Claude Opus 4" },
  { id: "anthropic/claude-sonnet-4-5",    label: "Claude Sonnet 4.5" },
  { id: "google/gemini-2.5-flash",        label: "Gemini 2.5 Flash" },
  { id: "google/gemini-2.5-pro",          label: "Gemini 2.5 Pro" },
  { id: "meta-llama/llama-3.3-70b-instruct", label: "Llama 3.3 70B" },
  { id: "deepseek/deepseek-r1",           label: "DeepSeek R1" },
  { id: "custom",                         label: "Custom model ID…" },
];

const PROVIDERS = [
  { id: "gemini",      label: "Gemini (direct)" },
  { id: "bedrock",     label: "AWS Bedrock" },
  { id: "openrouter",  label: "OpenRouter" },
  { id: "openai",      label: "OpenAI (direct)" },
  { id: "claude",      label: "Claude (direct)" },
];

const BEDROCK_MODELS = [
  { id: "anthropic.claude-sonnet-4-5-20250929-v1:0", label: "Claude Sonnet 4.5 (recommended)" },
  { id: "anthropic.claude-haiku-4-5-20251001-v1:0",  label: "Claude Haiku 4.5 (fast + cheap)" },
  { id: "amazon.nova-pro-v1:0",                      label: "Amazon Nova Pro" },
  { id: "amazon.nova-lite-v1:0",                     label: "Amazon Nova Lite (cheapest)" },
  { id: "anthropic.claude-3-5-sonnet-20241022-v2:0", label: "Claude 3.5 Sonnet v2 (legacy)" },
  { id: "custom",                                    label: "Custom model / profile ID…" },
];

interface Props {
  studyContext: string;
  onDetect: (
    file: File,
    dictFile: File | null,
    analystReportFile: File | null,
    outcomeColumn: string,
    provider: string,
    model: string,
    featureSelectionMethod: string,
  ) => void;
}

export function UploadForm({ studyContext, onDetect }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dictFile, setDictFile] = useState<File | null>(null);
  const [analystReportFile, setAnalystReportFile] = useState<File | null>(null);
  const [outcomeColumn, setOutcomeColumn] = useState("");
  const [provider, setProvider] = useState("gemini");
  const [orModel, setOrModel] = useState(OPENROUTER_MODELS[0].id);
  const [bedrockModel, setBedrockModel] = useState(BEDROCK_MODELS[0].id);
  const [customModel, setCustomModel] = useState("");
  const [featureSelectionMethod, setFeatureSelectionMethod] = useState("mrmr");

  function resolvedModel(): string {
    if (provider === "openrouter") return orModel === "custom" ? customModel.trim() : orModel;
    if (provider === "bedrock") return bedrockModel === "custom" ? customModel.trim() : bedrockModel;
    return "";
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (file) onDetect(file, dictFile, analystReportFile, outcomeColumn, provider, resolvedModel(), featureSelectionMethod);
  }

  const needsCustom = (provider === "openrouter" && orModel === "custom") ||
                      (provider === "bedrock" && bedrockModel === "custom");
  const canSubmit = file && (!needsCustom || customModel.trim());

  return (
    <div style={{ maxWidth: 640, margin: "3rem auto", padding: "0 1.5rem" }}>
      <p style={{ fontSize: "0.8rem", color: "#6b7280", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Step 2 of 4
      </p>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0 0 0.5rem" }}>
        Upload your dataset
      </h1>

      <div style={{
        background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 6,
        padding: "0.75rem 1rem", marginBottom: "1.5rem", fontSize: "0.9rem", color: "#1e40af",
      }}>
        <strong>Study:</strong> {studyContext}
      </div>

      <p style={{ color: "#4b5563", marginBottom: "1.5rem", lineHeight: 1.6 }}>
        Upload a CSV or Excel file. The system will scan column names to identify which
        column encodes treatment assignment — you'll confirm that before the analysis starts.
        You can also attach an existing analyst report for automated comparison.
      </p>

      <form onSubmit={handleSubmit}>
        {/* File */}
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

        {/* Analyst report */}
        <div style={{ marginBottom: "1.25rem" }}>
          <label htmlFor="analyst-report-file" style={{ fontWeight: 600, display: "block", marginBottom: "0.4rem" }}>
            Human analyst report <span style={{ fontWeight: 400, color: "#6b7280" }}>(optional)</span>
          </label>
          <input
            id="analyst-report-file"
            type="file"
            accept=".txt,.md,.markdown,.text,.rst,.json,text/plain,application/json"
            onChange={(e) => setAnalystReportFile(e.target.files?.[0] ?? null)}
            style={{ fontSize: "0.95rem" }}
          />
          <p style={{ fontSize: "0.8rem", color: "#9ca3af", margin: "0.3rem 0 0" }}>
            Supported in v1: plain UTF-8 text, Markdown, and JSON reports. If provided,
            Q-Trial will compare its findings against the uploaded analysis.
            {analystReportFile && (
              <span style={{ color: "#16a34a", marginLeft: "0.5rem" }}>✓ {analystReportFile.name}</span>
            )}
          </p>
        </div>

        {/* Data dictionary */}
        <div style={{ marginBottom: "1.25rem" }}>
          <label htmlFor="dict-file" style={{ fontWeight: 600, display: "block", marginBottom: "0.4rem" }}>
            Column dictionary <span style={{ fontWeight: 400, color: "#6b7280" }}>(optional)</span>
          </label>
          <input
            id="dict-file"
            type="file"
            accept=".json"
            onChange={(e) => setDictFile(e.target.files?.[0] ?? null)}
            style={{ fontSize: "0.95rem" }}
          />
          <p style={{ fontSize: "0.8rem", color: "#9ca3af", margin: "0.3rem 0 0" }}>
            JSON file mapping column names to descriptions, e.g.{" "}
            <code style={{ fontSize: "0.75rem" }}>{`{"status": "0=alive, 1=transplant, 2=dead"}`}</code>.
            Helps the agent interpret columns accurately.
            {dictFile && (
              <span style={{ color: "#16a34a", marginLeft: "0.5rem" }}>✓ {dictFile.name}</span>
            )}
          </p>
        </div>

        {/* Outcome column */}
        <div style={{ marginBottom: "1.25rem" }}>
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

        {/* Provider */}
        <div style={{ marginBottom: "1.25rem" }}>
          <label htmlFor="provider-select" style={{ fontWeight: 600, display: "block", marginBottom: "0.4rem" }}>
            AI provider
          </label>
          <select
            id="provider-select"
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            style={{
              padding: "0.5rem 0.75rem", width: "100%", fontSize: "0.95rem",
              border: "1px solid #d1d5db", borderRadius: 6, background: "#fff",
              boxSizing: "border-box",
            }}
          >
            {PROVIDERS.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
        </div>

        {/* OpenRouter model picker */}
        {provider === "openrouter" && (
          <div style={{ marginBottom: "1.25rem" }}>
            <label htmlFor="or-model-select" style={{ fontWeight: 600, display: "block", marginBottom: "0.4rem" }}>
              Model
            </label>
            <select
              id="or-model-select"
              value={orModel}
              onChange={(e) => setOrModel(e.target.value)}
              style={{
                padding: "0.5rem 0.75rem", width: "100%", fontSize: "0.95rem",
                border: "1px solid #d1d5db", borderRadius: 6, background: "#fff",
                boxSizing: "border-box",
              }}
            >
              {OPENROUTER_MODELS.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>

            {orModel === "custom" && (
              <input
                type="text"
                value={customModel}
                onChange={(e) => setCustomModel(e.target.value)}
                placeholder="e.g. mistralai/mistral-7b-instruct"
                style={{
                  marginTop: "0.5rem",
                  padding: "0.5rem 0.75rem", width: "100%", fontSize: "0.95rem",
                  border: "1px solid #d1d5db", borderRadius: 6, boxSizing: "border-box",
                }}
              />
            )}
            <p style={{ fontSize: "0.8rem", color: "#9ca3af", margin: "0.3rem 0 0" }}>
              Any model available on{" "}
              <a href="https://openrouter.ai/models" target="_blank" rel="noreferrer" style={{ color: "#6b7280" }}>
                openrouter.ai/models
              </a>
            </p>
          </div>
        )}

        {/* AWS Bedrock model picker */}
        {provider === "bedrock" && (
          <div style={{ marginBottom: "1.25rem" }}>
            <label htmlFor="bedrock-model-select" style={{ fontWeight: 600, display: "block", marginBottom: "0.4rem" }}>
              Model
            </label>
            <select
              id="bedrock-model-select"
              value={bedrockModel}
              onChange={(e) => setBedrockModel(e.target.value)}
              style={{
                padding: "0.5rem 0.75rem", width: "100%", fontSize: "0.95rem",
                border: "1px solid #d1d5db", borderRadius: 6, background: "#fff",
                boxSizing: "border-box",
              }}
            >
              {BEDROCK_MODELS.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>

            {bedrockModel === "custom" && (
              <input
                type="text"
                value={customModel}
                onChange={(e) => setCustomModel(e.target.value)}
                placeholder="e.g. anthropic.claude-3-haiku-20240307-v1:0"
                style={{
                  marginTop: "0.5rem",
                  padding: "0.5rem 0.75rem", width: "100%", fontSize: "0.95rem",
                  border: "1px solid #d1d5db", borderRadius: 6, boxSizing: "border-box",
                }}
              />
            )}
            <p style={{ fontSize: "0.8rem", color: "#9ca3af", margin: "0.3rem 0 0" }}>
              Requires <code>AWS_ACCESS_KEY_ID</code> + <code>AWS_SECRET_ACCESS_KEY</code> in <code>.env</code>,
              and the model enabled in your Bedrock console.
            </p>
          </div>
        )}

        {/* Feature Selection Method Picker */}
        <FeatureSelectionMethodPicker
          selectedMethod={featureSelectionMethod}
          onChange={setFeatureSelectionMethod}
        />

        <button
          type="submit"
          disabled={!canSubmit}
          style={{
            padding: "0.6rem 1.75rem", fontSize: "1rem", fontWeight: 600,
            background: canSubmit ? "#1d4ed8" : "#e5e7eb",
            color: canSubmit ? "#fff" : "#9ca3af",
            border: "none", borderRadius: 6, cursor: canSubmit ? "pointer" : "not-allowed",
          }}
        >
          Scan for treatment columns →
        </button>
      </form>
    </div>
  );
}
