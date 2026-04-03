"""
Architecture Decision Log (ADL) for the Q-Trial Clinical Data Analyst Agent.

Each entry documents a design decision, the alternatives considered, the
rationale chosen, and the clinical methodology standard it satisfies.

References:
  - ICH E9 (R1): Statistical Principles for Clinical Trials (1998, addendum 2019)
  - CONSORT 2010: Consolidated Standards of Reporting Trials
  - OCEBM: Oxford Centre for Evidence-Based Medicine Levels of Evidence (2011)
"""
from __future__ import annotations

_ADL_ENTRIES: list[dict] = [
    {
        "id": "ADL-001",
        "title": "Critic Loop (Synthesis Quality Self-Scoring)",
        "status": "Accepted",
        "context": (
            "The InsightSynthesisAgent produces a narrative summary of statistical "
            "findings. Without a quality gate, low-quality or incomplete syntheses "
            "could be presented to clinicians as authoritative conclusions."
        ),
        "decision": (
            "Implement a self-scoring critic loop: after synthesis, a second LLM call "
            "evaluates the output against a four-dimension rubric (completeness, clinical "
            "relevance, plain language, evidence grounding) and returns a score in [0, 1]. "
            "If the score falls below the configurable SYNTHESIS_QUALITY_THRESHOLD (default "
            "0.7), the synthesis step is re-run once. The score and rationale are recorded "
            "in SynthesisQualityScore and attached to the reproducibility log."
        ),
        "alternatives": [
            "Human-in-the-loop review before report delivery (rejected: adds latency, "
            "not feasible for automated pipelines).",
            "Fixed rubric checklist without LLM scoring (rejected: too rigid for "
            "heterogeneous study designs).",
        ],
        "consequences": (
            "One additional LLM call per run; at most two synthesis attempts. "
            "The re-run flag is surfaced in the report so reviewers know a quality "
            "gate was triggered."
        ),
        "references": [
            "ICH E9 §5.5 — Integrity of data and computer software validity",
            "CONSORT 2010 §15 — Outcomes and estimation",
        ],
    },
    {
        "id": "ADL-002",
        "title": "Treatment Column Blinding",
        "status": "Accepted",
        "context": (
            "In randomised controlled trials the treatment assignment column must not "
            "be used as a predictor or covariate in the primary analysis. Including it "
            "would violate the intention-to-treat principle and introduce allocation bias."
        ),
        "decision": (
            "Implement detect_treatment_columns() using a two-stage heuristic: "
            "(1) name-pattern matching against canonical terms "
            "(treatment/arm/group/intervention/control/allocation/randomized, "
            "case-insensitive), and (2) cardinality check (2–5 unique values, no "
            "single value exceeding 60% of rows). Detected columns are excluded from "
            "all statistical analyses and listed in treatment_columns_excluded on the "
            "FinalReportSchema. The user is shown the detected columns and may override "
            "before the pipeline runs."
        ),
        "alternatives": [
            "Manual column exclusion by the user only (rejected: error-prone, "
            "clinicians may not know which column encodes treatment).",
            "Exclude all categorical columns (rejected: over-broad, removes "
            "legitimate covariates such as sex and site).",
        ],
        "consequences": (
            "False positives are possible for columns that match the name pattern "
            "but are not treatment assignments; the confirmation modal allows the "
            "user to correct this. False negatives are possible for non-standard "
            "column names; the user can add columns manually."
        ),
        "references": [
            "ICH E9 §5.2 — Analysis sets (intention-to-treat principle)",
            "CONSORT 2010 §10 — Allocation concealment mechanism",
            "CONSORT 2010 §16 — Numbers analysed",
        ],
    },
    {
        "id": "ADL-003",
        "title": "Missing Data Strategy",
        "status": "Accepted",
        "context": (
            "Missing data is ubiquitous in clinical trial datasets. The choice of "
            "imputation strategy materially affects the validity of statistical "
            "conclusions. ICH E9 (R1) requires that the missing data strategy be "
            "pre-specified and justified."
        ),
        "decision": (
            "Adopt a three-tier missingness policy keyed on per-column missingness rate: "
            "(1) >50% missing → column excluded from all analyses (ExcludedColumn); "
            "(2) 20–50% missing → column retained in a high-missingness disclosure "
            "section but excluded from primary analysis (HighMissingnessColumn); "
            "(3) <20% missing → listwise deletion applied, rows dropped count disclosed "
            "in MissingnessDisclosure. "
            "No mean imputation or multiple imputation is performed. "
            "All thresholds and actions are disclosed in the report."
        ),
        "alternatives": [
            "Multiple imputation by chained equations (MICE) — rejected: introduces "
            "model assumptions that are difficult to validate without domain knowledge; "
            "results are harder to reproduce without full imputation model specification.",
            "Mean/median imputation — rejected: distorts variance estimates and "
            "correlation structure, violating ICH E9 §4.2 guidance on avoiding "
            "bias-inducing imputation.",
            "Complete-case analysis only (no disclosure) — rejected: non-disclosure "
            "of missingness violates CONSORT 2010 §12c.",
        ],
        "consequences": (
            "Analyses on high-missingness columns are not performed, which may reduce "
            "the breadth of findings. All exclusions are fully disclosed so reviewers "
            "can assess the impact."
        ),
        "references": [
            "ICH E9 (R1) §3.3 — Estimands and missing data",
            "ICH E9 §4.2 — Missing values and outliers",
            "CONSORT 2010 §12c — Methods for handling missing data",
        ],
    },
    {
        "id": "ADL-004",
        "title": "Clinical Search Term (CST) Translation",
        "status": "Accepted",
        "context": (
            "Statistical findings expressed in technical notation (p-values, "
            "correlation coefficients, test statistics) cannot be used directly "
            "as PubMed or Cochrane search queries. A translation step is required "
            "to convert findings into clinically meaningful search terms."
        ),
        "decision": (
            "Implement translate_findings_to_cst() which calls the LLM once per "
            "finding with a prompt that (a) injects the study_context, and (b) "
            "explicitly forbids raw statistical values (r=, p=, p<, chi-square, "
            "t-test, F-statistic, z-score, Pearson) in the output. "
            "On LLM failure the translation is marked translation_failed=True and "
            "the finding is skipped in literature validation. "
            "temperature=0 is used for determinism."
        ),
        "alternatives": [
            "Direct use of finding text as search query (rejected: PubMed does not "
            "index statistical notation; retrieval recall would be near zero).",
            "Rule-based extraction of clinical terms (rejected: insufficient for "
            "heterogeneous finding formats produced by the statistical agent).",
        ],
        "consequences": (
            "One LLM call per finding; cost scales linearly with finding count. "
            "Translation failures degrade literature coverage but do not halt the "
            "pipeline."
        ),
        "references": [
            "OCEBM 2011 — Levels of evidence (search strategy for systematic reviews)",
            "CONSORT 2010 §5 — Interventions (clinical terminology standards)",
        ],
    },
    {
        "id": "ADL-005",
        "title": "Evidence Strength Hierarchy",
        "status": "Accepted",
        "context": (
            "Retrieved literature articles vary widely in methodological quality. "
            "Presenting all articles as equally authoritative would mislead clinicians "
            "about the strength of evidence supporting or contradicting a finding."
        ),
        "decision": (
            "Implement score_evidence_strength() using a composite score (0–100) "
            "with three components: recency (year_score 0–40, linear from 1990 to "
            "current year), study type (study_type_score 0–40: meta-analysis=40, "
            "RCT=35, cohort=25, case study=10, unknown=0), and sample size "
            "(sample_size_score 0–20: ≥1000=20, ≥100=15, ≥10=8, else=2). "
            "Labels: ≥70 Strong, ≥45 Moderate, ≥20 Weak, <20 Insufficient. "
            "Plain-language format: '{Label} — based on a {year} {study_type} of "
            "{n} patients'."
        ),
        "alternatives": [
            "GRADE framework (rejected: requires manual domain-expert assessment "
            "of risk of bias, inconsistency, indirectness, and imprecision — not "
            "automatable from abstract metadata alone).",
            "Binary high/low quality split (rejected: too coarse for clinical "
            "decision support).",
        ],
        "consequences": (
            "Scores are heuristic approximations based on metadata available in "
            "article abstracts. They should be treated as indicative, not definitive. "
            "The plain-language format makes the basis of the score transparent."
        ),
        "references": [
            "OCEBM 2011 — Levels of evidence hierarchy",
            "ICH E9 §1.2 — Confirmatory and exploratory trials",
        ],
    },
    {
        "id": "ADL-006",
        "title": "Synthesis Quality Self-Scoring Threshold",
        "status": "Accepted",
        "context": (
            "The synthesis quality critic (ADL-001) requires a threshold below which "
            "a re-run is triggered. The threshold must be configurable to accommodate "
            "different deployment contexts (e.g., exploratory vs. regulatory-grade "
            "analyses)."
        ),
        "decision": (
            "Read the threshold from the SYNTHESIS_QUALITY_THRESHOLD environment "
            "variable at runtime (default 0.7). The value is validated to be in "
            "[0.0, 1.0]. The threshold and the actual score are both recorded in "
            "SynthesisQualityScore and surfaced in the report."
        ),
        "alternatives": [
            "Hardcoded threshold (rejected: inflexible across deployment contexts).",
            "User-supplied threshold per request (rejected: adds API surface area "
            "and risk of misconfiguration by non-technical users).",
        ],
        "consequences": (
            "Operators can tune the quality gate without code changes. "
            "Setting the threshold to 0.0 effectively disables re-runs."
        ),
        "references": [
            "ICH E9 §5.5 — Pre-specification of analysis parameters",
        ],
    },
    {
        "id": "ADL-007",
        "title": "BM25 Retrieval for Internal Evidence Verification",
        "status": "Accepted",
        "context": (
            "After the reasoning loop produces claims, those claims must be verified "
            "against the statistical evidence actually computed during the run — not "
            "against the LLM's parametric knowledge. A lightweight retrieval mechanism "
            "is needed that operates entirely on in-memory data."
        ),
        "decision": (
            "Build a runtime BM25 index over the concatenated statistical outputs "
            "(tool call results, static report, agent outputs) at the start of the "
            "reasoning phase. Each claim is scored against this index; claims with "
            "no supporting evidence chunk are flagged as 'unsupported'. "
            "Results are stored in ReasoningState.internal_verification."
        ),
        "alternatives": [
            "Dense vector retrieval (rejected: requires embedding model inference "
            "at runtime, adding latency and a GPU/API dependency).",
            "Exact string matching (rejected: too brittle for paraphrased claims).",
            "No internal verification (rejected: allows hallucinated claims to "
            "pass through to the report).",
        ],
        "consequences": (
            "BM25 is a sparse retrieval method; it may miss semantically equivalent "
            "evidence expressed in different vocabulary. The internal verification "
            "result is advisory — flagged claims are surfaced to the user but not "
            "automatically removed."
        ),
        "references": [
            "ICH E9 §5.5 — Integrity of data and computer software validity",
            "CONSORT 2010 §15 — Outcomes and estimation (evidence traceability)",
        ],
    },
    {
        "id": "ADL-008",
        "title": "Literature Validator Multi-Source Strategy",
        "status": "Accepted",
        "context": (
            "No single literature database provides complete coverage of clinical "
            "evidence. PubMed covers biomedical literature broadly; Cochrane CDSR "
            "covers systematic reviews and meta-analyses; ClinicalTrials.gov covers "
            "registered trials including unpublished results."
        ),
        "decision": (
            "LiteratureValidatorPipeline queries all four sources (PubMed, Semantic "
            "Scholar, Cochrane, ClinicalTrials.gov) for each Clinical Search Term. "
            "Per-source rate limiting is enforced (PubMed 0.35s, Cochrane 1.0s, "
            "ClinicalTrials 0.5s). Exponential backoff (1s/2s/4s, max 3 retries) "
            "is applied on transient failures. After three failures a source is "
            "marked unavailable for the remainder of the run and the event is logged "
            "to the ADL. A session cache keyed on (source, query_string) prevents "
            "duplicate HTTP calls."
        ),
        "alternatives": [
            "PubMed only (rejected: misses Cochrane meta-analyses which are the "
            "highest OCEBM evidence level).",
            "Parallel requests to all sources simultaneously (rejected: violates "
            "rate limits and risks IP bans from public APIs).",
        ],
        "consequences": (
            "Sequential querying increases latency proportionally to the number of "
            "CSTs. The cache mitigates this for repeated queries within a session. "
            "Source unavailability is disclosed in the reproducibility log."
        ),
        "references": [
            "OCEBM 2011 — Systematic reviews as highest evidence level",
            "ICH E9 §1.2 — Background and purpose of the guideline",
            "CONSORT 2010 §4 — Trial design",
        ],
    },
]


def build_adl() -> str:
    """Return the Architecture Decision Log as a Markdown string."""
    lines: list[str] = [
        "# Architecture Decision Log — Q-Trial Clinical Data Analyst Agent",
        "",
        "> This document records the key design decisions made during the "
        "development of the Q-Trial autonomous clinical data analyst pipeline. "
        "Each entry follows the ADR (Architecture Decision Record) format and "
        "cites the clinical methodology standard that motivated the decision.",
        "",
        "---",
        "",
    ]

    for entry in _ADL_ENTRIES:
        lines += [
            f"## {entry['id']}: {entry['title']}",
            "",
            f"**Status:** {entry['status']}",
            "",
            "### Context",
            "",
            entry["context"],
            "",
            "### Decision",
            "",
            entry["decision"],
            "",
            "### Alternatives Considered",
            "",
        ]
        for alt in entry["alternatives"]:
            lines.append(f"- {alt}")
        lines += [
            "",
            "### Consequences",
            "",
            entry["consequences"],
            "",
            "### Clinical Methodology References",
            "",
        ]
        for ref in entry["references"]:
            lines.append(f"- {ref}")
        lines += ["", "---", ""]

    return "\n".join(lines)
