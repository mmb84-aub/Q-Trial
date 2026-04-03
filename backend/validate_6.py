"""
Task 6 validation: test literature RAG directly with synthetic hypotheses.
No LLM API needed — only PubMed HTTP calls.
"""
from qtrial_backend.tools.literature.rag import (
    run_literature_rag,
    format_literature_for_agents,
    _queries_from_hypotheses,
)

# Simulate hypothesis dicts as produced by reasoning_state.hypotheses
hypotheses = [
    {"statement": "There is a significant difference in survival outcomes between treatment groups in primary biliary cholangitis", "hypothesis_id": "h1"},
    {"statement": "Serum albumin and bilirubin levels are prognostic markers for disease stage", "hypothesis_id": "h2"},
    {"statement": "Baseline clinical characteristics are imbalanced between treatment arms", "hypothesis_id": "h3"},
]

# Simulate a minimal preview
preview = {
    "schema": {
        "bili": "float64", "albumin": "float64", "trt": "int64",
        "status": "int64", "time": "float64", "stage": "int64",
    },
    "dataset_summary": "Primary biliary cholangitis clinical trial",
}
evidence = {}

print("=== QUERY GENERATION ===")
queries = _queries_from_hypotheses(hypotheses, preview, evidence)
for i, q in enumerate(queries):
    print(f"  [{i}] {q}")

print("\n=== PUBMED RETRIEVAL ===")
report = run_literature_rag(hypotheses, preview, evidence)
print(f"Summary: {report.summary}")
print(f"Articles: {report.total_retrieved}")
print(f"Queries used: {report.queries_used}")
print()

for art in report.articles:
    print(f"[{art.citation_alias}] {art.source.upper()} {art.paper_id}")
    print(f"  Title  : {art.title}")
    print(f"  Authors: {', '.join(art.authors[:3])}")
    print(f"  Year   : {art.year}")
    if art.tldr:
        print(f"  TLDR   : {art.tldr}")
    else:
        snippet = art.abstract_snippet[:120]
        print(f"  Snippet: {snippet}{'...' if len(art.abstract_snippet) > 120 else ''}")
    print()

print("=== FORMATTED AGENT BLOCK ===")
block = format_literature_for_agents(report)
print(block[:1500])
