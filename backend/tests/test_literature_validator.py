from qtrial_backend.agentic.literature_validator import LiteratureValidatorPipeline
from qtrial_backend.tools.literature.rag import LiteratureArticle


class _StubResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubClient:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, req):  # noqa: ANN001
        return _StubResponse(self._text)


def test_topically_supported_finding_is_not_labeled_novel(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.literature_validator.get_client",
        lambda provider: _StubClient('{"status":"Novel","rationale":"No direct precedent found."}'),
    )

    pipeline = LiteratureValidatorPipeline(provider="gemini")
    articles = [
        LiteratureArticle(
            source="pubmed",
            paper_id="1",
            title="Age and mortality risk in chronic heart failure",
            authors=["A"],
            year="2024",
            abstract_snippet=(
                "Older age was associated with higher mortality risk in patients with heart failure."
            ),
            citation_alias="lit[0]",
            search_query="age mortality heart failure",
        )
    ]

    status, rationale = pipeline._assign_grounding_status(
        "Older age was associated with higher mortality risk.",
        articles,
    )

    assert status == "Supported"
    assert "topically aligned" in rationale.lower()
