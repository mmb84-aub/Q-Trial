import time
from types import SimpleNamespace

from qtrial_backend.agentic.cst_translator import translate_findings_to_cst


class _FakeClient:
    def __init__(self, text: str, capture: dict[str, object] | None = None) -> None:
        self._text = text
        self._capture = capture if capture is not None else {}

    def generate(self, req):  # noqa: ANN001
        self._capture["user_prompt"] = req.user_prompt
        return SimpleNamespace(text=self._text)


class _SlowClient:
    def generate(self, req):  # noqa: ANN001
        time.sleep(0.05)
        return SimpleNamespace(text='{"term":"late term"}')


def test_empty_finding_list_returns_safely_without_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.cst_translator.get_client",
        lambda provider: (_ for _ in ()).throw(AssertionError("client should not be created")),
    )

    assert translate_findings_to_cst([], "Heart failure study.") == []


def test_all_filtered_findings_return_safely_without_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.cst_translator.get_client",
        lambda provider: (_ for _ in ()).throw(AssertionError("client should not be created")),
    )

    csts = translate_findings_to_cst(
        [
            "Deaths occurred much earlier (median 44.5 days vs.",
            "`time`: χ²=38.4916, p=0.0000",
            {"finding_text_plain": "1.18 mg/dL (p<0.001, adjusted p=0.00014)"},
        ],
        "Heart failure study.",
    )

    assert csts == []


def test_valid_clinical_finding_still_translates(monkeypatch) -> None:
    capture: dict[str, object] = {}
    monkeypatch.setattr(
        "qtrial_backend.agentic.cst_translator.get_client",
        lambda provider: _FakeClient('{"term":"serum creatinine mortality heart failure"}', capture),
    )

    csts = translate_findings_to_cst(
        [
            {
                "finding_text_plain": "Higher serum creatinine was associated with higher mortality risk.",
                "finding_category": "analytical",
                "claim_type": "association_claim",
                "variable": "serum_creatinine",
                "endpoint": "mortality",
                "significant": True,
            }
        ],
        "Heart failure study.",
    )

    assert len(csts) == 1
    assert csts[0].term == "serum creatinine mortality heart failure"
    assert "Higher serum creatinine" in str(capture["user_prompt"])


def test_malformed_and_raw_artifact_findings_do_not_freeze_cst(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.cst_translator.get_client",
        lambda provider: (_ for _ in ()).throw(AssertionError("client should not be created")),
    )

    csts = translate_findings_to_cst(
        [
            "Compared with survivors,",
            "platelets: chi-square=14.2, p=0.002",
        ],
        "Heart failure study.",
    )

    assert csts == []


def test_cst_translation_timeout_returns_failed_neutral_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.cst_translator.get_client",
        lambda provider: _SlowClient(),
    )

    csts = translate_findings_to_cst(
        ["Age was significantly associated with mortality."],
        "Heart failure study.",
        max_translation_seconds=0.01,
    )

    assert len(csts) == 1
    assert csts[0].translation_failed is True
    assert csts[0].term == ""
    assert "timed out" in (csts[0].failure_note or "")
