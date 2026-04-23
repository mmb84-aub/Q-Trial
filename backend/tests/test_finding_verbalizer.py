from types import SimpleNamespace

from qtrial_backend.agentic.finding_verbalizer import verbalize_statistical_findings


class _FakeClient:
    def __init__(self, text: str, capture: dict[str, object]) -> None:
        self._text = text
        self._capture = capture

    def generate(self, req):
        self._capture["system_prompt"] = req.system_prompt
        self._capture["user_prompt"] = req.user_prompt
        self._capture["payload"] = req.payload
        return SimpleNamespace(text=self._text)


def test_verbalizer_receives_structured_findings_not_dataset(monkeypatch) -> None:
    capture: dict[str, object] = {}
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_verbalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Age was significantly associated with the outcome."}]}',
            capture,
        ),
    )

    findings = [
        {
            "finding_id": "age",
            "finding_text_raw": "age; raw p=0.0001; adjusted p=0.0001",
            "variable": "age",
            "endpoint": None,
            "significant": True,
            "adjusted_p_value": 0.0001,
            "direction": "unknown",
            "analysis_type": "association",
        }
    ]
    verbalized = verbalize_statistical_findings(findings, "gemini")

    assert verbalized[0]["finding_text_plain"] == "Age was significantly associated with the outcome."
    user_prompt = str(capture["user_prompt"])
    assert '"variable": "age"' in user_prompt
    assert '"analysis_type": "association"' in user_prompt
    assert "a,b" not in user_prompt
    assert "rows" not in user_prompt


def test_verbalizer_supports_significant_conservative_sentence(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_verbalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Age was significantly associated with the outcome."}]}',
            {},
        ),
    )
    findings = verbalize_statistical_findings(
        [
            {
                "finding_id": "age",
                "finding_text_raw": "age; adjusted p=0.0001",
                "variable": "age",
                "endpoint": None,
                "significant": True,
                "adjusted_p_value": 0.0001,
                "direction": "unknown",
                "analysis_type": "association",
            }
        ],
        "gemini",
    )
    assert findings[0]["finding_text_plain"] == "Age was significantly associated with the outcome."


def test_verbalizer_supports_non_significant_conservative_sentence(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_verbalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"smoking","sentence":"Smoking did not show a statistically significant association with mortality."}]}',
            {},
        ),
    )
    findings = verbalize_statistical_findings(
        [
            {
                "finding_id": "smoking",
                "finding_text_raw": "smoking; adjusted p=0.41",
                "variable": "smoking",
                "endpoint": "mortality",
                "significant": False,
                "adjusted_p_value": 0.41,
                "direction": "unknown",
                "analysis_type": "association",
            }
        ],
        "gemini",
    )
    assert findings[0]["finding_text_plain"] == "Smoking did not show a statistically significant association with mortality."


def test_verbalizer_keeps_unknown_direction_neutral(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_verbalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"serum_creatinine","sentence":"Serum creatinine was significantly associated with mortality."}]}',
            {},
        ),
    )
    findings = verbalize_statistical_findings(
        [
            {
                "finding_id": "serum_creatinine",
                "finding_text_raw": "serum_creatinine; adjusted p=0.002",
                "variable": "serum_creatinine",
                "endpoint": "mortality",
                "significant": True,
                "adjusted_p_value": 0.002,
                "direction": "unknown",
                "analysis_type": "association",
            }
        ],
        "gemini",
    )
    assert "Higher" not in findings[0]["finding_text_plain"]
    assert "Lower" not in findings[0]["finding_text_plain"]


def test_verbalizer_allows_directional_wording_when_direction_is_known(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_verbalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"serum_creatinine","sentence":"Higher serum creatinine was associated with increased mortality risk."}]}',
            {},
        ),
    )
    findings = verbalize_statistical_findings(
        [
            {
                "finding_id": "serum_creatinine",
                "finding_text_raw": "serum_creatinine; adjusted p=0.002; OR=1.8",
                "variable": "serum_creatinine",
                "endpoint": "mortality",
                "significant": True,
                "adjusted_p_value": 0.002,
                "direction": "positive",
                "analysis_type": "association",
                "odds_ratio": 1.8,
            }
        ],
        "gemini",
    )
    assert findings[0]["finding_text_plain"] == "Higher serum creatinine was associated with increased mortality risk."


def test_verbalizer_rejects_extra_commentary_and_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_verbalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Age was significantly associated with the outcome. This should be monitored closely."}]}',
            {},
        ),
    )
    findings = verbalize_statistical_findings(
        [
            {
                "finding_id": "age",
                "finding_text_raw": "age; adjusted p=0.0001",
                "variable": "age",
                "endpoint": None,
                "significant": True,
                "adjusted_p_value": 0.0001,
                "direction": "unknown",
                "analysis_type": "association",
            }
        ],
        "gemini",
    )
    assert findings[0]["finding_text_plain"] == "Age was significantly associated with the outcome."


def test_verbalizer_rejects_directional_language_when_direction_unknown(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_verbalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Higher age was associated with increased mortality risk."}]}',
            {},
        ),
    )
    findings = verbalize_statistical_findings(
        [
            {
                "finding_id": "age",
                "finding_text_raw": "age; adjusted p=0.0001",
                "variable": "age",
                "endpoint": None,
                "significant": True,
                "adjusted_p_value": 0.0001,
                "direction": "unknown",
                "analysis_type": "association",
            }
        ],
        "gemini",
    )
    assert findings[0]["finding_text_plain"] == "Age was significantly associated with the outcome."


def test_verbalizer_handles_batching_and_id_mapping(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_verbalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Age was significantly associated with the outcome."},{"finding_id":"smoking","sentence":"Smoking did not show a statistically significant association with mortality."}]}',
            {},
        ),
    )
    findings = verbalize_statistical_findings(
        [
            {
                "finding_id": "age",
                "finding_text_raw": "age; adjusted p=0.0001",
                "variable": "age",
                "endpoint": None,
                "significant": True,
                "adjusted_p_value": 0.0001,
                "direction": "unknown",
                "analysis_type": "association",
            },
            {
                "finding_id": "smoking",
                "finding_text_raw": "smoking; adjusted p=0.41",
                "variable": "smoking",
                "endpoint": "mortality",
                "significant": False,
                "adjusted_p_value": 0.41,
                "direction": "unknown",
                "analysis_type": "association",
            },
        ],
        "gemini",
    )
    assert findings[0]["finding_text_plain"] == "Age was significantly associated with the outcome."
    assert findings[1]["finding_text_plain"] == "Smoking did not show a statistically significant association with mortality."


def test_verbalizer_preserves_raw_text_and_handles_malformed_output() -> None:
    from types import SimpleNamespace

    class _BrokenClient:
        def generate(self, req):
            return SimpleNamespace(text="not json")

    import qtrial_backend.agentic.finding_verbalizer as fv
    original = fv.get_client
    fv.get_client = lambda provider: _BrokenClient()
    try:
        findings = verbalize_statistical_findings(
            [
                {
                    "finding_id": "age",
                    "finding_text_raw": "age; adjusted p=0.0001",
                    "variable": "age",
                    "endpoint": None,
                    "significant": True,
                    "adjusted_p_value": 0.0001,
                    "direction": "unknown",
                    "analysis_type": "association",
                }
            ],
            "gemini",
        )
    finally:
        fv.get_client = original
    assert findings[0]["finding_text_raw"] == "age; adjusted p=0.0001"
    assert findings[0]["finding_text_plain"] == "Age was significantly associated with the outcome."
