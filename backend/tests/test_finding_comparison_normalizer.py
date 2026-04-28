from types import SimpleNamespace

from qtrial_backend.agentic.finding_comparison_normalizer import normalize_comparison_claims


class _FakeClient:
    def __init__(self, text: str, capture: dict[str, object]) -> None:
        self._text = text
        self._capture = capture

    def generate(self, req):
        self._capture["system_prompt"] = req.system_prompt
        self._capture["user_prompt"] = req.user_prompt
        self._capture["payload"] = req.payload
        return SimpleNamespace(text=self._text)


def test_comparison_normalizer_receives_structured_findings_only(monkeypatch) -> None:
    capture: dict[str, object] = {}
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_comparison_normalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Older age was associated with higher mortality risk."}]}',
            capture,
        ),
    )
    findings = normalize_comparison_claims(
        [
            {
                "finding_id": "age",
                "finding_text_raw": "age; adjusted p=0.0001",
                "claim_type": "association_claim",
                "variable": "age",
                "endpoint": "mortality",
                "significant_after_correction": True,
                "adjusted_p_value": 0.0001,
                "direction": "positive",
                "analysis_type": "association",
            }
        ],
        "gemini",
    )

    assert findings[0]["comparison_claim_text"] == "Older age was associated with higher mortality risk."
    user_prompt = str(capture["user_prompt"])
    assert '"variable": "age"' in user_prompt
    assert '"endpoint": "mortality"' in user_prompt
    assert '"analysis_type": "association"' in user_prompt
    assert "dataset rows" not in user_prompt


def test_age_finding_can_use_older_age_style_claim(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_comparison_normalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Older age was associated with higher mortality risk."}]}',
            {},
        ),
    )
    findings = normalize_comparison_claims(
        [
            {
                "finding_id": "age",
                "claim_type": "association_claim",
                "variable": "age",
                "endpoint": "mortality",
                "significant_after_correction": True,
                "direction": "positive",
                "adjusted_p_value": 0.0001,
            }
        ],
        "gemini",
    )
    assert findings[0]["comparison_claim_text"] == "Older age was associated with higher mortality risk."


def test_directional_fallback_is_used_when_model_response_is_too_neutral(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_comparison_normalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Age was associated with mortality."}]}',
            {},
        ),
    )
    findings = normalize_comparison_claims(
        [
            {
                "finding_id": "age",
                "claim_type": "association_claim",
                "variable": "age",
                "endpoint": "mortality",
                "significant_after_correction": True,
                "direction": "positive",
                "adjusted_p_value": 0.0001,
            }
        ],
        "gemini",
    )
    assert findings[0]["comparison_claim_text"] == "Older age was associated with higher mortality risk."


def test_ejection_fraction_finding_can_use_lower_phrase(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_comparison_normalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"ejection_fraction","sentence":"Lower ejection fraction was associated with higher mortality risk."}]}',
            {},
        ),
    )
    findings = normalize_comparison_claims(
        [
            {
                "finding_id": "ejection_fraction",
                "claim_type": "association_claim",
                "variable": "ejection_fraction",
                "endpoint": "mortality",
                "significant_after_correction": True,
                "direction": "negative",
                "adjusted_p_value": 0.0001,
            }
        ],
        "gemini",
    )
    assert findings[0]["comparison_claim_text"] == "Lower ejection fraction was associated with higher mortality risk."


def test_non_significant_finding_gets_neutral_non_significant_claim(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_comparison_normalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"smoking","sentence":"Smoking did not show a statistically significant association with mortality."}]}',
            {},
        ),
    )
    findings = normalize_comparison_claims(
        [
            {
                "finding_id": "smoking",
                "claim_type": "association_claim",
                "variable": "smoking",
                "endpoint": "mortality",
                "significant_after_correction": False,
                "direction": "unknown",
                "adjusted_p_value": 0.41,
            }
        ],
        "gemini",
    )
    assert findings[0]["comparison_claim_text"] == "Smoking did not show a statistically significant association with mortality."


def test_unknown_direction_stays_neutral(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_comparison_normalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Higher age was associated with increased mortality risk."}]}',
            {},
        ),
    )
    findings = normalize_comparison_claims(
        [
            {
                "finding_id": "age",
                "claim_type": "association_claim",
                "variable": "age",
                "endpoint": "mortality",
                "significant_after_correction": True,
                "direction": "unknown",
                "adjusted_p_value": 0.0001,
            }
        ],
        "gemini",
    )
    assert findings[0]["comparison_claim_text"] == "Age was significantly associated with mortality."


def test_only_association_claim_gets_comparison_claim_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_comparison_normalizer.get_client",
        lambda provider: _FakeClient('{"findings":[]}', {}),
    )
    findings = normalize_comparison_claims(
        [
            {
                "finding_id": "age",
                "claim_type": "association_claim",
                "variable": "age",
                "endpoint": "mortality",
                "significant_after_correction": True,
            },
            {
                "finding_id": "event_rate",
                "claim_type": "descriptive_claim",
                "finding_text_raw": "event rate 32.1%",
            },
        ],
        "gemini",
    )
    assert findings[0]["comparison_claim_text"] is not None
    assert findings[1]["comparison_claim_text"] is None


def test_malformed_or_overclaiming_output_falls_back_safely(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.finding_comparison_normalizer.get_client",
        lambda provider: _FakeClient(
            '{"findings":[{"finding_id":"age","sentence":"Age causes mortality. Clinicians should act on this."}]}',
            {},
        ),
    )
    findings = normalize_comparison_claims(
        [
            {
                "finding_id": "age",
                "claim_type": "association_claim",
                "variable": "age",
                "endpoint": "mortality",
                "significant_after_correction": True,
                "direction": "unknown",
            }
        ],
        "gemini",
    )
    assert findings[0]["comparison_claim_text"] == "Age was significantly associated with mortality."
