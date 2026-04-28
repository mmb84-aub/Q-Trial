import json

from fastapi.testclient import TestClient

from qtrial_backend import api
from qtrial_backend.agentic.report_comparison import build_comparison_report
from qtrial_backend.agentic.schemas import (
    AgentRunRecord,
    FinalReportSchema,
    InsightSynthesisOutput,
    PlanSchema,
    PlanStep,
    UnknownsOutput,
)


def _make_report() -> FinalReportSchema:
    return FinalReportSchema(
        provider="gemini",
        model="stub-model",
        plan=PlanSchema(
            dataset_summary="Test dataset",
            steps=[
                PlanStep(
                    step_number=1,
                    name="Stub",
                    goal="Stub",
                    inputs_used=["dataframe"],
                    expected_output_keys=["analysis_report"],
                    agent_to_call="DataQualityAgent",
                )
            ],
        ),
        agent_runs=[
            AgentRunRecord(
                step_number=1,
                agent="StubAgent",
                goal="Stub",
                output={"analysis_report": "stub"},
            )
        ],
        unknowns=UnknownsOutput(
            ranked_unknowns=[],
            explicit_assumptions=[],
            required_documents=[],
            summary="",
        ),
        final_insights=InsightSynthesisOutput(
            key_findings=[],
            risks_and_bias_signals=[],
            recommended_next_analyses=[],
            required_metadata_or_questions=[],
        ),
        clinical_analysis={
            "stage_3_corrections": {
                "corrected_findings": [
                    {
                        "finding_id": "mortality",
                        "adjusted_p_value": 0.01,
                        "effect_size": 0.42,
                        "significant_after_correction": True,
                    }
                ]
            }
        },
    )


def _stub_pipeline(captured: dict[str, object]):
    def _run_agentic_insights(*args, **kwargs):
        analyst_report_text = args[-2]
        analyst_report_name = args[-1]
        captured["analyst_report_text"] = analyst_report_text
        captured["analyst_report_name"] = analyst_report_name

        report = _make_report()
        if analyst_report_text and analyst_report_name:
            report = report.model_copy(
                update={
                    "comparison_report": build_comparison_report(
                        final_report=report,
                        analyst_report_name=analyst_report_name,
                        analyst_report_text=analyst_report_text,
                        provider="gemini",
                    )
                }
            )

        emit = args[9] if len(args) > 9 and callable(args[9]) else None
        if emit and analyst_report_text and analyst_report_name:
            emit(
                {
                    "type": "stage_complete",
                    "stage": "comparison",
                    "message": "Automated report comparison complete",
                }
            )
        return report

    return _run_agentic_insights


def test_run_analysis_keeps_analyst_report_optional(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(api, "build_static_report", lambda *args, **kwargs: ("static", None, None))
    monkeypatch.setattr(api, "run_statistical_agent_loop", lambda *args, **kwargs: ("loop", []))
    monkeypatch.setattr(api, "run_agentic_insights", _stub_pipeline(captured))

    client = TestClient(api.app)
    response = client.post(
        "/api/run",
        data={"study_context": "Test study"},
        files={"file": ("dataset.csv", b"a,b\n1,2\n", "text/csv")},
    )

    payload = response.json()
    assert response.status_code == 200
    assert captured["analyst_report_text"] is None
    assert captured["analyst_report_name"] is None
    assert payload["statistical_verification_report"] is None
    assert payload["comparison_report"] is None


def test_run_analysis_rejects_unsupported_analyst_report_format() -> None:
    client = TestClient(api.app)
    response = client.post(
        "/api/run",
        data={"study_context": "Test study"},
        files={
            "file": ("dataset.csv", b"a,b\n1,2\n", "text/csv"),
            "analyst_report_file": ("report.pdf", b"%PDF-1.4", "application/pdf"),
        },
    )

    assert response.status_code == 422
    assert "Unsupported analyst report format" in response.json()["detail"]


def test_run_analysis_accepts_utf8_plain_text_even_with_unlisted_extension(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(api, "build_static_report", lambda *args, **kwargs: ("static", None, None))
    monkeypatch.setattr(api, "run_statistical_agent_loop", lambda *args, **kwargs: ("loop", []))
    monkeypatch.setattr(api, "run_agentic_insights", _stub_pipeline(captured))

    client = TestClient(api.app)
    response = client.post(
        "/api/run",
        data={"study_context": "Test study"},
        files={
            "file": ("dataset.csv", b"a,b\n1,2\n", "text/csv"),
            "analyst_report_file": (
                "report.log",
                b"Results:\n- Mortality was statistically significant with p=0.01 [1].\n",
                "text/plain",
            ),
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert captured["analyst_report_name"] == "report.log"
    assert payload["comparison_report"]["analyst_report_name"] == "report.log"


def test_run_analysis_preserves_statistical_metadata_json(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _run_agentic_insights(*args, **kwargs):
        captured["metadata"] = args[5]
        return _make_report()

    monkeypatch.setattr(api, "build_static_report", lambda *args, **kwargs: ("static", None, None))
    monkeypatch.setattr(api, "run_statistical_agent_loop", lambda *args, **kwargs: ("loop", []))
    monkeypatch.setattr(api, "run_agentic_insights", _run_agentic_insights)

    client = TestClient(api.app)
    response = client.post(
        "/api/run",
        data={
            "study_context": "Test study",
            "metadata_json": json.dumps(
                {
                    "primary_endpoint": "DEATH_EVENT",
                    "time_column": "followup_time",
                    "event_column": "status",
                    "event_codes": [2],
                    "group_column": "allocation",
                    "status_mapping": {"0": "censored", "2": "death"},
                }
            ),
        },
        files={
            "file": (
                "dataset.csv",
                b"allocation,followup_time,status,DEATH_EVENT\n0,1,0,0\n1,2,2,1\n",
                "text/csv",
            )
        },
    )

    metadata = captured["metadata"]
    assert response.status_code == 200
    assert metadata.primary_endpoint == "DEATH_EVENT"
    assert metadata.time_column == "followup_time"
    assert metadata.event_column == "status"
    assert metadata.event_codes == [2]
    assert metadata.group_column == "allocation"
    assert metadata.status_mapping == {"0": "censored", "2": "death"}


def test_run_analysis_stream_passes_supported_analyst_report_through(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(api, "build_static_report", lambda *args, **kwargs: ("static", None, None))
    monkeypatch.setattr(api, "run_statistical_agent_loop", lambda *args, **kwargs: ("loop", []))
    monkeypatch.setattr(api, "run_agentic_insights", _stub_pipeline(captured))

    client = TestClient(api.app)
    response = client.post(
        "/api/run/stream",
        data={"study_context": "Test study"},
        files={
            "file": ("dataset.csv", b"a,b\n1,2\n", "text/csv"),
            "analyst_report_file": (
                "report.txt",
                b"Results:\n- Mortality was statistically significant with p=0.01 [1].\n",
                "text/plain",
            ),
        },
    )

    body = response.text
    assert response.status_code == 200
    assert captured["analyst_report_name"] == "report.txt"
    assert "Mortality was statistically significant" in str(captured["analyst_report_text"])
    assert '"stage": "comparison"' in body
    assert '"comparison_report"' in body
    assert '"analyst_report_name": "report.txt"' in body
