import pandas as pd

from qtrial_backend.report.static import build_static_report


def test_static_report_never_calls_outcome_treatment() -> None:
    df = pd.DataFrame(
        {
            "age": [50, 60, 70, 80],
            "DEATH_EVENT": [0, 1, 0, 1],
        }
    )
    report, _, _ = build_static_report(df, "toy")
    lowered = report.lower()
    assert "treatment = `death_event`" not in lowered
    assert "treatment = death_event" not in lowered

