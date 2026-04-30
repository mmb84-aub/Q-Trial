import pandas as pd

from qtrial_backend.api import _compute_protected_columns, _ensure_endpoint_selected


def test_protected_columns_include_endpoint_and_clinical_risk_factors() -> None:
    df = pd.DataFrame(
        {
            "age": [50, 60, 70],
            "serum_sodium": [135, 128, 140],
            "anaemia": [0, 1, 0],
            "DEATH_EVENT": [0, 1, 0],
        }
    )
    protected = _compute_protected_columns(
        df,
        endpoint_column="DEATH_EVENT",
        analyst_report_text="Age and hyponatremia (serum_sodium) were associated with mortality.",
        column_dict=None,
        meta=None,
    )
    assert "DEATH_EVENT" in protected
    assert "age" in protected
    assert "serum_sodium" in protected


def test_qubo_selected_df_keeps_protected_columns_even_if_excluded() -> None:
    df = pd.DataFrame({"age": [1, 2], "serum_sodium": [130, 140], "DEATH_EVENT": [0, 1], "x": [0.1, 0.2]})
    quantum = {"selected_columns": ["x"], "excluded_columns": ["age", "serum_sodium"], "n_selected": 1}
    selected = _ensure_endpoint_selected(df, quantum, "DEATH_EVENT", ["age", "serum_sodium"])
    assert set(selected.columns) >= {"x", "DEATH_EVENT", "age", "serum_sodium"}
    assert "protected_added_columns" in quantum
    assert set(quantum["protected_added_columns"]) == {"age", "serum_sodium"}

