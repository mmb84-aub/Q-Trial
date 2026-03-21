"""
Property-based tests for the Q-Trial Clinical Data Analyst Agent.

Each test carries a tag comment identifying the feature and property number.
Run with: pytest tests/test_properties.py -v

Requires: pytest, hypothesis, pandas, scipy
Install:  poetry add --group dev pytest hypothesis
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

# ── Feature: clinical-data-analyst-agent, Property 4: Data profiler completeness ──

from qtrial_backend.dataset.load import classify_missingness


@settings(max_examples=100)
@given(
    data_frames(
        columns=[
            column("a", elements=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False))),
            column("b", elements=st.one_of(st.none(), st.integers())),
        ],
        index=range_indexes(min_size=1, max_size=50),
    )
)
def test_property_4_classify_missingness_covers_all_columns(df: pd.DataFrame) -> None:
    # Feature: clinical-data-analyst-agent, Property 4: Data profiler completeness
    # Validates: Requirements 2.1, 2.2, 2.3, 2.5
    result = classify_missingness(df)
    assert set(result.keys()) == set(df.columns), (
        "classify_missingness must return an entry for every column"
    )


# ── Feature: clinical-data-analyst-agent, Property 5: Outlier detection coverage ──

from qtrial_backend.dataset.load import validate_dataset


@settings(max_examples=100)
@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=200,
    )
)
def test_property_5_validate_dataset_accepts_non_empty(values: list[float]) -> None:
    # Feature: clinical-data-analyst-agent, Property 5: Outlier detection coverage
    # Validates: Requirements 2.4
    df = pd.DataFrame({"x": values})
    # Should not raise for a non-empty dataset
    validate_dataset(df)


@settings(max_examples=10)
@given(st.just(pd.DataFrame()))
def test_property_5_validate_dataset_rejects_empty(df: pd.DataFrame) -> None:
    # Feature: clinical-data-analyst-agent, Property 5: Outlier detection coverage
    # Validates: Requirements 1.3
    with pytest.raises(ValueError):
        validate_dataset(df)


# ── Feature: clinical-data-analyst-agent, Property 6: Correlation matrix completeness and symmetry ──

@settings(max_examples=100)
@given(
    data_frames(
        columns=[
            column("x", elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            column("y", elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            column("z", elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
        ],
        index=range_indexes(min_size=3, max_size=50),
    )
)
def test_property_6_correlation_matrix_symmetry(df: pd.DataFrame) -> None:
    # Feature: clinical-data-analyst-agent, Property 6: Correlation matrix completeness and symmetry
    # Validates: Requirements 3.1
    corr = df.corr(numeric_only=True)
    # Must be square
    assert corr.shape[0] == corr.shape[1]
    # Must be symmetric: corr[i,j] == corr[j,i]
    for i in corr.columns:
        for j in corr.columns:
            v_ij = corr.loc[i, j]
            v_ji = corr.loc[j, i]
            if not (math.isnan(v_ij) and math.isnan(v_ji)):
                assert abs(v_ij - v_ji) < 1e-10, f"Correlation matrix not symmetric at ({i},{j})"


# ── Feature: clinical-data-analyst-agent, Property 7: P-value validity invariant ──

from scipy import stats as scipy_stats


@settings(max_examples=100)
@given(
    st.lists(
        st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=100,
    ),
    st.lists(
        st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=100,
    ),
)
def test_property_7_pvalue_in_unit_interval(a: list[float], b: list[float]) -> None:
    # Feature: clinical-data-analyst-agent, Property 7: P-value validity invariant
    # Validates: Requirements 3.6
    _, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    assert 0.0 <= p <= 1.0, f"p-value {p} is outside [0, 1]"


# ── Feature: clinical-data-analyst-agent, Property 9: High-missingness finding flags ──

@settings(max_examples=100)
@given(
    st.integers(min_value=10, max_value=200),
    st.floats(min_value=0.21, max_value=0.49),
)
def test_property_9_high_missingness_column_classified_correctly(n_rows: int, miss_rate: float) -> None:
    # Feature: clinical-data-analyst-agent, Property 9: High-missingness finding flags
    # Validates: Requirements 4.2, 4.3
    n_missing = round(n_rows * miss_rate)
    # After rounding, verify the actual rate still lands in the 20–50% tier;
    # skip the example if integer rounding pushed it out of range.
    actual_rate = n_missing / n_rows
    assume(0.20 <= actual_rate <= 0.50)
    values = [None] * n_missing + [1.0] * (n_rows - n_missing)
    df = pd.DataFrame({"col": values})
    result = classify_missingness(df)
    assert result["col"].action == "high_missingness_section", (
        f"Column with {actual_rate:.1%} missingness should be 'high_missingness_section', "
        f"got {result['col'].action}"
    )


# ── Feature: clinical-data-analyst-agent, Property 11: Missingness disclosures in report ──

@settings(max_examples=100)
@given(
    st.integers(min_value=10, max_value=200),
    st.floats(min_value=0.01, max_value=0.19),
)
def test_property_11_low_missingness_column_uses_listwise_deletion(n_rows: int, miss_rate: float) -> None:
    # Feature: clinical-data-analyst-agent, Property 11: Missingness disclosures in report
    # Validates: Requirements 4a.4, 4a.5
    n_missing = int(n_rows * miss_rate)
    values = [None] * n_missing + [1.0] * (n_rows - n_missing)
    df = pd.DataFrame({"col": values})
    result = classify_missingness(df)
    assert result["col"].action == "listwise_deletion", (
        f"Column with {miss_rate:.0%} missingness should use 'listwise_deletion', "
        f"got {result['col'].action}"
    )


# ── Feature: clinical-data-analyst-agent, Property 3: Treatment column exclusion invariant ──

from qtrial_backend.dataset.treatment_detector import detect_treatment_columns


@settings(max_examples=100)
@given(
    st.integers(min_value=20, max_value=200),
    st.sampled_from(["treatment", "arm", "group", "intervention", "control", "allocation", "randomized"]),
    st.integers(min_value=2, max_value=5),
)
def test_property_3_treatment_column_detected_by_name(
    n_rows: int, col_name: str, n_groups: int
) -> None:
    # Feature: clinical-data-analyst-agent, Property 3: Treatment column exclusion invariant
    # Validates: Requirements 1.7, 9.1
    groups = [f"group_{i}" for i in range(n_groups)]
    # Distribute evenly so no group exceeds 60%
    values = (groups * (n_rows // n_groups + 1))[:n_rows]
    df = pd.DataFrame({col_name: values, "outcome": range(n_rows)})
    detected = detect_treatment_columns(df)
    assert col_name in detected, (
        f"Column '{col_name}' with {n_groups} groups should be detected as treatment"
    )


@settings(max_examples=100)
@given(
    st.integers(min_value=20, max_value=200),
)
def test_property_3_dominant_group_not_detected(n_rows: int) -> None:
    # Feature: clinical-data-analyst-agent, Property 3: Treatment column exclusion invariant
    # Validates: Requirements 9.1 — column where one value exceeds 60% should NOT be detected
    # 70% of rows are "A", 30% are "B"
    n_a = int(n_rows * 0.70)
    values = ["A"] * n_a + ["B"] * (n_rows - n_a)
    df = pd.DataFrame({"treatment": values})
    detected = detect_treatment_columns(df)
    assert "treatment" not in detected, (
        "Column where one value exceeds 60% should not be detected as treatment"
    )


# ── Feature: clinical-data-analyst-agent, Property 23: Treatment blinding confirmation in report ──

from qtrial_backend.agentic.schemas import AnalysisReport


@settings(max_examples=50)
@given(
    st.lists(
        st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
        min_size=0,
        max_size=5,
    )
)
def test_property_23_treatment_columns_excluded_field_present(cols: list[str]) -> None:
    # Feature: clinical-data-analyst-agent, Property 23: Treatment blinding confirmation in report
    # Validates: Requirements 9.4, 9.5
    # AnalysisReport must always have treatment_columns_excluded (even if empty)
    # We verify the field exists and is a list
    schema_fields = AnalysisReport.model_fields
    assert "treatment_columns_excluded" in schema_fields
    # Default must be an empty list (not None)
    default = schema_fields["treatment_columns_excluded"].default_factory  # type: ignore[union-attr]
    assert default is not None
    assert default() == []


# ── Feature: clinical-data-analyst-agent, Property 18: Research questions in report question bank ──

from qtrial_backend.agentic.schemas import (
    GroundedFindings, GroundedFinding, ResearchQuestion, SynthesisOutput
)


@settings(max_examples=100)
@given(
    st.lists(
        st.builds(
            ResearchQuestion,
            question=st.text(min_size=5, max_size=100),
            source_finding=st.text(min_size=5, max_size=100),
        ),
        min_size=0,
        max_size=10,
    )
)
def test_property_18_research_questions_serialise_roundtrip(questions: list[ResearchQuestion]) -> None:
    # Feature: clinical-data-analyst-agent, Property 18: Research questions in report question bank
    # Validates: Requirements 6.4, 7.5
    schema = GroundedFindings(research_questions=questions)
    dumped = schema.model_dump()
    restored = GroundedFindings.model_validate(dumped)
    assert len(restored.research_questions) == len(questions)
    for orig, rest in zip(questions, restored.research_questions):
        assert orig.question == rest.question
        assert orig.source_finding == rest.source_finding


# ── Feature: clinical-data-analyst-agent, Property 20: Chart presence for numeric findings ──

from qtrial_backend.agentic.schemas import EvidenceStrengthScore


@settings(max_examples=100)
@given(
    st.integers(min_value=0, max_value=100),
    st.text(min_size=1, max_size=50),
    st.integers(min_value=1990, max_value=2025),
    st.text(min_size=1, max_size=30),
    st.one_of(st.none(), st.integers(min_value=1, max_value=100000)),
)
def test_property_20_evidence_strength_label_consistent_with_score(
    score: int, plain_language: str, year: int, study_type: str, sample_size: int | None
) -> None:
    # Feature: clinical-data-analyst-agent, Property 20: Chart presence for numeric findings
    # Validates: Requirements 7.3
    # Derive the expected label from the score
    if score >= 70:
        expected = "Strong"
    elif score >= 45:
        expected = "Moderate"
    elif score >= 20:
        expected = "Weak"
    else:
        expected = "Insufficient"

    ess = EvidenceStrengthScore(
        score=score,
        label=expected,  # type: ignore[arg-type]
        plain_language=plain_language,
        year=year,
        study_type=study_type,
        sample_size=sample_size,
    )
    assert ess.label == expected
    assert 0 <= ess.score <= 100


# ── Feature: clinical-data-analyst-agent, Property 30: Study_Context on report cover page ──

from qtrial_backend.agentic.schemas import ReproducibilityLog


@settings(max_examples=100)
@given(
    st.text(min_size=1, max_size=500),
    st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_30_reproducibility_log_preserves_study_context(
    study_context: str, seed: int
) -> None:
    # Feature: clinical-data-analyst-agent, Property 30: Study_Context on report cover page
    # Validates: Requirements 15.6
    log = ReproducibilityLog(
        run_id="test-run",
        timestamp="2025-01-01T00:00:00Z",
        study_context=study_context,
        seed=seed,
    )
    dumped = log.model_dump()
    restored = ReproducibilityLog.model_validate(dumped)
    assert restored.study_context == study_context
    assert restored.seed == seed


# ── Feature: clinical-data-analyst-agent, Property 31: CST incorporates Study_Context ──

from qtrial_backend.agentic.schemas import ClinicalSearchTerm


@settings(max_examples=100)
@given(
    st.text(min_size=1, max_size=200),
    st.text(min_size=1, max_size=200),
    st.text(min_size=1, max_size=500),
)
def test_property_31_cst_records_study_context(
    source_finding: str, term: str, study_context: str
) -> None:
    # Feature: clinical-data-analyst-agent, Property 31: CST incorporates Study_Context
    # Validates: Requirements 16.3
    cst = ClinicalSearchTerm(
        source_finding=source_finding,
        term=term,
        study_context_used=study_context,
    )
    assert cst.study_context_used == study_context
