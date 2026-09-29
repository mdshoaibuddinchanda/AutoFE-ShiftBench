from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from src.mechanism_audit import (
    ArithmeticCandidate,
    CandidateHistoryRecord,
    CandidateHistoryWriter,
    analytic_jacobian,
    check_derivative,
    estimate_candidate_history_storage,
    evaluate_arithmetic,
    finite_difference_jacobian,
    recommended_operator_isolation_configs,
    scaled_jacobian_norm,
    summarize_jacobian_norm,
    training_input_scale,
    validate_operator_isolation_budget,
)


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("add_numeric", np.array([[1.0, 1.0], [1.0, 1.0]])),
        ("subtract_numeric", np.array([[1.0, -1.0], [1.0, -1.0]])),
        ("multiply_numeric", np.array([[3.0, 2.0], [2.0, 5.0]])),
        ("divide_numeric", np.array([[1.0 / 3.0, -2.0 / 9.0], [0.5, -1.25]])),
    ],
)
def test_analytic_arithmetic_jacobians(operator, expected):
    values = np.array([[2.0, 3.0], [5.0, 2.0]])
    result = analytic_jacobian(ArithmeticCandidate("c", operator, ("a", "b")), values)
    np.testing.assert_allclose(result.values, expected)
    assert result.valid_rows.tolist() == [True, True]
    assert result.undefined_reasons == (None, None)


@pytest.mark.parametrize("operator", ["add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric"])
def test_checked_finite_differences_match_arithmetic_derivatives(operator):
    candidate = ArithmeticCandidate("c", operator, ("a", "b"))
    values = np.array([[0.7, 1.3], [2.0, -0.8], [-3.0, 4.2]])
    check = check_derivative(candidate, values, relative_step=1e-6)
    assert check.status == "pass"
    assert check.valid_rows == len(values)
    assert check.valid_entries == len(values) * 2
    assert check.max_absolute_error is not None and check.max_absolute_error < 1e-5


def test_division_zero_is_undefined_without_imputation():
    candidate = ArithmeticCandidate("div", "divide_numeric", ("a", "b"))
    values = np.array([[2.0, 0.0], [2.0, 2.0]])
    analytic = analytic_jacobian(candidate, values)
    numeric = finite_difference_jacobian(candidate, values)
    assert analytic.valid_rows.tolist() == [False, True]
    assert analytic.undefined_reasons[0] == "zero_denominator"
    assert numeric.valid_rows.tolist() == [False, True]
    assert np.isnan(analytic.values[0]).all()
    check = check_derivative(candidate, values)
    assert check.status == "pass"
    assert check.valid_rows == 1
    assert check.undefined_reason_counts["zero_denominator"] == 1


def test_scaled_norm_and_summary_keep_undefined_rows_explicitly():
    candidate = ArithmeticCandidate("mul", "multiply_numeric", ("a", "b"))
    values = np.array([[2.0, 3.0], [np.inf, 3.0]])
    derivative = analytic_jacobian(candidate, values)
    norms = scaled_jacobian_norm(derivative, input_scale=[2.0, 4.0])
    np.testing.assert_allclose(norms[0], np.sqrt((3 * 2) ** 2 + (2 * 4) ** 2))
    assert np.isnan(norms[1])
    summary = summarize_jacobian_norm(norms, undefined_reasons=derivative.undefined_reasons)
    assert summary.n_rows == 2
    assert summary.n_valid == 1
    assert summary.n_undefined == 1
    assert summary.undefined_reason_counts == {"nonfinite_parent": 1}
    with pytest.raises(ValueError, match="length"):
        summarize_jacobian_norm(norms, undefined_reasons=["too short"])


def test_training_scale_marks_constant_parent_undefined():
    scale = training_input_scale(np.array([[1.0, 2.0], [1.0, 4.0]]))
    assert np.isnan(scale[0])
    assert np.isfinite(scale[1])


def _history_record() -> CandidateHistoryRecord:
    return CandidateHistoryRecord(
        dataset="synthetic",
        split_policy="row_level",
        seed=42,
        fold=1,
        condition="clean",
        iteration=0,
        candidate_id="c0",
        parent_features=("a", "b"),
        operator="add_numeric",
        admissible=True,
        rejection_reason=None,
        selection_score=0.7,
        selection_score_status="finite",
        selection_decision="selected",
        candidate_seed=123,
    )


def test_candidate_history_requires_training_scope_and_streams_jsonl(tmp_path):
    record = _history_record()
    path = tmp_path / "history.jsonl.gz"
    with CandidateHistoryWriter(path) as writer:
        writer.append(record)
        assert writer.records_written == 1
        assert writer.uncompressed_bytes_written > 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.loads(stream.readline())
    assert payload["score_scope"] == "train"
    assert payload["candidate_id"] == "c0"
    assert payload["parent_features"] == ["a", "b"]
    with pytest.raises(ValueError, match="score_scope"):
        CandidateHistoryRecord(**{**record.to_dict(), "parent_features": ("a", "b"), "score_scope": "held_out"})


def test_history_storage_estimate_and_operator_budgets():
    estimate = estimate_candidate_history_storage(tasks=10, candidates_per_task=100)
    assert estimate.records == 1000
    assert estimate.uncompressed_bytes == 420_000
    assert estimate.compressed_bytes == 147_000
    specs = recommended_operator_isolation_configs()
    validate_operator_isolation_budget(specs)
    legacy = next(spec for spec in specs if spec.historical_alias == "AutoFE_NoMultiply")
    assert legacy.enabled_operators == ("add_numeric", "subtract_numeric")
    assert set(legacy.excluded_operators) == {"multiply_numeric", "divide_numeric"}
    names = {spec.name for spec in specs}
    assert {"AutoFE_Isolate_Add", "AutoFE_Isolate_Subtract", "AutoFE_Isolate_Multiply", "AutoFE_Isolate_Divide"} <= names
    assert {"AutoFE_LeaveOut_Add", "AutoFE_LeaveOut_Subtract", "AutoFE_LeaveOut_Multiply", "AutoFE_LeaveOut_Divide"} <= names


def test_candidate_history_rejects_undefined_selected_score():
    with pytest.raises(ValueError, match="selected candidates"):
        CandidateHistoryRecord(
            dataset="synthetic", split_policy="row_level", seed=1, fold=1,
            condition="clean", iteration=0, candidate_id="bad",
            parent_features=("a", "b"), operator="divide_numeric",
            admissible=True, rejection_reason=None, selection_score=None,
            selection_score_status="undefined", selection_decision="selected",
            candidate_seed=1,
        )
