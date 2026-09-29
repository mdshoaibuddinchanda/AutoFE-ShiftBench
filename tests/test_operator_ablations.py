from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.feature_engineering import (
    ARITHMETIC_PRIMITIVES,
    DFSConfig,
    _arithmetic_candidate_counts,
    expand_features_with_dfs,
)
from src.mechanism_audit import ArithmeticCandidate, evaluate_arithmetic
from src.pipeline_runner import PIPELINE_CONFIGS, _bounded_cache_key, run_experiment
from src.provenance import stable_digest
from src.reviewer1_analysis import build_corrected_result_note


OPS = {
    "add": "add_numeric",
    "subtract": "subtract_numeric",
    "multiply": "multiply_numeric",
    "divide": "divide_numeric",
}
ALL = tuple(OPS.values())


EXPECTED = {
    "AutoFE_Baseline": ALL,
    "AutoFE_NoMultiply": (OPS["add"], OPS["subtract"]),
    "AutoFE_Isolate_Add": (OPS["add"],),
    "AutoFE_Isolate_Subtract": (OPS["subtract"],),
    "AutoFE_Isolate_Multiply": (OPS["multiply"],),
    "AutoFE_Isolate_Divide": (OPS["divide"],),
    "AutoFE_LeaveOut_Add": (OPS["subtract"], OPS["multiply"], OPS["divide"]),
    "AutoFE_LeaveOut_Subtract": (OPS["add"], OPS["multiply"], OPS["divide"]),
    "AutoFE_LeaveOut_Multiply": (OPS["add"], OPS["subtract"], OPS["divide"]),
    "AutoFE_LeaveOut_Divide": (OPS["add"], OPS["subtract"], OPS["multiply"]),
}


def _synthetic_frame(path):
    rng = np.random.default_rng(1901)
    a = rng.normal(size=60)
    b = rng.normal(size=60)
    b[0] = 0.0
    c = rng.normal(size=60)
    frame = pd.DataFrame({"a": a, "b": b, "c": c, "target": (a + c > 0).astype(int)})
    frame.to_csv(path, index=False)
    path.with_name(f"{path.stem}_meta.json").write_text(
        json.dumps({"target_column": "target", "dataset_identity": {"provider": "operator-test"}}),
        encoding="utf-8",
    )


def test_operator_truth_table_matches_actual_runner_configs():
    assert set(PIPELINE_CONFIGS) >= set(EXPECTED)
    for name, expected in EXPECTED.items():
        config = PIPELINE_CONFIGS[name]
        assert tuple(config.trans_primitives) == expected
        assert len(config.trans_primitives) == len(set(config.trans_primitives))
    assert set(EXPECTED["AutoFE_NoMultiply"]) == {OPS["add"], OPS["subtract"]}
    assert OPS["divide"] not in EXPECTED["AutoFE_NoMultiply"]


def test_serialized_configs_cache_keys_task_ids_and_result_names_are_distinct(tmp_path):
    config_fingerprints = {name: stable_digest(asdict(PIPELINE_CONFIGS[name])) for name in EXPECTED}
    assert len(set(config_fingerprints.values())) == len(EXPECTED)
    cache_keys = {_bounded_cache_key("same-feature-task", name, "same-cache-fingerprint") for name in EXPECTED}
    assert len(cache_keys) == len(EXPECTED)
    task_ids = {
        name: stable_digest({"run_id": "operator", "pipeline": name, "seed": 42, "fold": 1})
        for name in EXPECTED
    }
    assert len(set(task_ids.values())) == len(EXPECTED)


def test_candidate_accounting_detects_duplicates_and_rejections():
    raw = pd.DataFrame({
        "ADD(a,b)": [3.0, 4.0],
        "ADD(a,b)-duplicate": [3.0, np.nan],
        "DIV(a,c)": [1.0, np.nan],
    })
    generated = [
        {"name": "ADD(a,b)", "primitive": "add_numeric", "parents": ["a", "b"]},
        {"name": "ADD(a,b)-duplicate", "primitive": "add_numeric", "parents": ["b", "a"]},
        {"name": "DIV(a,c)", "primitive": "divide_numeric", "parents": ["a", "c"]},
    ]
    counts = _arithmetic_candidate_counts(generated, raw, ["ADD(a,b)"])
    assert counts[OPS["add"]] == {"generated": 2, "rejected": 1, "eligible": 1, "selected": 1, "duplicates": 1}
    assert counts[OPS["divide"]] == {"generated": 1, "rejected": 1, "eligible": 0, "selected": 0, "duplicates": 0}


def test_division_zero_near_zero_and_nonfinite_values_have_consistent_rules():
    candidate = ArithmeticCandidate("division", OPS["divide"], ("a", "b"))
    values = np.array([[2.0, 0.0], [2.0, 1e-300], [np.inf, 2.0], [2.0, 1e-320]])
    result = evaluate_arithmetic(candidate, values)
    assert result.valid_rows.tolist() == [False, True, False, False]
    assert result.undefined_reasons[0] == "zero_denominator"
    assert result.undefined_reasons[2] == "nonfinite_parent"
    assert result.undefined_reasons[3] == "zero_denominator_or_nonfinite_parent_or_output"
    assert np.isfinite(result.values[1])


def test_subtraction_and_division_preserve_left_right_operand_order():
    subtract = evaluate_arithmetic(
        ArithmeticCandidate("sub", OPS["subtract"], ("left", "right")),
        np.array([[2.0, 5.0]]),
    )
    divide = evaluate_arithmetic(
        ArithmeticCandidate("div", OPS["divide"], ("left", "right")),
        np.array([[2.0, 5.0]]),
    )
    assert subtract.values.tolist() == [-3.0]
    assert divide.values.tolist() == [0.4]


def test_each_operator_variant_smoke_and_stable_metadata(tmp_path):
    data_path = tmp_path / "operators.csv"
    _synthetic_frame(data_path)
    x = pd.read_csv(data_path).drop(columns=["target"])
    y = pd.read_csv(data_path)["target"]
    observed = {}
    for name, expected in EXPECTED.items():
        _, _, first = expand_features_with_dfs(x.iloc[:40], x.iloc[40:], y.iloc[:40], PIPELINE_CONFIGS[name])
        _, _, second = expand_features_with_dfs(x.iloc[:40], x.iloc[40:], y.iloc[:40], PIPELINE_CONFIGS[name])
        assert first["operator_configuration"]["enabled_operators"] == list(expected)
        assert first["operator_candidate_counts"] == second["operator_candidate_counts"]
        counts = first["operator_candidate_counts"]
        for operator in ARITHMETIC_PRIMITIVES:
            if operator in expected:
                assert counts[operator]["generated"] > 0
            else:
                assert counts[operator]["generated"] == 0
        observed[name] = first["operator_candidate_counts"]
    assert observed["AutoFE_LeaveOut_Multiply"][OPS["divide"]]["generated"] > 0
    assert observed["AutoFE_LeaveOut_Divide"][OPS["multiply"]]["generated"] > 0
    _, _, raw_metadata = expand_features_with_dfs(x.iloc[:40], x.iloc[40:], y.iloc[:40], PIPELINE_CONFIGS["Raw"])
    assert raw_metadata["operator_configuration"]["enabled_operators"] == []
    assert all(value["generated"] == 0 for value in raw_metadata["operator_configuration"]["candidate_counts"].values())


def test_all_operator_variants_write_distinct_result_rows(tmp_path):
    data_path = tmp_path / "runner.csv"
    _synthetic_frame(data_path)
    manifest = run_experiment(
        {"operator": data_path}, run_id="operator-smoke", output_root=tmp_path / "runs",
        seeds=[42], folds=[1], conditions=(("clean", 0.0),), pipelines=tuple(EXPECTED),
        models=("logistic_regression",), n_splits=3, cache_policy="bounded", cache_max_bytes=10 * 1024 * 1024,
    )
    assert manifest["status"] == "complete"
    rows = [json.loads(line) for line in (tmp_path / "runs" / "operator-smoke" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    successful = [row for row in rows if row.get("status") == "success"]
    assert {row["pipeline"] for row in successful} == set(EXPECTED)
    assert all("operator_configuration" in row for row in successful)
    note = build_corrected_result_note(
        tmp_path / "runs" / "operator-smoke" / "results.jsonl",
        manifest_path=tmp_path / "runs" / "operator-smoke" / "manifest.json",
    )
    assert set(note["operator_ablation_audit"]) == set(EXPECTED)
    assert all(item["effects_status"] == "PENDING CORRECTED RUN" for item in note["operator_ablation_audit"].values())
