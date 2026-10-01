"""The separate mechanism run reproduces the runner's clean feature matrices."""

import gzip
import json
import sqlite3
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from provenance.run_mechanism_history import _histories_for_fold
from provenance import analyze_mechanism_history
from provenance import freeze_mechanism_scope_v1 as mechanism_freeze
from src.pipeline_runner import PIPELINE_CONFIGS, _prepare_matrices, _stable_perturbation_seed
from src.provenance import file_sha256, frame_sha256, stable_digest


def test_candidate_and_jacobian_histories_are_stable_and_match_runner(tmp_path):
    values = np.linspace(-2, 2, 60)
    frame = pd.DataFrame({"a": values, "b": np.cos(values)})
    labels = pd.Series((values > 0).astype(int), index=frame.index)
    x_train, y_train = frame.iloc[:48].copy(), labels.iloc[:48].copy()
    x_test = frame.iloc[48:].copy()
    identity = {"provider": "test", "requested_name": "synthetic"}
    kwargs = {
        "dataset": "synthetic", "split_policy": "row_level", "seed": 42,
        "fold": 1, "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "dataset_identity": identity,
    }
    first = _histories_for_fold(**kwargs, task_dir=tmp_path / "first")
    second = _histories_for_fold(**kwargs, task_dir=tmp_path / "second")
    assert first["candidate_count"] > 0
    assert first["candidate_count"] == second["candidate_count"]
    assert first["history_sha256"] == second["history_sha256"]
    assert first["jacobian_sha256"] == second["jacobian_sha256"]
    with gzip.open(tmp_path / "first" / "candidate_history.jsonl.gz", "rt", encoding="utf-8") as stream:
        candidates = [json.loads(line) for line in stream]
    with gzip.open(tmp_path / "first" / "candidate_jacobians.jsonl.gz", "rt", encoding="utf-8") as stream:
        jacobians = [json.loads(line) for line in stream]
    assert len(candidates) == len(jacobians) == first["candidate_count"]
    assert {row["operator"] for row in candidates} == {
        "add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric",
    }
    assert all(row["score_scope"] == "train" for row in candidates)
    assert all(row["candidate_id"] == jacobians[i]["candidate_id"] for i, row in enumerate(candidates))
    perturb_seed = _stable_perturbation_seed(identity, x_train, y_train, 42, 1, "clean")
    train_fe, test_fe, *_ = _prepare_matrices(
        x_train, y_train, x_test, family="clean", severity=0.0,
        perturbation_seed=perturb_seed, pipeline_name="AutoFE_Baseline",
        config=PIPELINE_CONFIGS["AutoFE_Baseline"],
    )
    assert first["train_matrix_sha256"] == frame_sha256(train_fe)
    assert first["test_matrix_sha256"] == frame_sha256(test_fe)


def test_clean_auc_pairing_uses_terminal_success_and_dataset_unit(tmp_path, monkeypatch):
    results = tmp_path / "results.jsonl"
    index = tmp_path / "manifest_outcomes.sqlite"
    rows = []
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE outcomes(task_key TEXT PRIMARY KEY, status TEXT)")
        for dataset, delta in (("d1", 0.05), ("d2", -0.02)):
            for model in ("a", "b"):
                for pipeline, auc in (("Raw", 0.7), ("AutoFE_Baseline", 0.7 + delta)):
                    key = f"{dataset}-{model}-{pipeline}"
                    connection.execute("INSERT INTO outcomes VALUES (?, 'success')", (key,))
                    rows.append({
                        "task_key": key, "status": "success", "condition": "clean",
                        "dataset": dataset, "seed": 42, "fold": 1, "model": model,
                        "pipeline": pipeline, "roc_auc": auc,
                        "train_matrix_sha256": "train", "test_matrix_sha256": "test",
                    })
        rows.append({"task_key": "old-failure", "status": "failed", "condition": "clean",
                     "dataset": "d1", "pipeline": "AutoFE_Baseline"})
    results.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    scores, hashes = analyze_mechanism_history._paired_clean_auc(results, 2)
    assert scores["d1"]["complete"] and scores["d2"]["complete"]
    assert abs(scores["d1"]["mean_auc_delta"] - 0.05) < 1e-12
    assert abs(scores["d2"]["mean_auc_delta"] + 0.02) < 1e-12
    assert hashes[("d1", 42, 1)] == ("train", "test")
    monkeypatch.setattr(analyze_mechanism_history, "PERMUTATIONS", 100)
    monkeypatch.setattr(analyze_mechanism_history, "BOOTSTRAPS", 100)
    first = analyze_mechanism_history._association([1, 2, 3, 4], [4, 3, 2, 1])
    second = analyze_mechanism_history._association([1, 2, 3, 4], [4, 3, 2, 1])
    assert first == second
    assert first["rho"] == -1.0


def _mechanism_fixture(tmp_path):
    scope = {
        "code_fingerprint": "source-v3", "seeds": [42], "folds": [1], "models": ["gaussian_nb"],
        "datasets": [{"name": "tiny", "csv_sha256": "csv-hash", "sidecar_sha256": "metadata-hash"}],
        "group_auc_ineligible_datasets": [],
        "analysis_sha256": {"provenance/run_mechanism_history.py": "script-hash"},
        "required_numerical_thread_environment": mechanism_freeze.THREAD_ENV,
    }
    configuration = {
        "datasets": ["tiny"], "seeds": [42], "folds": [1], "split_policy": "row_level",
        "condition": "clean", "pipeline": "AutoFE_Baseline", "code_fingerprint": "source-v3",
        "mechanism_script_sha256": "script-hash",
        "numerical_thread_environment": mechanism_freeze.THREAD_ENV,
    }
    manifest = {"status": "complete", "configuration": configuration,
                "configuration_fingerprint": stable_digest(configuration),
                "expected_tasks": 1, "counts": {"complete": 1, "skipped": 0}}
    task_dir = tmp_path / "mechanism" / "tasks" / "one"
    task_dir.mkdir(parents=True)
    history = task_dir / "history.jsonl.gz"
    history.write_bytes(gzip.compress(b'{"candidate_id":"one"}\n', mtime=0))
    jacobian = task_dir / "jacobian.jsonl.gz"
    jacobian.write_bytes(gzip.compress(b'{"selection_decision":"selected","jacobian":{"median":2.0}}\n', mtime=0))
    task = {
        "dataset": "tiny", "split_policy": "row_level", "seed": 42, "fold": 1,
        "condition": "clean", "pipeline": "AutoFE_Baseline", "status": "complete",
        "configuration_fingerprint": manifest["configuration_fingerprint"],
        "csv_sha256": "csv-hash", "sidecar_sha256": "metadata-hash",
        "train_matrix_sha256": "train-hash", "test_matrix_sha256": "test-hash",
        "history_path": history.name, "history_sha256": file_sha256(history),
        "jacobian_path": jacobian.name, "jacobian_sha256": file_sha256(jacobian),
    }
    status_path = task_dir / "status.json"
    status_path.write_text(json.dumps(task), encoding="utf-8")
    (task_dir.parents[1] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return scope, manifest, task, status_path


def test_mechanism_analysis_requires_exact_frozen_artifacts_and_grid(tmp_path):
    scope, _, _, status_path = _mechanism_fixture(tmp_path)
    exposures, hashes = analyze_mechanism_history._mechanism_exposures(
        status_path.parents[2], 1, scope=scope, split_policy="row_level")
    assert exposures["tiny"]["complete"] is True
    assert exposures["tiny"]["mean_fold_median_selected_scaled_jacobian_norm"] == 2.0
    assert hashes == {("tiny", 42, 1): ("train-hash", "test-hash")}


@pytest.mark.parametrize("field,value", [
    ("code_fingerprint", "old-source"), ("split_policy", "group_aware"),
    ("seeds", [123]), ("folds", [2]), ("mechanism_script_sha256", "old-script"),
])
def test_mechanism_analysis_rejects_mixed_source_policy_or_sampling_grid(tmp_path, field, value):
    scope, manifest, _, _ = _mechanism_fixture(tmp_path)
    manifest["configuration"][field] = value
    manifest["configuration_fingerprint"] = stable_digest(manifest["configuration"])
    with pytest.raises(ValueError, match="frozen source, policy or grid"):
        analyze_mechanism_history._validate_mechanism_manifest(manifest, scope, "row_level")


@pytest.mark.parametrize("damage", ["missing-task", "duplicate-task", "changed-history", "changed-jacobian"])
def test_mechanism_analysis_rejects_missing_duplicate_or_changed_evidence(tmp_path, damage):
    scope, _, task, status_path = _mechanism_fixture(tmp_path)
    if damage == "missing-task":
        status_path.unlink()
    elif damage == "duplicate-task":
        duplicate = status_path.parents[1] / "duplicate"
        duplicate.mkdir()
        (duplicate / "status.json").write_text(json.dumps(task), encoding="utf-8")
    else:
        filename = task["history_path" if damage == "changed-history" else "jacobian_path"]
        (status_path.parent / filename).write_bytes(b"damaged evidence")
    with pytest.raises(ValueError):
        analyze_mechanism_history._mechanism_exposures(
            status_path.parents[2], 1, scope=scope, split_policy="row_level")


def test_mechanism_analysis_rejects_unmatched_primary_fold_evidence(tmp_path, monkeypatch):
    scope, _, _, status_path = _mechanism_fixture(tmp_path)
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    results = tmp_path / "performance" / "results.jsonl"
    results.parent.mkdir()
    (results.parent / "manifest.json").write_text(json.dumps({
        "status": "complete", "code_fingerprint": "source-v3", "split_policy": "row_level",
    }), encoding="utf-8")
    monkeypatch.setattr(analyze_mechanism_history, "_paired_clean_auc", lambda *_args: ({}, {}))
    with pytest.raises(ValueError, match="fold coverage differ"):
        analyze_mechanism_history.analyze(
            results, results, status_path.parents[2], status_path.parents[2],
            tmp_path / "output.json", scope_path)
    assert not (tmp_path / "output.json").exists()


def test_mechanism_smoke_freeze_rejects_stale_source_and_modified_files(tmp_path):
    scope, manifest, task, status_path = _mechanism_fixture(tmp_path)
    smoke_manifest = deepcopy(manifest)
    smoke_manifest["expected_tasks"] = 2
    smoke_manifest["counts"] = {"complete": 2, "skipped": 0}
    smoke_manifest["configuration"]["datasets"] = list(mechanism_freeze.DATASETS)
    smoke_manifest["configuration_fingerprint"] = stable_digest(smoke_manifest["configuration"])
    mechanism_freeze._validate_smoke_manifest(smoke_manifest, scope, "row_level", "script-hash")
    stale = deepcopy(smoke_manifest)
    stale["configuration"]["code_fingerprint"] = "old-source"
    stale["configuration_fingerprint"] = stable_digest(stale["configuration"])
    with pytest.raises(RuntimeError, match="identity/count mismatch"):
        mechanism_freeze._validate_smoke_manifest(stale, scope, "row_level", "script-hash")
    mechanism_freeze._validate_smoke_task(status_path, task, manifest, scope, "row_level", "tiny")
    (status_path.parent / task["history_path"]).write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="artifact missing or changed"):
        mechanism_freeze._validate_smoke_task(status_path, task, manifest, scope, "row_level", "tiny")
