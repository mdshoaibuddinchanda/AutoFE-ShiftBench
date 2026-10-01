"""The separate mechanism run reproduces the runner's clean feature matrices."""

import gzip
import json
import sqlite3

import numpy as np
import pandas as pd

from provenance.run_mechanism_history import _histories_for_fold
from provenance import analyze_mechanism_history
from src.pipeline_runner import PIPELINE_CONFIGS, _prepare_matrices, _stable_perturbation_seed
from src.provenance import frame_sha256


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
