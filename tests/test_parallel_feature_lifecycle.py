from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import numpy as np
import pandas as pd

from src import pipeline_runner as runner


def _dataset(path):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(800, 10)), columns=[f"x{i}" for i in range(10)])
    frame["target"] = (frame.x0 + frame.x1 > 0).astype(int)
    frame.to_csv(path, index=False)
    return path


def _results(path):
    fields = ("train_matrix_sha256", "test_matrix_sha256", "prediction_sha256",
              "roc_auc", "f1", "operator_candidate_counts")
    return {(row["condition"], row["pipeline"], row["model"]):
            {field: row.get(field) for field in fields}
            for row in map(json.loads, path.read_text().splitlines()) if row["status"] == "success"}


def test_parallel_groups_reclaim_before_capacity_causes_valid_task_failures(tmp_path, monkeypatch):
    path = _dataset(tmp_path / "capacity.csv")
    preparation_calls = []
    original_prepare = runner._prepare_matrices

    def counted_prepare(*args, **kwargs):
        preparation_calls.append(kwargs["family"])
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(runner, "_prepare_matrices", counted_prepare)
    common = dict(data_paths={"capacity": path}, seeds=[42], folds=[1], n_splits=3,
                  conditions=(("clean", 0.0), ("gaussian_noise", 0.05), ("gaussian_noise", 0.1)),
                  pipelines=("Raw",), models=("gaussian_nb", "logistic_regression"),
                  cache_policy="bounded", cache_max_bytes=120_000,
                  durable_scheduler=True, cache_audit=True)
    parallel = runner.run_experiment(**common, run_id="parallel", output_root=tmp_path, workers=2)
    assert parallel["status"] == "complete"
    assert parallel["counts_by_status"]["success"] == 6
    assert parallel["counts_by_status"]["failed"] == 0
    assert len(preparation_calls) == 3  # DFS/preprocessing is not repeated on capacity retries.
    audits = list(parallel["cache_audit"]["artifacts"].values())
    assert len(audits) == 3
    assert all(item["build_count"] == 1 and item["hit_count"] == 1 for item in audits)
    assert all(item["memory_hit_count"] == 1 and item["deletion_observed"] for item in audits)
    assert not list((tmp_path / "parallel" / "cache_bounded").glob("*.payload"))
    assert parallel["cache_storage"]["high_water"]["high_water_bytes"] > 0
    rows = list(map(json.loads, (tmp_path / "parallel" / "results.jsonl").read_text().splitlines()))
    assert all(row["autofe_gen_time_s"] > 0 and row["preprocessing_time_s"] > 0 for row in rows)
    assert all(abs(row["preparation_time_s"] - row["autofe_gen_time_s"] - row["preprocessing_time_s"]) < 1e-8
               for row in rows)
    serial = runner.run_experiment(**common, run_id="serial", output_root=tmp_path, workers=1)
    assert serial["status"] == "complete"
    assert _results(tmp_path / "parallel" / "results.jsonl") == _results(tmp_path / "serial" / "results.jsonl")


def test_completed_fast_classifier_is_published_before_unfinished_first_submission(tmp_path, monkeypatch):
    path = _dataset(tmp_path / "dispatch.csv")
    release_first = threading.Event()
    first_started = threading.Event()
    published = []
    original_worker = runner._fit_and_score_worker

    def controlled_worker(payload):
        if payload["model"] == "gaussian_nb":
            first_started.set()
            release_first.wait(timeout=3)
        else:
            assert first_started.wait(timeout=3)
        return original_worker(payload)

    def publish_hook(phase, task):
        if phase == "after_scheduler_publish":
            published.append(task["model"])
            if task["model"] == "logistic_regression":
                release_first.set()

    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda max_workers, mp_context, **kwargs:
                        ThreadPoolExecutor(max_workers=max_workers))
    monkeypatch.setattr(runner, "_fit_and_score_worker", controlled_worker)
    manifest = runner.run_experiment(
        {"dispatch": path}, run_id="dispatch", output_root=tmp_path, workers=2,
        seeds=[42], folds=[1], n_splits=3, conditions=(("clean", 0.0),),
        pipelines=("Raw",), models=("gaussian_nb", "logistic_regression"),
        cache_policy="bounded", cache_max_bytes=120_000, durable_scheduler=True,
        failure_hook=publish_hook,
    )
    assert manifest["status"] == "complete"
    assert published == ["logistic_regression", "gaussian_nb"]


def test_serial_estimators_cannot_mutate_inputs_for_later_classifier(tmp_path, monkeypatch):
    path = _dataset(tmp_path / "mutation.csv")
    original_worker = runner._fit_and_score_worker
    seen = []

    def mutating_worker(payload):
        seen.append(payload["xtr"].copy())
        result = original_worker(payload)
        payload["xtr"][:] = 999
        payload["y_train_enc"][:] = 999
        return result

    monkeypatch.setattr(runner, "_fit_and_score_worker", mutating_worker)
    manifest = runner.run_experiment(
        {"mutation": path}, run_id="mutation", output_root=tmp_path,
        seeds=[42], folds=[1], n_splits=3, conditions=(("clean", 0.0),),
        pipelines=("Raw",), models=("gaussian_nb", "logistic_regression"),
    )
    assert manifest["status"] == "complete"
    np.testing.assert_array_equal(seen[0], seen[1])


def test_resume_reclaims_terminal_group_left_before_delete_and_new_admission(tmp_path):
    path = _dataset(tmp_path / "resume.csv")
    script = f"""
import os
from src.pipeline_runner import run_experiment
def stop(phase, task):
    if phase == 'after_scheduler_publish' and task['condition'] == 'clean' and task['model'] == 'logistic_regression':
        os._exit(77)
run_experiment({{'resume': {str(path)!r}}}, run_id='resume', output_root={str(tmp_path)!r},
    seeds=[42], folds=[1], n_splits=3, workers=2,
    conditions=(('clean',0.0),('gaussian_noise',0.05),('gaussian_noise',0.1)),
    pipelines=('Raw',), models=('gaussian_nb','logistic_regression'),
    cache_policy='bounded', cache_max_bytes=120000, durable_scheduler=True,
    scheduler_lease_seconds=30, failure_hook=stop)
"""
    first = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1])
    assert first.returncode == 77
    run_dir = tmp_path / "resume"
    assert list((run_dir / "cache_bounded").glob("*.payload"))
    with sqlite3.connect(run_dir / "scheduler.sqlite") as connection:
        before = dict(connection.execute(
            "SELECT task_key, final_result_digest FROM scheduler_tasks WHERE status='success'"))
    assert len(before) == 2
    resumed = subprocess.run([sys.executable, "-c", script.replace("failure_hook=stop", "failure_hook=None")],
                             cwd=Path(__file__).parents[1])
    assert resumed.returncode == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["counts_by_status"]["success"] == 6
    assert not list((run_dir / "cache_bounded").glob("*.payload"))
    with sqlite3.connect(run_dir / "scheduler.sqlite") as connection:
        after = dict(connection.execute(
            "SELECT task_key, final_result_digest FROM scheduler_tasks WHERE status='success'"))
        attempts = dict(connection.execute("SELECT task_key, COUNT(*) FROM scheduler_attempts GROUP BY task_key"))
    assert all(after[key] == digest and attempts[key] == 1 for key, digest in before.items())
