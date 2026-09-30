from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline_runner import _load_or_create_bounded_feature_cache, run_experiment
from src.cache_manager import CacheManager
from src.task_scheduler import TaskScheduler


def _write_dataset(path):
    rng = np.random.default_rng(41)
    signal = rng.normal(size=60)
    frame = pd.DataFrame({
        "x1": signal, "x2": rng.normal(size=60),
        "target": (signal > 0).astype(int),
    })
    frame.to_csv(path, index=False)
    path.with_name(f"{path.stem}_meta.json").write_text(json.dumps({
        "target_column": "target", "dataset_identity": {"provider": "test", "id": "bounded"},
    }), encoding="utf-8")


def test_bounded_cache_is_reclaimed_after_all_classifier_consumers(tmp_path):
    data_path = tmp_path / "bounded.csv"
    _write_dataset(data_path)
    manifest = run_experiment(
        {"bounded": data_path}, run_id="bounded-run", output_root=tmp_path / "runs",
        seeds=[42], folds=[1], conditions=(("clean", 0.0),),
        pipelines=("Raw",), models=("logistic_regression", "random_forest"), n_splits=3,
        cache_policy="bounded", cache_max_bytes=10 * 1024 * 1024,
        durable_scheduler=True,
    )
    assert manifest["status"] == "complete"
    assert manifest["counts_by_status"] == {
        "success": 2, "failed": 0, "skipped": 0, "timed_out": 0, "pending": 0,
    }
    cache_root = tmp_path / "runs" / "bounded-run" / "cache_bounded"
    assert not list(cache_root.glob("*.payload"))
    assert manifest["cache_storage"]["policy"] == "bounded_regenerable"
    assert manifest["cache_storage"]["final_cleanup"]["after_bytes"] >= 0
    scheduler = TaskScheduler(
        tmp_path / "runs" / "bounded-run" / "scheduler.sqlite",
        artifact_dir=tmp_path / "runs" / "bounded-run" / "scheduler_results",
    )
    states = [scheduler.task_state(task["task_key"]) for task in manifest["tasks"]]
    assert states and all(state and state["status"] == "success" for state in states)
    assert len(list((tmp_path / "runs" / "bounded-run" / "scheduler_results").glob("*.json"))) == 2


def test_cache_audit_proves_one_build_and_ten_consumer_fanout(tmp_path):
    data_path = tmp_path / "fanout.csv"
    _write_dataset(data_path)
    models = (
        "logistic_regression", "random_forest", "extra_trees", "linear_svm", "knn",
        "gaussian_nb", "mlp", "lightgbm", "xgboost", "catboost",
    )
    manifest = run_experiment(
        {"fanout": data_path}, run_id="fanout-run", output_root=tmp_path / "runs",
        seeds=[42], folds=[1], conditions=(("clean", 0.0),), pipelines=("AutoFE_Baseline",),
        models=models, n_splits=3, cache_policy="bounded", cache_max_bytes=10 * 1024 * 1024,
        durable_scheduler=True, cache_audit=True,
    )
    assert manifest["status"] == "complete"
    audit = manifest["cache_audit"]
    assert audit["enabled"] is True
    assert len(audit["artifacts"]) == 1
    record = next(iter(audit["artifacts"].values()))
    assert record["expected_consumers"] == 10
    assert record["build_count"] == 1
    assert record["hit_count"] == 9
    assert len(record["consumer_task_keys"]) == 10
    assert set(record["consumer_task_keys"]) == set(record["terminal_consumer_task_keys"])
    assert record["regeneration_count"] == 0
    assert record["active_reader_history"]
    assert all(item["active_readers"] >= 1 for item in record["active_reader_history"])
    assert record["deletion_observed"] is True
    assert record["deletion_time_utc"]
    assert not list((tmp_path / "runs" / "fanout-run" / "cache_bounded").glob("*.payload"))


def test_cache_reader_audit_does_not_scan_unrelated_artifacts(tmp_path, monkeypatch):
    manager = CacheManager(tmp_path / "cache")
    manager.put_bytes("unrelated", b"other payload")
    events = []
    def no_global_scan(*args, **kwargs):
        raise AssertionError("reader audit scanned the entire cache")
    monkeypatch.setattr(manager, "reconcile", no_global_scan)
    args = dict(
        feature_task_key="feature", pipeline_name="Raw",
        expected_manifest={"version": 1}, make_payload=lambda: ("payload",),
        audit_callback=lambda event, details: events.append((event, details)),
    )
    first, hit = _load_or_create_bounded_feature_cache(manager, **args)
    second, hit_again = _load_or_create_bounded_feature_cache(manager, **args)
    assert first == second == ("payload",)
    assert (hit, hit_again) == (False, True)
    assert [details["active_readers"] for event, details in events if event == "reader_acquired"] == [1, 1]


def test_bounded_results_match_retained_cache_results(tmp_path):
    data_path = tmp_path / "parity.csv"
    _write_dataset(data_path)
    common = dict(
        data_paths={"parity": data_path}, seeds=[42], folds=[1],
        conditions=(("clean", 0.0),), pipelines=("Raw",),
        models=("logistic_regression", "random_forest"), n_splits=3,
    )
    retained = run_experiment(
        **common, run_id="retain", output_root=tmp_path / "retained",
    )
    bounded = run_experiment(
        **common, run_id="bounded", output_root=tmp_path / "bounded",
        cache_policy="bounded", cache_max_bytes=10 * 1024 * 1024,
    )
    def successful(path):
        return {
            (row["dataset"], row["pipeline"], row["model"], row["condition"], row["seed"], row["fold"]): {
                field: row.get(field)
                for field in ("train_matrix_sha256", "test_matrix_sha256", "prediction_sha256", "roc_auc", "f1")
            }
            for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
            if row.get("status") == "success"
        }
    retained_rows = successful(tmp_path / "retained" / "retain" / "results.jsonl")
    bounded_rows = successful(tmp_path / "bounded" / "bounded" / "results.jsonl")
    assert retained["status"] == bounded["status"] == "complete"
    assert retained_rows == bounded_rows


def test_durable_runner_resumes_after_process_interrupt(tmp_path):
    data_path = tmp_path / "resume.csv"
    _write_dataset(data_path)
    repo = __file__
    script = textwrap.dedent(f"""
        import os
        from pathlib import Path
        from src.pipeline_runner import run_experiment
        data = Path({str(data_path)!r})
        def hook(phase, task):
            if phase == "phase2":
                os._exit(77)
        run_experiment(
            {{"resume": data}}, run_id="resume-run", output_root=Path({str(tmp_path / 'runs')!r}),
            seeds=[42], folds=[1], conditions=(("clean", 0.0),),
            pipelines=("Raw",), models=("logistic_regression",), n_splits=3,
            cache_policy="bounded", cache_max_bytes=10 * 1024 * 1024,
            durable_scheduler=True, scheduler_lease_seconds=30.0,
            failure_hook=hook,
        )
    """)
    first = subprocess.run([sys.executable, "-c", script], cwd=Path(repo).parents[1], check=False)
    assert first.returncode == 77
    time.sleep(0.5)
    resume_script = script.replace("failure_hook=hook", "failure_hook=None")
    second = subprocess.run([sys.executable, "-c", resume_script], cwd=Path(repo).parents[1], check=False)
    assert second.returncode == 0
    manifest = json.loads((tmp_path / "runs" / "resume-run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["counts_by_status"]["success"] == 1


def test_live_coordinator_cannot_be_reclaimed_by_second_process(tmp_path):
    data_path = tmp_path / "fenced.csv"
    _write_dataset(data_path)
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    script = textwrap.dedent(f"""
        import os, time
        from pathlib import Path
        from src.pipeline_runner import run_experiment
        ready = Path({str(ready)!r})
        release = Path({str(release)!r})
        def hook(phase, task):
            if phase == "phase2":
                ready.write_text("ready")
                deadline = time.monotonic() + 30
                while not release.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("test release missing")
                    time.sleep(0.05)
        run_experiment(
            {{"fenced": Path({str(data_path)!r})}}, run_id="fenced-run",
            output_root=Path({str(tmp_path / 'runs')!r}),
            seeds=[42], folds=[1], conditions=(("clean", 0.0),),
            pipelines=("Raw",), models=("logistic_regression",), n_splits=3,
            cache_policy="bounded", cache_max_bytes=10 * 1024 * 1024,
            durable_scheduler=True, scheduler_lease_seconds=600.0,
            failure_hook=hook if os.environ.get("BLOCK") == "1" else None,
        )
    """)
    env = dict(os.environ, BLOCK="1")
    first = subprocess.Popen(
        [sys.executable, "-c", script], cwd=Path(__file__).parents[1],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), first.communicate(timeout=2)
        second = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).parents[1],
            capture_output=True, text=True, timeout=30,
        )
        assert second.returncode != 0
        assert "Could not acquire coordinator lock" in second.stderr
        with sqlite3.connect(tmp_path / "runs" / "fenced-run" / "scheduler.sqlite") as conn:
            assert conn.execute("SELECT status FROM scheduler_tasks").fetchone()[0] == "running"
            assert conn.execute("SELECT COUNT(*) FROM scheduler_attempts").fetchone()[0] == 1
    finally:
        release.write_text("release")
        stdout, stderr = first.communicate(timeout=30)
    assert first.returncode == 0, (stdout, stderr)
    with sqlite3.connect(tmp_path / "runs" / "fenced-run" / "scheduler.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM scheduler_results").fetchone()[0] == 1


@pytest.mark.parametrize("split_policy", ["row_level", "group_aware"])
def test_cache_fanout_audit_survives_midgroup_process_resume(tmp_path, split_policy):
    data_path = tmp_path / "resume_fanout.csv"
    _write_dataset(data_path)
    count_path = tmp_path / "phase2-count.txt"
    models = (
        "logistic_regression", "random_forest", "extra_trees", "linear_svm", "knn",
        "gaussian_nb", "mlp", "lightgbm", "xgboost", "catboost",
    )
    script = textwrap.dedent(f"""
        import os
        from pathlib import Path
        from src.pipeline_runner import run_experiment
        data = Path({str(data_path)!r})
        count_path = Path({str(count_path)!r})
        def hook(phase, task):
            if phase == "phase2":
                count = int(count_path.read_text() or "0") if count_path.exists() else 0
                count_path.write_text(str(count + 1))
                if count + 1 == 6:
                    os._exit(77)
        run_experiment(
            {{"resume_fanout": data}}, run_id="resume-fanout", output_root=Path({str(tmp_path / 'runs')!r}),
            seeds=[42], folds=[1], conditions=(("clean", 0.0),),
            pipelines=("AutoFE_Baseline",), models={models!r}, n_splits=3,
            cache_policy="bounded", cache_max_bytes=10 * 1024 * 1024,
            durable_scheduler=True, scheduler_lease_seconds=30.0,
            cache_audit=True, failure_hook=hook, split_policy={split_policy!r},
        )
    """)
    first = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1], check=False)
    assert first.returncode == 77
    run_dir = tmp_path / "runs" / "resume-fanout"
    with sqlite3.connect(run_dir / "scheduler.sqlite") as conn:
        committed_before = dict(conn.execute(
            "SELECT task_key, final_result_digest FROM scheduler_tasks WHERE status='success'"
        ))
    assert len(committed_before) == 5
    assert len(list((run_dir / "cache_bounded").glob("*.payload"))) == 1
    time.sleep(0.5)
    resumed = subprocess.run(
        [sys.executable, "-c", script.replace("failure_hook=hook", "failure_hook=None")],
        cwd=Path(__file__).parents[1], check=False,
    )
    assert resumed.returncode == 0
    manifest = json.loads((tmp_path / "runs" / "resume-fanout" / "manifest.json").read_text(encoding="utf-8"))
    record = next(iter(manifest["cache_audit"]["artifacts"].values()))
    assert manifest["status"] == "complete"
    assert record["build_count"] == 1
    assert record["hit_count"] == 9
    assert record["regeneration_count"] == 0
    assert len(record["terminal_consumer_task_keys"]) == 10
    assert record["deletion_observed"] is True
    with sqlite3.connect(run_dir / "scheduler.sqlite") as conn:
        committed_after = dict(conn.execute(
            "SELECT task_key, final_result_digest FROM scheduler_tasks WHERE status='success'"
        ))
        attempts = dict(conn.execute(
            "SELECT task_key, COUNT(*) FROM scheduler_attempts GROUP BY task_key"
        ))
        authoritative = conn.execute("SELECT COUNT(*) FROM scheduler_results").fetchone()[0]
    assert len(committed_after) == authoritative == 10
    assert all(committed_after[key] == digest and attempts[key] == 1
               for key, digest in committed_before.items())
    success_rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()
                    if json.loads(line).get("status") == "success"]
    assert len(success_rows) == len({row["task_key"] for row in success_rows}) == 10


def test_scheduler_publication_before_checkpoint_is_repaired_on_resume(tmp_path):
    data_path = tmp_path / "publish_boundary.csv"
    _write_dataset(data_path)
    script = textwrap.dedent(f"""
        import os
        from pathlib import Path
        from src.pipeline_runner import run_experiment
        data = Path({str(data_path)!r})
        def hook(phase, task):
            if phase == "after_scheduler_publish":
                os._exit(78)
        run_experiment(
            {{"boundary": data}}, run_id="boundary-run", output_root=Path({str(tmp_path / 'runs')!r}),
            seeds=[42], folds=[1], conditions=(("clean", 0.0),), pipelines=("Raw",),
            models=("logistic_regression",), n_splits=3, cache_policy="bounded",
            cache_max_bytes=10 * 1024 * 1024, durable_scheduler=True,
            scheduler_lease_seconds=30.0, failure_hook=hook,
        )
    """)
    first = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1], check=False)
    assert first.returncode == 78
    resumed = subprocess.run(
        [sys.executable, "-c", script.replace("failure_hook=hook", "failure_hook=None")],
        cwd=Path(__file__).parents[1], check=False,
    )
    assert resumed.returncode == 0
    manifest = json.loads((tmp_path / "runs" / "boundary-run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["counts_by_status"]["success"] == 1
    rows = [json.loads(line) for line in (tmp_path / "runs" / "boundary-run" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({row["task_key"] for row in rows}) == 1


def test_integrated_runner_workers_overlap_and_publish_unique_rows(tmp_path):
    """The main runner, rather than the profiling harness, executes workers concurrently."""
    data_path = tmp_path / "concurrent.csv"
    _write_dataset(data_path)
    models = ("logistic_regression", "random_forest", "extra_trees", "linear_svm")
    manifest = run_experiment(
        {"concurrent": data_path}, run_id="concurrent-run", output_root=tmp_path / "runs",
        seeds=[42], folds=[1], conditions=(("clean", 0.0),), pipelines=("Raw",),
        models=models, n_splits=3, cache_policy="bounded", cache_max_bytes=32 * 1024 * 1024,
        durable_scheduler=True, cache_audit=True, workers=2,
    )
    assert manifest["status"] == "complete"
    rows = [json.loads(line) for line in (tmp_path / "runs" / "concurrent-run" / "results.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row.get("status") == "success"]
    assert len(rows) == len(models)
    assert len({row["task_key"] for row in rows}) == len(models)
    assert len({row["worker_pid"] for row in rows}) >= 2
    intervals = [(row["worker_started_unix"], row["worker_finished_unix"]) for row in rows]
    assert any(a0 < b1 and b0 < a1 for i, (a0, a1) in enumerate(intervals) for b0, b1 in intervals[i + 1:])
    assert manifest["counts_by_status"]["success"] == len(models)
