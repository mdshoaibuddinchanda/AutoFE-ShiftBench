"""Focused verification of the 600-second forced-restart evidence contract."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from provenance import large_dataset_recovery_v1 as recovery
from provenance import verify_reviewer1_launch_v2 as gate


THREAD_ENV = recovery.THREAD_ENV


def _completed_recovery_fixture(tmp_path):
    run_dir = tmp_path / "recovery-airlines-row-001"
    run_dir.mkdir()
    manifest = {
        "run_id": run_dir.name, "status": "complete", "expected_tasks": 20,
        "counts_by_status": {"success": 20, "failed": 0, "skipped": 0,
                             "timed_out": 0, "pending": 0},
        "code_fingerprint": "test-source",
        "configuration": {
            "split_policy": "row_level", "scheduler_lease_seconds": 600.0,
            "numerical_thread_environment": THREAD_ENV,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(20):
            stream.write(json.dumps({
                "task_key": f"task-{index}", "status": "success",
                "prediction_sha256": f"prediction-{index}",
                "train_matrix_sha256": f"train-{index}",
                "test_matrix_sha256": f"test-{index}",
            }) + "\n")
    with sqlite3.connect(run_dir / "scheduler.sqlite") as connection:
        connection.execute("CREATE TABLE scheduler_tasks(task_key TEXT, status TEXT)")
        connection.execute("CREATE TABLE scheduler_attempts(task_key TEXT, attempt_no INTEGER)")
        for index in range(20):
            connection.execute("INSERT INTO scheduler_tasks VALUES (?, 'success')", (f"task-{index}",))
            connection.execute("INSERT INTO scheduler_attempts VALUES (?, 1)", (f"task-{index}",))
    before = {"task-0": ("prediction-0", "train-0", "test-0"),
              "task-1": ("prediction-1", "train-1", "test-1")}
    return run_dir, before


def test_committed_precrash_cells_remain_once_only(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    run_dir, before = _completed_recovery_fixture(tmp_path)
    result = recovery._check_terminal(run_dir, before, "test-source")
    assert result["status"] == "passed"
    assert result["committed_successes_before_kill"] == 2
    with sqlite3.connect(run_dir / "scheduler.sqlite") as connection:
        connection.execute("INSERT INTO scheduler_attempts VALUES ('task-0', 2)")
    with pytest.raises(ValueError, match="refit"):
        recovery._check_terminal(run_dir, before, "test-source")


def test_launch_gate_checks_local_recovery_manifest_and_path(tmp_path, monkeypatch):
    run_dir, _ = _completed_recovery_fixture(tmp_path)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    item = {"run_id": run_dir.name, "run_manifest": f"{run_dir.name}/manifest.json"}
    assert gate._recovery_run_manifest_valid(item, "test-source")
    assert not gate._recovery_run_manifest_valid({**item, "run_manifest": "../outside.json"}, "test-source")
    assert not gate._recovery_run_manifest_valid(item, "different-source")


def test_launch_gate_detects_another_heavy_coordinator(monkeypatch):
    processes = [
        SimpleNamespace(pid=5, info={"cmdline": ["python.exe", "-m", "provenance.large_dataset_calibration_group_v3"]}),
        SimpleNamespace(pid=6, info={"cmdline": ["python.exe", "-m", "unrelated.module"]}),
    ]
    monkeypatch.setattr(gate.psutil, "process_iter", lambda _fields: processes)
    assert gate._active_heavy_coordinators() == [
        {"pid": 5, "module": "provenance.large_dataset_calibration_group_v3"}
    ]
