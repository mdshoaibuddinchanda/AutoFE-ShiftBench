from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.task_scheduler import (
    LeaseLost,
    ResultConflict,
    TaskIdentityMismatch,
    TaskScheduler,
    TaskSpec,
    classify_failure,
)


ROOT = Path(__file__).resolve().parents[1]


def _scheduler(tmp_path: Path, *, lease_seconds: float = 10.0, retry_backoff_seconds: float = 0.0) -> TaskScheduler:
    return TaskScheduler(
        tmp_path / "scheduler.sqlite",
        artifact_dir=tmp_path / "artifacts",
        lease_seconds=lease_seconds,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def _spec(key: str = "task-1", *, max_attempts: int = 3) -> TaskSpec:
    return TaskSpec(key, "run-1", {"dataset": "synthetic", "fold": 1}, max_attempts=max_attempts)


def test_register_is_immutable_and_database_uses_wal(tmp_path):
    scheduler = _scheduler(tmp_path)
    assert scheduler.register_tasks([_spec()], now=0.0) == 1
    assert scheduler.register_tasks([_spec()], now=1.0) == 0
    with pytest.raises(TaskIdentityMismatch):
        scheduler.register_tasks([TaskSpec("task-1", "run-1", {"changed": True})], now=2.0)
    with scheduler._connection() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL


def test_lease_heartbeat_expiry_and_reclaim(tmp_path):
    scheduler = _scheduler(tmp_path, lease_seconds=10.0)
    scheduler.register_tasks([_spec()], now=0.0)
    first = scheduler.claim_task("worker-a", now=100.0)
    assert first is not None
    assert scheduler.claim_task("worker-b", now=101.0) is None
    renewed = scheduler.heartbeat(first, now=109.0)
    assert renewed.lease_expires_at == 119.0
    assert scheduler.expire_leases(now=118.0) == []
    expired = scheduler.expire_leases(now=120.0)
    assert len(expired) == 1
    assert expired[0].next_status == "pending"
    second = scheduler.claim_task("worker-b", now=121.0)
    assert second is not None and second.attempt_no == 2
    with pytest.raises(LeaseLost):
        scheduler.heartbeat(first, now=121.0)


def test_retry_classification_and_attempt_budget(tmp_path):
    assert classify_failure("TimeoutError").retryable
    assert classify_failure("ValueError").retryable is False
    assert classify_failure("sqlite3.OperationalError", "database is locked").retryable
    assert classify_failure("UnexpectedError").classification == "unknown"

    scheduler = _scheduler(tmp_path, retry_backoff_seconds=2.0)
    scheduler.register_tasks([_spec(max_attempts=2)], now=0.0)
    first = scheduler.claim_task("worker", now=0.0)
    assert first is not None
    decision = scheduler.record_failure(first, "TimeoutError", "temporary service timeout", now=1.0)
    assert decision.classification == "transient"
    assert scheduler.task_state("task-1")["status"] == "pending"
    assert scheduler.task_state("task-1")["available_at"] == 3.0
    second = scheduler.claim_task("worker", now=3.0)
    assert second is not None and second.attempt_no == 2
    scheduler.record_failure(second, "ValueError", "bad input", now=4.0)
    assert scheduler.task_state("task-1")["status"] == "failed"


def test_normal_result_publication_is_atomic_and_idempotent(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.register_tasks([_spec()], now=0.0)
    lease = scheduler.claim_task("worker", now=1.0)
    assert lease is not None
    payload = {"metric": 0.75, "status": "ok"}
    outcome = scheduler.publish_result(lease, payload, now=2.0)
    assert outcome.attempt_no == 1
    artifact = Path(outcome.artifact_path)
    assert artifact.exists()
    envelope = json.loads(artifact.read_text(encoding="utf-8"))
    assert envelope["payload"] == payload
    assert scheduler.task_state("task-1")["status"] == "success"
    repeated = scheduler.publish_result(lease, payload, now=3.0)
    assert repeated.result_digest == outcome.result_digest
    with pytest.raises(ResultConflict):
        scheduler.publish_result(lease, {"metric": 0.76}, now=3.0)


def test_specific_claim_does_not_reconcile_unrelated_completed_artifacts(tmp_path, monkeypatch):
    scheduler = _scheduler(tmp_path)
    scheduler.register_tasks([_spec("first"), _spec("second")], now=0.0)
    first = scheduler.claim_task("worker", task_key="first", now=1.0)
    assert first is not None
    scheduler.publish_result(first, {"metric": 1.0}, now=2.0)
    def forbid_global_artifact_read(path):
        raise AssertionError(f"unrelated result artifact was rescanned: {path}")
    monkeypatch.setattr(scheduler, "_read_artifact", forbid_global_artifact_read)
    second = scheduler.claim_task("worker", task_key="second", now=3.0)
    assert second is not None and second.task_key == "second"
    assert scheduler.result_state("first") is not None


def test_specific_claim_adopts_its_own_published_crash_artifact(tmp_path):
    scheduler = _scheduler(tmp_path, lease_seconds=30.0)
    scheduler.register_tasks([_spec()], now=0.0)
    lease = scheduler.claim_task("worker", task_key="task-1", now=1.0)
    assert lease is not None
    with pytest.raises(RuntimeError):
        scheduler.publish_result(lease, {"metric": 1.0}, now=2.0, fault_stage="after_artifact")
    assert scheduler.result_state("task-1") is None
    assert scheduler.claim_task("replacement", task_key="task-1", now=3.0) is None
    assert scheduler.task_state("task-1")["status"] == "success"
    assert scheduler.result_state("task-1")["recovered"] == 1
    with scheduler._connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM scheduler_attempts WHERE task_key='task-1'").fetchone()[0] == 1


def test_process_crash_after_artifact_rename_is_reconciled(tmp_path):
    db = tmp_path / "scheduler.sqlite"
    artifacts = tmp_path / "artifacts"
    scheduler = TaskScheduler(db, artifact_dir=artifacts, lease_seconds=10.0)
    scheduler.register_tasks([_spec()], now=0.0)
    script = f"""
from src.task_scheduler import TaskScheduler
s = TaskScheduler(r'{db}', artifact_dir=r'{artifacts}', lease_seconds=10.0)
lease = s.claim_task('crashed-worker', now=100.0)
s.publish_result(lease, {{'metric': 1.0}}, now=100.0, fault_stage='after_artifact')
"""
    child = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert child.returncode != 0
    report = scheduler.reconcile(now=200.0)
    assert report.expired_attempts == 1
    assert report.adopted_results == 1
    assert report.invalid_artifacts == ()
    assert scheduler.task_state("task-1")["status"] == "success"
    assert scheduler.result_state("task-1")["recovered"] == 1


def test_process_crash_before_rename_expires_and_retries(tmp_path):
    db = tmp_path / "scheduler.sqlite"
    artifacts = tmp_path / "artifacts"
    scheduler = TaskScheduler(db, artifact_dir=artifacts, lease_seconds=10.0)
    scheduler.register_tasks([_spec(max_attempts=2)], now=0.0)
    script = f"""
from src.task_scheduler import TaskScheduler
s = TaskScheduler(r'{db}', artifact_dir=r'{artifacts}', lease_seconds=10.0)
lease = s.claim_task('crashed-worker', now=100.0)
s.publish_result(lease, {{'metric': 1.0}}, now=100.0, fault_stage='after_temp')
"""
    child = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert child.returncode != 0
    report = scheduler.reconcile(now=200.0, temp_max_age_seconds=0.0)
    assert report.expired_attempts == 1
    assert report.adopted_results == 0
    assert report.removed_temp_files == 1
    assert scheduler.task_state("task-1")["status"] == "pending"
    retry = scheduler.claim_task("replacement-worker", now=201.0)
    assert retry is not None and retry.attempt_no == 2
    scheduler.publish_result(retry, {"metric": 1.0}, now=202.0)
    assert scheduler.task_state("task-1")["status"] == "success"


def test_new_process_reclaims_foreign_live_lease(tmp_path):
    scheduler = _scheduler(tmp_path, lease_seconds=3600.0)
    scheduler.register_tasks([_spec()], now=0.0)
    lease = scheduler.claim_task("old-process", now=100.0)
    assert lease is not None
    reclaimed = scheduler.reclaim_foreign_leases("new-process", now=101.0)
    assert len(reclaimed) == 1
    assert scheduler.task_state("task-1")["status"] == "pending"
    replacement = scheduler.claim_task("new-process", now=102.0)
    assert replacement is not None and replacement.attempt_no == 2


def test_reconciliation_reports_invalid_artifact_without_adopting(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.register_tasks([_spec()], now=0.0)
    lease = scheduler.claim_task("worker", now=1.0)
    assert lease is not None
    path = scheduler._artifact_path(lease.run_id, lease.task_key)
    path.write_text("{not-json", encoding="utf-8")
    report = scheduler.reconcile(now=2.0)
    assert report.adopted_results == 0
    assert report.invalid_artifacts
    assert scheduler.task_state("task-1")["status"] == "running"
