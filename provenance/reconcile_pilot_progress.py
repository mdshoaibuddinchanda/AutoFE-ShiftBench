"""Safely reconcile compatible completed cells into a repaired pilot run.

This migration is deliberately explicit: it validates the scientific
configuration and overlapping numerical results, rewrites task fingerprints
for the new implementation revision, and records the old implementation
identity on every imported row.  It never imports failed cells or feature
cache files as completed model evaluations.
"""

from __future__ import annotations

import argparse
import json
import sys
import sqlite3
import time
import secrets
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import record_task
from src.pipeline_runner import _save_run_manifest, _safe_result_write
from src.provenance import atomic_write_json, cache_fingerprint
from src.task_scheduler import TaskScheduler, TaskSpec, _digest_json


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reconcile(old_dir: Path, target_dir: Path) -> dict:
    old_manifest = json.loads((old_dir / "manifest.json").read_text(encoding="utf-8"))
    target_manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    if old_manifest.get("run_id") != target_manifest.get("run_id"):
        raise ValueError("old and target run identities differ")
    if old_manifest.get("configuration_fingerprint") != target_manifest.get("configuration_fingerprint"):
        raise ValueError("scientific configuration fingerprints differ")
    old_hashes = {(d.get("dataset"), d.get("source_csv_sha256"), d.get("source_metadata_sha256")) for d in old_manifest.get("datasets", [])}
    target_hashes = {(d.get("dataset"), d.get("source_csv_sha256"), d.get("source_metadata_sha256")) for d in target_manifest.get("datasets", [])}
    if not target_hashes.issubset(old_hashes):
        raise ValueError("target dataset source identities are incompatible")
    # The interrupted target may have published only the first dataset's
    # manifest entry.  Fill missing entries from the validated old manifest;
    # this is metadata recovery, not a data substitution.
    by_dataset = {d.get("dataset"): d for d in target_manifest.get("datasets", [])}
    for dataset in old_manifest.get("datasets", []):
        by_dataset.setdefault(dataset.get("dataset"), dataset)
    target_manifest["datasets"] = list(by_dataset.values())

    old_rows = {row["task_key"]: row for row in _read_rows(old_dir / "results.jsonl") if row.get("status") == "success"}
    target_rows = {row["task_key"]: row for row in _read_rows(target_dir / "results.jsonl") if row.get("status") == "success"}
    overlap = sorted(set(old_rows) & set(target_rows))
    comparison_fields = ("prediction_sha256", "train_matrix_sha256", "test_matrix_sha256", "accuracy", "roc_auc", "pr_auc", "log_loss", "f1", "precision", "recall", "mcc")
    mismatches = []
    for key in overlap:
        for field in comparison_fields:
            if old_rows[key].get(field) != target_rows[key].get(field):
                mismatches.append((key, field, old_rows[key].get(field), target_rows[key].get(field)))
    if mismatches:
        raise ValueError(f"overlapping completed cells disagree: {mismatches[:3]}")

    old_tasks = {task["task_key"]: task for task in old_manifest.get("tasks", [])}
    target_tasks = {task["task_key"]: task for task in target_manifest.get("tasks", [])}
    db_path = target_dir / "checkpoints.sqlite"
    index_path = target_dir / "results_index.sqlite"
    imported = []
    for key, row in old_rows.items():
        if key in target_rows:
            continue
        task = old_tasks.get(key)
        if task is None:
            raise ValueError(f"missing task manifest for completed result {key}")
        current_task = dict(task)
        current_task["configuration_fingerprint"] = target_manifest["configuration_fingerprint"]
        current_task["code_fingerprint"] = target_manifest["code_fingerprint"]
        current_task["runtime_fingerprint"] = target_manifest["runtime_fingerprint"]
        current_fp = cache_fingerprint(current_task)
        recovered = dict(row)
        recovered["task_fingerprint"] = current_fp
        recovered["original_task_fingerprint"] = row.get("task_fingerprint")
        recovered["recovered_from_run_dir"] = str(old_dir)
        recovered["recovered_from_implementation_commit"] = old_manifest.get("code_commit")
        recovered["recovery_compatibility"] = "exact overlap hashes and metrics; same configuration and dataset identities; runner-only implementation delta"
        _safe_result_write(target_dir / "results.jsonl", recovered, index_path=index_path)
        for phase in ("phase1", "phase2"):
            record_task(
                db_path, run_id=target_manifest["run_id"], task_key=key, phase=phase,
                status="success", dataset=task["dataset"], seed=task["seed"], fold=task["fold"],
                condition=task["condition"], pipeline=task["pipeline"], model=task["model"],
                manifest_fingerprint=current_fp,
            )
        imported.append((key, current_task, recovered))

    # Re-register completed tasks and publish scheduler artifacts so the
    # durable scheduler, not a JSONL mirror, is authoritative for completion.
    scheduler = TaskScheduler(target_dir / "scheduler.sqlite", artifact_dir=target_dir / "scheduler_results", lease_seconds=3600.0)
    worker_id = f"reconcile-pid-{__import__('os').getpid()}"
    scheduler.reconcile()
    scheduler.reclaim_foreign_leases(worker_id)
    completed_for_scheduler = []
    merged_rows = {row["task_key"]: row for row in _read_rows(target_dir / "results.jsonl") if row.get("status") == "success"}
    for key, recovered in merged_rows.items():
        task = target_tasks.get(key) or old_tasks.get(key)
        if task is None:
            raise ValueError(f"missing task manifest for completed result {key}")
        completed_for_scheduler.append((key, task, recovered))
    # Register all missing task identities in one transaction.  The original
    # migration used claim/publish per cell, which is correct but needlessly
    # fsync-heavy for thousands of already validated results.
    registrations = []
    for key, task, _recovered in completed_for_scheduler:
        with sqlite3.connect(target_dir / "scheduler.sqlite") as conn:
            existing = conn.execute("SELECT spec_json, run_id, max_attempts FROM scheduler_tasks WHERE task_key=?", (key,)).fetchone()
        if existing:
            scheduler_payload = json.loads(existing[0])
            scheduler_run_id = existing[1]
            scheduler_max_attempts = int(existing[2])
        else:
            scheduler_payload = task
            scheduler_run_id = target_manifest["run_id"]
            scheduler_max_attempts = 3
        registrations.append(TaskSpec(task_key=key, run_id=scheduler_run_id, payload=scheduler_payload, max_attempts=scheduler_max_attempts))
    scheduler.register_tasks(registrations)

    # Publish in bounded transactions.  Artifacts are atomically renamed and
    # fsynced before the corresponding SQLite commit; SQLite remains the
    # authority for completion.  Re-running this block is idempotent because
    # scheduler_results.task_key is unique and successful tasks are skipped.
    db_path = target_dir / "scheduler.sqlite"
    published = 0
    for offset in range(0, len(completed_for_scheduler), 250):
        batch = completed_for_scheduler[offset:offset + 250]
        artifacts = []
        now = time.time()
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            for key, task, recovered in batch:
                row = conn.execute("SELECT status, run_id, spec_json FROM scheduler_tasks WHERE task_key=?", (key,)).fetchone()
                if row is None:
                    raise RuntimeError(f"scheduler registration missing for {key}")
                done = conn.execute("SELECT 1 FROM scheduler_results WHERE task_key=?", (key,)).fetchone()
                if done or row[0] == "success":
                    continue
                token = secrets.token_hex(16)
                artifact_path, digest = scheduler._write_artifact(
                    run_id=row[1], task_key=key, attempt_no=1,
                    lease_token=token, payload=recovered, created_at=now,
                    fault_stage=None,
                )
                artifacts.append((key, token, artifact_path, digest))
            conn.execute("BEGIN IMMEDIATE")
            try:
                for key, token, artifact_path, digest in artifacts:
                    conn.execute(
                        "INSERT OR IGNORE INTO scheduler_attempts(task_key, attempt_no, worker_id, lease_token, status, claimed_at, heartbeat_at, lease_expires_at, finished_at, retryable) VALUES (?,1,?,?,?,?,?,?,?,0)",
                        (key, worker_id, token, "success", now, now, now + scheduler.lease_seconds, now),
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO scheduler_results(task_key, attempt_no, result_digest, artifact_path, published_at, recovered) VALUES (?,1,?,?,?,1)",
                        (key, digest, str(artifact_path), now),
                    )
                    conn.execute(
                        "UPDATE scheduler_tasks SET status='success', final_result_digest=?, final_artifact_path=?, updated_at=? WHERE task_key=? AND status <> 'success'",
                        (digest, str(artifact_path), now, key),
                    )
                    conn.execute(
                        "INSERT INTO scheduler_events(event_time, task_key, attempt_no, event_type, details_json) VALUES (?,?,?,?,?)",
                        (now, key, 1, "result_published", json.dumps({"artifact_path": str(artifact_path), "result_digest": digest, "recovered": True}, sort_keys=True)),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        published += len(artifacts)

    outcomes = list(target_manifest.get("tasks", []))
    outcome_keys = {row.get("task_key") for row in outcomes}
    for key, task, recovered in completed_for_scheduler:
        if key not in outcome_keys:
            outcomes.append({**task, "status": "success", "recovered": True, "recovered_from_implementation_commit": old_manifest.get("code_commit")})
    target_manifest["recovery"] = {
        "performed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run_dir": str(old_dir),
        "source_code_commit": old_manifest.get("code_commit"),
        "target_code_commit": target_manifest.get("code_commit"),
        "source_completed_successes": len(old_rows),
        "overlap_cells_compared": len(overlap),
        "overlap_cells_matching": len(overlap) - len(mismatches),
        "imported_completed_cells": len(imported),
        "source_failed_cells_not_imported": sum(1 for row in _read_rows(old_dir / "results.jsonl") if row.get("status") == "failed"),
        "source_cache_artifacts_not_imported_as_completion": True,
    }
    _save_run_manifest(target_dir, target_manifest, outcomes)
    return target_manifest["recovery"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=Path("corrected_runs/four_dataset_pilot/pilot-row_level-001-preheartbeat-aborted"))
    parser.add_argument("--target", type=Path, default=Path("corrected_runs/four_dataset_pilot/pilot-row_level-001"))
    args = parser.parse_args()
    report = reconcile(args.old, args.target)
    out = Path("provenance/pilot_checkpoint_recovery_merge.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
