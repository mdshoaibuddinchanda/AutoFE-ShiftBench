"""Read-only accounting of both four-dataset pilot ledgers and result artifacts."""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.task_scheduler import TaskScheduler

RESULT_PARITY_FIELDS = (
    "status", "dataset", "seed", "fold", "condition", "pipeline", "model",
    "split_policy", "n_train", "n_test", "n_retained", "n_generated",
    "train_matrix_sha256", "test_matrix_sha256", "prediction_sha256",
    "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "mcc",
    "precision", "recall", "f1", "log_loss", "brier_score", "train_auc",
)


def _readonly(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def audit_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    mirror: dict[str, dict[str, Any]] = {}
    mirror_duplicates: list[str] = []
    mirror_failures: list[dict[str, Any]] = []
    with (run_dir / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "success":
                key = row["task_key"]
                if key in mirror:
                    mirror_duplicates.append(key)
                mirror[key] = row
            else:
                mirror_failures.append(row)

    with _readonly(run_dir / "scheduler.sqlite") as scheduler:
        tasks = {row["task_key"]: dict(row) for row in scheduler.execute("SELECT * FROM scheduler_tasks")}
        results = {row["task_key"]: dict(row) for row in scheduler.execute("SELECT * FROM scheduler_results")}
        attempts = [dict(row) for row in scheduler.execute("SELECT * FROM scheduler_attempts")]
    with _readonly(run_dir / "checkpoints.sqlite") as checkpoints:
        phase2 = {
            row["task_key"]: row["status"]
            for row in checkpoints.execute("SELECT task_key,status FROM task_state WHERE phase='phase2'")
        }

    first_claim_by_task: dict[str, float] = {}
    for attempt in attempts:
        key = attempt["task_key"]
        claimed = float(attempt["claimed_at"])
        first_claim_by_task[key] = min(first_claim_by_task.get(key, claimed), claimed)

    disagreements: list[dict[str, Any]] = []
    datasets: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"success": 0, "first_worker_started_unix": None, "last_worker_finished_unix": None,
                 "first_claimed_unix": None, "last_result_published_unix": None,
                 "fit_seconds": 0.0, "worker_seconds": 0.0}
    )
    model_fit_times: dict[str, list[float]] = collections.defaultdict(list)
    for key, task in tasks.items():
        status = task["status"]
        result = results.get(key)
        row = mirror.get(key)
        checkpoint_status = phase2.get(key)
        if status == "success":
            missing = [
                name for name, present in (("scheduler_result", result is not None),
                                           ("jsonl_success", row is not None),
                                           ("phase2_success", checkpoint_status == "success"))
                if not present
            ]
            if missing:
                disagreements.append({"task_key": key, "issue": "missing_success_evidence", "missing": missing})
                continue
            artifact = Path(result["artifact_path"])
            if not artifact.is_absolute():
                # Scheduler paths were recorded relative to the original
                # repository root.  Audits may run from a detached worktree.
                artifact = run_dir / "scheduler_results" / artifact.name
            try:
                envelope = TaskScheduler._read_artifact(artifact)
                if (envelope["task_key"] != key or envelope["result_digest"] != result["result_digest"]
                        or envelope["payload"].get("task_key") != key):
                    raise ValueError("result identity/digest mismatch")
            except (OSError, ValueError, KeyError) as exc:
                disagreements.append({"task_key": key, "issue": "artifact_invalid", "error": str(exc)})
                continue
            different = [field for field in RESULT_PARITY_FIELDS
                         if envelope["payload"].get(field) != row.get(field)]
            if different:
                disagreements.append({"task_key": key, "issue": "jsonl_artifact_value_mismatch", "fields": different})
            dataset = datasets[row["dataset"]]
            dataset["success"] += 1
            claimed = first_claim_by_task.get(key)
            published = float(result["published_at"])
            if claimed is not None:
                dataset["first_claimed_unix"] = (
                    claimed if dataset["first_claimed_unix"] is None
                    else min(claimed, dataset["first_claimed_unix"])
                )
            dataset["last_result_published_unix"] = (
                published if dataset["last_result_published_unix"] is None
                else max(published, dataset["last_result_published_unix"])
            )
            start = float(row.get("worker_started_unix") or 0)
            end = float(row.get("worker_finished_unix") or 0)
            dataset["first_worker_started_unix"] = (
                start if dataset["first_worker_started_unix"] is None
                else min(start, dataset["first_worker_started_unix"])
            )
            dataset["last_worker_finished_unix"] = (
                end if dataset["last_worker_finished_unix"] is None
                else max(end, dataset["last_worker_finished_unix"])
            )
            dataset["fit_seconds"] += float(row.get("train_time_s") or 0)
            dataset["worker_seconds"] += float(row.get("worker_elapsed_s") or 0)
            model_fit_times[row["model"]].append(float(row.get("train_time_s") or 0))
        elif result is not None or row is not None or checkpoint_status == "success":
            disagreements.append({"task_key": key, "issue": "non_success_has_success_evidence", "status": status})

    for key in set(results) | set(mirror) | set(phase2):
        if key not in tasks:
            disagreements.append({"task_key": key, "issue": "unregistered_evidence"})
    for key in mirror_duplicates:
        disagreements.append({"task_key": key, "issue": "duplicate_jsonl_success"})
    if manifest["status"] == "complete":
        manifest_successes = manifest["counts_by_status"].get("success", 0)
        if manifest_successes != len(results) or manifest.get("expected_tasks") not in (None, len(tasks)):
            disagreements.append({
                "task_key": None, "issue": "final_manifest_count_mismatch",
                "manifest_successes": manifest_successes,
                "manifest_expected_tasks": manifest.get("expected_tasks"),
                "registered_tasks": len(tasks), "scheduler_results": len(results),
            })

    for value in datasets.values():
        first = value["first_worker_started_unix"]
        last = value["last_worker_finished_unix"]
        value["worker_span_seconds"] = last - first if first is not None and last is not None else None
        claimed = value["first_claimed_unix"]
        published = value["last_result_published_unix"]
        value["scheduler_wall_span_seconds"] = (
            published - claimed if claimed is not None and published is not None else None
        )

    first_claimed = min(first_claim_by_task.values(), default=None)
    last_published = max((float(result["published_at"]) for result in results.values()), default=None)

    audit = manifest.get("cache_audit", {}).get("artifacts", {})
    logical_groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in audit.values():
        logical_groups[(record["feature_task_key"], record["pipeline"])].append(record)
    cache = {
        "audit_identities": len(audit),
        "logical_feature_groups": len(logical_groups),
        "builds": sum(record["build_count"] for record in audit.values()),
        "hits": sum(record["hit_count"] for record in audit.values()),
        "extra_builds_across_identity_migrations": sum(
            max(0, sum(record["build_count"] for record in group) - 1)
            for group in logical_groups.values()
        ),
        "deletion_observed": sum(bool(record["deletion_observed"]) for record in audit.values()),
        "remaining_payload_files": len(list((run_dir / "cache_bounded").glob("*.payload"))),
    }
    attempt_statuses = collections.Counter((row["status"], row["error_type"]) for row in attempts)
    success_attempts = collections.Counter(row["task_key"] for row in attempts if row["status"] == "success")
    model_timings = {
        model: {
            "count": len(times),
            "fit_seconds": sum(times),
            "median_fit_seconds": statistics.median(times),
            "p90_fit_seconds": sorted(times)[min(len(times) - 1, int(0.9 * len(times)))],
        }
        for model, times in model_fit_times.items()
    }
    return {
        "run_id": manifest["run_id"],
        "manifest_status": manifest["status"],
        "manifest_counts_by_status": manifest["counts_by_status"],
        "scheduler_task_statuses": dict(collections.Counter(row["status"] for row in tasks.values())),
        "scheduler_results": len(results),
        "scheduler_first_claimed_unix": first_claimed,
        "scheduler_last_result_published_unix": last_published,
        "scheduler_wall_span_seconds": (
            last_published - first_claimed if first_claimed is not None and last_published is not None else None
        ),
        "phase2_successes": sum(status == "success" for status in phase2.values()),
        "jsonl_successes": len(mirror),
        "jsonl_failed_attempt_rows": len(mirror_failures),
        "attempt_statuses": {f"{status}:{error or 'none'}": count
                             for (status, error), count in attempt_statuses.items()},
        "duplicate_successful_attempt_tasks": sorted(key for key, count in success_attempts.items() if count > 1),
        "disagreements": disagreements,
        "datasets": dict(datasets),
        "models": model_timings,
        "cache": cache,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing pilot-row_level-001 and pilot-group_aware-001")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path")
    args = parser.parse_args()
    report = {
        policy: audit_run(args.root / f"pilot-{policy}-001")
        for policy in ("row_level", "group_aware")
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "summary": {policy: {
                "scheduler_results": row["scheduler_results"],
                "phase2_successes": row["phase2_successes"],
                "jsonl_successes": row["jsonl_successes"],
                "disagreements": len(row["disagreements"]),
            } for policy, row in report.items()},
        }, sort_keys=True))
    else:
        print(encoded)


if __name__ == "__main__":
    main()
