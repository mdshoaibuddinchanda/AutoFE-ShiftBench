"""Summarize one corrected, run-scoped benchmark directory."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def check_progress(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Corrected run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = manifest.get("configuration", {})
    expected = (
        len(configuration.get("datasets", []))
        * len(configuration.get("seeds", []))
        * len(configuration.get("folds", []))
        * len(configuration.get("conditions", []))
        * len(configuration.get("pipelines", []))
        * len(configuration.get("models", []))
    )
    ledger = manifest.get("task_ledger")
    if ledger:
        if ledger.get("format") != "sqlite" or ledger.get("path") != "manifest_outcomes.sqlite":
            raise ValueError("Unsupported task ledger declared by manifest")
        with sqlite3.connect(root / ledger["path"]) as connection:
            status_counts = {
                str(status): int(count) for status, count in connection.execute(
                    "SELECT status, COUNT(*) FROM outcomes GROUP BY status"
                )
            }
        recorded = sum(status_counts.values())
    else:
        unique_tasks = {
            (row.get("dataset"), row.get("seed"), row.get("fold"), row.get("condition"),
             row.get("pipeline"), row.get("model")): row.get("status")
            for row in manifest.get("tasks", [])
        }
        status_counts = {
            status: sum(value == status for value in unique_tasks.values())
            for status in ("success", "failed", "skipped", "timed_out")
        }
        recorded = len(unique_tasks)
    task_counts = {name: status_counts.get(name, 0) for name in ("success", "failed", "skipped", "timed_out")}
    task_counts["remaining"] = max(expected - recorded, 0)

    phases: dict[str, dict[str, int]] = {}
    scheduler_states: dict[str, int] = {}
    scheduler_recovered = 0
    db_path = root / "checkpoints.sqlite"
    if db_path.exists():
        with sqlite3.connect(db_path) as connection:
            for phase, status, count in connection.execute(
                "SELECT phase, status, COUNT(*) FROM task_state GROUP BY phase, status"
            ):
                phases.setdefault(phase, {})[status] = count
    scheduler_path = root / "scheduler.sqlite"
    if scheduler_path.exists():
        with sqlite3.connect(scheduler_path) as connection:
            for status, count in connection.execute("SELECT status, COUNT(*) FROM scheduler_tasks GROUP BY status"):
                scheduler_states[str(status)] = int(count)
            row = connection.execute("SELECT COUNT(*) FROM scheduler_results WHERE recovered=1").fetchone()
            scheduler_recovered = int(row[0] if row else 0)

    results_path = root / "results.jsonl"
    result_rows = 0
    result_index = root / "results_index.sqlite"
    if result_index.exists():
        with sqlite3.connect(result_index) as connection:
            result_rows = int(connection.execute("SELECT COUNT(*) FROM result_keys").fetchone()[0])
    elif results_path.exists():
        with results_path.open("r", encoding="utf-8") as stream:
            result_rows = sum(bool(line.strip()) for line in stream)

    live_status = manifest.get("status")
    if ledger and not str(live_status).startswith("blocked_"):
        if task_counts["remaining"] == 0:
            live_status = "completed_with_failures" if task_counts["failed"] else "complete"
        else:
            live_status = "running_with_failures" if task_counts["failed"] else "running"
    summary = {
        "run_id": manifest.get("run_id"), "status": live_status,
        "experiment_scope": manifest.get("experiment_scope"),
        "expected_tasks": expected, "task_counts": task_counts,
        "phase_states": phases, "scheduler_states": scheduler_states,
        "scheduler_recovered_results": scheduler_recovered,
        "result_rows": result_rows,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Path to corrected_runs/<run-id>; historical caches are not read")
    args = parser.parse_args()
    check_progress(args.run_dir)


if __name__ == "__main__":
    main()
