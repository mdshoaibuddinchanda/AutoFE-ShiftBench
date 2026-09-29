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
    unique_tasks = {
        (row.get("dataset"), row.get("seed"), row.get("fold"), row.get("condition"),
         row.get("pipeline"), row.get("model")): row.get("status")
        for row in manifest.get("tasks", [])
    }
    task_counts = {
        "success": sum(status == "success" for status in unique_tasks.values()),
        "failed": sum(status == "failed" for status in unique_tasks.values()),
    }
    task_counts["remaining"] = max(expected - len(unique_tasks), 0)

    phases: dict[str, dict[str, int]] = {}
    db_path = root / "checkpoints.sqlite"
    if db_path.exists():
        with sqlite3.connect(db_path) as connection:
            for phase, status, count in connection.execute(
                "SELECT phase, status, COUNT(*) FROM task_state GROUP BY phase, status"
            ):
                phases.setdefault(phase, {})[status] = count

    results_path = root / "results.jsonl"
    result_rows = 0
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as stream:
            result_rows = sum(bool(line.strip()) for line in stream)

    summary = {
        "run_id": manifest.get("run_id"), "status": manifest.get("status"),
        "experiment_scope": manifest.get("experiment_scope"),
        "expected_tasks": expected, "task_counts": task_counts,
        "phase_states": phases, "result_rows": result_rows,
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
