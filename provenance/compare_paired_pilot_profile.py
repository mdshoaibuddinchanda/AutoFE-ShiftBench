"""Compare two bounded pilot profiles on identical logical cells and outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = ("dataset", "seed", "fold", "condition", "pipeline", "model", "split_policy")
EXACT_FIELDS = (
    "train_matrix_sha256", "test_matrix_sha256", "prediction_sha256",
    "n_train", "n_test", "n_retained", "n_generated", "status",
)
METRIC_FIELDS = (
    "roc_auc", "pr_auc", "f1", "accuracy", "balanced_accuracy", "precision",
    "recall", "mcc", "brier_score", "log_loss", "train_auc",
)


def _success_rows(run_dir: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    with (run_dir / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "success":
                continue
            identity = tuple(row.get(field) for field in IDENTITY_FIELDS)
            if identity in rows:
                raise ValueError(f"duplicate successful cell in {run_dir}: {identity}")
            rows[identity] = row
    return rows


def compare(baseline: Path, optimized: Path) -> dict[str, Any]:
    before = _success_rows(baseline)
    after = _success_rows(optimized)
    baseline_profile = json.loads((baseline / "paired_profile_summary.json").read_text(encoding="utf-8"))
    optimized_profile = json.loads((optimized / "paired_profile_summary.json").read_text(encoding="utf-8"))
    for field in ("configuration", "dataset_sha256", "python"):
        if baseline_profile.get(field) != optimized_profile.get(field):
            raise ValueError(f"paired profiles differ in {field}")
    mismatches: list[dict[str, Any]] = []
    for identity in sorted(before.keys() & after.keys()):
        left, right = before[identity], after[identity]
        for field in EXACT_FIELDS:
            if left.get(field) != right.get(field):
                mismatches.append({"cell": identity, "field": field,
                                   "baseline": left.get(field), "optimized": right.get(field)})
        for field in METRIC_FIELDS:
            a, b = left.get(field), right.get(field)
            if a is None or b is None:
                equal = a is None and b is None
            else:
                equal = math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
            if not equal:
                mismatches.append({"cell": identity, "field": field,
                                   "baseline": a, "optimized": b})
    return {
        "baseline_run_id": baseline_profile["run_id"],
        "optimized_run_id": optimized_profile["run_id"],
        "baseline_code_fingerprint": baseline_profile.get("code_fingerprint"),
        "optimized_code_fingerprint": optimized_profile.get("code_fingerprint"),
        "configuration_equal": True,
        "dataset_sha256_equal": True,
        "baseline_cells": len(before),
        "optimized_cells": len(after),
        "matched_cells": len(before.keys() & after.keys()),
        "baseline_only": [list(key) for key in sorted(before.keys() - after.keys())],
        "optimized_only": [list(key) for key in sorted(after.keys() - before.keys())],
        "output_mismatches": mismatches,
        "baseline_elapsed_s": baseline_profile["elapsed_s"],
        "optimized_elapsed_s": optimized_profile["elapsed_s"],
        "wall_speedup": baseline_profile["elapsed_s"] / optimized_profile["elapsed_s"],
        "baseline_valid_cells_per_hour": baseline_profile["valid_cells_per_hour"],
        "optimized_valid_cells_per_hour": optimized_profile["valid_cells_per_hour"],
        "baseline_peak_process_tree_rss_mib": baseline_profile["peak_process_tree_rss_mib"],
        "optimized_peak_process_tree_rss_mib": optimized_profile["peak_process_tree_rss_mib"],
        "baseline_peak_system_gpu_used_mib": baseline_profile["peak_system_gpu_used_mib"],
        "optimized_peak_system_gpu_used_mib": optimized_profile["peak_system_gpu_used_mib"],
        "baseline_cache_builds": baseline_profile["cache_builds"],
        "optimized_cache_builds": optimized_profile["cache_builds"],
        "baseline_cache_hits": baseline_profile["cache_hits"],
        "optimized_cache_hits": optimized_profile["cache_hits"],
        "baseline_cache_deletions": baseline_profile["cache_deletions"],
        "optimized_cache_deletions": optimized_profile["cache_deletions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.optimized)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
