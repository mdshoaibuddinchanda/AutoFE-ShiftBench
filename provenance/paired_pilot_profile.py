"""Run one bounded, paired pilot workload from the current working tree.

Run baseline mode with the post-retry source retaining both original global
scans as the working directory, then run local_claim, local_cache, and both
modes with the staged source.  All four
invocations use the same absolute data path, grid, P12 interpreter, and host;
run IDs and output directories must differ.
The script has a Windows spawn guard and never touches the 5,600-cell pilot IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import psutil

from src.cache_manager import CacheManager
from src.pipeline_runner import PIPELINE_CONFIGS, run_experiment
from src.task_scheduler import TaskScheduler


PILOT_PIPELINES = (
    "Raw", "Raw_CapMatched", "AutoFE_MI", "AutoFE_Random",
    "AutoFE_Baseline", "AutoFE_NoMultiply", "AutoFE_Isolate_Add",
    "AutoFE_Isolate_Subtract", "AutoFE_Isolate_Multiply",
    "AutoFE_Isolate_Divide", "AutoFE_LeaveOut_Add",
    "AutoFE_LeaveOut_Subtract", "AutoFE_LeaveOut_Multiply",
    "AutoFE_LeaveOut_Divide",
)
PILOT_MODELS = (
    "logistic_regression", "random_forest", "extra_trees", "linear_svm",
    "knn", "gaussian_nb", "mlp", "lightgbm", "xgboost", "catboost",
)


def _gpu_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return int(completed.stdout.splitlines()[0].strip()) if completed.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--datasets", nargs="+", default=[
        "sonar", "heart-disease", "haberman", "ionosphere",
    ])
    parser.add_argument(
        "--mode", choices=("baseline", "local_claim", "local_cache", "both"), required=True,
        help="Select the post-retry baseline or staged optimizations to measure",
    )
    args = parser.parse_args()
    if any(name not in PIPELINE_CONFIGS for name in PILOT_PIPELINES):
        raise RuntimeError("The frozen fourteen-pipeline configuration is unavailable")
    data_paths = {name: args.data_root / f"{name}.csv" for name in args.datasets}
    if any(not path.exists() for path in data_paths.values()):
        raise FileNotFoundError(data_paths)
    if args.mode == "baseline" and hasattr(CacheManager, "active_lease_count"):
        raise RuntimeError("baseline mode requires the original global-scan working tree")
    if args.mode != "baseline" and not hasattr(CacheManager, "active_lease_count"):
        raise RuntimeError("staged modes require the repaired working tree")
    if args.mode == "local_claim":
        # Keep the staged task-local claim, but restore the historical global
        # cache audit for a one-change comparison.
        CacheManager.active_lease_count = lambda self, key: self.reconcile()["active_lease_count"]
    elif args.mode == "local_cache":
        # Keep the staged artifact-local reader audit, but restore global
        # scheduler recovery on every claim for a one-change comparison.
        full_reconcile = TaskScheduler.reconcile
        TaskScheduler.reconcile = lambda self, **kwargs: full_reconcile(
            self, now=kwargs.get("now"), temp_max_age_seconds=kwargs.get("temp_max_age_seconds", 3600.0),
        )

    stop = threading.Event()
    peaks = {"rss_bytes": 0, "system_gpu_used_mib": 0, "processes": 0}

    def monitor() -> None:
        parent = psutil.Process(os.getpid())
        next_gpu = 0.0
        while not stop.wait(0.1):
            processes = [parent, *parent.children(recursive=True)]
            resident = 0
            for process in processes:
                try:
                    resident += process.memory_info().rss
                except psutil.Error:
                    continue
            peaks["rss_bytes"] = max(peaks["rss_bytes"], resident)
            peaks["processes"] = max(peaks["processes"], len(processes))
            if time.monotonic() >= next_gpu:
                gpu_used = _gpu_used_mib()
                if gpu_used is not None:
                    peaks["system_gpu_used_mib"] = max(peaks["system_gpu_used_mib"], gpu_used)
                next_gpu = time.monotonic() + 5.0

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.perf_counter()
    try:
        manifest = run_experiment(
            data_paths, run_id=args.run_id, output_root=args.output_root,
            seeds=[42], folds=[1], conditions=(("clean", 0.0),),
            pipelines=PILOT_PIPELINES, models=PILOT_MODELS, n_splits=5,
            split_policy="group_aware", cache_policy="bounded",
            cache_max_bytes=8 * 1024**3, durable_scheduler=True,
            cache_audit=True, scheduler_lease_seconds=600.0,
            scheduler_max_attempts=3, workers=4, use_gpu=False,
        )
    finally:
        stop.set()
        monitor_thread.join(timeout=4)
    elapsed = time.perf_counter() - started
    audit = manifest["cache_audit"]["artifacts"]
    report = {
        "run_id": args.run_id,
        "mode": args.mode,
        "source_root": str(Path.cwd()),
        "python": sys.executable,
        "code_commit": manifest["code_commit"],
        "code_fingerprint": manifest["code_fingerprint"],
        "configuration": manifest["configuration"],
        "datasets": list(data_paths),
        "data_paths": {key: str(value) for key, value in data_paths.items()},
        "dataset_sha256": {
            record["dataset"]: record["dataset_identity"]["saved_csv_sha256"]
            for record in manifest["datasets"]
        },
        "status": manifest["status"],
        "counts": manifest["counts_by_status"],
        "elapsed_s": elapsed,
        "valid_cells_per_hour": manifest["counts_by_status"]["success"] * 3600 / elapsed,
        "peak_process_tree_rss_mib": peaks["rss_bytes"] / 2**20,
        "peak_system_gpu_used_mib": peaks["system_gpu_used_mib"],
        "peak_process_count": peaks["processes"],
        "cache_builds": sum(row["build_count"] for row in audit.values()),
        "cache_hits": sum(row["hit_count"] for row in audit.values()),
        "cache_deletions": sum(bool(row["deletion_observed"]) for row in audit.values()),
    }
    report_path = args.output_root / args.run_id / "paired_profile_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
