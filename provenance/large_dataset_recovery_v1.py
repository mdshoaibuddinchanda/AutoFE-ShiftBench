"""Bounded forced-restart proof on airlines under the frozen 600-second lease.

This is diagnostic code verification, not a corrected performance run. The
coordinator is deliberately terminated after committed successes, then the
identical run ID/configuration is resumed and its retained cells are checked.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "corrected_runs" / "large_recovery"
REPORT = ROOT / "provenance" / "large_dataset_recovery_v1.json"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}
MODELS = (
    "logistic_regression", "random_forest", "extra_trees", "linear_svm", "knn",
    "gaussian_nb", "mlp", "lightgbm", "xgboost", "catboost",
)
EXPECTED = 2 * len(MODELS)


def _worker(policy: str, run_id: str) -> None:
    for name, value in THREAD_ENV.items():
        os.environ[name] = value
    from src.pipeline_runner import run_experiment

    manifest = run_experiment(
        data_paths={"airlines": ROOT / "data" / "raw" / "airlines.csv"},
        output_root=OUTPUT_ROOT, run_id=run_id,
        seeds=[42], folds=[1], conditions=[("clean", 0.0)], n_splits=5,
        pipelines=("Raw", "AutoFE_Baseline"), models=MODELS,
        split_policy=policy, cache_policy="bounded", cache_max_bytes=8 * 1024**3,
        durable_scheduler=True, cache_audit=True, scheduler_lease_seconds=600.0,
        workers=4, use_gpu=False, manifest_policy="detailed",
    )
    print(json.dumps({"run_id": run_id, "status": manifest["status"],
                      "counts": manifest["counts_by_status"]}))


def _start_worker(policy: str, run_id: str, phase: str) -> subprocess.Popen:
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    command = [sys.executable, "-m", "provenance.large_dataset_recovery_v1",
               "--worker", "--policy", policy, "--run-id", run_id]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT / f"{run_id}.{phase}.coordinator.log"
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )


def _stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.kill()
    except (NameError, psutil.NoSuchProcess):
        pass
    process.wait(timeout=30)


def _manifest(run_dir: Path) -> dict | None:
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _successes(run_dir: Path) -> dict[str, tuple[str, str, str]]:
    path = run_dir / "results.jsonl"
    if not path.exists():
        return {}
    successes = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "success":
                key = str(row["task_key"])
                identity = tuple(str(row[field]) for field in (
                    "prediction_sha256", "train_matrix_sha256", "test_matrix_sha256"))
                if key in successes and successes[key] != identity:
                    raise ValueError(f"Conflicting successful result for task: {key}")
                successes[key] = identity
    return successes


def _wait_for_successes(process: subprocess.Popen, run_dir: Path, *, timeout_s: float = 3600) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        manifest = _manifest(run_dir)
        if manifest and 1 <= manifest.get("counts_by_status", {}).get("success", 0) < EXPECTED:
            return manifest
        if process.poll() is not None:
            raise RuntimeError(f"Recovery worker exited before forced restart: {process.returncode}")
        time.sleep(2)
    raise TimeoutError("No committed success appeared before the forced-restart deadline")


def _check_terminal(run_dir: Path, before: dict[str, tuple[str, str, str]],
                    source_fingerprint: str) -> dict:
    manifest = _manifest(run_dir)
    if not manifest or manifest.get("status") != "complete":
        raise ValueError(f"Resumed recovery run did not complete: {run_dir}")
    counts = manifest["counts_by_status"]
    if (manifest.get("expected_tasks") != EXPECTED or counts.get("success") != EXPECTED
            or any(counts.get(status, 0) for status in ("failed", "skipped", "timed_out", "pending"))
            or manifest.get("code_fingerprint") != source_fingerprint
            or manifest.get("configuration", {}).get("scheduler_lease_seconds") != 600.0
            or manifest.get("configuration", {}).get("numerical_thread_environment") != THREAD_ENV):
        raise ValueError(f"Recovered run differs from frozen recovery contract: {run_dir}")
    after = _successes(run_dir)
    if len(after) != EXPECTED or any(after.get(key) != identity for key, identity in before.items()):
        raise ValueError("A committed pre-crash result was lost or refit with different output")
    with sqlite3.connect(run_dir / "scheduler.sqlite") as connection:
        rows = connection.execute(
            "SELECT task_key, COUNT(*) FROM scheduler_attempts GROUP BY task_key"
        ).fetchall()
        attempts = dict(rows)
        statuses = dict(connection.execute(
            "SELECT status, COUNT(*) FROM scheduler_tasks GROUP BY status"
        ).fetchall())
    if statuses != {"success": EXPECTED} or any(attempts.get(key) != 1 for key in before):
        raise ValueError("Scheduler refit a committed pre-crash cell or lost terminal coverage")
    return {
        "run_id": run_dir.name, "status": "passed",
        "committed_successes_before_kill": len(before),
        "terminal_successes_after_resume": len(after),
        "pre_crash_successes_retained_without_refit": True,
        "scheduler_statuses": statuses,
        "run_manifest": str((run_dir / "manifest.json").relative_to(ROOT)).replace("\\", "/"),
    }


def _prove_policy(policy: str, source_fingerprint: str, version: str) -> dict:
    suffix = "row" if policy == "row_level" else "group"
    run_id = f"recovery-airlines-{suffix}-{version}"
    run_dir = OUTPUT_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"Preserve existing recovery evidence; use a new versioned ID: {run_dir}")
    started = time.monotonic()
    first = _start_worker(policy, run_id, "forced_stop")
    try:
        _wait_for_successes(first, run_dir)
    finally:
        _stop_process_tree(first)
    before = _successes(run_dir)
    if not before or len(before) >= EXPECTED:
        raise ValueError("Forced restart did not interrupt an in-progress bounded run")
    resumed = _start_worker(policy, run_id, "resume")
    try:
        resumed.wait(timeout=7200)
        if resumed.returncode:
            raise RuntimeError(f"Recovery worker failed with exit code {resumed.returncode}: {run_id}")
    finally:
        _stop_process_tree(resumed)
    result = _check_terminal(run_dir, before, source_fingerprint)
    result["elapsed_s_including_forced_restart"] = round(time.monotonic() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=("row_level", "group_aware"), help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--version", default="001", help="Three-digit diagnostic ID version")
    args = parser.parse_args()
    if len(args.version) != 3 or not args.version.isdecimal():
        parser.error("--version must be three decimal digits")
    if args.worker:
        if not args.policy or not args.run_id:
            parser.error("worker needs policy and run ID")
        _worker(args.policy, args.run_id)
        return
    if Path(sys.executable).resolve() != Path(r"D:\Conda\p12\python.exe").resolve():
        raise RuntimeError("Use the existing p12 interpreter")
    for name, value in THREAD_ENV.items():
        os.environ[name] = value
    from src.provenance import code_fingerprint, file_sha256

    scope = json.loads((ROOT / "provenance" / "reviewer1_launch_scope_v2.json").read_text(encoding="utf-8"))
    source_fingerprint = code_fingerprint(ROOT)
    if source_fingerprint != scope["code_fingerprint"]:
        raise RuntimeError("Frozen source fingerprint changed before recovery proof")
    input_path = ROOT / "data" / "raw" / "airlines.csv"
    sidecar_path = ROOT / "data" / "raw" / "airlines_meta.json"
    source_row = next(item for item in scope["datasets"] if item["name"] == "airlines")
    if (file_sha256(input_path) != source_row["csv_sha256"]
            or file_sha256(sidecar_path) != source_row["sidecar_sha256"]):
        raise RuntimeError("Recovery input changed from the frozen dataset")
    results = [_prove_policy(policy, source_fingerprint, args.version)
               for policy in ("row_level", "group_aware")]
    report = {
        "artifact_type": "large_dataset_forced_restart_v1",
        "status": "passed", "scientific_use": "diagnostic recovery proof only",
        "host": platform.node(), "python_executable": str(Path(sys.executable).resolve()),
        "code_fingerprint": source_fingerprint,
        "dataset": "airlines", "csv_sha256": source_row["csv_sha256"],
        "sidecar_sha256": source_row["sidecar_sha256"],
        "numerical_thread_environment": THREAD_ENV,
        "scheduler_lease_seconds": 600, "workers": 4,
        "task_design": "one seed, one fold, clean, Raw and AutoFE_Baseline, ten classifiers",
        "runs": results,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
