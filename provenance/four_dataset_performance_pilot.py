"""Run the bounded four-dataset pilot through the integrated benchmark runner.

This script is intentionally separate from the frozen full-grid launcher.  It
uses the pre-specified four smallest group-AUC-eligible datasets, one seed and
fold, all ten primary conditions, the frozen fourteen pipelines, and all ten
classifiers under each split policy.  Results stay under ``corrected_runs/``
and are disposable pilot evidence, not corrected scientific estimates.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline_runner import PIPELINE_CONFIGS, PRIMARY_CONDITIONS, run_experiment
from src.provenance import file_sha256


DATA = ROOT / "data" / "raw"
OUT = ROOT / "corrected_runs" / "four_dataset_pilot"
DATASETS = ("sonar", "heart-disease", "haberman", "ionosphere")
PIPELINES = (
    "Raw", "Raw_CapMatched", "AutoFE_MI", "AutoFE_Random", "AutoFE_Baseline",
    "AutoFE_NoMultiply", "AutoFE_Isolate_Add", "AutoFE_Isolate_Subtract",
    "AutoFE_Isolate_Multiply", "AutoFE_Isolate_Divide", "AutoFE_LeaveOut_Add",
    "AutoFE_LeaveOut_Subtract", "AutoFE_LeaveOut_Multiply", "AutoFE_LeaveOut_Divide",
)
MODELS = (
    "logistic_regression", "random_forest", "extra_trees", "linear_svm", "knn",
    "gaussian_nb", "mlp", "lightgbm", "xgboost", "catboost",
)


def _status(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    success = [row for row in rows if row.get("status") == "success"]
    by_dataset = {}
    for dataset in DATASETS:
        subset = [row for row in success if row.get("dataset") == dataset]
        by_dataset[dataset] = {
            "completed": len(subset),
            "train_time_s": sum(float(row.get("train_time_s", 0.0)) for row in subset),
            "infer_time_s": sum(float(row.get("infer_time_s", 0.0)) for row in subset),
            "autofe_generation_s": sum(float(row.get("autofe_gen_time_s", 0.0)) for row in subset),
        }
    intervals = [(row.get("worker_started_unix"), row.get("worker_finished_unix")) for row in success]
    intervals = [(float(a), float(b)) for a, b in intervals if a is not None and b is not None]
    overlap_pairs = sum(
        1 for i, (a0, a1) in enumerate(intervals)
        for b0, b1 in intervals[i + 1:] if a0 < b1 and b0 < a1
    )
    audit = manifest.get("cache_audit", {}).get("artifacts", {})
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "counts": manifest.get("counts_by_status", {}),
        "code_commit": manifest.get("code_commit"),
        "configuration_fingerprint": manifest.get("configuration_fingerprint"),
        "started_utc": manifest.get("created_utc"),
        "completed_rows": len(success),
        "by_dataset": by_dataset,
        "worker_pids": sorted({row.get("worker_pid") for row in success if row.get("worker_pid")}),
        "overlapping_worker_interval_pairs": overlap_pairs,
        "cache_groups": len(audit),
        "cache_builds": sum(int(row.get("build_count", 0)) for row in audit.values()),
        "cache_hits": sum(int(row.get("hit_count", 0)) for row in audit.values()),
        "cache_deletions": sum(1 for row in audit.values() if row.get("deletion_observed")),
    }


def main() -> None:
    if len(PIPELINES) != 14 or set(PIPELINES) != {
        name for name in PIPELINE_CONFIGS
        if name in set(PIPELINES)
    }:
        raise RuntimeError("pilot pipeline list does not match the frozen fourteen-pipeline set")
    data_paths = {name: DATA / f"{name}.csv" for name in DATASETS}
    missing = [str(path) for path in data_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    identities = {}
    for name, path in data_paths.items():
        sidecar = path.with_name(f"{path.stem}_meta.json")
        identities[name] = {
            "csv_sha256": file_sha256(path),
            "sidecar_sha256": file_sha256(sidecar) if sidecar.exists() else None,
            "rows_bytes": path.stat().st_size,
        }
    selection = {
        "rule": "From configured datasets with <=1000 rows, require canonical group split, class support, and group-AUC support at seed 42; sort by (row_count, dataset_name) and select first four. No pilot scores were inspected.",
        "datasets": list(DATASETS),
        "identities": identities,
        "seed": 42,
        "fold": 1,
        "conditions": [list(item) for item in PRIMARY_CONDITIONS],
        "pipelines": list(PIPELINES),
        "models": list(MODELS),
        "intended_cells_per_policy": len(DATASETS) * len(PRIMARY_CONDITIONS) * len(PIPELINES) * len(MODELS),
        "intended_cells_both_policies": 2 * len(DATASETS) * len(PRIMARY_CONDITIONS) * len(PIPELINES) * len(MODELS),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    started = time.time()
    runs = []
    common = dict(
        data_paths=data_paths,
        output_root=OUT,
        seeds=[42], folds=[1], conditions=PRIMARY_CONDITIONS,
        pipelines=PIPELINES, models=MODELS, n_splits=5,
        cache_policy="bounded", cache_max_bytes=8 * 1024**3,
        durable_scheduler=True, cache_audit=True,
        scheduler_lease_seconds=3600.0, scheduler_max_attempts=3,
        workers=4, use_gpu=False,
    )
    for split_policy in ("row_level", "group_aware"):
        run_id = f"pilot-{split_policy}-001"
        run_started = time.time()
        run_experiment(**common, run_id=run_id, split_policy=split_policy)
        run_dir = OUT / run_id
        entry = _status(run_dir)
        entry.update({"split_policy": split_policy, "elapsed_s": time.time() - run_started})
        runs.append(entry)
    report = {
        "artifact_type": "four_dataset_performance_pilot",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"hostname": os.environ.get("COMPUTERNAME"), "pid": os.getpid(), "python": os.sys.executable},
        "selection": selection,
        "runs": runs,
        "elapsed_s_total": time.time() - started,
        "scientific_status": "pilot execution and timing evidence only; corrected performance effects remain PENDING CORRECTED RUN",
    }
    (ROOT / "provenance" / "four_dataset_performance_pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Four-dataset performance pilot",
        "",
        "This is bounded execution and timing evidence; corrected performance effects remain **PENDING CORRECTED RUN**.",
        "",
        f"- Datasets: {', '.join(DATASETS)}",
        "- Selection: smallest four datasets meeting canonical group split, class support, and group-AUC support at seed 42; no scores inspected.",
        f"- Cells: {selection['intended_cells_per_policy']:,} per split policy; {selection['intended_cells_both_policies']:,} both policies.",
        f"- Outer workers: 4; seed/fold: 42/1; cache cap: 8 GiB.",
        "",
        "## Run results",
        "",
        "| Split policy | Status | Completed | Failed | Pending | Elapsed (s) | Worker PIDs | Overlap pairs | Cache builds | Cache hits | Deletions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in runs:
        counts = entry.get("counts", {})
        lines.append(
            f"| {entry.get('split_policy')} | {entry.get('status')} | {entry.get('completed_rows', 0):,} | {counts.get('failed', 0):,} | {counts.get('pending', 0):,} | {entry.get('elapsed_s', 0.0):.1f} | {len(entry.get('worker_pids', []))} | {entry.get('overlapping_worker_interval_pairs', 0):,} | {entry.get('cache_builds', 0):,} | {entry.get('cache_hits', 0):,} | {entry.get('cache_deletions', 0):,} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "The pilot uses one seed and one fold for coverage and timing. It does not establish corrected ROC-AUC effects, Jacobian associations, or full-grid duration. Large-dataset calibration and friend-PC measurements are separate launch gates.",
    ]
    (ROOT / "provenance" / "four_dataset_performance_pilot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
