"""Clean group-aware slow-tail calibration after the concurrent-memory incident.

The completed row-level -002 calibration is retained. The interrupted
group-aware -002 directory remains a failed diagnostic and is never resumed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}
for name, value in THREAD_ENV.items():
    os.environ[name] = value

from src.pipeline_runner import run_experiment
from src.provenance import code_fingerprint, file_sha256


DATASETS = ("airlines", "covertype")
PIPELINES = ("Raw", "AutoFE_Baseline", "AutoFE_LeaveOut_Multiply")
MODELS = (
    "logistic_regression", "random_forest", "extra_trees", "linear_svm", "knn",
    "gaussian_nb", "mlp", "lightgbm", "xgboost", "catboost",
)
CONDITIONS = (("clean", 0.0), ("gaussian_noise", 0.10))
RUN_ID = "calibration-group_aware-003"


def main() -> None:
    if Path(sys.executable).resolve() != Path(r"D:\Conda\p12\python.exe").resolve():
        raise RuntimeError("Use the existing p12 interpreter")
    frozen = json.loads((ROOT / "provenance" / "reviewer1_launch_scope_v2.json").read_text(encoding="utf-8"))
    if code_fingerprint(ROOT) != frozen["code_fingerprint"]:
        raise RuntimeError("Frozen source fingerprint differs from clean calibration")
    row_path = ROOT / "corrected_runs" / "large_calibration" / "calibration-row_level-002" / "manifest.json"
    row = json.loads(row_path.read_text(encoding="utf-8"))
    if (row.get("status") != "complete" or row.get("expected_tasks") != 120
            or row.get("counts_by_status", {}).get("success") != 120
            or row.get("code_fingerprint") != frozen["code_fingerprint"]):
        raise RuntimeError("The row-level -002 calibration must be complete before group -003")
    paths = {name: ROOT / "data" / "raw" / f"{name}.csv" for name in DATASETS}
    hashes = {name: {"csv_sha256": file_sha256(path),
                     "sidecar_sha256": file_sha256(path.with_name(f"{name}_meta.json"))}
              for name, path in paths.items()}
    for item in frozen["datasets"]:
        if item["name"] in hashes and hashes[item["name"]] != {
            "csv_sha256": item["csv_sha256"], "sidecar_sha256": item["sidecar_sha256"]
        }:
            raise RuntimeError(f"Frozen input changed: {item['name']}")
    output_root = ROOT / "corrected_runs" / "large_calibration"
    if (output_root / RUN_ID).exists():
        raise FileExistsError(f"Preserve the existing group -003 attempt: {output_root / RUN_ID}")
    started = time.monotonic()
    manifest = run_experiment(
        data_paths=paths, output_root=output_root, run_id=RUN_ID,
        seeds=[42], folds=[1], conditions=CONDITIONS,
        pipelines=PIPELINES, models=MODELS, n_splits=5,
        split_policy="group_aware", cache_policy="bounded", cache_max_bytes=8 * 1024**3,
        durable_scheduler=True, cache_audit=True, scheduler_lease_seconds=3600.0,
        workers=4, use_gpu=False,
    )
    report = {
        "artifact_type": "clean_group_aware_large_calibration_v3",
        "status": manifest.get("status"), "run_id": RUN_ID,
        "code_fingerprint": frozen["code_fingerprint"],
        "source_datasets": list(DATASETS), "dataset_hashes": hashes,
        "numerical_thread_environment": THREAD_ENV,
        "workers": 4, "scheduler_lease_seconds": 3600,
        "elapsed_s": round(time.monotonic() - started, 3),
        "counts_by_status": manifest.get("counts_by_status"),
        "row_level_reference": "calibration-row_level-002",
        "abandoned_concurrent_group_attempt": "calibration-group_aware-002",
        "scientific_use": "diagnostic throughput and resource evidence only",
    }
    out = ROOT / "provenance" / "large_dataset_calibration_group_v3.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
