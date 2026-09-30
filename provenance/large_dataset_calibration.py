"""Bounded slow-tail calibration for airlines and covertype.

This is a separate preflight run, not part of the four-dataset pilot or the
frozen benchmark.  It fixes one seed/fold, two representative conditions, the
Raw/central/expensive-ablation pipelines, and all classifier types under both
split policies.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline_runner import run_experiment
from src.provenance import file_sha256


DATA = ROOT / "data" / "raw"
DATASETS = ("airlines", "covertype")
PIPELINES = ("Raw", "AutoFE_Baseline", "AutoFE_LeaveOut_Multiply")
MODELS = (
    "logistic_regression", "random_forest", "extra_trees", "linear_svm", "knn",
    "gaussian_nb", "mlp", "lightgbm", "xgboost", "catboost",
)
CONDITIONS = (("clean", 0.0), ("gaussian_noise", 0.10))


def main() -> None:
    paths = {name: DATA / f"{name}.csv" for name in DATASETS}
    hashes = {name: {"csv_sha256": file_sha256(path), "sidecar_sha256": file_sha256(path.with_name(f"{path.stem}_meta.json"))} for name, path in paths.items()}
    common = dict(
        data_paths=paths, output_root=ROOT / "corrected_runs" / "large_calibration",
        seeds=[42], folds=[1], conditions=CONDITIONS, pipelines=PIPELINES,
        models=MODELS, n_splits=5, cache_policy="bounded", cache_max_bytes=8 * 1024**3,
        durable_scheduler=True, cache_audit=True, scheduler_lease_seconds=3600.0,
        workers=4, use_gpu=False,
    )
    rows = []
    started = time.time()
    for policy in ("row_level", "group_aware"):
        run_id = f"calibration-{policy}-001"
        t0 = time.time()
        manifest = run_experiment(**common, run_id=run_id, split_policy=policy)
        rows.append({"run_id": run_id, "split_policy": policy, "elapsed_s": time.time() - t0, "status": manifest.get("status"), "counts": manifest.get("counts_by_status", {})})
    report = {
        "artifact_type": "large_dataset_slow_tail_calibration",
        "datasets": list(DATASETS), "dataset_hashes": hashes,
        "selection": "airlines and covertype prespecified as large group-AUC-eligible datasets; no pilot scores used",
        "conditions": [list(x) for x in CONDITIONS], "pipelines": list(PIPELINES), "models": list(MODELS),
        "seed": 42, "fold": 1, "workers": 4, "runs": rows,
        "elapsed_s_total": time.time() - started,
        "scientific_status": "slow-tail calibration only; corrected effects remain PENDING CORRECTED RUN",
    }
    out = ROOT / "provenance" / "large_dataset_calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
