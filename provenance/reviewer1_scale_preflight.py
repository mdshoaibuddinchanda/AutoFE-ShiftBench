"""Bounded scale preflight on a large configured dataset.

Runs one seed/fold/condition and three pipelines under both split policies.
This is a timing and artifact-size measurement only; it is not benchmark
evidence and does not launch the full campaign.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.pipeline_runner import run_experiment
from src.provenance import code_fingerprint, current_git_commit


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = "airlines"
    data_path = ROOT / "data" / "raw" / f"{dataset}.csv"
    output_root = ROOT / "corrected_runs" / "reviewer1_scale_preflight"
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for policy in ("row_level", "group_aware"):
        run_id = f"{dataset}-{policy}"
        started = time.perf_counter()
        manifest = run_experiment(
            {dataset: data_path}, run_id=run_id, output_root=output_root,
            seeds=[42], folds=[1], conditions=(("clean", 0.0),),
            pipelines=("Raw", "Raw_CapMatched", "AutoFE_Baseline"),
            models=("logistic_regression",), n_splits=5, split_policy=policy,
        )
        run_dir = output_root / run_id
        runs.append({
            "run_id": run_id, "split_policy": policy,
            "elapsed_s": time.perf_counter() - started,
            "status": manifest["status"], "expected_tasks": manifest["expected_tasks"],
            "counts_by_status": manifest.get("counts_by_status"),
            "results_bytes": (run_dir / "results.jsonl").stat().st_size,
            "manifest_bytes": (run_dir / "manifest.json").stat().st_size,
            "checkpoint_bytes": (run_dir / "checkpoints.sqlite").stat().st_size,
            "cache_bytes": sum(path.stat().st_size for path in (run_dir / "cache").glob("*")),
        })
    payload = {
        "artifact_type": "reviewer1_scale_preflight",
        "environment": r"D:\Conda\p12", "dataset": dataset,
        "rows": 100000, "seed": 42, "requested_fold": 1, "n_splits": 5,
        "condition": "clean", "pipelines": ["Raw", "Raw_CapMatched", "AutoFE_Baseline"],
        "models": ["logistic_regression"], "code_commit": current_git_commit(ROOT),
        "code_fingerprint": code_fingerprint(ROOT), "runs": runs,
        "full_campaign_started": False,
    }
    output = ROOT / "provenance" / "reviewer1_scale_preflight_manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
