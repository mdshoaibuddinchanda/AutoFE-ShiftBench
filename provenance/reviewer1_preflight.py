"""Bounded real-data gate for the corrected two-track runner.

This command intentionally runs one small dataset, one seed, one requested
fold, one condition, three pipelines, and one model under each split policy.
It measures artifact sizes and candidate-history volume without launching the
full campaign.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data_loader import load_csv_dataset
from src.feature_engineering import DFSConfig, expand_features_with_dfs
from src.mechanism_audit import CandidateHistoryWriter, estimate_candidate_history_storage
from src.pipeline_runner import run_experiment
from src.provenance import code_fingerprint, current_git_commit


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = "sonar"
    data_path = ROOT / "data" / "raw" / f"{dataset}.csv"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    output_root = ROOT / "corrected_runs" / "reviewer1_preflight"
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for policy in ("row_level", "group_aware"):
        run_id = f"{dataset}-{policy}"
        started = time.perf_counter()
        manifest = run_experiment(
            {dataset: data_path}, run_id=run_id, output_root=output_root,
            seeds=[42], folds=[1], conditions=(("clean", 0.0),),
            pipelines=("Raw", "Raw_CapMatched", "AutoFE_Baseline"),
            models=("logistic_regression",), n_splits=5, split_policy=policy,
        )
        elapsed = time.perf_counter() - started
        run_dir = output_root / run_id
        records.append({
            "run_id": run_id, "split_policy": policy, "elapsed_s": elapsed,
            "status": manifest["status"], "counts": manifest["counts"],
            "counts_by_status": manifest.get("counts_by_status"),
            "expected_tasks": manifest["expected_tasks"],
            "result_bytes": (run_dir / "results.jsonl").stat().st_size,
            "cache_bytes": sum(path.stat().st_size for path in (run_dir / "cache").glob("*")),
        })

    # Exercise the candidate-history schema on the same real dataset's first
    # training partition.  This is a bounded mechanism audit, not benchmark
    # output and not a held-out selection operation.
    frame = load_csv_dataset(data_path)
    target = "target" if "target" in frame.columns else "target_label"
    X = frame.drop(columns=[target]).select_dtypes(include=["number", "bool"])
    y = frame[target].reset_index(drop=True)
    x_train = X.iloc[: min(100, len(X))].reset_index(drop=True)
    x_test = X.iloc[min(100, len(X)): min(130, len(X))].reset_index(drop=True)
    history_path = output_root / "candidate_history_preflight.jsonl.gz"
    with CandidateHistoryWriter(history_path) as writer:
        _, _, metadata = expand_features_with_dfs(
            x_train, x_test, y.iloc[: len(x_train)],
            config=DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1),
            audit_context={"dataset": dataset, "split_policy": "row_level", "seed": 42, "fold": 1, "condition": "clean"},
            candidate_history_writer=writer,
        )
        history = {
            "records_written": writer.records_written,
            "uncompressed_bytes": writer.uncompressed_bytes_written,
            "compressed_bytes": history_path.stat().st_size,
            "metadata": metadata.get("candidate_history"),
        }
    history["full_grid_estimate"] = asdict(estimate_candidate_history_storage(
        tasks=612_500, candidates_per_task=max(1, history["records_written"])
    ))
    payload = {
        "artifact_type": "reviewer1_bounded_preflight",
        "environment": r"D:\Conda\p12",
        "dataset": dataset, "n_splits": 5, "seed": 42,
        "code_commit": current_git_commit(ROOT),
        "code_fingerprint": code_fingerprint(ROOT),
        "runs": records, "candidate_history": history,
        "full_campaign_started": False,
    }
    out = ROOT / "provenance" / "reviewer1_preflight_manifest.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
