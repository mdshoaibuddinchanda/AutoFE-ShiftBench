from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.pipeline_runner import run_experiment
from src.task_scheduler import TaskScheduler


def _write_dataset(path):
    rng = np.random.default_rng(41)
    signal = rng.normal(size=60)
    frame = pd.DataFrame({
        "x1": signal, "x2": rng.normal(size=60),
        "target": (signal > 0).astype(int),
    })
    frame.to_csv(path, index=False)
    path.with_name(f"{path.stem}_meta.json").write_text(json.dumps({
        "target_column": "target", "dataset_identity": {"provider": "test", "id": "bounded"},
    }), encoding="utf-8")


def test_bounded_cache_is_reclaimed_after_all_classifier_consumers(tmp_path):
    data_path = tmp_path / "bounded.csv"
    _write_dataset(data_path)
    manifest = run_experiment(
        {"bounded": data_path}, run_id="bounded-run", output_root=tmp_path / "runs",
        seeds=[42], folds=[1], conditions=(("clean", 0.0),),
        pipelines=("Raw",), models=("logistic_regression", "random_forest"), n_splits=3,
        cache_policy="bounded", cache_max_bytes=10 * 1024 * 1024,
        durable_scheduler=True,
    )
    assert manifest["status"] == "complete"
    assert manifest["counts_by_status"] == {
        "success": 2, "failed": 0, "skipped": 0, "timed_out": 0, "pending": 0,
    }
    cache_root = tmp_path / "runs" / "bounded-run" / "cache_bounded"
    assert not list(cache_root.glob("*.payload"))
    assert manifest["cache_storage"]["policy"] == "bounded_regenerable"
    assert manifest["cache_storage"]["final_cleanup"]["after_bytes"] >= 0
    scheduler = TaskScheduler(
        tmp_path / "runs" / "bounded-run" / "scheduler.sqlite",
        artifact_dir=tmp_path / "runs" / "bounded-run" / "scheduler_results",
    )
    states = [scheduler.task_state(task["task_key"]) for task in manifest["tasks"]]
    assert states and all(state and state["status"] == "success" for state in states)
    assert len(list((tmp_path / "runs" / "bounded-run" / "scheduler_results").glob("*.json"))) == 2
