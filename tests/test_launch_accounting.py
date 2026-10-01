"""Full-grid accounting and bounded manifest behavior."""

import json
import sqlite3

import numpy as np
import pandas as pd

from src.check_progress import check_progress
from src.pipeline_runner import CompactOutcomeLedger, _save_run_manifest, run_experiment


def _dataset(path, values, labels):
    pd.DataFrame({"value": values, "target": labels}).to_csv(path, index=False)
    path.with_name(f"{path.stem}_meta.json").write_text(
        json.dumps({"target_column": "target", "dataset_identity": {"provider": "test"}}),
        encoding="utf-8",
    )


def test_group_auc_infeasible_cells_are_skipped_and_next_dataset_runs(tmp_path):
    infeasible = tmp_path / "infeasible.csv"
    feasible = tmp_path / "feasible.csv"
    _dataset(infeasible, [0, 0, 1, 2, 3, 4], [0, 0, 0, 0, 1, 1])
    _dataset(feasible, np.arange(90), [0, 1] * 45)
    manifest = run_experiment(
        {"infeasible": infeasible, "feasible": feasible},
        run_id="group-skip", output_root=tmp_path / "runs", seeds=[42],
        folds=[1, 2, 3], n_splits=3, conditions=(("clean", 0.0),),
        pipelines=("Raw",), models=("logistic_regression",),
        split_policy="group_aware", cache_policy="bounded", durable_scheduler=True,
        manifest_policy="compact", cache_max_bytes=4 * 1024 * 1024,
    )
    assert manifest["status"] == "complete"
    assert manifest["counts_by_status"]["skipped"] == 3
    assert manifest["counts_by_status"]["success"] == 3
    rows = [json.loads(line) for line in (
        tmp_path / "runs" / "group-skip" / "results.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    skipped = [row for row in rows if row["status"] == "skipped"]
    assert len(skipped) == 3
    assert {row["dataset"] for row in skipped} == {"infeasible"}
    assert all(row["split_policy"] == "group_aware" for row in skipped)
    assert all(row["train_indices_sha256"] and row["test_indices_sha256"] for row in skipped)
    assert all("dataset_ineligible_for_all_configured_seed_auc" in row["skip_reason"] for row in skipped)
    progress = check_progress(tmp_path / "runs" / "group-skip")
    assert progress["task_counts"]["remaining"] == 0
    assert progress["task_counts"]["skipped"] == 3
    resumed = run_experiment(
        {"infeasible": infeasible, "feasible": feasible},
        run_id="group-skip", output_root=tmp_path / "runs", seeds=[42],
        folds=[1, 2, 3], n_splits=3, conditions=(("clean", 0.0),),
        pipelines=("Raw",), models=("logistic_regression",),
        split_policy="group_aware", cache_policy="bounded", durable_scheduler=True,
        manifest_policy="compact", cache_max_bytes=4 * 1024 * 1024,
    )
    assert resumed["counts_by_status"] == manifest["counts_by_status"]
    assert len((tmp_path / "runs" / "group-skip" / "results.jsonl").read_text().splitlines()) == 6


def test_one_unsupported_seed_excludes_all_configured_group_auc_cells(tmp_path, monkeypatch):
    from src import pipeline_runner

    path = tmp_path / "seed_sensitive.csv"
    _dataset(path, np.arange(90), [0, 1] * 45)
    original = pipeline_runner.get_group_stratified_splits

    def seed_sensitive(*args, **kwargs):
        splits, status = original(*args, **kwargs)
        if args[3] == 123:
            status = dict(status, auc_status="infeasible", reasons=["seed_specific_auc_infeasible"])
        return splits, status

    monkeypatch.setattr(pipeline_runner, "get_group_stratified_splits", seed_sensitive)
    manifest = run_experiment(
        {"seed_sensitive": path}, run_id="all-seed-skip", output_root=tmp_path / "runs",
        seeds=[42, 123], folds=[1], n_splits=3, conditions=(("clean", 0.0),),
        pipelines=("Raw",), models=("logistic_regression",),
        split_policy="group_aware", cache_policy="bounded", durable_scheduler=True,
        manifest_policy="compact", cache_max_bytes=4 * 1024 * 1024,
    )
    assert manifest["counts_by_status"]["skipped"] == 2
    statuses = manifest["datasets"][0]["group_fold_status_by_seed"]
    assert statuses["42"]["auc_status"] == "supported"
    assert statuses["123"]["auc_status"] == "infeasible"


def test_compact_manifest_keeps_per_task_evidence_in_sqlite(tmp_path):
    ledger = CompactOutcomeLedger(tmp_path / "manifest_outcomes.sqlite")
    base = {"run_id": "ledger", "configuration": {
        "datasets": ["d"], "seeds": [1], "folds": [1], "conditions": [["clean", 0.0]],
        "pipelines": ["Raw"], "models": list(range(1000)),
    }}
    _save_run_manifest(tmp_path, base, ledger)
    initial_bytes = (tmp_path / "manifest.json").stat().st_size
    for number in range(1000):
        ledger.upsert({"task_key": f"key-{number}", "status": "success"})
        _save_run_manifest(tmp_path, base, ledger)
    _save_run_manifest(tmp_path, base, ledger, force=True)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.stat().st_size - initial_bytes < 300
    assert manifest["counts_by_status"]["success"] == 1000
    assert manifest["tasks"] == []
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] == 1000
    reopened = CompactOutcomeLedger(ledger.path)
    reopened.upsert({"task_key": "key-0", "status": "failed"})
    _save_run_manifest(tmp_path, base, reopened, force=True)
    revised = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert revised["counts_by_status"]["success"] == 999
    assert revised["counts_by_status"]["failed"] == 1


def test_corrected_table_gap_and_operator_figure_use_result_schema(tmp_path, monkeypatch):
    from src.generate_tables import generate_table_9_overfitting_gap
    from src import plotting_q1

    names = [
        "Raw", "Raw_CapMatched", "AutoFE_Baseline", "AutoFE_MI", "AutoFE_Random",
        "AutoFE_NoMultiply", "AutoFE_Isolate_Add", "AutoFE_Isolate_Subtract",
        "AutoFE_Isolate_Multiply", "AutoFE_Isolate_Divide", "AutoFE_LeaveOut_Add",
        "AutoFE_LeaveOut_Subtract", "AutoFE_LeaveOut_Multiply", "AutoFE_LeaveOut_Divide",
    ]
    rows = [
        {"dataset": dataset, "seed": 42, "fold": 1, "condition": "clean",
         "model": "logistic_regression", "pipeline": pipeline, "roc_auc": 0.7,
         "train_auc": 0.8}
        for dataset in ("d1", "d2") for pipeline in names
    ]
    frame = pd.DataFrame(rows)
    assert "0.1000" in generate_table_9_overfitting_gap(frame)
    captured = {}

    def capture(fig, out_dir, stem, dpi=300):
        captured["labels"] = [label.get_text() for label in fig.axes[0].get_yticklabels()]
        captured["stem"] = stem
        plotting_q1.plt.close(fig)

    monkeypatch.setattr(plotting_q1, "_save_figure", capture)
    plotting_q1.plot_fig10_ablation(frame, tmp_path)
    assert captured["stem"] == "Figure_10_Ablation_Study"
    assert captured["labels"] == names
