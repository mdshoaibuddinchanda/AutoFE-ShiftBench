import json
import pytest

from provenance.audit_four_dataset_pilot import audit_run
from provenance.compare_paired_pilot_profile import compare
from src.checkpoint import record_task
from src.task_scheduler import TaskScheduler, TaskSpec


def _row(prediction_hash="prediction"):
    return {
        "run_id": "profile", "task_key": "task", "status": "success",
        "dataset": "tiny", "seed": 42, "fold": 1, "condition": "clean",
        "pipeline": "Raw", "model": "logistic_regression", "split_policy": "group_aware",
        "n_train": 8, "n_test": 4, "n_retained": 2, "n_generated": 0,
        "train_matrix_sha256": "train", "test_matrix_sha256": "test",
        "prediction_sha256": prediction_hash, "roc_auc": 0.75, "f1": 0.5,
        "accuracy": 0.5, "worker_started_unix": 1.0, "worker_finished_unix": 2.0,
        "train_time_s": 0.2, "worker_elapsed_s": 0.3,
    }


def test_read_only_pilot_auditor_checks_result_parity(tmp_path):
    run = tmp_path / "pilot-group_aware-001"
    run.mkdir()
    scheduler = TaskScheduler(run / "scheduler.sqlite", artifact_dir=run / "scheduler_results")
    scheduler.register_tasks([TaskSpec("task", "profile", {"dataset": "tiny"})])
    lease = scheduler.claim_task("worker", task_key="task")
    assert lease is not None
    scheduler.publish_result(lease, _row())
    record_task(run / "checkpoints.sqlite", run_id="profile", task_key="task",
                phase="phase2", status="success", dataset="tiny")
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "profile", "status": "complete", "counts_by_status": {"success": 1},
        "cache_audit": {"artifacts": {}},
    }))
    (run / "results.jsonl").write_text(json.dumps(_row()) + "\n")
    report = audit_run(run)
    assert report["scheduler_results"] == report["phase2_successes"] == report["jsonl_successes"] == 1
    assert report["disagreements"] == []
    assert report["scheduler_wall_span_seconds"] >= 0
    assert report["datasets"]["tiny"]["scheduler_wall_span_seconds"] >= 0
    (run / "results.jsonl").write_text(json.dumps(_row("changed")) + "\n")
    changed = audit_run(run)
    assert changed["disagreements"] == [{
        "task_key": "task", "issue": "jsonl_artifact_value_mismatch", "fields": ["prediction_sha256"],
    }]
    metric_changed = _row()
    metric_changed["brier_score"] = 0.2
    (run / "results.jsonl").write_text(json.dumps(metric_changed) + "\n")
    changed = audit_run(run)
    assert changed["disagreements"] == [{
        "task_key": "task", "issue": "jsonl_artifact_value_mismatch", "fields": ["brier_score"],
    }]
    (run / "results.jsonl").write_text(json.dumps(_row()) + "\n")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["counts_by_status"]["success"] = 0
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert audit_run(run)["disagreements"] == [{
        "task_key": None, "issue": "final_manifest_count_mismatch",
        "manifest_successes": 0, "manifest_expected_tasks": None,
        "registered_tasks": 1, "scheduler_results": 1,
    }]


def test_paired_profile_comparator_rejects_changed_prediction(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "results.jsonl").write_text(json.dumps(_row()) + "\n")
    (after / "results.jsonl").write_text(json.dumps(_row()) + "\n")
    for path, elapsed in ((before, 4.0), (after, 2.0)):
        (path / "paired_profile_summary.json").write_text(json.dumps({
            "run_id": path.name, "elapsed_s": elapsed, "valid_cells_per_hour": 3600 / elapsed,
            "peak_process_tree_rss_mib": 100, "peak_system_gpu_used_mib": 0,
            "cache_builds": 1, "cache_hits": 0, "cache_deletions": 1,
        }))
    report = compare(before, after)
    assert report["matched_cells"] == 1
    assert report["output_mismatches"] == []
    assert report["wall_speedup"] == 2.0
    (after / "results.jsonl").write_text(json.dumps(_row("changed")) + "\n")
    changed = compare(before, after)
    assert [item["field"] for item in changed["output_mismatches"]] == ["prediction_sha256"]
    after_profile_path = after / "paired_profile_summary.json"
    after_profile = json.loads(after_profile_path.read_text())
    after_profile["dataset_sha256"] = {"tiny": "changed"}
    after_profile_path.write_text(json.dumps(after_profile))
    with pytest.raises(ValueError, match="dataset_sha256"):
        compare(before, after)
