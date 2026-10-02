from __future__ import annotations

import tempfile
import time
import unittest
import json
from pathlib import Path

from src.task_manifest import (
    ExecutionConfig,
    ManifestConflictError,
    ManifestStore,
    build_task_records,
    run_callable_with_timeout,
    scientific_task_id,
)


def _slow(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


class TaskManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.db = self.directory / "manifest.db"
        self.manifest = self.directory / "manifest.jsonl"

    def _store(self, records):
        store = ManifestStore(self.db, manifest_path=self.manifest)
        config = {"protocol_version": "test_protocol", "seed_scheme_version": "test_seed", "execution": ExecutionConfig(max_workers=1, max_attempts=2).to_dict()}
        store.create_run("run_test", config, records)
        return store

    def test_cartesian_manifest_is_complete_and_reordering_is_stable(self) -> None:
        kwargs = dict(
            datasets=["d1", "d2"], seeds=[1, 2], folds=[1],
            conditions=[("clean", 0.0), ("gaussian_noise", 0.1)],
            pipelines=["Raw", "AutoFE_Baseline"], models=["logistic_regression"],
        )
        first = build_task_records(**kwargs)
        second = build_task_records(**{**kwargs, "datasets": ["d2", "d1"], "seeds": [2, 1]})
        self.assertEqual(len(first), 24)  # 8 precompute + 16 model tasks
        self.assertEqual({r["scientific_task_id"] for r in first}, {r["scientific_task_id"] for r in second})
        self.assertEqual(len({r["scientific_task_id"] for r in first}), len(first))

    def test_claim_failure_retry_and_durable_result_are_accounted_once(self) -> None:
        records = build_task_records(["d1"], [1], [1], [("clean", 0.0)], ["Raw"], ["logistic_regression"], include_precompute=False)
        store = self._store(records)
        task_id = records[0]["scientific_task_id"]
        attempt = store.claim_task("run_test", task_id, worker_id="w1")
        self.assertIsNotNone(attempt)
        store.record_failure("run_test", task_id, attempt, failure_class="worker_exception", retry=True)
        self.assertEqual(store.get_task("run_test", task_id)["state"], "pending")
        attempt2 = store.claim_task("run_test", task_id, worker_id="w2")
        payload = {"scientific_task_id": task_id, "attempt_id": attempt2, "status": "success", "roc_auc": 0.75}
        self.assertTrue(store.commit_result("run_test", task_id, attempt2, payload))
        self.assertFalse(store.commit_result("run_test", task_id, attempt2, payload))
        self.assertEqual(store.get_task("run_test", task_id)["state"], "completed")
        self.assertEqual(store.attempt_counts("run_test")["failed"], 1)
        with self.assertRaises(ManifestConflictError):
            store.commit_result("run_test", task_id, attempt, {**payload, "roc_auc": 0.1})

    def test_dependency_failure_and_state_counts_reconcile(self) -> None:
        records = build_task_records(["d1"], [1], [1], [("clean", 0.0)], ["Raw"], ["logistic_regression"])
        store = self._store(records)
        pre = next(r for r in records if r["task_kind"] == "precompute")
        model = next(r for r in records if r["task_kind"] == "model")
        pre_attempt = store.claim_task("run_test", pre["scientific_task_id"], worker_id="w1")
        store.record_failure("run_test", pre["scientific_task_id"], pre_attempt, failure_class="precompute_failure")
        self.assertEqual(store.propagate_dependency_failure("run_test", pre["scientific_task_id"]), 1)
        counts = store.state_counts("run_test")
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["skipped"], 1)
        self.assertEqual(counts["intended"], 2)
        self.assertEqual(store.get_task("run_test", model["scientific_task_id"])["outcome_reason"], "dependency_failure")

    def test_timeout_contains_child_process(self) -> None:
        outcome = run_callable_with_timeout(_slow, args=(2.0,), timeout_seconds=0.1)
        self.assertEqual(outcome["state"], "timeout")

    def test_execution_config_rejects_invalid_limits(self) -> None:
        with self.assertRaises(ValueError):
            ExecutionConfig(max_workers=0)
        self.assertEqual(scientific_task_id({"dataset": "d1"})[:5], "task_")

    def test_existing_run_rejects_changed_configuration_and_task_grid(self) -> None:
        records = build_task_records(["d1"], [1], [1], [("clean", 0.0)], ["Raw"], ["logistic_regression"], include_precompute=False)
        store = self._store(records)
        with self.assertRaises(ManifestConflictError):
            store.create_run("run_test", {"protocol_version": "changed", "seed_scheme_version": "test_seed", "execution": ExecutionConfig(max_workers=1, max_attempts=2).to_dict()}, records)
        changed_records = build_task_records(["d1", "d2"], [1], [1], [("clean", 0.0)], ["Raw"], ["logistic_regression"], include_precompute=False)
        with self.assertRaises(ManifestConflictError):
            store.create_run("run_test", {"protocol_version": "test_protocol", "seed_scheme_version": "test_seed", "execution": ExecutionConfig(max_workers=1, max_attempts=2).to_dict()}, changed_records)

    def test_planned_missing_dataset_is_explicitly_skipped(self) -> None:
        records = build_task_records(["missing"], [1], [1], [("clean", 0.0)], ["Raw"], ["logistic_regression"], data_paths={"missing": self.directory / "absent.csv"})
        store = self._store(records)
        counts = store.state_counts("run_test")
        self.assertEqual(counts["skipped"], len(records))
        self.assertEqual(counts["pending"], 0)
        self.assertTrue(all(record["planned_skip_reason"] == "dataset_unavailable" for record in records))

    def test_stale_recovery_respects_max_attempts(self) -> None:
        records = build_task_records(["d1"], [1], [1], [("clean", 0.0)], ["Raw"], ["logistic_regression"], include_precompute=False)
        store = self._store(records)
        task_id = records[0]["scientific_task_id"]
        first = store.claim_task("run_test", task_id, worker_id="dead-1")
        with store._connect() as connection:
            connection.execute("UPDATE attempts SET started_at = ? WHERE run_id = ? AND attempt_id = ?", ("2000-01-01T00:00:00+00:00", "run_test", first))
        self.assertEqual(store.recover_stale_attempts("run_test", stale_after_seconds=1), 1)
        second = store.claim_task("run_test", task_id, worker_id="dead-2")
        with store._connect() as connection:
            connection.execute("UPDATE attempts SET started_at = ? WHERE run_id = ? AND attempt_id = ?", ("2000-01-01T00:00:00+00:00", "run_test", second))
        self.assertEqual(store.recover_stale_attempts("run_test", stale_after_seconds=1), 1)
        self.assertEqual(store.get_task("run_test", task_id)["state"], "failed")

    def test_stale_attempt_recovery_and_result_reconciliation(self) -> None:
        records = build_task_records(["d1"], [1], [1], [("clean", 0.0)], ["Raw"], ["logistic_regression"], include_precompute=False)
        store = self._store(records)
        task_id = records[0]["scientific_task_id"]
        old_attempt = store.claim_task("run_test", task_id, worker_id="dead")
        with store._connect() as connection:
            connection.execute("UPDATE attempts SET started_at = ? WHERE run_id = ? AND attempt_id = ?", ("2000-01-01T00:00:00+00:00", "run_test", old_attempt))
        self.assertEqual(store.recover_stale_attempts("run_test", stale_after_seconds=1), 1)
        self.assertEqual(store.get_task("run_test", task_id)["state"], "pending")
        with self.assertRaises(ManifestConflictError):
            store.commit_result("run_test", task_id, old_attempt, {"scientific_task_id": task_id, "attempt_id": old_attempt, "status": "success"})
        new_attempt = store.claim_task("run_test", task_id, worker_id="new")
        ledger = self.directory / "results.jsonl"
        result = {"scientific_task_id": task_id, "attempt_id": new_attempt, "status": "success", "roc_auc": 0.8}
        ledger.write_text(json.dumps(result) + "\n{" , encoding="utf-8")
        report = store.reconcile_result_ledger("run_test", ledger)
        self.assertEqual(report["valid_results_committed"], 1)
        self.assertEqual(report["malformed_lines"], 1)
        self.assertEqual(store.get_task("run_test", task_id)["state"], "completed")

    def test_snapshot_reads_run_tasks_attempts_and_durable_results_together(self) -> None:
        records = build_task_records(["d1"], [1], [1], [("clean", 0.0)], ["Raw"], ["logistic_regression"], include_precompute=False)
        store = self._store(records)
        task_id = records[0]["scientific_task_id"]
        attempt = store.claim_task("run_test", task_id, worker_id="snapshot")
        store.commit_result("run_test", task_id, attempt, {"status": "success", "roc_auc": 0.75})
        snapshot = store.snapshot("run_test")
        self.assertEqual(snapshot["run"]["run_id"], "run_test")
        self.assertEqual(len(snapshot["tasks"]), 1)
        self.assertEqual(len(snapshot["attempts"]), 1)
        self.assertEqual(len(snapshot["durable_results"]), 1)
        self.assertEqual(snapshot["tasks"][0]["state"], "completed")


if __name__ == "__main__":
    unittest.main()
