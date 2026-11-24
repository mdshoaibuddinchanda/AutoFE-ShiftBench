import json
import sqlite3
from pathlib import Path

import pytest

from src.pipeline_runner import _record_manifest_failure
from src.task_manifest import ExecutionConfig, ManifestStore, build_task_records


def fixture(tmp_path, *, precompute=True):
    records = build_task_records(["d"],[42],[1],[("clean",0)],["Raw"],["logistic_regression"],include_precompute=precompute)
    store = ManifestStore(tmp_path/"manifest.db")
    store.create_run("r",{"execution":ExecutionConfig(max_attempts=2).to_dict()},records)
    return store,records


def test_retryable_precompute_does_not_skip_models(tmp_path):
    store,records = fixture(tmp_path)
    pre,model = records
    attempt = store.claim_task("r",pre["scientific_task_id"])
    task = {"manifest_db":str(store.db_path),"run_id":"r","scientific_task_id":pre["scientific_task_id"]}
    _record_manifest_failure(task,attempt,failure_class="precompute_failure",retry=True)
    assert store.get_task("r",pre["scientific_task_id"])["state"] == "pending"
    assert store.get_task("r",model["scientific_task_id"])["state"] == "pending"
    attempt2 = store.claim_task("r",pre["scientific_task_id"])
    store.commit_result("r",pre["scientific_task_id"],attempt2,{"stage":"precompute","status":"completed"})
    assert store.claim_task("r",model["scientific_task_id"]) is not None
    assert store.attempt_counts("r")["failed"] == 1


def test_terminal_dependency_failure_is_idempotent(tmp_path):
    store,records = fixture(tmp_path)
    pre,model = records
    attempt = store.claim_task("r",pre["scientific_task_id"])
    store.record_failure("r",pre["scientific_task_id"],attempt,failure_class="precompute_failure")
    assert store.propagate_dependency_failure("r",pre["scientific_task_id"]) == 1
    assert store.propagate_dependency_failure("r",pre["scientific_task_id"]) == 0

