import json
import queue
from unittest.mock import patch

import pytest

from src.pipeline_runner import writer_process
from src.task_manifest import ManifestConflictError, ManifestStore
from tests.test_critical_recovery import fixture


def result(store,records):
    task=records[0]["scientific_task_id"]
    attempt=store.claim_task("r",task)
    return {"run_id":"r","scientific_task_id":task,"attempt_id":attempt,"status":"success","roc_auc":.8,
        "dataset":"d","seed":42,"fold":1,"condition":"clean","pipeline":"Raw","model":"logistic_regression","split_policy":"stratified"}


def test_export_failure_does_not_lose_authoritative_commit(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    store,records=fixture(tmp_path,precompute=False)
    payload=result(store,records)
    delivery=queue.Queue()
    delivery.put(payload)
    delivery.put("DONE")
    with patch("src.pipeline_runner.os.fsync",side_effect=OSError("injected export failure")):
        with pytest.raises(OSError):
            writer_process(delivery,tmp_path/"results.jsonl",store.db_path,"r")
    assert store.get_task("r",payload["scientific_task_id"])["state"] == "completed"
    assert len(store.snapshot("r")["durable_results"]) == 1


def test_missing_or_truncated_export_rebuilds_from_committed_results(tmp_path):
    store,records=fixture(tmp_path,precompute=False)
    payload=result(store,records)
    store.commit_result("r",payload["scientific_task_id"],payload["attempt_id"],payload)
    ledger=tmp_path/"results.jsonl"
    ledger.write_text('{"truncated":')
    report=store.export_durable_results("r",ledger)
    assert report["malformed_lines"] == 1
    assert json.loads(ledger.read_text()) == payload
    assert store.export_durable_results("r",ledger)["rows_added"] == 0


def test_conflicting_export_is_not_silently_overwritten(tmp_path):
    store,records=fixture(tmp_path,precompute=False)
    payload=result(store,records)
    store.commit_result("r",payload["scientific_task_id"],payload["attempt_id"],payload)
    ledger=tmp_path/"results.jsonl"
    ledger.write_text(json.dumps({**payload,"roc_auc":.1})+"\n")
    with pytest.raises(ManifestConflictError):
        store.export_durable_results("r",ledger)


def test_duplicate_delivery_does_not_duplicate_export(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    store,records=fixture(tmp_path,precompute=False)
    payload=result(store,records)
    delivery=queue.Queue()
    for item in (payload,payload,"DONE"):
        delivery.put(item)
    writer_process(delivery,tmp_path/"results.jsonl",store.db_path,"r")
    assert len((tmp_path/"results.jsonl").read_text().splitlines()) == 1


def test_transaction_failure_rolls_back_result_task_and_attempt_together(tmp_path):
    store,records=fixture(tmp_path,precompute=False)
    payload=result(store,records)
    with patch.object(store,"_finish_attempt",side_effect=RuntimeError("injected before task transition")):
        with pytest.raises(RuntimeError):
            store.commit_result("r",payload["scientific_task_id"],payload["attempt_id"],payload)
    snapshot=store.snapshot("r")
    assert snapshot["durable_results"] == []
    assert snapshot["tasks"][0]["state"] == snapshot["attempts"][0]["state"] == "running"


def test_legacy_fsynced_row_reconciles_before_fencing(tmp_path):
    store,records=fixture(tmp_path,precompute=False)
    payload=result(store,records)
    ledger=tmp_path/"results.jsonl"
    ledger.write_text(json.dumps(payload)+"\n")
    with store._connect() as connection:
        connection.execute("UPDATE attempts SET started_at='2000-01-01T00:00:00+00:00'")
    assert store.reconcile_result_ledger("r",ledger)["valid_results_committed"] == 1
    assert store.recover_stale_attempts("r",stale_after_seconds=1) == 0
    assert store.get_task("r",payload["scientific_task_id"])["state"] == "completed"


def test_late_stale_worker_cannot_commit_after_takeover(tmp_path):
    store,records=fixture(tmp_path,precompute=False)
    payload=result(store,records)
    with store._connect() as connection:
        connection.execute("UPDATE attempts SET started_at='2000-01-01T00:00:00+00:00'")
    store.recover_stale_attempts("r",stale_after_seconds=1)
    second=store.claim_task("r",payload["scientific_task_id"])
    with pytest.raises(ManifestConflictError):
        store.commit_result("r",payload["scientific_task_id"],payload["attempt_id"],payload)
    assert store.get_task("r",payload["scientific_task_id"])["active_attempt_id"] == second
