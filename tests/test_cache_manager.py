from __future__ import annotations

import json
import os
import threading
import time

import pytest

from src.cache_manager import CacheBusyError, CacheManager, CacheNotReadyError


def test_put_publishes_only_atomic_ready_artifact_and_reads_through_lease(tmp_path):
    manager = CacheManager(tmp_path / "cache")
    record = manager.put_bytes("task/raw", b"payload", {"pipeline": "Raw"})
    assert record.payload_path.exists() and record.manifest_path.exists() and record.ready_path.exists()
    assert not list((tmp_path / "cache").glob("*.tmp"))
    with manager.lease("task/raw", owner="test") as lease:
        assert lease.read_bytes() == b"payload"
        report = manager.reconcile()
        assert report["ready_count"] == 1
        assert report["active_lease_count"] == 1
    assert manager.reconcile()["active_lease_count"] == 0


def test_partial_artifact_is_never_read_as_ready(tmp_path):
    manager = CacheManager(tmp_path / "cache")
    digest = manager._digest("partial")
    (tmp_path / "cache" / f"{digest}.payload").write_bytes(b"unfinished")
    with pytest.raises(CacheNotReadyError):
        manager.acquire("partial")
    assert digest in manager.reconcile()["incomplete"]


def test_two_consumer_leases_form_refcount_and_block_cleanup(tmp_path):
    manager = CacheManager(tmp_path / "cache", lease_ttl_seconds=60)
    manager.put_text("shared", "value")
    first = manager.acquire("shared", owner="one")
    second = manager.acquire("shared", owner="two")
    digest = manager._digest("shared")
    assert manager.reconcile()["leased"][digest] == 2
    blocked = manager.cleanup(max_age_seconds=0, now=time.time() + 1, dry_run=False)
    assert digest in blocked["skipped_leased"]
    first.release()
    assert manager.reconcile()["leased"][digest] == 1
    second.release()
    deleted = manager.cleanup(max_age_seconds=0, now=time.time() + 2, dry_run=False)
    assert digest in deleted["deleted_digests"]
    assert not (tmp_path / "cache" / f"{digest}.payload").exists()


def test_replacing_a_leased_artifact_is_rejected(tmp_path):
    manager = CacheManager(tmp_path / "cache")
    manager.put_text("replace", "old")
    lease = manager.acquire("replace")
    with pytest.raises(CacheBusyError):
        manager.put_text("replace", "new")
    lease.release()
    assert manager.put_text("replace", "new").payload_bytes == 3


def test_heartbeat_extends_lease_and_release_removes_refcount(tmp_path):
    manager = CacheManager(tmp_path / "cache", lease_ttl_seconds=1)
    manager.put_text("heartbeat", "value")
    lease = manager.acquire("heartbeat")
    expiry = lease.heartbeat(ttl_seconds=30)
    assert expiry > time.time() + 20
    assert manager.reconcile()["active_lease_count"] == 1
    lease.release()
    assert manager.reconcile()["active_lease_count"] == 0


def test_malformed_lease_is_conservative_and_prevents_deletion(tmp_path):
    manager = CacheManager(tmp_path / "cache")
    manager.put_text("protected", "value")
    digest = manager._digest("protected")
    bad_lease = manager.leases_dir / f"{digest}.bad.lease.json"
    bad_lease.write_text("not json", encoding="utf-8")
    result = manager.cleanup(max_age_seconds=0, now=time.time() + 1, dry_run=False)
    assert digest in result["skipped_leased"]
    assert (tmp_path / "cache" / f"{digest}.payload").exists()


def test_reconcile_and_cleanup_remove_old_temp_and_incomplete_debris(tmp_path):
    manager = CacheManager(tmp_path / "cache", temp_ttl_seconds=1)
    digest = manager._digest("debris")
    partial = tmp_path / "cache" / f"{digest}.payload"
    partial.write_bytes(b"partial")
    temp = tmp_path / "cache" / ".writer.tmp"
    temp.write_bytes(b"temp")
    old = time.time() - 10
    os.utime(partial, (old, old))
    os.utime(temp, (old, old))
    report = manager.reconcile(now=time.time())
    assert digest in report["incomplete"]
    assert str(temp) in report["temp_files"]
    result = manager.cleanup(max_age_seconds=1, now=time.time(), dry_run=False)
    assert digest in result["deleted_digests"]
    assert str(temp) in result["old_temp_files"]
    assert not partial.exists() and not temp.exists()


def test_high_water_monitor_and_lru_cleanup_are_bounded(tmp_path):
    manager = CacheManager(tmp_path / "cache", max_bytes=300)
    manager.put_text("a", "a" * 80)
    time.sleep(0.01)
    manager.put_text("b", "b" * 80)
    over = manager.high_water()
    assert over["current_bytes"] > 300
    assert over["over_limit"] is True
    result = manager.cleanup(max_age_seconds=10_000, max_bytes=300, dry_run=False)
    assert result["after_bytes"] <= 300
    assert result["high_water"]["high_water_bytes"] >= over["current_bytes"]


def test_dry_run_reports_without_deleting(tmp_path):
    manager = CacheManager(tmp_path / "cache")
    record = manager.put_text("dry", "value")
    result = manager.cleanup(max_age_seconds=0, now=time.time() + 1, dry_run=True)
    assert record.digest in result["deleted_digests"]
    assert record.payload_path.exists()


def test_get_or_create_builds_once_across_concurrent_consumers(tmp_path):
    manager = CacheManager(tmp_path / "cache")
    calls = {"count": 0}
    guard = threading.Lock()

    def factory():
        with guard:
            calls["count"] += 1
        time.sleep(0.02)
        return b"shared-payload"

    results = []

    def consume():
        results.append(manager.get_or_create_bytes("fanout", factory))

    threads = [threading.Thread(target=consume) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert calls["count"] == 1
    assert sorted(hit for _payload, hit in results) == [False, True, True, True]
    assert {payload for payload, _hit in results} == {b"shared-payload"}


def test_admission_control_does_not_evict_an_in_use_artifact(tmp_path):
    manager = CacheManager(tmp_path / "cache", max_bytes=200)
    manager.put_text("in-use", "x" * 40)
    lease = manager.acquire("in-use", owner="consumer")
    try:
        with pytest.raises(CacheBusyError):
            manager.get_or_create_bytes("next", lambda: b"y" * 40)
        assert manager.reconcile()["active_lease_count"] == 1
        assert manager._paths("in-use")[1].exists()
    finally:
        lease.release()
