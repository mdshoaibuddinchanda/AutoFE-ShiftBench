"""Bounded, lease-aware storage for run-scoped feature-cache artifacts.

The current runner owns its cache read/write path.  This module is an isolated
storage layer for a future integration: it gives writers atomic temp/ready
states, consumers explicit leases, reconciliation of interrupted writes, and
bounded cleanup with high-water accounting.  It intentionally does not edit
or call ``pipeline_runner``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import time
from typing import Any, Iterator, Mapping


class CacheError(RuntimeError):
    """Base error for cache lifecycle failures."""


class CacheNotReadyError(CacheError):
    """Raised when a consumer requests an artifact without a valid ready state."""


class CacheBusyError(CacheError):
    """Raised when a writer or cleanup operation encounters an active lease."""


_KEY_RE = re.compile(r"^[A-Za-z0-9_.:/@+=,-]{1,512}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _now_seconds() -> float:
    return time.time()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    with temp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_time(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class CacheRecord:
    """One ready cache artifact and its accounting metadata."""

    key: str
    digest: str
    payload_path: Path
    manifest_path: Path
    ready_path: Path
    payload_bytes: int
    created_at: float
    last_access_at: float
    metadata: dict[str, Any]


class CacheLease:
    """A consumer lease whose presence protects an artifact from cleanup."""

    def __init__(self, manager: "CacheManager", record: CacheRecord, token: str, lease_path: Path, expires_at: float):
        self._manager = manager
        self.record = record
        self.token = token
        self.lease_path = lease_path
        self.expires_at = expires_at
        self._released = False

    def read_bytes(self) -> bytes:
        if self._released:
            raise CacheError("Cannot read from a released cache lease")
        self._manager._assert_lease_active(self)
        payload = self.record.payload_path.read_bytes()
        if _sha256_bytes(payload) != self.record.metadata.get("payload_sha256"):
            raise CacheError(f"Cache payload checksum changed while leased: {self.record.key}")
        return payload

    def heartbeat(self, ttl_seconds: float | None = None) -> float:
        """Extend this lease and return its new epoch expiry."""
        if self._released:
            raise CacheError("Cannot heartbeat a released cache lease")
        ttl = self._manager.lease_ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0:
            raise ValueError("Lease TTL must be positive")
        now = _now_seconds()
        self.expires_at = now + ttl
        _atomic_json(self.lease_path, {
            "key": self.record.key,
            "digest": self.record.digest,
            "token": self.token,
            "created_at": self.record.metadata.get("created_at", _utc_now()),
            "heartbeat_at": now,
            "expires_at": self.expires_at,
        })
        return self.expires_at

    def release(self) -> None:
        if not self._released:
            self.lease_path.unlink(missing_ok=True)
            self._released = True

    def __enter__(self) -> "CacheLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class CacheManager:
    """Manage ready cache artifacts with leases and bounded cleanup.

    The manager stores payloads as ``<digest>.payload``, metadata as
    ``<digest>.manifest.json``, and a final ``<digest>.ready`` marker.  Readers
    accept an artifact only when all three exist and the manifest/payload
    checksums agree.  Temp files and incomplete final files are never read.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        lease_ttl_seconds: float = 3600.0,
        temp_ttl_seconds: float = 3600.0,
        max_bytes: int | None = None,
        lock_timeout_seconds: float = 10.0,
    ):
        self.root = Path(root)
        self.leases_dir = self.root / "leases"
        self.locks_dir = self.root / "locks"
        self.high_water_path = self.root / "storage.highwater.json"
        self.lease_ttl_seconds = float(lease_ttl_seconds)
        self.temp_ttl_seconds = float(temp_ttl_seconds)
        self.max_bytes = max_bytes
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        if self.lease_ttl_seconds <= 0 or self.temp_ttl_seconds <= 0:
            raise ValueError("Lease and temp TTLs must be positive")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.root.mkdir(parents=True, exist_ok=True)
        self.leases_dir.mkdir(exist_ok=True)
        self.locks_dir.mkdir(exist_ok=True)

    @staticmethod
    def _digest(key: str) -> str:
        if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
            raise ValueError("Cache key must be a nonempty safe string (no path traversal)")
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _paths(self, key: str) -> tuple[str, Path, Path, Path]:
        digest = self._digest(key)
        return digest, self.root / f"{digest}.payload", self.root / f"{digest}.manifest.json", self.root / f"{digest}.ready"

    @contextmanager
    def _artifact_lock(self, digest: str) -> Iterator[None]:
        """Take a small cross-process key lock for lease/cleanup decisions."""
        lock = self.locks_dir / f"{digest}.lock"
        deadline = _now_seconds() + self.lock_timeout_seconds
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if _now_seconds() - lock.stat().st_mtime > self.lock_timeout_seconds * 2:
                        lock.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if _now_seconds() >= deadline:
                    raise CacheBusyError(f"Timed out acquiring cache lock for {digest}")
                time.sleep(0.01)
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)

    def _lease_files(self, digest: str) -> list[Path]:
        return sorted(self.leases_dir.glob(f"{digest}.*.lease.json"))

    def _active_lease_files(self, digest: str, now: float | None = None) -> tuple[list[Path], list[Path]]:
        now = _now_seconds() if now is None else now
        active: list[Path] = []
        invalid: list[Path] = []
        for path in self._lease_files(digest):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if _parse_time(record.get("expires_at"), 0.0) > now:
                    active.append(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # Unknown lease state is treated conservatively as active.
                invalid.append(path)
        return active, invalid

    def _assert_lease_active(self, lease: CacheLease) -> None:
        if lease._released or lease.expires_at <= _now_seconds():
            raise CacheError(f"Cache lease expired: {lease.record.key}")
        if not lease.lease_path.exists():
            raise CacheError(f"Cache lease was released: {lease.record.key}")

    def _read_record(self, key: str, *, touch: bool = True) -> CacheRecord:
        digest, payload_path, manifest_path, ready_path = self._paths(key)
        if not (payload_path.exists() and manifest_path.exists() and ready_path.exists()):
            raise CacheNotReadyError(f"Cache artifact is not ready: {key}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CacheNotReadyError(f"Cache metadata is unreadable: {key}") from exc
        if manifest.get("state") != "ready" or manifest.get("key") != key or ready.get("state") != "ready":
            raise CacheNotReadyError(f"Cache artifact is not in ready state: {key}")
        payload_size = payload_path.stat().st_size
        if int(manifest.get("payload_bytes", -1)) != payload_size:
            raise CacheNotReadyError(f"Cache payload size mismatch: {key}")
        if _sha256_bytes(payload_path.read_bytes()) != manifest.get("payload_sha256"):
            raise CacheNotReadyError(f"Cache payload checksum mismatch: {key}")
        if ready.get("manifest_sha256") != _sha256_bytes(manifest_path.read_bytes()):
            raise CacheNotReadyError(f"Cache ready marker mismatch: {key}")
        if touch:
            manifest["last_access_epoch"] = _now_seconds()
            manifest["last_access_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            ready["manifest_sha256"] = _sha256_bytes(manifest_path.read_bytes())
            _atomic_json(ready_path, ready)
        return CacheRecord(
            key=key, digest=digest, payload_path=payload_path, manifest_path=manifest_path,
            ready_path=ready_path, payload_bytes=payload_size,
            created_at=_parse_time(manifest.get("created_epoch")),
            last_access_at=_parse_time(manifest.get("last_access_epoch"), _parse_time(manifest.get("created_epoch"))),
            metadata=manifest,
        )

    def put_bytes(self, key: str, payload: bytes, metadata: Mapping[str, Any] | None = None) -> CacheRecord:
        """Atomically publish one ready artifact, refusing replacement while leased."""
        if not isinstance(payload, bytes):
            raise TypeError("put_bytes requires bytes; serialize objects before calling it")
        digest, payload_path, manifest_path, ready_path = self._paths(key)
        now = _now_seconds()
        with self._artifact_lock(digest):
            active, invalid = self._active_lease_files(digest, now)
            if active or invalid:
                raise CacheBusyError(f"Cannot replace leased cache artifact: {key}")
            base = dict(metadata or {})
            base.update({
                "key": key, "digest": digest, "state": "ready", "schema_version": 1,
                "payload_bytes": len(payload), "payload_sha256": _sha256_bytes(payload),
                "created_epoch": now, "created_at": _utc_now(),
                "last_access_epoch": now, "last_access_at": _utc_now(),
            })
            temp_payload = self.root / f".{digest}.{os.getpid()}.{secrets.token_hex(8)}.payload.tmp"
            temp_manifest = self.root / f".{digest}.{os.getpid()}.{secrets.token_hex(8)}.manifest.tmp"
            temp_ready = self.root / f".{digest}.{os.getpid()}.{secrets.token_hex(8)}.ready.tmp"
            try:
                with temp_payload.open("wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                with temp_manifest.open("wb") as stream:
                    stream.write((json.dumps(base, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_payload, payload_path)
                os.replace(temp_manifest, manifest_path)
                ready = {"state": "ready", "key": key, "digest": digest, "manifest_sha256": _sha256_bytes(manifest_path.read_bytes())}
                temp_ready.write_text(json.dumps(ready, sort_keys=True) + "\n", encoding="utf-8")
                with temp_ready.open("r+b") as stream:
                    os.fsync(stream.fileno())
                os.replace(temp_ready, ready_path)
            finally:
                temp_payload.unlink(missing_ok=True)
                temp_manifest.unlink(missing_ok=True)
                temp_ready.unlink(missing_ok=True)
        return self._read_record(key)

    def put_text(self, key: str, text: str, metadata: Mapping[str, Any] | None = None) -> CacheRecord:
        return self.put_bytes(key, text.encode("utf-8"), metadata)

    def acquire(self, key: str, *, owner: str | None = None, ttl_seconds: float | None = None) -> CacheLease:
        """Acquire a refcounted lease; active lease files protect cleanup."""
        ttl = self.lease_ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0:
            raise ValueError("Lease TTL must be positive")
        digest, _, _, _ = self._paths(key)
        with self._artifact_lock(digest):
            record = self._read_record(key)
            token = secrets.token_hex(16)
            lease_path = self.leases_dir / f"{digest}.{token}.lease.json"
            expires = _now_seconds() + ttl
            _atomic_json(lease_path, {
                "key": key, "digest": digest, "token": token,
                "owner": owner or f"pid:{os.getpid()}", "created_at": _utc_now(),
                "created_epoch": _now_seconds(), "expires_at": expires,
            })
        return CacheLease(self, record, token, lease_path, expires)

    @contextmanager
    def lease(self, key: str, *, owner: str | None = None, ttl_seconds: float | None = None) -> Iterator[CacheLease]:
        handle = self.acquire(key, owner=owner, ttl_seconds=ttl_seconds)
        try:
            yield handle
        finally:
            handle.release()

    def _artifact_digests(self) -> set[str]:
        digests = {path.name[:-len(".payload")] for path in self.root.glob("*.payload")}
        digests.update(path.name[:-len(".manifest.json")] for path in self.root.glob("*.manifest.json"))
        digests.update(path.name[:-len(".ready")] for path in self.root.glob("*.ready"))
        return digests

    def reconcile(self, *, now: float | None = None) -> dict[str, Any]:
        """Inspect ready, partial, orphaned, corrupt, and leased artifacts."""
        now = _now_seconds() if now is None else now
        ready: list[dict[str, Any]] = []
        incomplete: list[str] = []
        corrupt: list[str] = []
        leased: dict[str, int] = {}
        for digest in sorted(self._artifact_digests()):
            files = {
                "payload": self.root / f"{digest}.payload",
                "manifest": self.root / f"{digest}.manifest.json",
                "ready": self.root / f"{digest}.ready",
            }
            active, invalid = self._active_lease_files(digest, now)
            if active or invalid:
                leased[digest] = len(active) + len(invalid)
            if all(path.exists() for path in files.values()):
                try:
                    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
                    marker = json.loads(files["ready"].read_text(encoding="utf-8"))
                    valid = (
                        manifest.get("state") == "ready" and marker.get("state") == "ready"
                        and marker.get("manifest_sha256") == _sha256_bytes(files["manifest"].read_bytes())
                        and int(manifest.get("payload_bytes", -1)) == files["payload"].stat().st_size
                        and manifest.get("payload_sha256") == _sha256_bytes(files["payload"].read_bytes())
                    )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    valid = False
                if valid:
                    artifact_bytes = sum(files[name].stat().st_size for name in files)
                    ready.append({
                        "digest": digest, "key": manifest.get("key"),
                        "bytes": int(artifact_bytes),
                        "payload_bytes": int(manifest["payload_bytes"]),
                        "created_epoch": _parse_time(manifest.get("created_epoch")),
                        "last_access_epoch": _parse_time(manifest.get("last_access_epoch"), _parse_time(manifest.get("created_epoch"))),
                        "lease_count": leased.get(digest, 0),
                    })
                else:
                    corrupt.append(digest)
            else:
                incomplete.append(digest)
        temp_files = [str(path) for path in self.root.glob(".*.tmp")]
        return {
            "root": str(self.root), "ready": ready, "incomplete": incomplete,
            "corrupt": corrupt, "leased": leased, "temp_files": temp_files,
            "ready_count": len(ready), "active_lease_count": sum(leased.values()),
            "current_bytes": self._usage_bytes(), "observed_epoch": now,
        }

    def _usage_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if not path.is_file() or path == self.high_water_path:
                continue
            if self.leases_dir in path.parents or self.locks_dir in path.parents:
                continue
            try:
                total += path.stat().st_size
            except OSError:
                pass
        return total

    def high_water(self, *, limit_bytes: int | None = None, now: float | None = None) -> dict[str, Any]:
        """Update and return current/peak byte usage and configured limits."""
        now = _now_seconds() if now is None else now
        current = self._usage_bytes()
        if limit_bytes is not None and limit_bytes < 0:
            raise ValueError("limit_bytes must be nonnegative")
        previous: dict[str, Any] = {}
        if self.high_water_path.exists():
            try:
                previous = json.loads(self.high_water_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                previous = {}
        limit = self.max_bytes if limit_bytes is None else limit_bytes
        report = {
            "current_bytes": current,
            "high_water_bytes": max(current, int(previous.get("high_water_bytes", 0))),
            "limit_bytes": limit,
            "over_limit": bool(limit is not None and current > limit),
            "observed_at": _utc_now(),
            "observed_epoch": now,
        }
        _atomic_json(self.high_water_path, report)
        return report

    def cleanup(
        self,
        *,
        max_age_seconds: float | None = None,
        max_bytes: int | None = None,
        now: float | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Remove only unleased stale/over-budget ready artifacts and old debris.

        Cleanup is lease-safe: every candidate takes the same per-artifact lock
        used by lease acquisition, rechecks active leases, and skips leased or
        malformed lease state.  ``dry_run=True`` reports candidates without
        changing files.
        """
        now = _now_seconds() if now is None else now
        age_limit = self.temp_ttl_seconds if max_age_seconds is None else float(max_age_seconds)
        byte_limit = self.max_bytes if max_bytes is None else max_bytes
        if age_limit < 0 or (byte_limit is not None and byte_limit < 0):
            raise ValueError("Cleanup age and byte limits must be nonnegative")
        before = self.reconcile(now=now)
        candidates: list[tuple[str, str, int, float]] = []
        for item in before["ready"]:
            age = now - item["last_access_epoch"]
            if age >= age_limit:
                candidates.append((item["digest"], "expired", item["bytes"], item["last_access_epoch"]))
        usage = before["current_bytes"]
        if byte_limit is not None and usage > byte_limit:
            for item in sorted(before["ready"], key=lambda row: row["last_access_epoch"]):
                if item["digest"] not in {candidate[0] for candidate in candidates}:
                    candidates.append((item["digest"], "high_water", item["bytes"], item["last_access_epoch"]))
                projected = usage - sum(candidate[2] for candidate in candidates)
                if projected <= byte_limit:
                    break
        # Incomplete/corrupt artifacts can be left by a terminated writer or
        # external damage.  They are eligible only after the same age bound and
        # are always checked for leases before removal.
        debris = set(before["incomplete"]) | set(before["corrupt"])
        for digest in sorted(debris):
            paths = [
                self.root / f"{digest}.payload",
                self.root / f"{digest}.manifest.json",
                self.root / f"{digest}.ready",
            ]
            existing = [path for path in paths if path.exists()]
            if existing and all(now - path.stat().st_mtime >= age_limit for path in existing):
                candidates.append((digest, "debris", sum(path.stat().st_size for path in existing), 0.0))
        deleted: list[str] = []
        skipped_leased: list[str] = []
        skipped_recent: list[str] = []
        reclaimed = 0
        for digest, reason, size, _ in candidates:
            lock = self.locks_dir / f"{digest}.lock"
            try:
                with self._artifact_lock(digest):
                    active, invalid = self._active_lease_files(digest, now)
                    if active or invalid:
                        skipped_leased.append(digest)
                        continue
                    if dry_run:
                        deleted.append(digest)
                        reclaimed += size
                        continue
                    for suffix in (".payload", ".manifest.json", ".ready"):
                        (self.root / f"{digest}{suffix}").unlink(missing_ok=True)
                    deleted.append(digest)
                    reclaimed += size
            except CacheBusyError:
                skipped_recent.append(digest)
        # Old temp files are safe to remove because no reader accepts them.
        old_temp: list[str] = []
        for path in self.root.glob(".*.tmp"):
            try:
                if now - path.stat().st_mtime >= age_limit:
                    old_temp.append(str(path))
                    if not dry_run:
                        path.unlink(missing_ok=True)
            except OSError:
                continue
        after_usage = self._usage_bytes()
        expired_leases: list[str] = []
        for lease_path in self.leases_dir.glob("*.lease.json"):
            try:
                lease_record = json.loads(lease_path.read_text(encoding="utf-8"))
                if _parse_time(lease_record.get("expires_at"), now + 1) <= now:
                    expired_leases.append(str(lease_path))
                    if not dry_run:
                        lease_path.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # Malformed lease state is intentionally retained and remains
                # conservative in _active_lease_files.
                continue
        high_water = self.high_water(limit_bytes=byte_limit, now=now)
        return {
            "dry_run": dry_run, "before_bytes": before["current_bytes"],
            "after_bytes": after_usage, "reclaimed_bytes": reclaimed,
            "candidate_count": len(candidates), "deleted_digests": deleted,
            "skipped_leased": skipped_leased, "skipped_busy": skipped_recent,
            "old_temp_files": old_temp, "expired_lease_files": expired_leases,
            "high_water": high_water,
        }
