"""Durable task leases and crash recovery for corrected benchmark runs.

The runner integrates this WAL-backed scheduler alongside its phase
checkpoint ledger:

* tasks are claimed with a renewable lease and an attempt token;
* a worker can publish a result through an atomic file rename followed by a
  SQLite transaction;
* a process that dies after the rename is recovered by reconciliation; and
* expired or retryable failures return to ``pending`` until their attempt
  budget is exhausted.

All timestamps are Unix seconds.  Production callers may use wall-clock time;
tests can inject deterministic timestamps without changing the state machine.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


TASK_STATUSES = frozenset({"pending", "running", "success", "failed", "timed_out"})
ATTEMPT_STATUSES = frozenset({"running", "success", "failed", "timed_out"})
RESULT_SCHEMA_VERSION = 1


class SchedulerError(RuntimeError):
    """Base class for scheduler errors."""


class TaskIdentityMismatch(SchedulerError):
    """A task key was registered with a different immutable specification."""


class LeaseLost(SchedulerError):
    """The caller no longer owns the task lease."""


class ResultConflict(SchedulerError):
    """A different result already exists for the task."""


class InjectedCrash(SchedulerError):
    """Fault-injection marker used by process-boundary tests."""


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Immutable task identity and retry budget."""

    task_key: str
    run_id: str
    payload: dict[str, Any]
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.task_key or not self.run_id:
            raise ValueError("task_key and run_id must be non-empty")
        if not isinstance(self.payload, dict):
            raise TypeError("TaskSpec.payload must be a dictionary")
        if not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")


@dataclass(frozen=True, slots=True)
class Lease:
    """Worker ownership token returned by ``claim_task``."""

    task_key: str
    run_id: str
    attempt_no: int
    worker_id: str
    lease_token: str
    claimed_at: float
    lease_expires_at: float


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Classification and action for a failed attempt."""

    classification: str
    retryable: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PublishOutcome:
    """Result publication receipt."""

    task_key: str
    attempt_no: int
    artifact_path: str
    result_digest: str
    recovered: bool = False


@dataclass(frozen=True, slots=True)
class ExpiredAttempt:
    task_key: str
    attempt_no: int
    worker_id: str
    lease_expires_at: float
    next_status: str


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    scanned_attempts: int
    adopted_results: int
    expired_attempts: int
    invalid_artifacts: tuple[str, ...]
    orphan_artifacts: tuple[str, ...]
    removed_temp_files: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_json(value: Any) -> str:
    return _digest_bytes(_canonical_json(value).encode("utf-8"))


def classify_failure(error_type: str, error_summary: str = "") -> RetryDecision:
    """Classify a failure without retrying arbitrary deterministic errors.

    Transient I/O, timeout, lock, and worker-process failures are retryable.
    Input/schema/programming errors are permanent.  Unknown errors are kept
    non-retryable until a caller explicitly adds a classification rule, which
    prevents an expensive run from looping indefinitely on a deterministic bug.
    """

    name = str(error_type).rsplit(".", 1)[-1]
    summary = str(error_summary).lower()
    if name in {"TimeoutError", "ConnectionError", "BrokenPipeError", "InterruptedError", "BlockingIOError"}:
        return RetryDecision("transient", True, f"{name} is a transient worker/I/O failure")
    if name in {"InjectedCrash", "WorkerCrashed", "ProcessLookupError", "EOFError", "LeaseExpired"}:
        return RetryDecision("worker_crash", True, f"{name} indicates worker loss")
    if name in {"OperationalError", "DatabaseError"} and any(word in summary for word in ("locked", "busy", "timeout")):
        return RetryDecision("transient", True, "SQLite or external resource was temporarily unavailable")
    if name in {"MemoryError", "Resource temporarily unavailable"}:
        return RetryDecision("transient", True, f"{name} may resolve after resource recovery")
    if name in {"ValueError", "TypeError", "KeyError", "IndexError", "AssertionError", "FileNotFoundError", "TaskIdentityMismatch"}:
        return RetryDecision("permanent", False, f"{name} indicates deterministic task or input failure")
    return RetryDecision("unknown", False, f"{name} is not in the retry allowlist")


class TaskScheduler:
    """SQLite-WAL-backed task scheduler with durable result artifacts."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        artifact_dir: str | Path | None = None,
        lease_seconds: float = 900.0,
        retry_backoff_seconds: float = 0.0,
        busy_timeout_ms: int = 30_000,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else self.db_path.parent / "task_results"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if not (lease_seconds > 0.0):
            raise ValueError("lease_seconds must be positive")
        if retry_backoff_seconds < 0.0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if busy_timeout_ms < 1:
            raise ValueError("busy_timeout_ms must be positive")
        self.lease_seconds = float(lease_seconds)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.initialize()

    @contextlib.contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            # WAL and FULL synchronous are set for every connection because
            # SQLite PRAGMAs are connection-scoped except journal_mode.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduler_tasks (
                    task_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    spec_hash TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL CHECK(max_attempts >= 1),
                    status TEXT NOT NULL CHECK(status IN ('pending','running','success','failed','timed_out')),
                    available_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    final_result_digest TEXT,
                    final_artifact_path TEXT
                );
                CREATE TABLE IF NOT EXISTS scheduler_attempts (
                    task_key TEXT NOT NULL REFERENCES scheduler_tasks(task_key),
                    attempt_no INTEGER NOT NULL CHECK(attempt_no >= 1),
                    worker_id TEXT NOT NULL,
                    lease_token TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN ('running','success','failed','timed_out')),
                    claimed_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    lease_expires_at REAL NOT NULL,
                    finished_at REAL,
                    retryable INTEGER,
                    error_type TEXT,
                    error_summary TEXT,
                    PRIMARY KEY(task_key, attempt_no)
                );
                CREATE TABLE IF NOT EXISTS scheduler_results (
                    task_key TEXT PRIMARY KEY REFERENCES scheduler_tasks(task_key),
                    attempt_no INTEGER NOT NULL,
                    result_digest TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    published_at REAL NOT NULL,
                    recovered INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS scheduler_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time REAL NOT NULL,
                    task_key TEXT,
                    attempt_no INTEGER,
                    event_type TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS scheduler_tasks_ready_idx
                    ON scheduler_tasks(status, available_at, created_at);
                CREATE INDEX IF NOT EXISTS scheduler_attempts_expiry_idx
                    ON scheduler_attempts(status, lease_expires_at);
                """
            )

    @staticmethod
    def _begin(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _commit(conn: sqlite3.Connection) -> None:
        conn.commit()

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        if conn.in_transaction:
            conn.rollback()

    def _event(
        self,
        conn: sqlite3.Connection,
        *,
        now: float,
        event_type: str,
        task_key: str | None = None,
        attempt_no: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO scheduler_events(event_time, task_key, attempt_no, event_type, details_json) VALUES (?, ?, ?, ?, ?)",
            (now, task_key, attempt_no, event_type, _canonical_json(details or {})),
        )

    def register_tasks(self, tasks: Iterable[TaskSpec], *, now: float | None = None) -> int:
        """Register immutable task specs; reject changed identities on resume."""

        timestamp = time.time() if now is None else float(now)
        task_list = list(tasks)
        inserted = 0
        with self._connection() as conn:
            self._begin(conn)
            try:
                for spec in task_list:
                    spec_json = _canonical_json(spec.payload)
                    spec_hash = _digest_json({"run_id": spec.run_id, "payload": spec.payload, "max_attempts": spec.max_attempts})
                    row = conn.execute(
                        "SELECT run_id, spec_hash, max_attempts FROM scheduler_tasks WHERE task_key=?",
                        (spec.task_key,),
                    ).fetchone()
                    if row:
                        if row["run_id"] != spec.run_id or row["spec_hash"] != spec_hash or row["max_attempts"] != spec.max_attempts:
                            raise TaskIdentityMismatch(f"Task specification changed for {spec.task_key!r}")
                        continue
                    conn.execute(
                        """
                        INSERT INTO scheduler_tasks(
                            task_key, run_id, spec_json, spec_hash, max_attempts, status,
                            available_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                        """,
                        (spec.task_key, spec.run_id, spec_json, spec_hash, spec.max_attempts, timestamp, timestamp, timestamp),
                    )
                    self._event(conn, now=timestamp, task_key=spec.task_key, event_type="registered", details={"run_id": spec.run_id})
                    inserted += 1
                self._commit(conn)
            except Exception:
                self._rollback(conn)
                raise
        return inserted

    def _artifact_path(self, run_id: str, task_key: str) -> Path:
        identity = f"{run_id}\0{task_key}".encode("utf-8")
        return self.artifact_dir / f"{hashlib.sha256(identity).hexdigest()}.json"

    def _expire_leases_tx(self, conn: sqlite3.Connection, *, now: float) -> list[ExpiredAttempt]:
        rows = conn.execute(
            """
            SELECT a.task_key, a.attempt_no, a.worker_id, a.lease_expires_at,
                   t.max_attempts
            FROM scheduler_attempts AS a
            JOIN scheduler_tasks AS t ON t.task_key=a.task_key
            WHERE a.status='running' AND a.lease_expires_at <= ?
            ORDER BY a.task_key, a.attempt_no
            """,
            (now,),
        ).fetchall()
        expired: list[ExpiredAttempt] = []
        for row in rows:
            next_status = "pending" if row["attempt_no"] < row["max_attempts"] else "timed_out"
            conn.execute(
                "UPDATE scheduler_attempts SET status='timed_out', finished_at=?, retryable=1, error_type='LeaseExpired', error_summary=? WHERE task_key=? AND attempt_no=? AND status='running'",
                (now, "Lease expired before result publication", row["task_key"], row["attempt_no"]),
            )
            conn.execute(
                "UPDATE scheduler_tasks SET status=?, available_at=?, updated_at=? WHERE task_key=? AND status='running'",
                (next_status, now, now, row["task_key"]),
            )
            self._event(
                conn, now=now, task_key=row["task_key"], attempt_no=row["attempt_no"],
                event_type="lease_expired", details={"next_status": next_status, "worker_id": row["worker_id"]},
            )
            expired.append(ExpiredAttempt(row["task_key"], row["attempt_no"], row["worker_id"], row["lease_expires_at"], next_status))
        return expired

    def claim_task(self, worker_id: str, *, task_key: str | None = None, now: float | None = None) -> Lease | None:
        """Atomically claim a ready task, optionally requiring a specific key."""

        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        timestamp = time.time() if now is None else float(now)
        # A crashed worker may have renamed its result before dying.  Recover
        # that artifact before permitting a replacement attempt to overwrite it.
        # Startup performs a full audit.  Before an individual claim, only
        # that task's artifact can affect whether the claim is safe.  A full
        # directory scan here grows quadratically with completed task count.
        self.reconcile(now=timestamp, task_key=task_key)
        with self._connection() as conn:
            self._begin(conn)
            try:
                self._expire_leases_tx(conn, now=timestamp)
                if task_key is None:
                    row = conn.execute(
                        """
                        SELECT task_key, run_id, max_attempts
                        FROM scheduler_tasks
                        WHERE status='pending' AND available_at <= ?
                        ORDER BY created_at, task_key
                        LIMIT 1
                        """,
                        (timestamp,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT task_key, run_id, max_attempts
                        FROM scheduler_tasks
                        WHERE task_key=? AND status='pending' AND available_at <= ?
                        LIMIT 1
                        """,
                        (task_key, timestamp),
                    ).fetchone()
                if row is None:
                    self._commit(conn)
                    return None
                previous = conn.execute(
                    "SELECT COALESCE(MAX(attempt_no), 0) AS max_attempt FROM scheduler_attempts WHERE task_key=?",
                    (row["task_key"],),
                ).fetchone()["max_attempt"]
                attempt_no = int(previous) + 1
                if attempt_no > row["max_attempts"]:
                    conn.execute("UPDATE scheduler_tasks SET status='timed_out', updated_at=? WHERE task_key=?", (timestamp, row["task_key"]))
                    self._commit(conn)
                    return None
                token = secrets.token_urlsafe(32)
                expires = timestamp + self.lease_seconds
                conn.execute(
                    """
                    INSERT INTO scheduler_attempts(
                        task_key, attempt_no, worker_id, lease_token, status,
                        claimed_at, heartbeat_at, lease_expires_at
                    ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                    """,
                    (row["task_key"], attempt_no, worker_id, token, timestamp, timestamp, expires),
                )
                conn.execute("UPDATE scheduler_tasks SET status='running', updated_at=? WHERE task_key=?", (timestamp, row["task_key"]))
                self._event(conn, now=timestamp, task_key=row["task_key"], attempt_no=attempt_no, event_type="claimed", details={"worker_id": worker_id, "lease_expires_at": expires})
                self._commit(conn)
                return Lease(row["task_key"], row["run_id"], attempt_no, worker_id, token, timestamp, expires)
            except Exception:
                self._rollback(conn)
                raise

    def heartbeat(self, lease: Lease, *, now: float | None = None) -> Lease:
        """Extend a live lease and return its updated receipt."""

        timestamp = time.time() if now is None else float(now)
        with self._connection() as conn:
            self._begin(conn)
            try:
                row = conn.execute(
                    "SELECT worker_id, status, lease_expires_at FROM scheduler_attempts WHERE task_key=? AND attempt_no=? AND lease_token=?",
                    (lease.task_key, lease.attempt_no, lease.lease_token),
                ).fetchone()
                if row is None or row["worker_id"] != lease.worker_id or row["status"] != "running" or row["lease_expires_at"] <= timestamp:
                    raise LeaseLost(f"Lease is no longer live for {lease.task_key!r} attempt {lease.attempt_no}")
                expires = timestamp + self.lease_seconds
                conn.execute(
                    "UPDATE scheduler_attempts SET heartbeat_at=?, lease_expires_at=? WHERE task_key=? AND attempt_no=? AND lease_token=?",
                    (timestamp, expires, lease.task_key, lease.attempt_no, lease.lease_token),
                )
                self._event(conn, now=timestamp, task_key=lease.task_key, attempt_no=lease.attempt_no, event_type="heartbeat", details={"lease_expires_at": expires})
                self._commit(conn)
                return Lease(lease.task_key, lease.run_id, lease.attempt_no, lease.worker_id, lease.lease_token, lease.claimed_at, expires)
            except Exception:
                self._rollback(conn)
                raise

    renew_lease = heartbeat

    def _write_artifact(
        self,
        *,
        run_id: str,
        task_key: str,
        attempt_no: int,
        lease_token: str,
        payload: dict[str, Any],
        created_at: float,
        fault_stage: str | None,
    ) -> tuple[Path, str]:
        result_json = _canonical_json(payload)
        result_digest = _digest_bytes(result_json.encode("utf-8"))
        envelope = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": run_id,
            "task_key": task_key,
            "attempt_no": attempt_no,
            "lease_token": lease_token,
            "created_at": created_at,
            "result_digest": result_digest,
            "payload": payload,
        }
        artifact_path = self._artifact_path(run_id, task_key)
        temp_path = artifact_path.with_name(f".{artifact_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        encoded = (_canonical_json(envelope) + "\n").encode("utf-8")
        try:
            with temp_path.open("wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if fault_stage == "after_temp":
                raise InjectedCrash("fault injected after durable temporary artifact write")
            os.replace(temp_path, artifact_path)
            self._fsync_directory(artifact_path.parent)
            if fault_stage == "after_artifact":
                raise InjectedCrash("fault injected after durable artifact rename")
        finally:
            # A killed process may leave a temp file; reconciliation cleans
            # old files, while a normal exception can remove this one now.
            if temp_path.exists() and fault_stage != "after_temp":
                temp_path.unlink(missing_ok=True)
        return artifact_path, result_digest

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def publish_result(
        self,
        lease: Lease,
        payload: dict[str, Any],
        *,
        now: float | None = None,
        fault_stage: str | None = None,
    ) -> PublishOutcome:
        """Atomically publish a result artifact and its committed DB state.

        ``fault_stage`` exists only for process-boundary tests and should be
        omitted in production.  A crash after rename leaves a self-describing
        artifact that ``reconcile`` can safely adopt.
        """

        if fault_stage not in {None, "after_temp", "after_artifact"}:
            raise ValueError("fault_stage must be None, 'after_temp', or 'after_artifact'")
        timestamp = time.time() if now is None else float(now)
        requested_digest = _digest_json(payload)
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT t.run_id, t.status AS task_status, a.status AS attempt_status, a.worker_id,
                       a.lease_expires_at
                FROM scheduler_tasks AS t
                JOIN scheduler_attempts AS a ON a.task_key=t.task_key
                WHERE t.task_key=? AND a.attempt_no=? AND a.lease_token=?
                """,
                (lease.task_key, lease.attempt_no, lease.lease_token),
            ).fetchone()
        if row is None or row["worker_id"] != lease.worker_id:
            raise LeaseLost(f"Unknown lease for {lease.task_key!r} attempt {lease.attempt_no}")
        if row["attempt_status"] == "success" or row["task_status"] == "success":
            with self._connection() as conn:
                result = conn.execute("SELECT * FROM scheduler_results WHERE task_key=?", (lease.task_key,)).fetchone()
            if result:
                if result["result_digest"] != requested_digest:
                    raise ResultConflict(f"A different result already exists for {lease.task_key!r}")
                return PublishOutcome(lease.task_key, lease.attempt_no, result["artifact_path"], result["result_digest"], recovered=bool(result["recovered"]))
            raise SchedulerError(f"Task {lease.task_key!r} is marked successful without a published result")
        if row["attempt_status"] != "running" or row["lease_expires_at"] <= timestamp:
            raise LeaseLost(f"Lease is no longer live for {lease.task_key!r} attempt {lease.attempt_no}")
        artifact_path, result_digest = self._write_artifact(
            run_id=lease.run_id, task_key=lease.task_key, attempt_no=lease.attempt_no,
            lease_token=lease.lease_token, payload=payload, created_at=timestamp,
            fault_stage=fault_stage,
        )
        with self._connection() as conn:
            self._begin(conn)
            try:
                row = conn.execute(
                    "SELECT status, worker_id, lease_expires_at FROM scheduler_attempts WHERE task_key=? AND attempt_no=? AND lease_token=?",
                    (lease.task_key, lease.attempt_no, lease.lease_token),
                ).fetchone()
                if row is None or row["worker_id"] != lease.worker_id or row["status"] != "running" or row["lease_expires_at"] <= timestamp:
                    raise LeaseLost(f"Lease was lost before DB publication for {lease.task_key!r}")
                existing = conn.execute("SELECT result_digest, artifact_path FROM scheduler_results WHERE task_key=?", (lease.task_key,)).fetchone()
                if existing and existing["result_digest"] != result_digest:
                    raise ResultConflict(f"A different result already exists for {lease.task_key!r}")
                if existing:
                    self._commit(conn)
                    return PublishOutcome(lease.task_key, lease.attempt_no, existing["artifact_path"], existing["result_digest"], recovered=bool(False))
                conn.execute(
                    "INSERT INTO scheduler_results(task_key, attempt_no, result_digest, artifact_path, published_at, recovered) VALUES (?, ?, ?, ?, ?, 0)",
                    (lease.task_key, lease.attempt_no, result_digest, str(artifact_path), timestamp),
                )
                conn.execute(
                    "UPDATE scheduler_attempts SET status='success', finished_at=?, retryable=0 WHERE task_key=? AND attempt_no=? AND lease_token=?",
                    (timestamp, lease.task_key, lease.attempt_no, lease.lease_token),
                )
                conn.execute(
                    "UPDATE scheduler_tasks SET status='success', final_result_digest=?, final_artifact_path=?, updated_at=? WHERE task_key=?",
                    (result_digest, str(artifact_path), timestamp, lease.task_key),
                )
                self._event(conn, now=timestamp, task_key=lease.task_key, attempt_no=lease.attempt_no, event_type="result_published", details={"artifact_path": str(artifact_path), "result_digest": result_digest})
                self._commit(conn)
            except Exception:
                self._rollback(conn)
                raise
        return PublishOutcome(lease.task_key, lease.attempt_no, str(artifact_path), result_digest)

    def record_failure(
        self,
        lease: Lease,
        error_type: str,
        error_summary: str = "",
        *,
        now: float | None = None,
    ) -> RetryDecision:
        """Durably classify a failure and schedule the next attempt if allowed."""

        timestamp = time.time() if now is None else float(now)
        decision = classify_failure(error_type, error_summary)
        with self._connection() as conn:
            self._begin(conn)
            try:
                row = conn.execute(
                    """
                    SELECT a.status, a.worker_id, t.max_attempts
                    FROM scheduler_attempts AS a JOIN scheduler_tasks AS t ON t.task_key=a.task_key
                    WHERE a.task_key=? AND a.attempt_no=? AND a.lease_token=?
                    """,
                    (lease.task_key, lease.attempt_no, lease.lease_token),
                ).fetchone()
                if row is None or row["worker_id"] != lease.worker_id or row["status"] != "running":
                    raise LeaseLost(f"Cannot record failure for lost lease {lease.task_key!r}")
                retry = decision.retryable and lease.attempt_no < row["max_attempts"]
                next_status = "pending" if retry else "failed"
                available_at = timestamp + (self.retry_backoff_seconds * (2 ** max(0, lease.attempt_no - 1)) if retry else 0.0)
                conn.execute(
                    "UPDATE scheduler_attempts SET status='failed', finished_at=?, retryable=?, error_type=?, error_summary=? WHERE task_key=? AND attempt_no=? AND lease_token=?",
                    (timestamp, int(decision.retryable), str(error_type), str(error_summary)[:2000], lease.task_key, lease.attempt_no, lease.lease_token),
                )
                conn.execute(
                    "UPDATE scheduler_tasks SET status=?, available_at=?, updated_at=? WHERE task_key=?",
                    (next_status, available_at, timestamp, lease.task_key),
                )
                self._event(conn, now=timestamp, task_key=lease.task_key, attempt_no=lease.attempt_no, event_type="failed", details={"classification": decision.classification, "retryable": decision.retryable, "next_status": next_status, "error_type": str(error_type)})
                self._commit(conn)
            except Exception:
                self._rollback(conn)
                raise
        return decision

    def expire_leases(self, *, now: float | None = None) -> list[ExpiredAttempt]:
        """Mark expired running attempts timed out and return them to the queue."""

        timestamp = time.time() if now is None else float(now)
        with self._connection() as conn:
            self._begin(conn)
            try:
                expired = self._expire_leases_tx(conn, now=timestamp)
                self._commit(conn)
                return expired
            except Exception:
                self._rollback(conn)
                raise

    def reclaim_foreign_leases(self, worker_id: str, *, now: float | None = None) -> list[ExpiredAttempt]:
        """Reclaim live leases owned by a prior process on this single-host run.

        The corrected runner has one coordinator per run directory.  On a new
        process after a forced termination, a prior lease may still have time
        remaining even though its worker no longer exists.  This explicit
        startup operation converts only leases owned by a different worker ID
        to the same pending/timed-out state used by expiry; it is intentionally
        not a multi-host coordination primitive.
        """
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        timestamp = time.time() if now is None else float(now)
        with self._connection() as conn:
            self._begin(conn)
            try:
                rows = conn.execute(
                    """
                    SELECT a.task_key, a.attempt_no, a.worker_id, a.lease_expires_at,
                           t.max_attempts
                    FROM scheduler_attempts AS a
                    JOIN scheduler_tasks AS t ON t.task_key=a.task_key
                    WHERE a.status='running' AND a.worker_id <> ?
                    ORDER BY a.task_key, a.attempt_no
                    """,
                    (worker_id,),
                ).fetchall()
                reclaimed: list[ExpiredAttempt] = []
                for row in rows:
                    next_status = "pending" if row["attempt_no"] < row["max_attempts"] else "timed_out"
                    conn.execute(
                        "UPDATE scheduler_attempts SET status='timed_out', finished_at=?, retryable=1, error_type='WorkerReplaced', error_summary=? WHERE task_key=? AND attempt_no=? AND status='running'",
                        (timestamp, "Prior runner process was replaced before result publication", row["task_key"], row["attempt_no"]),
                    )
                    conn.execute(
                        "UPDATE scheduler_tasks SET status=?, available_at=?, updated_at=? WHERE task_key=? AND status='running'",
                        (next_status, timestamp, timestamp, row["task_key"]),
                    )
                    self._event(
                        conn, now=timestamp, task_key=row["task_key"], attempt_no=row["attempt_no"],
                        event_type="worker_reclaimed", details={"next_status": next_status, "worker_id": row["worker_id"]},
                    )
                    reclaimed.append(ExpiredAttempt(row["task_key"], row["attempt_no"], row["worker_id"], row["lease_expires_at"], next_status))
                self._commit(conn)
                return reclaimed
            except Exception:
                self._rollback(conn)
                raise

    @staticmethod
    def _read_artifact(path: Path) -> dict[str, Any]:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        required = {"schema_version", "run_id", "task_key", "attempt_no", "lease_token", "result_digest", "payload"}
        if not required <= set(envelope):
            raise ValueError("artifact envelope is missing required fields")
        if envelope["schema_version"] != RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported artifact schema version")
        if not isinstance(envelope["payload"], dict):
            raise ValueError("artifact payload must be a dictionary")
        observed = _digest_json(envelope["payload"])
        if observed != envelope["result_digest"]:
            raise ValueError("artifact payload digest mismatch")
        return envelope

    def reconcile(
        self, *, now: float | None = None, temp_max_age_seconds: float = 3600.0,
        task_key: str | None = None,
    ) -> ReconciliationReport:
        """Recover durable artifacts and expire stale leases.

        A named task checks only its attempt artifacts before that task is
        claimed.  The default full audit also scans orphan files and debris;
        it is used at coordinator startup and for unqualified queue claims.
        """

        timestamp = time.time() if now is None else float(now)
        if temp_max_age_seconds < 0:
            raise ValueError("temp_max_age_seconds cannot be negative")
        expired = self.expire_leases(now=timestamp)
        invalid: list[str] = []
        orphan: list[str] = []
        adopted = 0
        scanned = 0
        temp_removed = 0
        with self._connection() as conn:
            query = """
                SELECT a.task_key, a.attempt_no, a.lease_token, a.status,
                       t.run_id, t.status AS task_status
                FROM scheduler_attempts AS a JOIN scheduler_tasks AS t ON t.task_key=a.task_key
                WHERE a.status IN ('running','timed_out','failed')
            """
            parameters: tuple[Any, ...] = ()
            if task_key is not None:
                query += " AND a.task_key=?"
                parameters = (task_key,)
            query += " ORDER BY a.task_key, a.attempt_no"
            rows = conn.execute(query, parameters).fetchall()
        known_paths: set[Path] = set()
        for row in rows:
            scanned += 1
            path = self._artifact_path(row["run_id"], row["task_key"])
            known_paths.add(path)
            if not path.exists():
                continue
            try:
                envelope = self._read_artifact(path)
                if (
                    envelope["run_id"] != row["run_id"]
                    or envelope["task_key"] != row["task_key"]
                    or int(envelope["attempt_no"]) != int(row["attempt_no"])
                    or envelope["lease_token"] != row["lease_token"]
                ):
                    raise ValueError("artifact identity does not match scheduler attempt")
            except Exception as exc:
                invalid.append(f"{path.name}: {exc}")
                continue
            with self._connection() as conn:
                self._begin(conn)
                try:
                    existing = conn.execute("SELECT result_digest FROM scheduler_results WHERE task_key=?", (row["task_key"],)).fetchone()
                    if existing:
                        if existing["result_digest"] != envelope["result_digest"]:
                            raise ResultConflict(f"reconciliation found conflicting result for {row['task_key']!r}")
                        self._commit(conn)
                        continue
                    conn.execute(
                        "INSERT INTO scheduler_results(task_key, attempt_no, result_digest, artifact_path, published_at, recovered) VALUES (?, ?, ?, ?, ?, 1)",
                        (row["task_key"], row["attempt_no"], envelope["result_digest"], str(path), timestamp),
                    )
                    conn.execute(
                        "UPDATE scheduler_attempts SET status='success', finished_at=?, retryable=0 WHERE task_key=? AND attempt_no=?",
                        (timestamp, row["task_key"], row["attempt_no"]),
                    )
                    conn.execute(
                        "UPDATE scheduler_tasks SET status='success', final_result_digest=?, final_artifact_path=?, updated_at=? WHERE task_key=?",
                        (envelope["result_digest"], str(path), timestamp, row["task_key"]),
                    )
                    self._event(conn, now=timestamp, task_key=row["task_key"], attempt_no=row["attempt_no"], event_type="result_recovered", details={"artifact_path": str(path), "result_digest": envelope["result_digest"]})
                    self._commit(conn)
                    adopted += 1
                except Exception:
                    self._rollback(conn)
                    raise
        if task_key is None:
            for path in self.artifact_dir.glob(".*.tmp"):
                try:
                    age = max(0.0, timestamp - path.stat().st_mtime)
                    if age >= temp_max_age_seconds:
                        path.unlink(missing_ok=True)
                        temp_removed += 1
                except OSError:
                    continue
            for path in self.artifact_dir.glob("*.json"):
                if path in known_paths:
                    continue
                try:
                    envelope = self._read_artifact(path)
                    with self._connection() as conn:
                        exists = conn.execute("SELECT 1 FROM scheduler_tasks WHERE task_key=? AND run_id=?", (envelope["task_key"], envelope["run_id"])).fetchone()
                    if not exists:
                        orphan.append(path.name)
                except Exception:
                    invalid.append(f"{path.name}: unrecognized orphan artifact")
        return ReconciliationReport(len(rows), adopted, len(expired), tuple(invalid), tuple(orphan), temp_removed)

    def task_state(self, task_key: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM scheduler_tasks WHERE task_key=?", (task_key,)).fetchone()
        return dict(row) if row else None

    def attempt_state(self, task_key: str, attempt_no: int) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM scheduler_attempts WHERE task_key=? AND attempt_no=?", (task_key, attempt_no)).fetchone()
        return dict(row) if row else None

    def result_state(self, task_key: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM scheduler_results WHERE task_key=?", (task_key,)).fetchone()
        return dict(row) if row else None

    def events(self, task_key: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as conn:
            if task_key is None:
                rows = conn.execute("SELECT * FROM scheduler_events ORDER BY event_id").fetchall()
            else:
                rows = conn.execute("SELECT * FROM scheduler_events WHERE task_key=? ORDER BY event_id", (task_key,)).fetchall()
        return [dict(row) for row in rows]
