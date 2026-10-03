"""Manifest, attempt, state, and recovery primitives for benchmark runs.

The manifest is the authoritative accounting layer.  A result ledger remains a
human-readable export, while SQLite supplies transactional task claims,
attempt history, durable result identity, and stale-worker fencing.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import time
import traceback as traceback_module
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION
from src.artifact_integrity import artifact_lock, atomic_bytes, dataset_identity, file_sha256


MANIFEST_SCHEMA_VERSION = "task_manifest_logical_history_v2"
TASK_IDENTITY_VERSION = "scientific_task_identity_data_v2"
ATTEMPT_SCHEMA_VERSION = "attempt_history_v1"
TASK_STATES = ("pending", "running", "completed", "failed", "timeout", "skipped")
TERMINAL_STATES = {"completed", "failed", "timeout", "skipped"}
RETRYABLE_FAILURES = {"worker_exception", "metric_failure", "precompute_failure", "timeout", "result_write_failure", "coordinator_crash"}


class ManifestError(RuntimeError):
    """Base class for manifest consistency errors."""


class ManifestConflictError(ManifestError):
    """Raised for duplicate claims, incompatible identities, or stale writers."""


class _ClosingConnection(sqlite3.Connection):
    """Commit/rollback and deterministically close this process-owned handle."""

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


@dataclass(frozen=True)
class ExecutionConfig:
    """Declared stopping and retry controls persisted with a run."""

    max_workers: int = 1
    task_timeout_seconds: float | None = None
    run_wall_time_seconds: float | None = None
    max_attempts: int = 1
    retryable_failure_classes: tuple[str, ...] = tuple(sorted(RETRYABLE_FAILURES))
    stop_after_tasks: int | None = None
    stale_after_seconds: float = 3600.0
    graceful_stop: bool = True

    def __post_init__(self) -> None:
        if self.max_workers < 1 or self.max_attempts < 1:
            raise ValueError("max_workers and max_attempts must be positive")
        if self.task_timeout_seconds is not None and self.task_timeout_seconds <= 0:
            raise ValueError("task_timeout_seconds must be positive")
        if self.run_wall_time_seconds is not None and self.run_wall_time_seconds <= 0:
            raise ValueError("run_wall_time_seconds must be positive")
        if self.stop_after_tasks is not None and self.stop_after_tasks < 1:
            raise ValueError("stop_after_tasks must be positive")
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retryable_failure_classes"] = list(self.retryable_failure_classes)
        return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value: Any, prefix: str, length: int = 24) -> str:
    return prefix + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:length]


def scientific_task_id(identity: Mapping[str, Any]) -> str:
    """Derive a deterministic ID from scientific identity only."""
    return _hash({"identity_version": TASK_IDENTITY_VERSION, **dict(identity)}, "task_")


def run_id_for(config: Mapping[str, Any]) -> str:
    """Derive a reproducible run identity from protocol and execution config."""
    return _hash({"manifest_schema": MANIFEST_SCHEMA_VERSION, **dict(config)}, "run_")


def attempt_id_for(run_id: str, task_id: str, attempt_number: int) -> str:
    return _hash({"run_id": run_id, "task_id": task_id, "attempt_number": int(attempt_number)}, "attempt_")


def _normalise_condition(value: Any) -> tuple[str, float, str]:
    if isinstance(value, Mapping):
        family = str(value["shift_family"])
        severity = float(value.get("severity", 0.0))
        condition = str(value.get("condition", family if severity == 0.0 else f"{family}_{severity}"))
        return family, severity, condition
    if len(value) == 2:
        family, severity = value
        severity = float(severity)
        return str(family), severity, str(family) if severity == 0.0 else f"{family}_{severity}"
    family, severity, condition = value
    return str(family), float(severity), str(condition)


def build_task_records(
    datasets: Iterable[str],
    seeds: Iterable[int],
    folds: Iterable[int],
    conditions: Iterable[Any],
    pipelines: Iterable[str],
    models: Iterable[str],
    *,
    pipeline_identity: Callable[[str], str] | None = None,
    pipeline_metadata: Callable[[str], Mapping[str, Any]] | None = None,
    data_paths: Mapping[str, str | Path] | None = None,
    data_identities: Mapping[str, Mapping[str, Any]] | None = None,
    protocol_version: str = EVALUATION_PROTOCOL_VERSION,
    seed_scheme_version: str = SEED_SCHEME_VERSION,
    include_precompute: bool = True,
) -> list[dict[str, Any]]:
    """Build a complete deterministic manifest for a bounded or full grid."""
    datasets = [str(value) for value in datasets]
    seeds = [int(value) for value in seeds]
    folds = [int(value) for value in folds]
    conditions = [_normalise_condition(value) for value in conditions]
    pipelines = [str(value) for value in pipelines]
    models = [str(value) for value in models]
    records: list[dict[str, Any]] = []
    precompute_ids: dict[tuple[Any, ...], str] = {}
    for dataset in datasets:
        available = data_paths is None or (dataset in data_paths and Path(data_paths[dataset]).exists())
        data_signature = dict(data_identities[dataset]) if data_identities is not None else (dataset_identity(data_paths[dataset]) if data_paths is not None and available else {"availability": "unverified" if data_paths is None else "unavailable", "fingerprint": None})
        dataset_reason = None if available else "dataset_unavailable"
        for seed in seeds:
            for fold in folds:
                for shift_family, severity, condition in conditions:
                    base = {
                        "dataset": dataset,
                        "data_path": None if data_paths is None else str(data_paths.get(dataset)),
                        "data_identity": data_signature,
                        "protocol_version": protocol_version,
                        "seed_scheme_version": seed_scheme_version,
                        "split_policy": shift_family if shift_family in {"covariate_shift", "population_shift"} else "stratified",
                        "seed": seed,
                        "fold": fold,
                        "shift_family": shift_family,
                        "severity": severity,
                        "condition": condition,
                    }
                    pre_identity = {**base, "stage": "precompute", "pipeline": "__all__", "model": "__all__"}
                    pre_id = scientific_task_id(pre_identity)
                    precompute_ids[(dataset, seed, fold, condition)] = pre_id
                    if include_precompute:
                        records.append({
                            **pre_identity,
                            "scientific_task_id": pre_id,
                            "task_kind": "precompute",
                            "stage": "precompute",
                            "pipeline": "__all__",
                            "model": "__all__",
                            "initial_state": "skipped" if dataset_reason else "pending",
                            "planned_skip_reason": dataset_reason,
                            "depends_on": [],
                        })
                    for pipeline in pipelines:
                        identity = pipeline_identity(pipeline) if pipeline_identity else pipeline
                        metadata = dict(pipeline_metadata(pipeline) if pipeline_metadata else {})
                        for model in models:
                            task_identity = {
                                **base,
                                "stage": "model",
                                "pipeline": pipeline,
                                "pipeline_identity": identity,
                                "operator_set_id": metadata.get("operator_set_id"),
                                "cap_policy_version": metadata.get("cap_policy_version"),
                                "model": model,
                            }
                            task_id = scientific_task_id(task_identity)
                            records.append({
                                **task_identity,
                                "scientific_task_id": task_id,
                                "task_kind": "model",
                                "stage": "model",
                                "initial_state": "skipped" if dataset_reason else "pending",
                                "planned_skip_reason": dataset_reason,
                                "depends_on": [pre_id],
                            })
    ids = [record["scientific_task_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ManifestConflictError("Scientific task identity collision in manifest")
    return records


class ManifestStore:
    """SQLite-backed manifest and append-only attempt state machine."""

    def __init__(self, db_path: str | Path, *, manifest_path: str | Path | None = None, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.manifest_path = None if manifest_path is None else Path(manifest_path)
        if read_only:
            if not self.db_path.is_file():
                raise FileNotFoundError(self.db_path)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        target = self.db_path.resolve().as_uri()+"?mode=ro" if self.read_only else str(self.db_path)
        connection = sqlite3.connect(target, timeout=30.0, uri=self.read_only, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    protocol_version TEXT NOT NULL,
                    seed_scheme_version TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    run_id TEXT NOT NULL,
                    scientific_task_id TEXT NOT NULL,
                    task_kind TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    planned_skip_reason TEXT,
                    outcome_reason TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    active_attempt_id TEXT,
                    result_ref TEXT,
                    checkpoint_identity TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, scientific_task_id),
                    CHECK (state IN ('pending','running','completed','failed','timeout','skipped'))
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    run_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    scientific_task_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    state TEXT NOT NULL,
                    worker_id TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    elapsed_seconds REAL,
                    outcome TEXT,
                    exception_type TEXT,
                    exception_message TEXT,
                    traceback TEXT,
                    timeout_seconds REAL,
                    result_ref TEXT,
                    PRIMARY KEY (run_id, attempt_id),
                    UNIQUE (run_id, scientific_task_id, attempt_number)
                );
                CREATE TABLE IF NOT EXISTS durable_results (
                    run_id TEXT NOT NULL,
                    scientific_task_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_ref TEXT,
                    durable_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, scientific_task_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_attempt_id ON attempts(run_id, attempt_id);
                CREATE TABLE IF NOT EXISTS evidence_anchors (
                    run_id TEXT NOT NULL, anchor_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, anchor_id)
                );
                CREATE TABLE IF NOT EXISTS task_events (
                    event_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL, scientific_task_id TEXT NOT NULL,
                    state TEXT NOT NULL, attempt_id TEXT, outcome_reason TEXT,
                    recorded_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_events_run ON task_events(run_id,event_order);
                """
            )

    def _event(self,connection,run_id,task_id,state,attempt_id=None,reason=None):
        cursor=connection.execute("INSERT INTO task_events (run_id,scientific_task_id,state,attempt_id,outcome_reason,recorded_at) VALUES (?,?,?,?,?,?)",(run_id,task_id,state,attempt_id,reason,_now()))
        return cursor.lastrowid

    def create_run(self, run_id: str, config: Mapping[str, Any], records: Iterable[Mapping[str, Any]], *, manifest_path: str | Path | None = None) -> int:
        records = [dict(record) for record in records]
        now = _now()
        config = dict(config)
        config.setdefault("manifest_schema_version", MANIFEST_SCHEMA_VERSION)
        with self._connect() as connection:
            existing = connection.execute("SELECT config_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            encoded_config = _canonical(config)
            if existing is not None:
                if existing["config_json"] != encoded_config:
                    raise ManifestConflictError(f"run_id {run_id} already exists with a different configuration")
                existing_ids = {row[0] for row in connection.execute("SELECT scientific_task_id FROM tasks WHERE run_id = ?", (run_id,)).fetchall()}
                requested_ids = {str(record["scientific_task_id"]) for record in records}
                if existing_ids != requested_ids:
                    raise ManifestConflictError(f"run_id {run_id} already exists with an incomplete or different manifest")
                return int(connection.execute("SELECT COUNT(*) FROM tasks WHERE run_id = ?", (run_id,)).fetchone()[0])
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, MANIFEST_SCHEMA_VERSION, config.get("protocol_version", EVALUATION_PROTOCOL_VERSION), config.get("seed_scheme_version", SEED_SCHEME_VERSION), encoded_config, "created", now, now),
            )
            for record in records:
                task_id = str(record["scientific_task_id"])
                state = str(record.get("initial_state", "pending"))
                if state not in TASK_STATES:
                    raise ManifestError(f"invalid initial task state: {state}")
                connection.execute(
                    "INSERT INTO tasks (run_id, scientific_task_id, task_kind, stage, payload_json, state, planned_skip_reason, outcome_reason, attempt_count, active_attempt_id, result_ref, checkpoint_identity, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?, ?)",
                    (run_id, task_id, record.get("task_kind", "model"), record.get("stage", "model"), _canonical(record), state, record.get("planned_skip_reason"), record.get("planned_skip_reason"), record.get("checkpoint_identity"), now, now),
                )
                self._event(connection,run_id,task_id,state,reason=record.get("planned_skip_reason"))
        target = Path(manifest_path or self.manifest_path) if (manifest_path or self.manifest_path) else None
        if target is not None and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(_canonical(record) + "\n")
        return len(records)

    def set_run_status(self, run_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?", (status, _now(), run_id))

    def record_evidence_anchor(self, run_id: str, evidence: Mapping[str, Any]) -> str:
        encoded=_canonical(dict(evidence))
        anchor_id="evidence_"+hashlib.sha256(encoded.encode()).hexdigest()
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM runs WHERE run_id=?",(run_id,)).fetchone() is None:
                raise ManifestError("Evidence anchor references unknown run")
            connection.execute("INSERT OR IGNORE INTO evidence_anchors VALUES (?,?,?,?)",(run_id,anchor_id,encoded,_now()))
        return anchor_id

    def evidence_anchor(self,run_id: str,anchor_id: str) -> dict[str,Any] | None:
        with self._connect() as connection:
            try:
                row=connection.execute("SELECT payload_json FROM evidence_anchors WHERE run_id=? AND anchor_id=?",(run_id,anchor_id)).fetchone()
            except sqlite3.OperationalError:
                return None
        return None if row is None else json.loads(row["payload_json"])

    def run_config(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT config_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ManifestError(f"Unknown run: {run_id}")
        return json.loads(row["config_json"])

    def get_task(self, run_id: str, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE run_id = ? AND scientific_task_id = ?", (run_id, task_id)).fetchone()
        return None if row is None else dict(row)

    def task_payload(self, run_id: str, task_id: str) -> dict[str, Any]:
        task = self.get_task(run_id, task_id)
        if task is None:
            raise ManifestError(f"Unknown task: {task_id}")
        return json.loads(task["payload_json"])

    def snapshot(self, run_id: str) -> dict[str, Any]:
        """Read one consistent SQLite snapshot for analysis and provenance.

        All three tables are read under one SQLite transaction so a coverage
        report cannot mix task states from one coordinator moment with attempt
        or durable-result rows from another.
        """
        captured_at = _now()
        with self._connect() as connection:
            connection.execute("BEGIN")
            run = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                raise ManifestError(f"Unknown run: {run_id}")
            tasks = connection.execute("SELECT * FROM tasks WHERE run_id = ? ORDER BY scientific_task_id", (run_id,)).fetchall()
            attempts = connection.execute("SELECT * FROM attempts WHERE run_id = ? ORDER BY attempt_id", (run_id,)).fetchall()
            durable_results = connection.execute("SELECT * FROM durable_results WHERE run_id = ? ORDER BY scientific_task_id", (run_id,)).fetchall()
            try:
                events=connection.execute("SELECT * FROM task_events WHERE run_id=? ORDER BY event_order",(run_id,)).fetchall()
            except sqlite3.OperationalError:
                events=[]
        return {
            "captured_at": captured_at,
            "run": dict(run),
            "tasks": [dict(row) for row in tasks],
            "attempts": [dict(row) for row in attempts],
            "durable_results": [dict(row) for row in durable_results],
            "task_events":[dict(row) for row in events],
        }

    def claim_task(self, run_id: str, task_id: str, *, worker_id: str = "unknown", timeout_seconds: float | None = None) -> str | None:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT state, attempt_count, stage FROM tasks WHERE run_id = ? AND scientific_task_id = ?", (run_id, task_id)).fetchone()
            if row is None:
                raise ManifestError(f"Unknown task: {task_id}")
            if row["state"] != "pending":
                return None
            attempt_number = int(row["attempt_count"]) + 1
            attempt_id = attempt_id_for(run_id, task_id, attempt_number)
            connection.execute("UPDATE tasks SET state = 'running', attempt_count = ?, active_attempt_id = ?, updated_at = ? WHERE run_id = ? AND scientific_task_id = ?", (attempt_number, attempt_id, now, run_id, task_id))
            connection.execute("INSERT INTO attempts (run_id, attempt_id, scientific_task_id, attempt_number, stage, state, worker_id, started_at, timeout_seconds) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)", (run_id, attempt_id, task_id, attempt_number, row["stage"], worker_id, now, timeout_seconds))
            self._event(connection,run_id,task_id,"running",attempt_id)
            return attempt_id

    def _finish_attempt(self, connection: sqlite3.Connection, run_id: str, attempt_id: str, state: str, *, outcome: str | None = None, exception_type: str | None = None, exception_message: str | None = None, traceback_text: str | None = None, result_ref: str | None = None) -> None:
        row = connection.execute("SELECT started_at FROM attempts WHERE run_id = ? AND attempt_id = ?", (run_id, attempt_id)).fetchone()
        if row is None:
            raise ManifestError(f"Unknown attempt: {attempt_id}")
        started = datetime.fromisoformat(row["started_at"])
        elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        connection.execute("UPDATE attempts SET state = ?, finished_at = ?, elapsed_seconds = ?, outcome = ?, exception_type = ?, exception_message = ?, traceback = ?, result_ref = ? WHERE run_id = ? AND attempt_id = ?", (state, _now(), elapsed, outcome, exception_type, exception_message, traceback_text, result_ref, run_id, attempt_id))

    def record_failure(self, run_id: str, task_id: str, attempt_id: str, *, failure_class: str, exception: BaseException | None = None, timeout_seconds: float | None = None, retry: bool = False, result_ref: str | None = None) -> str:
        if failure_class not in RETRYABLE_FAILURES and failure_class not in {"user_requested_stop", "dependency_failure", "infeasible"}:
            failure_class = "worker_exception"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT state, active_attempt_id, attempt_count FROM tasks WHERE run_id = ? AND scientific_task_id = ?", (run_id, task_id)).fetchone()
            if row is None:
                raise ManifestError(f"Unknown task: {task_id}")
            if row["state"] != "running" or row["active_attempt_id"] != attempt_id:
                raise ManifestConflictError("stale or duplicate failure update")
            exception_type = type(exception).__name__ if exception else None
            exception_message = str(exception) if exception else None
            traceback_text = "".join(traceback_module.format_exception(exception)) if exception else None
            attempt_state = "timeout" if failure_class == "timeout" else "failed"
            self._finish_attempt(connection, run_id, attempt_id, attempt_state, outcome=failure_class, exception_type=exception_type, exception_message=exception_message, traceback_text=traceback_text, result_ref=result_ref)
            run_config_row = connection.execute("SELECT config_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            try:
                max_attempts = int(json.loads(run_config_row["config_json"]).get("execution", {}).get("max_attempts", 1)) if run_config_row else 1
            except (TypeError, ValueError, json.JSONDecodeError):
                max_attempts = 1
            retry_allowed = retry and failure_class in RETRYABLE_FAILURES and int(row["attempt_count"]) < max_attempts
            next_state = "pending" if retry_allowed else attempt_state
            connection.execute("UPDATE tasks SET state = ?, outcome_reason = ?, active_attempt_id = NULL, updated_at = ? WHERE run_id = ? AND scientific_task_id = ?", (next_state, failure_class, _now(), run_id, task_id))
            self._event(connection,run_id,task_id,next_state,attempt_id,failure_class)
            return next_state

    def mark_skipped(self, run_id: str, task_id: str, *, reason: str) -> None:
        with self._connect() as connection:
            changed=connection.execute("UPDATE tasks SET state = 'skipped', outcome_reason = ?, active_attempt_id = NULL, updated_at = ? WHERE run_id = ? AND scientific_task_id = ? AND state IN ('pending','failed','timeout')", (reason, _now(), run_id, task_id)).rowcount
            if changed:
                self._event(connection,run_id,task_id,"skipped",reason=reason)

    def propagate_dependency_failure(self, run_id: str, dependency_task_id: str, *, reason: str = "dependency_failure") -> int:
        changed = 0
        with self._connect() as connection:
            dependency = connection.execute("SELECT state FROM tasks WHERE run_id=? AND scientific_task_id=?",(run_id,dependency_task_id)).fetchone()
            if dependency is None or dependency["state"] not in {"failed","timeout","skipped"}:
                return 0
            rows = connection.execute("SELECT scientific_task_id, payload_json FROM tasks WHERE run_id = ? AND state = 'pending'", (run_id,)).fetchall()
            for row in rows:
                payload = json.loads(row["payload_json"])
                if dependency_task_id in payload.get("depends_on", []):
                    updated=connection.execute("UPDATE tasks SET state = 'skipped', outcome_reason = ?, updated_at = ? WHERE run_id = ? AND scientific_task_id = ? AND state='pending'", (reason, _now(), run_id, row["scientific_task_id"])).rowcount
                    if updated:
                        self._event(connection,run_id,row["scientific_task_id"],"skipped",reason=reason)
                        changed += 1
        return changed

    def commit_result(self, run_id: str, task_id: str, attempt_id: str, result: Mapping[str, Any], *, result_ref: str | None = None) -> bool:
        payload = dict(result)
        encoded = _canonical(payload)
        payload_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = connection.execute("SELECT state, active_attempt_id FROM tasks WHERE run_id = ? AND scientific_task_id = ?", (run_id, task_id)).fetchone()
            if task is None:
                raise ManifestError(f"Unknown task: {task_id}")
            existing = connection.execute("SELECT payload_hash, attempt_id FROM durable_results WHERE run_id = ? AND scientific_task_id = ?", (run_id, task_id)).fetchone()
            if existing is not None:
                if existing["payload_hash"] != payload_hash or existing["attempt_id"] != attempt_id:
                    raise ManifestConflictError("conflicting durable result for scientific task")
                return False
            if task["state"] != "running" or task["active_attempt_id"] != attempt_id:
                raise ManifestConflictError("stale worker cannot commit this task")
            connection.execute("INSERT INTO durable_results VALUES (?, ?, ?, ?, ?, ?, ?)", (run_id, task_id, attempt_id, payload_hash, encoded, result_ref, _now()))
            self._finish_attempt(connection, run_id, attempt_id, "completed", outcome="success", result_ref=result_ref)
            connection.execute("UPDATE tasks SET state = 'completed', outcome_reason = 'success', active_attempt_id = NULL, result_ref = ?, updated_at = ? WHERE run_id = ? AND scientific_task_id = ?", (result_ref, _now(), run_id, task_id))
            self._event(connection,run_id,task_id,"completed",attempt_id,"success")
            return True

    def export_durable_results(self, run_id: str, ledger_path: str | Path) -> dict[str, int]:
        """Repair the idempotent export from SQLite's transactional authority.

        Only a truncated final line is recoverable. Conflicting complete rows
        are rejected; originals are retained before any repair publication.
        """
        path = Path(ledger_path)
        with artifact_lock(path.with_suffix(path.suffix+".lock")):
            snapshot = self.snapshot(run_id)
            tasks = {t["scientific_task_id"]:t for t in snapshot["tasks"]}
            authoritative = {}
            for row in snapshot["durable_results"]:
                task = tasks.get(row["scientific_task_id"])
                if task is None or task["stage"] != "model":
                    continue
                payload = json.loads(row["payload_json"])
                payload.setdefault("run_id",run_id)
                payload.setdefault("scientific_task_id",row["scientific_task_id"])
                payload.setdefault("attempt_id",row["attempt_id"])
                authoritative[row["scientific_task_id"]] = payload
            rows,seen = [],{}
            malformed,duplicates = 0,0
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            nonempty = [i for i,line in enumerate(lines) if line.strip()]
            for index,line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    payload=json.loads(line)
                    if not isinstance(payload,dict):
                        raise ManifestConflictError("Non-object ledger record")
                except json.JSONDecodeError:
                    if index != nonempty[-1] or line.rstrip().endswith("}"):
                        raise ManifestConflictError("Malformed nontruncated ledger evidence")
                    malformed += 1
                    continue
                if payload.get("run_id") != run_id:
                    rows.append(payload)
                    continue
                task_id=payload.get("scientific_task_id")
                if task_id not in authoritative or _canonical(payload) != _canonical(authoritative[task_id]):
                    raise ManifestConflictError("Ledger payload conflicts with authoritative result")
                if task_id in seen:
                    duplicates += 1
                    continue
                seen[task_id]=payload
                rows.append(payload)
            missing=sorted(set(authoritative)-set(seen))
            rows.extend(authoritative[task_id] for task_id in missing)
            if missing or malformed or duplicates:
                if path.exists():
                    backup=path.with_name(path.name+".recovery_"+file_sha256(path)[:16])
                    if not backup.exists():
                        atomic_bytes(backup,path.read_bytes())
                atomic_bytes(path,("".join(_canonical(row)+"\n" for row in rows)).encode("utf-8"))
            return {"rows_added":len(missing),"malformed_lines":malformed,"duplicates_removed":duplicates}

    def recover_stale_attempts(self, run_id: str, *, stale_after_seconds: float, retry: bool = True) -> int:
        cutoff = time.time() - stale_after_seconds
        recovered = 0
        with self._connect() as connection:
            run_row = connection.execute("SELECT config_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            try:
                max_attempts = int(json.loads(run_row["config_json"]).get("execution", {}).get("max_attempts", 1)) if run_row else 1
            except (TypeError, ValueError, json.JSONDecodeError):
                max_attempts = 1
            rows = connection.execute("SELECT attempt_id, scientific_task_id, started_at FROM attempts WHERE run_id = ? AND state = 'running'", (run_id,)).fetchall()
            for row in rows:
                started = datetime.fromisoformat(row["started_at"]).timestamp()
                if started <= cutoff:
                    task = connection.execute("SELECT attempt_count, stage, active_attempt_id FROM tasks WHERE run_id = ? AND scientific_task_id = ?", (run_id, row["scientific_task_id"])).fetchone()
                    self._finish_attempt(connection, run_id, row["attempt_id"], "failed", outcome="coordinator_crash")
                    if task is None or task["active_attempt_id"] != row["attempt_id"]:
                        continue
                    retry_allowed = retry and task is not None and int(task["attempt_count"]) < max_attempts
                    next_state = "pending" if retry_allowed else "failed"
                    connection.execute("UPDATE tasks SET state = ?, outcome_reason = 'coordinator_crash', active_attempt_id = NULL, updated_at = ? WHERE run_id = ? AND scientific_task_id = ?", (next_state, _now(), run_id, row["scientific_task_id"]))
                    self._event(connection,run_id,row["scientific_task_id"],next_state,row["attempt_id"],"coordinator_crash")
                    if next_state == "failed" and task is not None and task["stage"] == "precompute":
                        dependents = connection.execute("SELECT scientific_task_id, payload_json FROM tasks WHERE run_id = ? AND state = 'pending'", (run_id,)).fetchall()
                        for dependent in dependents:
                            payload = json.loads(dependent["payload_json"])
                            if row["scientific_task_id"] in payload.get("depends_on", []):
                                connection.execute("UPDATE tasks SET state = 'skipped', outcome_reason = 'dependency_failure', updated_at = ? WHERE run_id = ? AND scientific_task_id = ?", (_now(), run_id, dependent["scientific_task_id"]))
                                self._event(connection,run_id,dependent["scientific_task_id"],"skipped",reason="dependency_failure")
                    recovered += 1
        return recovered

    def request_stop(self, run_id: str, *, reason: str = "user_requested_stop") -> None:
        self.set_run_status(run_id, "stopping")
        with self._connect() as connection:
            connection.execute("UPDATE tasks SET outcome_reason = ? WHERE run_id = ? AND state = 'pending'", (reason, run_id))

    def resume(self, run_id: str) -> None:
        self.set_run_status(run_id, "running")

    def state_counts(self, run_id: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT state, COUNT(*) AS n FROM tasks WHERE run_id = ? GROUP BY state", (run_id,)).fetchall()
        result = {state: 0 for state in TASK_STATES}
        result.update({row["state"]: int(row["n"]) for row in rows})
        result["intended"] = sum(result[state] for state in TASK_STATES)
        return result

    def attempt_counts(self, run_id: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT state, COUNT(*) AS n FROM attempts WHERE run_id = ? GROUP BY state", (run_id,)).fetchall()
        return {row["state"]: int(row["n"]) for row in rows}

    def reconcile_result_ledger(self, run_id: str, ledger_path: str | Path) -> dict[str, Any]:
        """Reconcile durable JSONL records and detect truncated/duplicate lines."""
        path = Path(ledger_path)
        valid = 0
        malformed = 0
        conflicts = 0
        sqlite_first = self.run_config(run_id).get("durability_protocol") == "sqlite_result_outbox_v2"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                task_id = result.get("scientific_task_id") or result.get("task_id")
                attempt_id = result.get("attempt_id")
                if not task_id or not attempt_id:
                    continue
                if result.get("run_id") not in (None,run_id):
                    continue
                try:
                    if sqlite_first:
                        with self._connect() as connection:
                            saved=connection.execute("SELECT payload_json FROM durable_results WHERE run_id=? AND scientific_task_id=?",(run_id,task_id)).fetchone()
                        if saved is None or saved["payload_json"] != _canonical(result):
                            raise ManifestConflictError("SQLite-first ledger lacks matching authoritative commit")
                        continue
                    if self.commit_result(run_id, task_id, attempt_id, result, result_ref=str(path)):
                        valid += 1
                except ManifestConflictError:
                    conflicts += 1
        # A completion marker without its durable payload is recoverable only by
        # returning the task to pending; the attempt history remains intact.
        with self._connect() as connection:
            rows = connection.execute("SELECT scientific_task_id FROM tasks WHERE run_id = ? AND state = 'completed' AND NOT EXISTS (SELECT 1 FROM durable_results WHERE durable_results.run_id = tasks.run_id AND durable_results.scientific_task_id = tasks.scientific_task_id)", (run_id,)).fetchall()
            for row in rows:
                connection.execute("UPDATE tasks SET state = 'pending', outcome_reason = 'missing_durable_result', updated_at = ? WHERE run_id = ? AND scientific_task_id = ?", (_now(), run_id, row["scientific_task_id"]))
                self._event(connection,run_id,row["scientific_task_id"],"pending",reason="missing_durable_result")
        return {"valid_results_committed": valid, "malformed_lines": malformed, "conflicts": conflicts, "state_counts": self.state_counts(run_id)}


def _callable_entry(target: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any], queue: Any) -> None:
    try:
        value = target(*args, **dict(kwargs))
        queue.put({"state": "completed", "value": value})
    except BaseException as exc:  # pragma: no cover - exercised in child
        queue.put({"state": "failed", "exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": "".join(traceback_module.format_exception(exc))})


def run_callable_with_timeout(target: Callable[..., Any], args: tuple[Any, ...] = (), kwargs: Mapping[str, Any] | None = None, *, timeout_seconds: float) -> dict[str, Any]:
    """Contain a callable in a child process and terminate it on timeout."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    context = mp.get_context("spawn")
    queue = context.Queue()

    process = context.Process(target=_callable_entry, args=(target, args, dict(kwargs or {}), queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        return {"state": "timeout", "exitcode": process.exitcode}
    try:
        return queue.get_nowait()
    except Exception:
        return {"state": "failed", "exitcode": process.exitcode, "exception_type": "ChildProcessError", "exception_message": "child exited without an outcome"}
