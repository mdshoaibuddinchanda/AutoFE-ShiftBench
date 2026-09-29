"""Run-scoped task state; successful completion is distinct from attempts/failures."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


LEGACY_DB_PATH = Path("reports/cache.db")


def init_db(db_path: str | Path = LEGACY_DB_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_state (
                run_id TEXT NOT NULL,
                task_key TEXT NOT NULL,
                phase TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                dataset TEXT NOT NULL,
                seed INTEGER,
                fold INTEGER,
                condition TEXT,
                pipeline TEXT,
                model TEXT,
                manifest_fingerprint TEXT,
                exception_type TEXT,
                error_summary TEXT,
                PRIMARY KEY (run_id, task_key, phase)
            )
        """)


def has_success(
    db_path: str | Path,
    run_id: str,
    task_key: str,
    phase: str,
    expected_fingerprint: str | None = None,
) -> bool:
    try:
        with sqlite3.connect(db_path, timeout=30) as conn:
            row = conn.execute(
                "SELECT status, manifest_fingerprint FROM task_state WHERE run_id=? AND task_key=? AND phase=?",
                (run_id, task_key, phase),
            ).fetchone()
    except sqlite3.OperationalError:
        return False
    return bool(
        row
        and row[0] == "success"
        and (expected_fingerprint is None or row[1] == expected_fingerprint)
    )


def record_task(
    db_path: str | Path,
    *,
    run_id: str,
    task_key: str,
    phase: str,
    status: str,
    dataset: str,
    seed: int | None = None,
    fold: int | None = None,
    condition: str | None = None,
    pipeline: str | None = None,
    model: str | None = None,
    manifest_fingerprint: str | None = None,
    exception_type: str | None = None,
    error_summary: str | None = None,
) -> None:
    if status not in {"success", "failed"}:
        raise ValueError("Task state status must be 'success' or 'failed'")
    init_db(db_path)
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.execute("""
            INSERT INTO task_state (
                run_id, task_key, phase, status, dataset, seed, fold, condition,
                pipeline, model, manifest_fingerprint, exception_type, error_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, task_key, phase) DO UPDATE SET
                status=excluded.status,
                manifest_fingerprint=excluded.manifest_fingerprint,
                exception_type=excluded.exception_type,
                error_summary=excluded.error_summary
        """, (
            run_id, task_key, phase, status, dataset, seed, fold, condition,
            pipeline, model, manifest_fingerprint, exception_type, error_summary,
        ))


def get_task_state(db_path: str | Path, run_id: str, task_key: str, phase: str) -> dict[str, Any] | None:
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM task_state WHERE run_id=? AND task_key=? AND phase=?",
            (run_id, task_key, phase),
        ).fetchone()
    return dict(row) if row else None
