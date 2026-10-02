import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION

DB_PATH = Path("reports/cache.db")


@contextmanager
def _connection():
    """Open a checkpoint connection and always close its Windows file handle."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS completed_tasks (
                task_hash TEXT PRIMARY KEY,
                dataset TEXT,
                seed INTEGER,
                fold INTEGER,
                condition TEXT,
                pipeline TEXT,
                model TEXT
            )
        """)

def compute_hash(
    dataset: str,
    seed: int,
    fold: int,
    condition: str,
    pipeline: str,
    model: str,
    split_policy: str,
) -> str:
    return (
        f"{EVALUATION_PROTOCOL_VERSION}|{SEED_SCHEME_VERSION}|{dataset}|{split_policy}|"
        f"{seed}|{fold}|{condition}|{pipeline}|{model}"
    )

def has_run(
    dataset: str,
    seed: int,
    fold: int,
    condition: str,
    pipeline: str,
    model: str,
    split_policy: str,
) -> bool:
    h = compute_hash(dataset, seed, fold, condition, pipeline, model, split_policy)
    try:
        with _connection() as conn:
            cur = conn.execute("SELECT 1 FROM completed_tasks WHERE task_hash = ?", (h,))
            return cur.fetchone() is not None
    except sqlite3.OperationalError:
        return False

def log_run(
    dataset: str,
    seed: int,
    fold: int,
    condition: str,
    pipeline: str,
    model: str,
    split_policy: str,
) -> None:
    h = compute_hash(dataset, seed, fold, condition, pipeline, model, split_policy)
    with _connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO completed_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
            (h, dataset, seed, fold, condition, pipeline, model)
        )
