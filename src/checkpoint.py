import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("reports/cache.db")

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
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

def compute_hash(dataset: str, seed: int, fold: int, condition: str, pipeline: str, model: str) -> str:
    return f"{dataset}_{seed}_{fold}_{condition}_{pipeline}_{model}"

def has_run(dataset: str, seed: int, fold: int, condition: str, pipeline: str, model: str) -> bool:
    h = compute_hash(dataset, seed, fold, condition, pipeline, model)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute("SELECT 1 FROM completed_tasks WHERE task_hash = ?", (h,))
            return cur.fetchone() is not None
    except sqlite3.OperationalError:
        return False

def log_run(dataset: str, seed: int, fold: int, condition: str, pipeline: str, model: str) -> None:
    h = compute_hash(dataset, seed, fold, condition, pipeline, model)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO completed_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
            (h, dataset, seed, fold, condition, pipeline, model)
        )
