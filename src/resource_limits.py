"""Exact allocation budgets: refuse infeasible work without changing its inputs."""
import os
import psutil
import shutil
from pathlib import Path
from contextlib import contextmanager


class ResourceLimitError(MemoryError):
    pass


def dense_budget():
    configured=os.environ.get('AUTOFE_DENSE_BUDGET_BYTES')
    return int(configured) if configured else min(1024**3,int(psutil.virtual_memory().available*.4))


def require_bytes(required,*,purpose,budget=None):
    budget=dense_budget() if budget is None else budget
    if required > budget:
        raise ResourceLimitError(f'{purpose} requires {required} exact bytes; budget {budget}; no sampling/grouping/precision fallback')


@contextmanager
def disk_write_reservation(path, required=0):
    """Serialize project payload writes and refuse reserve-consuming writes.

    Callers flush before releasing this lease. External applications can still
    consume space; this is a guarded write, not a filesystem quota.
    """
    reserve = int(os.environ.get('AUTOFE_MIN_FREE_BYTES', '0'))
    if not reserve:
        yield
        return
    from src.artifact_integrity import artifact_lock
    target = Path(path).resolve()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = os.environ.get('AUTOFE_DISK_WRITE_LOCK')
    if not lock:
        raise ResourceLimitError('Free-space guard requires a shared write lock')
    with artifact_lock(lock, timeout_seconds=60):
        free = shutil.disk_usage(parent).free
        if free - required < reserve:
            raise ResourceLimitError(f'Write of {required} bytes at {target} would breach free-space reserve {reserve}; free {free}')
        yield
