"""Exact allocation budgets: refuse infeasible work without changing its inputs."""
import os
import psutil


class ResourceLimitError(MemoryError):
    pass


def dense_budget():
    configured=os.environ.get('AUTOFE_DENSE_BUDGET_BYTES')
    return int(configured) if configured else min(1024**3,int(psutil.virtual_memory().available*.4))


def require_bytes(required,*,purpose,budget=None):
    budget=dense_budget() if budget is None else budget
    if required > budget:
        raise ResourceLimitError(f'{purpose} requires {required} exact bytes; budget {budget}; no sampling/grouping/precision fallback')
