"""Centralized filesystem paths for new AutoFE-ShiftBench runs.

Historical artifacts retain their original locations.  These helpers only
define destinations for newly created runs and analysis outputs.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str) -> Path:
    """Resolve a configured path relative to the repository, never the CWD."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


REPORTS_ROOT = _project_path(os.environ.get("AUTOFE_REPORTS_BASE", "reports"))
RUNS_ROOT = REPORTS_ROOT / "runs"
FULL_RUNS_ROOT = RUNS_ROOT / "full"
PILOT_RUNS_ROOT = RUNS_ROOT / "pilot"
ANALYSIS_ROOT = REPORTS_ROOT / "analysis"
AUDITS_ROOT = REPORTS_ROOT / "audits"
FIGURE_INPUTS_ROOT = REPORTS_ROOT / "figure_inputs"


def run_root(kind: str, run_id: str) -> Path:
    """Return the stable destination for a new full or pilot run."""
    # Reject both separator styles even when tests run on POSIX, because the
    # launcher and its primary deployment target are Windows.
    if (not run_id or "/" in run_id or "\\" in run_id
            or Path(run_id).name != run_id or run_id in {".", ".."}):
        raise ValueError(f"run_id must be a single safe directory name: {run_id!r}")
    if kind == "full":
        return FULL_RUNS_ROOT / run_id
    if kind == "pilot":
        return PILOT_RUNS_ROOT / run_id
    raise ValueError(f"Unknown run kind: {kind!r}")
