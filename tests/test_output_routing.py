"""Non-training tests for stable experiment output routing."""

from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest

from src.paths import run_root
from src.task_manifest import ManifestConflictError, ManifestStore


def test_full_and_pilot_run_roots_are_stable_and_separate():
    project_root = Path(__file__).resolve().parents[1]
    assert run_root("full", "experiment_001") == project_root / "reports/runs/full/experiment_001"
    assert run_root("pilot", "pilot_001") == project_root / "reports/runs/pilot/pilot_001"
    assert run_root("full", "experiment_001") == run_root("full", "experiment_001")


@pytest.mark.parametrize("run_id", ["", ".", "..", "nested\\run", "nested/run"])
def test_run_id_rejects_path_traversal(run_id):
    with pytest.raises(ValueError):
        run_root("full", run_id)


def test_manifest_resume_reuses_run_and_rejects_incompatible_protocol(tmp_path):
    store = ManifestStore(tmp_path / "benchmark.db")
    config = {"protocol_version": "protocol-a", "seed_scheme_version": "seed-a", "datasets": ["adult"]}
    records = [{"scientific_task_id": "task-1", "task_kind": "model", "stage": "model"}]

    assert store.create_run("experiment_001", config, records) == 1
    assert store.create_run("experiment_001", config, records) == 1

    with pytest.raises(ManifestConflictError):
        store.create_run(
            "experiment_001",
            {**config, "protocol_version": "protocol-b"},
            records,
        )


def test_worker_log_routing_is_run_local():
    launcher = Path("run_project.ps1").read_text(encoding="utf-8")
    pilot = Path("pilot_run.ps1").read_text(encoding="utf-8")
    assert '"logs/workers"' in launcher
    assert '"logs/workers"' in pilot
    assert "AUTOFE_WORKER_LOG_DIR" in launcher
    assert "AUTOFE_WORKER_LOG_DIR" in pilot


def test_project_root_does_not_depend_on_current_working_directory(tmp_path):
    probe = "from src.paths import PROJECT_ROOT; print(PROJECT_ROOT)"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    isolated = tmp_path / "unicode space Ω"
    isolated.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=isolated,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(Path(__file__).resolve().parents[1])


def test_plan_only_launcher_is_non_mutating(tmp_path):
    """A clean clone can inspect the plan without installing, downloading, or writing."""
    if not shutil.which("pwsh"):
        pytest.skip("PowerShell is not available")
    source = Path(__file__).resolve().parents[1] / "run_project.ps1"
    clone = tmp_path / "clean clone space Ω"
    clone.mkdir()
    launcher = clone / "run_project.ps1"
    launcher.write_bytes(source.read_bytes())
    before = sorted(path.relative_to(clone).as_posix() for path in clone.rglob("*"))
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(launcher), "-PlanOnly", "-RunId", "ci_plan"],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
    )
    after = sorted(path.relative_to(clone).as_posix() for path in clone.rglob("*"))
    assert before == after == ["run_project.ps1"]
    assert "no download or benchmark command was executed" in result.stdout
