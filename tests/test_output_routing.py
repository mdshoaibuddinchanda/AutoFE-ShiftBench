"""Non-training tests for stable experiment output routing."""

from pathlib import Path

import pytest

from src.paths import run_root
from src.task_manifest import ManifestConflictError, ManifestStore


def test_full_and_pilot_run_roots_are_stable_and_separate():
    assert run_root("full", "experiment_001") == Path("reports/runs/full/experiment_001")
    assert run_root("pilot", "pilot_001") == Path("reports/runs/pilot/pilot_001")
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
