"""Read-only identity and resource gate for one frozen Reviewer #1 run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.provenance import code_fingerprint, file_sha256


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "provenance" / "reviewer1_launch_scope_v2.json"


def verify(run_id: str) -> dict:
    scope = json.loads(SCOPE.read_text(encoding="utf-8"))
    run = next((item for item in scope["runs"] if item["run_id"] == run_id), None)
    if run is None:
        raise ValueError(f"Run ID is not in the frozen seven-run scope: {run_id}")
    checks = {}
    checks["existing_p12"] = Path(sys.executable).resolve() == Path(r"D:\Conda\p12\python.exe").resolve()
    checks["code_fingerprint"] = code_fingerprint(ROOT) == scope["code_fingerprint"]
    checks["dataset_list"] = file_sha256(ROOT / "config" / "dataset_list.yaml") == scope["dataset_list_sha256"]
    checks["group_seed_audit"] = file_sha256(ROOT / "provenance" / "group_seed_grid_audit_v1.json") == scope["group_seed_audit_sha256"]
    checks["datasets"] = all(
        file_sha256(ROOT / "data" / "raw" / f"{item['name']}.csv") == item["csv_sha256"]
        and file_sha256(ROOT / "data" / "raw" / f"{item['name']}_meta.json") == item["sidecar_sha256"]
        for item in scope["datasets"]
    )
    checks["analysis_definitions"] = all(
        file_sha256(ROOT / name) == digest for name, digest in scope["analysis_sha256"].items()
    )
    checks["numerical_threads"] = all(
        os.environ.get(name) == value
        for name, value in scope["required_numerical_thread_environment"].items()
    )
    host_path = ROOT / scope["host_probe_path"]
    host = json.loads(host_path.read_text(encoding="utf-8"))
    checks["host_probe_hash"] = file_sha256(host_path) == scope["host_probe_sha256"]
    checks["host_identity"] = (
        host.get("computer_name", "").casefold() == platform.node().casefold()
        and Path(host.get("python_executable", "")).resolve() == Path(sys.executable).resolve()
    )
    # Two times the projected seven-run result/checkpoint volume plus the
    # bounded live cache cap is a conservative local capacity gate. Candidate
    # histories are a separate scope and are not silently budgeted as output.
    storage = scope["storage"]
    required_gib = (2 * storage["scaled_results_checkpoints_projection_gib_all_seven_runs"]
                    + storage["bounded_live_cache_cap_gib_per_run"]
                    + storage["separate_mechanism_history_reservation_gib"])
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    checks["disk_margin"] = free_gib >= required_gib

    package_mismatches = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.lstrip().startswith("#"):
            continue
        name, required = line.split("==", 1)
        try:
            installed = version(name.strip())
        except PackageNotFoundError:
            installed = None
        if installed != required.strip():
            package_mismatches.append({"package": name.strip(), "required": required.strip(), "installed": installed})
    checks["requirements_pins"] = not package_mismatches

    calibration = {}
    for policy in ("row_level", "group_aware"):
        path = ROOT / "corrected_runs" / "large_calibration" / f"calibration-{policy}-002" / "manifest.json"
        if not path.exists():
            calibration[policy] = {"status": "missing"}
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        calibration[policy] = {
            "status": item.get("status"),
            "expected": item.get("expected_tasks"),
            "success": item.get("counts_by_status", {}).get("success"),
            "code_fingerprint_matches": item.get("code_fingerprint") == scope["code_fingerprint"],
            "threads_match": item.get("configuration", {}).get("numerical_thread_environment") == scope["required_numerical_thread_environment"],
        }
    checks["large_calibration"] = all(
        item.get("status") == "complete" and item.get("expected") == 120
        and item.get("success") == 120 and item.get("code_fingerprint_matches")
        and item.get("threads_match")
        for item in calibration.values()
    )
    mechanism_path = ROOT / "provenance" / "mechanism_history_scope_v1.json"
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8")) if mechanism_path.exists() else {}
    checks["mechanism_history_scope"] = (
        mechanism.get("status") == "ready"
        and mechanism.get("code_fingerprint") == scope["code_fingerprint"]
        and mechanism.get("dataset_hash_digest") == scope["dataset_hash_digest"]
        and mechanism.get("mechanism_script_sha256") == file_sha256(
            ROOT / "provenance" / "run_mechanism_history.py"
        )
        and mechanism.get("association_analysis_script_sha256") == file_sha256(
            ROOT / "provenance" / "analyze_mechanism_history.py"
        )
    )

    existing = ROOT / "corrected_runs" / run_id / "manifest.json"
    if existing.exists():
        prior = json.loads(existing.read_text(encoding="utf-8"))
        checks["run_identity"] = (
            prior.get("code_fingerprint") == scope["code_fingerprint"]
            and prior.get("configuration", {}).get("split_policy") == run["split_policy"]
            and prior.get("experiment_scope") == (
                "primary_training_corruption" if run["scope"] == "primary" else run["scope"]
            )
        )
    else:
        checks["run_identity"] = not (ROOT / "corrected_runs" / run_id).exists()
    return {
        "run_id": run_id, "ready_for_frozen_command": all(checks.values()),
        "checks": checks, "free_gib": round(free_gib, 2),
        "minimum_free_gib": round(required_gib, 2),
        "package_mismatches": package_mismatches,
        "large_calibration": calibration,
        "mechanism_history_status": mechanism.get("status", "missing"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = verify(args.run_id)
    print(json.dumps(result, indent=2))
    if not result["ready_for_frozen_command"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
