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

import psutil

from src.provenance import code_fingerprint, file_sha256


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "provenance" / "reviewer1_launch_scope_v5.json"
HEAVY_COORDINATOR_MODULES = {
    "src.pipeline_runner", "provenance.large_dataset_calibration",
    "provenance.large_dataset_calibration_group_v3",
    "provenance.large_dataset_recovery_v1", "provenance.run_mechanism_history",
    "provenance.final_optimization_verification",
    "provenance.adaptive_resource_verification",
}


def _active_heavy_coordinators() -> list[dict]:
    active = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            args = process.info.get("cmdline") or []
            for index, value in enumerate(args[:-1]):
                if value == "-m" and args[index + 1] in HEAVY_COORDINATOR_MODULES:
                    active.append({"pid": process.pid, "module": args[index + 1]})
                    break
            else:
                known_scripts = {name.split('.')[-1] + '.py' for name in HEAVY_COORDINATOR_MODULES}
                if any(Path(value).name in known_scripts for value in args[1:]):
                    active.append({'pid': process.pid, 'module': 'direct heavy script'})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return active


def _recovery_run_manifest_valid(item: dict, source_fingerprint: str) -> bool:
    relative = item.get("run_manifest")
    run_id = item.get("run_id", "")
    if not relative or not isinstance(relative, str) or not run_id:
        return False
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    policy = ("row_level" if run_id.startswith("recovery-airlines-row-") else
              "group_aware" if run_id.startswith("recovery-airlines-group-") else None)
    return bool(
        policy and manifest.get("run_id") == run_id and manifest.get("status") == "complete"
        and manifest.get("expected_tasks") == 20
        and manifest.get("counts_by_status", {}).get("success") == 20
        and manifest.get("code_fingerprint") == source_fingerprint
        and manifest.get("configuration", {}).get("split_policy") == policy
        and manifest.get("configuration", {}).get("scheduler_lease_seconds") == 600.0
    )


def _recovery_scientific_parity_valid(recovery: dict, scope_path: Path) -> bool:
    """Bind the new Airline parity summary to the actual frozen result files."""
    from provenance.final_optimization_verification import rows, FIELDS, MODELS
    if recovery.get('scope_sha256') != file_sha256(scope_path):
        return False
    summaries = recovery.get('historical_calibration_scientific_parity', [])
    if (len(summaries) != 2 or {r.get('policy') for r in summaries} != {'row_level', 'group_aware'}):
        return False
    try:
        for policy, suffix, version in (('row_level', 'row', '002'), ('group_aware', 'group', '003')):
            item = next(r for r in recovery['runs'] if r['run_id'].startswith(f'recovery-airlines-{suffix}-'))
            directory = (ROOT / item['run_manifest']).parent
            manifest = json.loads((directory / 'manifest.json').read_text())
            config = manifest['configuration']
            if (config.get('datasets') != ['airlines'] or config.get('pipelines') != ['Raw', 'AutoFE_Baseline']
                    or config.get('models') != list(MODELS) or config.get('seeds') != [42]
                    or config.get('folds') != [1] or config.get('n_splits') != 5
                    or config.get('conditions') != [['clean', 0.0]] or config.get('use_gpu') is not False):
                return False
            current = rows(directory / 'results.jsonl')
            prior = rows(ROOT / 'corrected_runs' / 'large_calibration' /
                         f'calibration-{policy}-{version}' / 'results.jsonl')
            if len(current) != 20 or any(key not in prior or any(
                    current[key].get(field) != prior[key].get(field) for field in FIELDS) for key in current):
                return False
    except (OSError, ValueError, KeyError, StopIteration):
        return False
    return True


def verify(run_id: str, scope_path: Path | None = None) -> dict:
    scope = json.loads((scope_path or SCOPE).read_text(encoding="utf-8"))
    if scope.get('adaptive_verification_path'):
        from provenance.verify_adaptive_launch import verify as verify_adaptive
        return verify_adaptive(run_id, scope_path or SCOPE)
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
    active_heavy = _active_heavy_coordinators()
    checks["one_heavy_coordinator"] = not active_heavy
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
                    + storage.get("cache_publication_temporary_allowance_gib", 0)
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
        calibration_version = "002" if policy == "row_level" else "003"
        path = ROOT / "corrected_runs" / "large_calibration" / f"calibration-{policy}-{calibration_version}" / "manifest.json"
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
    historical_calibration_passed = all(
        item.get("status") == "complete" and item.get("expected") == 120
        and item.get("success") == 120 and item.get("code_fingerprint_matches")
        and item.get("threads_match")
        for item in calibration.values()
    )
    optimization_path = scope.get('optimization_verification_path')
    if optimization_path:
        path = ROOT / optimization_path
        optimized = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        saved_evidence_valid = False
        if optimized.get('status') == 'passed' and optimized.get('per_run_evidence'):
            try:
                from provenance.final_optimization_verification import verify_saved_report
                rechecked = verify_saved_report(optimized)
                saved_evidence_valid = (rechecked['per_run_evidence'] == optimized['per_run_evidence']
                                        and rechecked['scientific_fields'] == optimized.get('scientific_fields'))
            except (OSError, ValueError, KeyError):
                pass
        checks['bounded_optimization_parity'] = (
            saved_evidence_valid
            and
            optimized.get('status') == 'passed'
            and optimized.get('code_fingerprint') == scope['code_fingerprint']
            and optimized.get('workers') == 4
            and optimized.get('numerical_thread_environment') == scope['required_numerical_thread_environment']
            and len(optimized.get('small_dataset_parity', [])) == 2
            and {r.get('policy') for r in optimized.get('small_dataset_parity', [])} == {'row_level', 'group_aware'}
            and all(r.get('compared_cells') == 560 and r.get('mismatches') == [] for r in optimized.get('small_dataset_parity', []))
            and len(optimized.get('large_dataset_parity', [])) == 2
            and {r.get('policy') for r in optimized.get('large_dataset_parity', [])} == {'row_level', 'group_aware'}
            and all(r.get('compared_cells') == 10 and r.get('mismatches') == [] for r in optimized.get('large_dataset_parity', []))
            and all(file_sha256(ROOT / 'data' / 'raw' / f'{name}.csv') == digest
                    for name, digest in optimized.get('data_hashes', {}).items())
            and len(optimized.get('data_hashes', {})) == 5
        )
        # Old 120-cell timing reports are explicitly historical after a source
        # change. New-source large data/parity and recovery are separate gates.
        checks['historical_large_calibration'] = all(
            item.get('status') == 'complete' and item.get('expected') == 120 and item.get('success') == 120
            for item in calibration.values())
    else:
        checks['large_calibration'] = historical_calibration_passed
    recovery_path = ROOT / scope.get('recovery_evidence_path', 'provenance/large_dataset_recovery_v1.json')
    recovery = json.loads(recovery_path.read_text(encoding="utf-8")) if recovery_path.exists() else {}
    airline = next(item for item in scope["datasets"] if item["name"] == "airlines")
    recovery_runs = recovery.get("runs", [])
    checks["large_dataset_forced_restart"] = (
        recovery.get("artifact_type") == "large_dataset_forced_restart_v1"
        and recovery.get("status") == "passed"
        and recovery.get("host", "").casefold() == platform.node().casefold()
        and Path(recovery.get("python_executable", "")).resolve() == Path(sys.executable).resolve()
        and recovery.get("code_fingerprint") == scope["code_fingerprint"]
        and recovery.get("csv_sha256") == airline["csv_sha256"]
        and recovery.get("sidecar_sha256") == airline["sidecar_sha256"]
        and recovery.get("numerical_thread_environment") == scope["required_numerical_thread_environment"]
        and recovery.get("scheduler_lease_seconds") == 600
        and recovery.get("workers") == 4
        and len(recovery_runs) == 2
        and {"row" if item.get("run_id", "").startswith("recovery-airlines-row-") else
             "group" if item.get("run_id", "").startswith("recovery-airlines-group-") else "invalid"
             for item in recovery_runs} == {"row", "group"}
        and all(item.get("status") == "passed"
                and item.get("committed_successes_before_kill", 0) > 0
                and item.get("terminal_successes_after_resume") == 20
                and item.get("pre_crash_successes_retained_without_refit")
                and _recovery_run_manifest_valid(item, scope["code_fingerprint"])
                for item in recovery_runs)
    )
    if optimization_path:
        checks['large_airlines_scientific_parity'] = (
            _recovery_scientific_parity_valid(recovery, scope_path or SCOPE)
            and
            len(recovery.get('historical_calibration_scientific_parity', [])) == 2
            and all(r.get('compared_cells') == 20 and r.get('mismatches') == []
                    for r in recovery.get('historical_calibration_scientific_parity', [])))
    mechanism_path = ROOT / scope.get('mechanism_scope_path', 'provenance/mechanism_history_scope_v1.json')
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
        "large_dataset_recovery": recovery.get("status", "missing"),
        "active_heavy_coordinators": active_heavy,
        "mechanism_history_status": mechanism.get("status", "missing"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument('--scope', type=Path, default=SCOPE)
    args = parser.parse_args()
    result = verify(args.run_id, args.scope)
    print(json.dumps(result, indent=2))
    if not result["ready_for_frozen_command"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
