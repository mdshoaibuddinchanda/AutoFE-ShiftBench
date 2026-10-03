"""Portable dataset, code, environment, and artifact provenance helpers.

The package is deliberately metadata-only: it fingerprints available local
bytes and reports unavailable or unverified inputs without downloading or
inventing source metadata.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION
from src.task_manifest import ManifestError, ManifestStore


PROVENANCE_SCHEMA_VERSION = "provenance_package_v1"
DATASET_REGISTRY_SCHEMA_VERSION = "dataset_registry_v1"
LINEAGE_SCHEMA_VERSION = "artifact_lineage_v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: str | Path, repo_root: str | Path) -> str:
    root = Path(repo_root).resolve()
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate.name


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def collect_code_identity(repo_root: str | Path = ".") -> dict[str, Any]:
    """Return the source identity actually visible at package/run creation."""
    root = Path(repo_root).resolve()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _git(root, "diff", "--binary")
    cached = _git(root, "diff", "--cached", "--binary")
    identity_material = "\n".join((status, diff, cached))
    return {
        "schema_version": "code_identity_v1",
        "commit": _git(root, "rev-parse", "HEAD") or None,
        "branch": _git(root, "branch", "--show-current") or None,
        "dirty_worktree": bool(status),
        "status_porcelain": status.splitlines(),
        "worktree_fingerprint_sha256": hashlib.sha256(identity_material.encode("utf-8")).hexdigest(),
    }


def _dataset_entries(dataset_list_path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(dataset_list_path.read_text(encoding="utf-8")) or {}
    entries = raw.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("dataset registry requires a datasets list")
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"name": entry})
        elif isinstance(entry, Mapping) and entry.get("name"):
            normalized.append({str(key): value for key, value in entry.items()})
        else:
            raise ValueError(f"Invalid dataset entry: {entry!r}")
    return normalized


def _inspect_csv(path: Path) -> dict[str, Any]:
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    rows = 0
    class_counts: dict[str, int] = {}
    for chunk in pd.read_csv(path, chunksize=10_000):
        if not columns:
            columns = [str(value) for value in chunk.columns]
            dtypes = {str(key): str(value) for key, value in chunk.dtypes.items()}
        rows += len(chunk)
        target = "target_label" if "target_label" in chunk.columns else "target" if "target" in chunk.columns else None
        if target is not None:
            for key, value in chunk[target].astype(str).value_counts(dropna=False).items():
                class_counts[str(key)] = class_counts.get(str(key), 0) + int(value)
    target_name = "target_label" if "target_label" in columns else "target" if "target" in columns else None
    return {"row_count": rows, "column_count": len(columns), "columns": columns, "dtypes": dtypes, "target_column": target_name, "class_counts": class_counts}


def build_dataset_registry(
    dataset_list_path: str | Path = "config/dataset_list.yaml",
    *,
    data_root: str | Path = "data/raw",
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Fingerprint available CSVs and explicitly label unavailable inputs."""
    root = Path(repo_root).resolve()
    list_path = Path(dataset_list_path)
    if not list_path.is_absolute():
        list_path = root / list_path
    raw_root = Path(data_root)
    if not raw_root.is_absolute():
        raw_root = root / raw_root
    datasets: list[dict[str, Any]] = []
    for configured in _dataset_entries(list_path):
        name = str(configured["name"])
        path = raw_root / f"{name}.csv"
        record: dict[str, Any] = {
            "name": name,
            "path": relative_path(path, root),
            "source": {key: value for key, value in configured.items() if key != "name"},
            "source_metadata_status": "configured" if len(configured) > 1 else "unconfigured",
            "verified": False,
        }
        if not path.exists():
            record.update({"status": "unavailable", "unavailable_reason": "raw_csv_missing"})
        else:
            try:
                record.update({"status": "verified", "verified": True, "byte_size": path.stat().st_size, "sha256": file_sha256(path), "schema": _inspect_csv(path)})
            except Exception as exc:
                record.update({"status": "unverified", "unavailable_reason": f"schema_or_hash_error:{type(exc).__name__}:{exc}"})
        datasets.append(record)
    registry = {
        "schema_version": DATASET_REGISTRY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_list_path": relative_path(list_path, root),
        "data_root": relative_path(raw_root, root),
        "loader": "src.data_loader.load_csv_dataset",
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "seed_scheme_version": SEED_SCHEME_VERSION,
        "datasets": datasets,
    }
    registry["registry_id"] = "dataset_registry_" + canonical_sha256({key: value for key, value in registry.items() if key != "generated_at"})[:24]
    return registry


def _requirement_names(path: Path) -> list[str]:
    names: list[str] = []
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.split("==", 1)[0].split("<", 1)[0].strip())
    return names


def collect_environment_identity(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    requirements = root / "requirements.txt"
    resolved: dict[str, str | None] = {}
    for name in _requirement_names(requirements):
        try:
            resolved[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            resolved[name] = None
    return {
        "schema_version": "environment_identity_v1",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV") or "P12_or_direct_conda_prefix_unreported",
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "cpu_count": os.cpu_count(),
        "hardware_thread_settings": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT") if os.environ.get(key) is not None},
        "requirements_path": relative_path(requirements, root),
        "requirements_sha256": file_sha256(requirements) if requirements.exists() else None,
        "resolved_dependencies": resolved,
        "portable_reconstruction": "Install requirements.txt in Conda P12; no machine-specific prefix is recorded.",
    }


def _artifact(path: Path, root: Path, role: str, *, required: bool = False, expected_sha256: str | None = None) -> dict[str, Any]:
    item = {"role": role, "path": relative_path(path, root), "required": required}
    if not path.exists():
        item.update({"status": "missing"})
        return item
    actual = file_sha256(path)
    item.update({"status": "valid" if expected_sha256 in (None, actual) else "conflict", "byte_size": path.stat().st_size, "sha256": actual})
    if expected_sha256 is not None:
        item["expected_sha256"] = expected_sha256
    return item


def _cache_inventory(cache_root: Path, root: Path, *, max_files: int = 10_000) -> dict[str, Any]:
    """Capture cache identity without copying cache bytes into the package."""
    if not cache_root.exists():
        return {"status": "unavailable", "path": relative_path(cache_root, root), "reason": "cache_root_missing", "entries": []}
    files = sorted(path for path in cache_root.rglob("*") if path.is_file())
    entries = [_artifact(path, root, "cache_artifact") for path in files[:max_files]]
    return {
        "status": "verified" if len(files) <= max_files else "truncated_unverified",
        "path": relative_path(cache_root, root),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "listed_file_count": len(entries),
        "max_files": max_files,
        "entries": entries,
    }


def _manifest_summary(manifest_db: Path, run_id: str) -> dict[str, Any]:
    snapshot = ManifestStore(manifest_db).snapshot(run_id)
    task_ids = [row["scientific_task_id"] for row in snapshot["tasks"]]
    counts: dict[str, int] = {}
    for row in snapshot["tasks"]:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {
        "run_id": run_id,
        "captured_at": snapshot["captured_at"],
        "run_status": snapshot["run"]["status"],
        "run_config": json.loads(snapshot["run"]["config_json"]),
        "task_counts": counts,
        "attempt_count": len(snapshot["attempts"]),
        "durable_result_count": len(snapshot["durable_results"]),
        "task_id_fingerprint_sha256": hashlib.sha256("\n".join(sorted(task_ids)).encode("utf-8")).hexdigest(),
        "manifest_schema_version": snapshot["run"]["schema_version"],
        "protocol_version": snapshot["run"]["protocol_version"],
        "seed_scheme_version": snapshot["run"]["seed_scheme_version"],
    }


def build_provenance_package(
    *,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    dataset_list_path: str | Path = "config/dataset_list.yaml",
    manifest_db: str | Path | None = None,
    manifest_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    run_id: str | None = None,
    analysis_dirs: Iterable[str | Path] = (),
) -> Path:
    root = Path(repo_root).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    code = collect_code_identity(root)
    environment = collect_environment_identity(root)
    registry = build_dataset_registry(dataset_list_path, repo_root=root)
    artifacts: list[dict[str, Any]] = []
    manifest_summary = None
    if manifest_db is not None and run_id is not None:
        db_path = Path(manifest_db)
        if not db_path.is_absolute():
            db_path = root / db_path
        manifest_summary = _manifest_summary(db_path, run_id)
        artifacts.append(_artifact(db_path, root, "manifest_sqlite", required=True))
    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        if not manifest_file.is_absolute():
            manifest_file = root / manifest_file
        artifacts.append(_artifact(manifest_file, root, "manifest_jsonl", required=False))
    if ledger_path is not None:
        ledger = Path(ledger_path)
        if not ledger.is_absolute():
            ledger = root / ledger
        artifacts.append(_artifact(ledger, root, "authoritative_result_ledger", required=True))
    for analysis_dir in analysis_dirs:
        directory = Path(analysis_dir)
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    artifacts.append(_artifact(path, root, "analysis_output", required=False))
    for dataset in registry["datasets"]:
        path = root / dataset["path"]
        artifacts.append(_artifact(path, root, "dataset_bytes", required=False, expected_sha256=dataset.get("sha256")))
    cache_inventory = _cache_inventory(root / "data" / "cache", root)
    artifacts.extend(cache_inventory.pop("entries", []))
    artifact_inventory = {"schema_version": "artifact_inventory_v1", "cache_inventory": cache_inventory, "artifacts": artifacts}
    nodes = [{"id": "datasets", "role": "dataset_bytes"}, {"id": "cache", "role": "split_corruption_feature_cache"}, {"id": "manifest", "role": "task_manifest"}, {"id": "results", "role": "authoritative_results"}, {"id": "analysis", "role": "analysis_outputs"}]
    edges = [{"from": "datasets", "to": "cache", "relationship": "split_corruption_preprocessing_features"}, {"from": "cache", "to": "results", "relationship": "fitted_features_and_diagnostics"}, {"from": "datasets", "to": "manifest", "relationship": "configured_inputs"}, {"from": "manifest", "to": "results", "relationship": "task_identity_and_durability"}, {"from": "results", "to": "analysis", "relationship": "authoritative_snapshot"}]
    lineage = {"schema_version": LINEAGE_SCHEMA_VERSION, "nodes": nodes, "edges": edges, "compatibility": {"protocol_version": EVALUATION_PROTOCOL_VERSION, "seed_scheme_version": SEED_SCHEME_VERSION}}
    package = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root_relative": ".",
        "run_id": run_id,
        "code_identity": code,
        "environment": environment,
        "dataset_registry_id": registry["registry_id"],
        "manifest_summary": manifest_summary,
        "artifact_inventory_sha256": canonical_sha256(artifact_inventory),
        "lineage_schema_version": LINEAGE_SCHEMA_VERSION,
    }
    (output / "provenance.json").write_text(_canonical(package), encoding="utf-8")
    (output / "dataset_registry.json").write_text(_canonical(registry), encoding="utf-8")
    (output / "code_identity.json").write_text(_canonical(code), encoding="utf-8")
    (output / "environment.json").write_text(_canonical(environment), encoding="utf-8")
    (output / "artifact_inventory.json").write_text(_canonical(artifact_inventory), encoding="utf-8")
    (output / "lineage.json").write_text(_canonical(lineage), encoding="utf-8")
    commands = [
        "python -m src.pipeline_runner --dry-run-manifest --max-datasets 2 --max-seeds 1 --max-folds 1 --max-conditions 2 --max-workers 1",
        "python -m src.provenance_cli verify --manifest-db <manifest.db> --ledger <results.jsonl> --run-id <run_id>",
        "python -m src.stats_analysis --sensitivity --manifest-db <manifest.db> --ledger <results.jsonl> --run-id <run_id>",
    ]
    (output / "reproduction_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# AutoFE-ShiftBench provenance package\n\nThis metadata package fingerprints available dataset bytes, code identity, environment settings, manifests, result ledgers, and analysis outputs. Missing datasets and unavailable historical ledgers remain explicitly unverified. It contains no benchmark data or caches.\n", encoding="utf-8")
    return output


def verify_provenance(
    *,
    repo_root: str | Path = ".",
    dataset_list_path: str | Path = "config/dataset_list.yaml",
    manifest_db: str | Path | None = None,
    ledger_path: str | Path | None = None,
    run_id: str | None = None,
    package_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only integrity check for current provenance and run artifacts."""
    root = Path(repo_root).resolve()
    registry = build_dataset_registry(dataset_list_path, repo_root=root)
    checks: list[dict[str, Any]] = []
    checks.extend({"name": f"dataset:{item['name']}", "status": item["status"]} for item in registry["datasets"])
    snapshot = None
    if manifest_db is not None and ledger_path is not None and run_id is not None:
        try:
            db_path = Path(manifest_db)
            ledger_file = Path(ledger_path)
            if not db_path.is_absolute():
                db_path = root / db_path
            if not ledger_file.is_absolute():
                ledger_file = root / ledger_file
            before = file_sha256(ledger_file)
            snapshot = ManifestStore(db_path).snapshot(run_id)
            after = file_sha256(ledger_file)
            checks.append({"name": "manifest_ledger_snapshot", "status": "valid" if before == after else "conflict", "ledger_sha256": after, "task_count": len(snapshot["tasks"]), "durable_result_count": len(snapshot["durable_results"])})
            protocol_ok = snapshot["run"]["protocol_version"] == EVALUATION_PROTOCOL_VERSION
            seed_ok = snapshot["run"]["seed_scheme_version"] == SEED_SCHEME_VERSION
            checks.append({"name": "protocol_compatibility", "status": "valid" if protocol_ok else "conflict", "observed": snapshot["run"]["protocol_version"], "expected": EVALUATION_PROTOCOL_VERSION})
            checks.append({"name": "seed_scheme_compatibility", "status": "valid" if seed_ok else "conflict", "observed": snapshot["run"]["seed_scheme_version"], "expected": SEED_SCHEME_VERSION})
            task_protocols: set[str] = set()
            task_seed_schemes: set[str] = set()
            for task in snapshot["tasks"]:
                try:
                    payload = json.loads(task["payload_json"])
                except json.JSONDecodeError:
                    checks.append({"name": f"task_payload:{task['scientific_task_id']}", "status": "malformed"})
                    continue
                if payload.get("protocol_version") is not None:
                    task_protocols.add(str(payload["protocol_version"]))
                if payload.get("seed_scheme_version") is not None:
                    task_seed_schemes.add(str(payload["seed_scheme_version"]))
            checks.append({"name": "task_protocol_compatibility", "status": "valid" if task_protocols == {EVALUATION_PROTOCOL_VERSION} else "conflict", "observed": sorted(task_protocols), "expected": [EVALUATION_PROTOCOL_VERSION]})
            checks.append({"name": "task_seed_scheme_compatibility", "status": "valid" if task_seed_schemes == {SEED_SCHEME_VERSION} else "conflict", "observed": sorted(task_seed_schemes), "expected": [SEED_SCHEME_VERSION]})
            task_ids = {str(row["scientific_task_id"]) for row in snapshot["tasks"]}
            with ledger_file.open(encoding="utf-8") as handle:
                seen: set[str] = set()
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        checks.append({"name": f"ledger_line:{line_number}", "status": "malformed"})
                        continue
                    if result.get("run_id") != run_id:
                        continue
                    task_id = result.get("scientific_task_id")
                    if not task_id or task_id not in task_ids:
                        checks.append({"name": f"ledger_line:{line_number}", "status": "unrecognized_task"})
                    elif task_id in seen:
                        checks.append({"name": f"ledger_task:{task_id}", "status": "duplicate"})
                    seen.add(task_id)
        except (OSError, ManifestError, json.JSONDecodeError) as exc:
            checks.append({"name": "manifest_ledger_snapshot", "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
    if package_dir is not None:
        inventory_path = Path(package_dir) / "artifact_inventory.json"
        if inventory_path.exists():
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            cache_status = inventory.get("cache_inventory", {}).get("status")
            if cache_status:
                checks.append({"name": "cache_inventory", "status": cache_status})
            for item in inventory.get("artifacts", []):
                path = root / item["path"]
                if item.get("status") == "missing":
                    checks.append({"name": item["role"], "status": "missing", "path": item["path"]})
                elif path.exists() and item.get("sha256") == file_sha256(path):
                    checks.append({"name": item["role"], "status": "valid", "path": item["path"]})
                else:
                    checks.append({"name": item["role"], "status": "conflict", "path": item["path"]})
    counts: dict[str, int] = {}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    overall = "valid" if not any(status in counts for status in ("conflict", "error", "malformed", "duplicate", "unrecognized_task", "unverified", "truncated_unverified")) else "attention_required"
    return {"schema_version": "provenance_verification_v1", "overall_status": overall, "counts": counts, "checks": checks, "code_identity": collect_code_identity(root), "registry_id": registry["registry_id"]}
