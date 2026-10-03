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


PROVENANCE_SCHEMA_VERSION = "provenance_package_anchored_v2"
DATASET_REGISTRY_SCHEMA_VERSION = "dataset_registry_v1"
LINEAGE_SCHEMA_VERSION = "artifact_task_lineage_v2"


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
        return "external/"+hashlib.sha256(str(candidate).encode()).hexdigest()[:24]+"/"+candidate.name


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def collect_code_identity(repo_root: str | Path = ".") -> dict[str, Any]:
    """Return the source identity actually visible at package/run creation."""
    root = Path(repo_root).resolve()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _git(root, "diff", "--binary")
    cached = _git(root, "diff", "--cached", "--binary")
    names=_git(root,"ls-files","--cached","--others","--exclude-standard").splitlines()
    if not names:
        names=[p.relative_to(root).as_posix() for p in root.rglob("*.py") if not set(p.relative_to(root).parts)&{".git",".venv","data","reports","__pycache__"}]
    contents={name:file_sha256(root/name) for name in sorted(set(names)) if (root/name).is_file()}
    identity_material = _canonical({"status":status,"diff":diff,"cached":cached,"contents":contents})
    return {
        "schema_version": "code_identity_v1",
        "commit": _git(root, "rev-parse", "HEAD") or None,
        "branch": _git(root, "branch", "--show-current") or None,
        "dirty_worktree": bool(status),
        "status_porcelain": status.splitlines(),
        "worktree_fingerprint_sha256": hashlib.sha256(identity_material.encode("utf-8")).hexdigest(),
        "source_content_hashes": contents,
        "identity_status": "verified_git_content" if _git(root,"rev-parse","HEAD") else "unverified_commit_identity",
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


def _artifact(path: Path, root: Path, role: str, *, required: bool = False, expected_sha256: str | None = None, hash_file=None) -> dict[str, Any]:
    item = {"role": role, "path": relative_path(path, root), "required": required}
    if not path.exists():
        item.update({"status": "missing"})
        return item
    actual = (hash_file or file_sha256)(path)
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
    snapshot = ManifestStore(manifest_db,read_only=True).snapshot(run_id)
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


def build_provenance_package(**kwargs) -> Path:
    """Create a manifest-anchored package of the actual run dependencies."""
    from src.provenance_evidence import build_package
    return build_package(**kwargs)


def verify_provenance(**kwargs) -> dict[str, Any]:
    """Read-only verification against authoritative results and package anchors."""
    from src.provenance_evidence import verify
    return verify(**kwargs)
