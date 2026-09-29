"""Stable identities and cache manifests for reproducible benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "autofe-shiftbench-protocol-2"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(
    dataset_identity: dict[str, Any],
    repetition_seed: int,
    fold: int,
    condition: str,
    protocol_version: str = PROTOCOL_VERSION,
) -> int:
    """Return a process-independent 32-bit seed for a task's perturbation."""
    payload = {
        "dataset_identity": dataset_identity,
        "repetition_seed": int(repetition_seed),
        "fold": int(fold),
        "condition": condition,
        "protocol_version": protocol_version,
    }
    return int.from_bytes(hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:4], "big")


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame: Any) -> str:
    """Hash a pandas frame including column order, dtypes, index, and values."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame_sha256 expects a pandas DataFrame")
    digest = hashlib.sha256()
    digest.update(canonical_json({
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_name": str(frame.index.name),
    }).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True, categorize=True).values.tobytes())
    return digest.hexdigest()


def vector_sha256(values: Any) -> str:
    import pandas as pd

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    digest = hashlib.sha256()
    digest.update(canonical_json({"dtype": str(series.dtype), "index": [str(i) for i in series.index]}).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(series, index=True, categorize=True).values.tobytes())
    return digest.hexdigest()


def index_sha256(indices: Any) -> str:
    return stable_digest([int(index) for index in indices])


def code_fingerprint(root: str | Path) -> str:
    """Hash source/config inputs so dirty code cannot reuse a prior cache."""
    root = Path(root)
    paths = sorted([
        *root.glob("*.py"), *root.glob("src/**/*.py"),
        *root.glob("config/*.yaml"), root / "requirements.txt",
    ])
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def current_git_commit(root: str | Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(root), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def cache_fingerprint(manifest: dict[str, Any]) -> str:
    # Full-source checksums are recorded for provenance, but the source CSV hash
    # includes y_test. Cache identity instead uses the training labels, train/test
    # features, and split hashes. This keeps held-out labels out of cache decisions.
    ignored = {"status", "cache_fingerprint", "source_csv_sha256", "saved_csv_sha256"}

    def strip_provenance_checksums(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip_provenance_checksums(child)
                for key, child in value.items()
                if key not in ignored
            }
        if isinstance(value, list):
            return [strip_provenance_checksums(child) for child in value]
        return value

    return stable_digest(strip_provenance_checksums(manifest))


def verify_cache_manifest(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    expected_fingerprint = cache_fingerprint(expected)
    observed_fingerprint = observed.get("cache_fingerprint") or cache_fingerprint(observed)
    if observed_fingerprint != expected_fingerprint:
        raise ValueError(
            "Cached task manifest does not match current dataset, split, protocol, configuration, or code fingerprint"
        )
