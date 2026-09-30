"""Run isolated, provenance-tracked corrected benchmark experiments.

The default experiment is the training-only corruption study. Domain partitions,
feature availability ablations, and majority-label relabeling are deliberately
excluded from this primary condition grid.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError as FutureTimeoutError
import multiprocessing as mp
from importlib.metadata import PackageNotFoundError, version as package_version
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

from src.checkpoint import has_success, init_db, record_task
from src.cache_manager import CacheManager, CacheNotReadyError
from src.task_scheduler import Lease, LeaseLost, TaskScheduler, TaskSpec
from src.data_loader import inspect_target_proxy_candidates, load_csv_dataset, load_dataset_names
from src.evaluation import compute_classification_metrics
from src.feature_engineering import DFSConfig, expand_features_with_dfs
from src.group_splits import (
    GroupSplitInfeasibleError,
    assert_group_fold_integrity,
    assess_group_fold_feasibility,
    canonical_feature_group_ids,
    summarize_groups,
    get_group_stratified_splits,
)
from src.model import build_model
from src.preprocessing import _build_preprocessor, _to_dense_array
from src.provenance import (
    PROTOCOL_VERSION,
    atomic_write_json,
    cache_fingerprint,
    code_fingerprint,
    current_git_commit,
    file_sha256,
    frame_sha256,
    index_sha256,
    stable_digest,
    stable_seed,
    vector_sha256,
    verify_cache_manifest,
)
from src.shift_generator import apply_perturbation
from src.splitters import (
    assert_fold_integrity,
    get_covariate_splits,
    get_population_splits,
    get_stratified_splits,
)


def _fit_and_score_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Fit one classifier task in a bounded child process.

    The parent process owns task leases, cache publication, checkpoints, and
    result rows.  Workers receive only the already prepared numeric matrices
    for one task, so no mutable scheduler or cache state crosses the process
    boundary.  Timestamps and the PID are returned as evidence of actual
    overlap in the integrated runner.
    """
    started_at = time.time()
    started = time.perf_counter()
    model_name = str(payload["model"])
    seed = int(payload["seed"])
    use_gpu = bool(payload.get("use_gpu", False))
    xtr = np.asarray(payload["xtr"], dtype=np.float32)
    xte = np.asarray(payload["xte"], dtype=np.float32)
    y_train_enc = np.asarray(payload["y_train_enc"])
    y_test_enc = np.asarray(payload["y_test_enc"])
    classes = np.asarray(payload["classes"])
    model = build_model(
        model_name,
        random_state=seed,
        use_gpu=bool(use_gpu and model_name in {"xgboost", "catboost"}),
    )
    fit_started = time.perf_counter()
    model.fit(xtr, y_train_enc)
    train_time = time.perf_counter() - fit_started
    infer_started = time.perf_counter()
    y_pred = model.predict(xte)
    y_proba = model.predict_proba(xte) if hasattr(model, "predict_proba") else np.zeros(
        (len(xte), len(classes))
    )
    infer_time = time.perf_counter() - infer_started
    train_pred = model.predict(xtr)
    train_proba = model.predict_proba(xtr) if hasattr(model, "predict_proba") else np.zeros(
        (len(y_train_enc), len(classes))
    )
    test_metrics = compute_classification_metrics(y_test_enc, y_pred, y_proba, classes)
    train_metrics = compute_classification_metrics(y_train_enc, train_pred, train_proba, classes)
    return {
        "status": "success",
        "pid": os.getpid(),
        "worker_started_unix": started_at,
        "worker_finished_unix": time.time(),
        "worker_elapsed_s": time.perf_counter() - started,
        "model_backend": "gpu" if use_gpu and model_name in {"xgboost", "catboost"} else "cpu",
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "y_pred": np.asarray(y_pred),
        "test_metrics": test_metrics,
        "train_auc": train_metrics.get("roc_auc"),
    }


PRIMARY_CONDITIONS: tuple[tuple[str, float], ...] = (
    ("clean", 0.0),
    ("gaussian_noise", 0.01), ("gaussian_noise", 0.05), ("gaussian_noise", 0.10),
    ("missing_values", 0.05), ("missing_values", 0.10), ("missing_values", 0.20),
    ("label_noise", 0.05), ("label_noise", 0.10), ("label_noise", 0.20),
)
DOMAIN_PARTITION_CONDITIONS: tuple[tuple[str, float], ...] = (
    ("covariate_partition", 0.0), ("population_partition", 0.0),
)
SEPARATE_EXPERIMENT_CONDITIONS: tuple[tuple[str, float], ...] = (
    ("feature_availability_ablation", 0.20), ("majority_label_relabeling", 0.0),
)

PIPELINE_CONFIGS: dict[str, DFSConfig] = {
    # Raw retains every preprocessed input column. The historical implementation
    # capped it at 20 base columns, making its 100-feature selectors no-ops.
    "Raw": DFSConfig(enable_dfs=False, selection_method="none", max_features=None, max_base_features=None),
    "Raw_Variance": DFSConfig(enable_dfs=False, selection_method="variance", max_features=100, max_base_features=None),
    "Raw_MI": DFSConfig(enable_dfs=False, selection_method="mi", max_features=100, max_base_features=None),
    # Matched-cap raw baseline: same 20-column input cap and 100-feature
    # selector budget as the default DFS condition, but without synthesis.
    "Raw_CapMatched": DFSConfig(enable_dfs=False, selection_method="variance", max_features=100, max_base_features=20),
    "AutoFE_Baseline": DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1),
    "AutoFE_MI": DFSConfig(enable_dfs=True, selection_method="mi", max_features=100, depth=1),
    "AutoFE_Random": DFSConfig(enable_dfs=True, selection_method="random", max_features=100, depth=1),
    "AutoFE_NoMultiply": DFSConfig(
        enable_dfs=True, selection_method="variance", max_features=100, depth=1,
        trans_primitives=["add_numeric", "subtract_numeric"],
    ),
}

# Operator-isolation scopes use the same depth, base-feature cap, output cap,
# and variance selector.  They are secondary diagnostics and do not alter the
# frozen Raw-vs-Baseline primary core.
for _operator in ("add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric"):
    _short = _operator.removesuffix("_numeric").title()
    PIPELINE_CONFIGS[f"AutoFE_Isolate_{_short}"] = DFSConfig(
        enable_dfs=True, selection_method="variance", max_features=100,
        max_base_features=20, depth=1, trans_primitives=[_operator],
    )
    PIPELINE_CONFIGS[f"AutoFE_LeaveOut_{_short}"] = DFSConfig(
        enable_dfs=True, selection_method="variance", max_features=100,
        max_base_features=20, depth=1,
        trans_primitives=[item for item in ("add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric") if item != _operator],
    )


def _condition_name(family: str, severity: float) -> str:
    return family if severity == 0.0 else f"{family}_{severity:.2f}"


def _domain_fold_status(
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> dict[str, Any]:
    """Summarize class support for a transductive domain partition."""
    labels = y.astype(str).to_numpy()
    global_classes = set(labels)
    fold_support = []
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        train_classes = set(labels[np.asarray(train_idx, dtype=np.int64)])
        test_classes = set(labels[np.asarray(test_idx, dtype=np.int64)])
        fold_support.append({
            "fold": fold_no,
            "train_class_count": len(train_classes),
            "test_class_count": len(test_classes),
            "train_has_all_classes": train_classes == global_classes,
            "test_has_all_classes": test_classes == global_classes,
            "test_has_at_least_two_classes": len(test_classes) >= 2,
        })
    auc_supported = bool(fold_support) and all(
        item["train_has_all_classes"]
        and item["test_has_all_classes"]
        and item["test_has_at_least_two_classes"]
        for item in fold_support
    )
    return {
        "fold_class_support": fold_support,
        "auc_status": "supported" if auc_supported else "infeasible",
        "auc_reason": (
            "all_train_and_test_folds_contain_all_classes"
            if auc_supported else "one_or_more_domain_folds_lacks_class_support_for_auc"
        ),
    }


def _resolve_target_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    """Resolve a single target explicitly; refuse a silent last-column fallback."""
    if requested and requested in frame.columns:
        return requested
    if "target_label" in frame.columns:
        return "target_label"
    if "target" in frame.columns:
        return "target"
    raise KeyError("Target column is not declared in metadata and neither 'target' nor 'target_label' exists")


def _task_dataset_checksum(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> str:
    """Hash task-visible data without using held-out labels for cache decisions."""
    return stable_digest({
        "x_train": frame_sha256(x_train),
        "y_train": vector_sha256(y_train),
        "x_test": frame_sha256(x_test),
        "train_indices": index_sha256(train_idx),
        "test_indices": index_sha256(test_idx),
    })


def _stable_perturbation_seed(
    dataset_identity: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    repetition_seed: int,
    fold: int,
    condition: str,
) -> int:
    """Derive corruption randomness from dataset identity and training rows only.

    The saved CSV checksum includes held-out labels and would let an evaluation
    label change alter the training corruption. The training-only fingerprint
    distinguishes changed local data while leaving held-out features/labels out.
    """
    safe_identity = {
        key: value for key, value in dataset_identity.items()
        if key not in {"saved_csv_sha256", "csv_sha256", "source_csv_sha256"}
    }
    safe_identity["training_rows_sha256"] = stable_digest({
        "x_train": frame_sha256(x_train),
        "y_train": vector_sha256(y_train),
    })
    return stable_seed(safe_identity, repetition_seed, fold, condition)


def _prepare_matrices(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    family: str,
    severity: float,
    perturbation_seed: int,
    pipeline_name: str,
    config: DFSConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, LabelEncoder, dict[str, Any], pd.DataFrame]:
    """Fit every learned step on training rows and transform held-out X only."""
    if not x_train.index.equals(y_train.index):
        raise ValueError("Training X/y row index and order differ before perturbation")
    x_train_corrupt, y_train_corrupt = apply_perturbation(
        x_train, y_train, shift_family=family, severity=severity, random_state=perturbation_seed,
    )
    if not x_train_corrupt.index.equals(y_train_corrupt.index):
        raise ValueError("Perturbation changed training X/y row alignment")

    # The label vocabulary is learned from training labels only. Test labels are
    # transformed solely after the fitted model predicts, for metric calculation.
    encoder = LabelEncoder().fit(y_train.astype(str))
    y_train_encoded = encoder.transform(y_train_corrupt.astype(str))

    preprocessor = _build_preprocessor(x_train_corrupt, encoding="onehot", scale_numeric=True)
    train_arr = _to_dense_array(preprocessor.fit_transform(x_train_corrupt))
    test_arr = _to_dense_array(preprocessor.transform(x_test))
    clean_test_arr = _to_dense_array(preprocessor.transform(x_test))
    columns = preprocessor.get_feature_names_out().tolist()
    x_train_prepped = pd.DataFrame(train_arr, columns=columns).reset_index(drop=True)
    x_test_prepped = pd.DataFrame(test_arr, columns=columns).reset_index(drop=True)
    x_test_clean = pd.DataFrame(clean_test_arr, columns=columns).reset_index(drop=True)
    y_train_series = pd.Series(y_train_encoded, index=x_train_prepped.index)

    cfg = DFSConfig(**asdict(config))
    cfg.random_seed = perturbation_seed
    x_train_fe, x_test_fe, metadata = expand_features_with_dfs(
        x_train_prepped, x_test_prepped, y_train_series, config=cfg,
    )
    return (
        x_train_fe.reset_index(drop=True), x_test_fe.reset_index(drop=True),
        y_train_encoded, encoder, metadata, x_test_clean,
    )


def prepare_task(
    frame: pd.DataFrame,
    *,
    target_column: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    family: str,
    severity: float,
    perturbation_seed: int,
    pipeline_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder, dict[str, Any], pd.DataFrame]:
    """Public preparation seam used by poison/alignment regression tests."""
    if target_column not in frame.columns:
        raise KeyError(f"Resolved target column {target_column!r} is absent")
    X = frame.drop(columns=[target_column])
    if target_column in X.columns:
        raise AssertionError("Resolved target must be absent at the split/preprocessing boundary")
    if np.intersect1d(train_indices, test_indices).size:
        raise AssertionError("Train/test row indices overlap")
    x_train, y_train = X.iloc[train_indices].copy(), frame[target_column].iloc[train_indices].copy()
    x_test, y_test = X.iloc[test_indices].copy(), frame[target_column].iloc[test_indices].copy()
    x_train_fe, x_test_fe, y_train_encoded, encoder, metadata, x_test_clean = _prepare_matrices(
        x_train, y_train, x_test,
        family=family, severity=severity, perturbation_seed=perturbation_seed,
        pipeline_name=pipeline_name, config=PIPELINE_CONFIGS[pipeline_name],
    )
    try:
        y_test_encoded = encoder.transform(y_test.astype(str))
    except ValueError as exc:
        raise ValueError("Held-out labels contain a class absent from training labels") from exc
    return x_train_fe, x_test_fe, y_train_encoded, y_test_encoded, encoder, metadata, x_test_clean


def _cache_paths(run_dir: Path, feature_task_key: str, pipeline_name: str) -> tuple[Path, Path]:
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{feature_task_key}_{pipeline_name}"
    return cache_dir / f"{stem}.pkl", cache_dir / f"{stem}.manifest.json"


def _load_or_create_feature_cache(
    run_dir: Path,
    feature_task_key: str,
    pipeline_name: str,
    expected_manifest: dict[str, Any],
    make_payload: Callable[[], tuple[Any, ...]],
) -> tuple[Any, ...]:
    cache_path, manifest_path = _cache_paths(run_dir, feature_task_key, pipeline_name)
    expected = dict(expected_manifest, pipeline=pipeline_name)
    expected["cache_fingerprint"] = cache_fingerprint(expected)
    rejected_reason = None
    if cache_path.exists() and manifest_path.exists():
        try:
            observed = json.loads(manifest_path.read_text(encoding="utf-8"))
            verify_cache_manifest(expected, observed)
            return pd.read_pickle(cache_path)
        except Exception as exc:
            # A stale, malformed, or unreadable cache is never accepted. It is
            # removed and regenerated from the current task inputs.
            rejected_reason = f"{type(exc).__name__}: {str(exc)}"[:500]
            cache_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
    payload = make_payload()
    temp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    pd.to_pickle(payload, temp_path)
    os.replace(temp_path, cache_path)
    if rejected_reason:
        expected["cache_regenerated_after_rejection"] = rejected_reason
    atomic_write_json(manifest_path, expected)
    return payload


def _bounded_cache_key(feature_task_key: str, pipeline_name: str, cache_fingerprint_value: str) -> str:
    """Stable cache identity for the lease-aware bounded cache."""
    return f"{feature_task_key}/{pipeline_name}/{cache_fingerprint_value}"


def _load_or_create_bounded_feature_cache(
    manager: CacheManager,
    *,
    feature_task_key: str,
    pipeline_name: str,
    expected_manifest: dict[str, Any],
    make_payload: Callable[[], tuple[Any, ...]],
    audit_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[tuple[Any, ...], bool]:
    """Read/create a lease-validated cache and return ``(payload, cache_hit)``.

    The lease covers the complete serialized read.  The caller may then delete
    the artifact after all compatible classifier consumers have reached a
    durable terminal state; no worker can observe a partially written file.
    """
    expected_manifest = dict(expected_manifest)
    expected_manifest["cache_fingerprint"] = cache_fingerprint(expected_manifest)
    cache_fp = str(expected_manifest["cache_fingerprint"])
    key = _bounded_cache_key(feature_task_key, pipeline_name, cache_fp)
    encoded, cache_hit = manager.get_or_create_bytes(
        key,
        lambda: pickle.dumps(make_payload(), protocol=pickle.HIGHEST_PROTOCOL),
        metadata={
            "pipeline": pipeline_name,
            "cache_fingerprint": cache_fp,
            "protocol_version": PROTOCOL_VERSION,
            "expected_manifest": expected_manifest,
        },
    )
    with manager.lease(key, owner=f"pid:{os.getpid()}") as lease:
        if audit_callback is not None:
            audit_callback("reader_acquired", {
                "cache_key": key, "active_readers": manager.reconcile().get("active_lease_count", 0),
            })
        payload = pickle.loads(lease.read_bytes())
    if not isinstance(payload, tuple):
        raise ValueError("bounded feature cache payload must be a tuple")
    if audit_callback is not None:
        audit_callback("hit" if cache_hit else "build", {"cache_key": key})
    return payload, cache_hit


def _initialize_result_index(index_path: Path, results_path: Path) -> None:
    """Rebuild a compact task-key index once at process startup."""
    import sqlite3

    with sqlite3.connect(index_path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS result_keys (task_key TEXT NOT NULL, status TEXT NOT NULL, PRIMARY KEY(task_key, status))")
        if results_path.exists():
            with results_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = row.get("task_key")
                    if key:
                        connection.execute(
                            "INSERT OR IGNORE INTO result_keys(task_key, status) VALUES (?, ?)",
                            (str(key), str(row.get("status", "unknown"))),
                        )


def _safe_result_write(path: Path, row: dict[str, Any], *, index_path: Path | None = None) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): json_safe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(child) for child in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    task_key = row.get("task_key")
    if index_path is not None and task_key:
        import sqlite3

        status = str(row.get("status", "unknown"))
        with sqlite3.connect(index_path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM result_keys WHERE task_key=? AND status=?", (str(task_key), status)
            ).fetchone()
        if exists:
            return
        # Append first, then index.  If the process dies between these two
        # operations, the next startup scan repairs the index before resume.
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        with sqlite3.connect(index_path) as connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO result_keys(task_key, status) VALUES (?, ?)", (str(task_key), status)
            ).rowcount
        if inserted == 0:
            # A duplicate append can only occur after a crash; leave the
            # append-only evidence intact and let analysis use the unique key.
            return
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False, default=str) + "\n")
        stream.flush()


def _scheduler_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe task result for durable scheduler publication."""
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): json_safe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(child) for child in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return json_safe(row)


def _recover_scheduler_payload(scheduler: TaskScheduler, task_key: str) -> dict[str, Any] | None:
    """Read a verified scheduler result for checkpoint repair on resume."""
    state = scheduler.result_state(task_key)
    if not state:
        return None
    path = Path(state["artifact_path"])
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    payload = envelope.get("payload")
    return payload if isinstance(payload, dict) and payload.get("task_key") == task_key else None


def _record_failure(
    *, run_dir: Path, db_path: Path, run_id: str, task_key: str,
    phase: str, dataset: str, seed: int, fold: int, condition: str,
    pipeline: str, model: str, fingerprint: str, exc: BaseException,
    result_index_path: Path | None = None,
) -> dict[str, Any]:
    summary = " ".join(str(exc).split())[:1000]
    record_task(
        db_path, run_id=run_id, task_key=task_key, phase=phase, status="failed",
        dataset=dataset, seed=seed, fold=fold, condition=condition,
        pipeline=pipeline, model=model, manifest_fingerprint=fingerprint,
        exception_type=type(exc).__name__, error_summary=summary,
    )
    row = {
        "run_id": run_id, "task_key": task_key, "dataset": dataset, "seed": seed,
        "fold": fold, "condition": condition, "pipeline": pipeline, "model": model,
        "phase": phase, "status": "failed", "exception_type": type(exc).__name__,
        "error_summary": summary,
    }
    _safe_result_write(run_dir / "results.jsonl", row, index_path=result_index_path)
    return row


def _dataset_identity(dataset_name: str, csv_path: Path, sidecar_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_checksum = file_sha256(csv_path)
    sidecar: dict[str, Any] = {}
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    identity = sidecar.get("dataset_identity", {})
    if not identity:
        identity = {"provider": "local_csv", "requested_name": dataset_name}
    identity = dict(identity)
    identity["saved_csv_sha256"] = source_checksum
    return identity, sidecar


def _compact_group_status(status: dict[str, Any]) -> dict[str, Any]:
    """Keep fold diagnostics bounded while retaining a reproducible conflict digest."""
    compact = dict(status)
    conflict_ids = list(compact.pop("conflicting_label_group_ids", []))
    compact["conflicting_label_group_id_count"] = len(conflict_ids)
    compact["conflicting_label_group_id_sha256"] = stable_digest(conflict_ids)
    compact["conflicting_label_group_id_examples"] = conflict_ids[:5]
    return compact


def _save_run_manifest(run_dir: Path, base: dict[str, Any], outcomes: list[dict[str, Any]]) -> None:
    latest: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        latest[outcome["task_key"]] = outcome
    outcomes[:] = list(latest.values())
    # Keep the historical ``counts`` shape stable for existing consumers while
    # publishing a complete status accounting block for corrected campaigns.
    counts: dict[str, int] = {"success": 0, "failed": 0}
    counts_by_status: dict[str, int] = {
        "success": 0, "failed": 0, "skipped": 0, "timed_out": 0, "pending": 0,
    }
    for outcome in outcomes:
        counts[outcome["status"]] = counts.get(outcome["status"], 0) + 1
        if outcome.get("status") in counts_by_status:
            counts_by_status[outcome["status"]] += 1
    configuration = base.get("configuration", {})
    expected = (
        len(configuration.get("datasets", []))
        * len(configuration.get("seeds", []))
        * len(configuration.get("folds", []))
        * len(configuration.get("conditions", []))
        * len(configuration.get("pipelines", []))
        * len(configuration.get("models", []))
    )
    counts_by_status["pending"] = max(expected - len(outcomes), 0)
    forced_status = base.get("status") if base.get("status", "").startswith("blocked_") else None
    if forced_status:
        status = forced_status
    elif len(outcomes) < expected:
        status = "running_with_failures" if counts["failed"] else "running"
    else:
        status = "completed_with_failures" if counts["failed"] else "complete"
    phase_counts: dict[str, int] = {}
    for outcome in outcomes:
        phase = outcome.get("phase")
        if phase:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
    manifest = dict(
        base, status=status, expected_tasks=expected, counts=counts,
        counts_by_status=counts_by_status,
        task_accounting={
            "expected": expected,
            "terminal": sum(counts_by_status[name] for name in ("success", "failed", "skipped", "timed_out")),
            "counts_by_status": counts_by_status,
            "counts_by_phase": phase_counts,
            "pending_definition": "expected task keys absent from the current-attempt task list",
        },
        tasks=outcomes,
    )
    # Windows antivirus/indexer/read-only observers can briefly hold the
    # destination during ``os.replace``.  Retry the atomic publication instead
    # of converting that transient filesystem condition into a task failure.
    manifest_path = run_dir / "manifest.json"
    for attempt in range(8):
        try:
            atomic_write_json(manifest_path, manifest)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def _upsert_outcome(outcomes: list[dict[str, Any]], outcome: dict[str, Any]) -> None:
    """Keep one current manifest entry per task while retaining attempt rows in JSONL."""
    outcomes[:] = [row for row in outcomes if row.get("task_key") != outcome["task_key"]]
    outcomes.append(outcome)


def _runtime_versions() -> dict[str, str]:
    packages = (
        "numpy", "pandas", "scikit-learn", "scipy", "featuretools", "woodwork",
        "xgboost", "lightgbm", "catboost",
    )
    versions: dict[str, str] = {
        "python": platform.python_version(), "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = package_version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def run_experiment(
    data_paths: dict[str, str | Path],
    *,
    run_id: str,
    output_root: str | Path = "corrected_runs",
    seeds: list[int] | None = None,
    folds: list[int] | None = None,
    conditions: tuple[tuple[str, float], ...] = PRIMARY_CONDITIONS,
    pipelines: tuple[str, ...] = ("Raw", "AutoFE_Baseline"),
    models: tuple[str, ...] = ("logistic_regression",),
    n_splits: int = 5,
    split_policy: str = "row_level",
    cache_policy: str = "retain",
    cache_max_bytes: int | None = None,
    cache_lease_ttl_seconds: float = 3600.0,
    durable_scheduler: bool = False,
    scheduler_lease_seconds: float = 3600.0,
    scheduler_max_attempts: int = 3,
    cache_audit: bool = False,
    use_gpu: bool = False,
    workers: int = 1,
    failure_hook: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a small or full grid with explicit task status and isolated outputs."""
    if split_policy not in {"row_level", "group_aware"}:
        raise ValueError("split_policy must be 'row_level' or 'group_aware'")
    if cache_policy not in {"retain", "bounded"}:
        raise ValueError("cache_policy must be 'retain' or 'bounded'")
    if cache_policy == "bounded" and cache_max_bytes is not None and cache_max_bytes <= 0:
        raise ValueError("cache_max_bytes must be positive when supplied")
    if scheduler_lease_seconds <= 0:
        raise ValueError("scheduler_lease_seconds must be positive")
    if scheduler_max_attempts < 1:
        raise ValueError("scheduler_max_attempts must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    seeds = seeds or [42, 123, 456, 789, 2025]
    folds = folds or list(range(1, n_splits + 1))
    selected_families = {family for family, _severity in conditions}
    primary_families = {"clean", "gaussian_noise", "missing_values", "label_noise"}
    domain_families = {"covariate_partition", "population_partition"}
    availability_families = {"feature_availability_ablation"}
    relabel_families = {"majority_label_relabeling"}
    scope_sets = {
        "primary_training_corruption": primary_families,
        "transductive_domain_partition": domain_families,
        "feature_availability_ablation": availability_families,
        "majority_label_relabeling": relabel_families,
    }
    selected_scopes = [scope for scope, families_in_scope in scope_sets.items()
                       if selected_families and selected_families.issubset(families_in_scope)]
    if len(selected_scopes) != 1:
        raise ValueError(
            "A run must contain conditions from exactly one scope: primary corruption, "
            "domain partition, feature-availability ablation, or majority-label relabeling"
        )
    experiment_scope = selected_scopes[0]
    if split_policy == "group_aware" and experiment_scope == "transductive_domain_partition":
        # PCA/K-means partitions are fitted on the complete target-free feature
        # table. Applying those row-level partitions here would silently bypass
        # the group-aware zero-overlap guarantee. A group-constrained
        # transductive protocol needs its own prespecified assignment rule.
        raise ValueError(
            "transductive_domain_partition currently supports row_level only; "
            "group-aware transductive folds require a separately prespecified "
            "group-constrained assignment protocol"
        )
    unknown_pipelines = sorted(set(pipelines).difference(PIPELINE_CONFIGS))
    if unknown_pipelines:
        raise ValueError(f"Unknown pipelines: {unknown_pipelines}")
    if not data_paths:
        raise ValueError("At least one dataset path is required")
    normalized_data_paths = {name: Path(path) for name, path in data_paths.items()}
    missing_paths = [str(path) for path in normalized_data_paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Configured datasets are missing; no run was started: {missing_paths}")
    source_checksums = {name: file_sha256(path) for name, path in normalized_data_paths.items()}
    sidecar_paths = {
        name: path.with_name(f"{path.stem}_meta.json")
        for name, path in normalized_data_paths.items()
    }
    sidecar_checksums = {
        name: file_sha256(path) if path.exists() else None
        for name, path in sidecar_paths.items()
    }
    run_dir = Path(output_root) / run_id
    existing_manifest = None
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists():
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("run_id") != run_id:
                raise ValueError("Existing run directory has a different run ID")
            existing_manifest = existing
        elif any(run_dir.iterdir()):
            raise ValueError("Existing run directory has no manifest; choose a new run ID")
    source_fingerprint = code_fingerprint(Path.cwd())
    runtime_info = _runtime_versions()
    runtime_fingerprint = stable_digest(runtime_info)
    config_data = {
        "protocol": PROTOCOL_VERSION, "datasets": list(normalized_data_paths),
        "conditions": [list(x) for x in conditions],
        "pipelines": list(pipelines), "models": list(models), "seeds": seeds,
        "folds": folds, "n_splits": n_splits,
        "split_policy": split_policy,
        "cache_policy": cache_policy,
        "cache_max_bytes": cache_max_bytes,
        "durable_scheduler": durable_scheduler,
        "scheduler_lease_seconds": scheduler_lease_seconds,
        "scheduler_max_attempts": scheduler_max_attempts,
        "cache_audit": bool(cache_audit),
        "use_gpu": bool(use_gpu),
        "workers": int(workers),
        "pipeline_configs": {name: asdict(PIPELINE_CONFIGS[name]) for name in pipelines},
    }
    config_fingerprint = stable_digest(config_data)
    if existing_manifest and (
        existing_manifest.get("configuration_fingerprint") != config_fingerprint
        or existing_manifest.get("code_fingerprint") != source_fingerprint
        or existing_manifest.get("runtime_fingerprint") != runtime_fingerprint
    ):
        raise ValueError("Existing run ID has a different code/configuration/runtime identity; choose a new run ID")
    if existing_manifest:
        previous_identities = {
            row.get("dataset"): (row.get("source_csv_sha256"), row.get("source_metadata_sha256"))
            for row in existing_manifest.get("datasets", [])
        }
        changed = [
            name for name in source_checksums
            if name in previous_identities
            and previous_identities[name] != (source_checksums[name], sidecar_checksums[name])
        ]
        if changed:
            raise ValueError(
                f"Dataset source or metadata checksum changed for {changed}; use a new run ID to preserve provenance"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "checkpoints.sqlite"
    result_index_path = run_dir / "results_index.sqlite"
    _initialize_result_index(result_index_path, run_dir / "results.jsonl")
    init_db(db_path)
    cache_manager = (
        CacheManager(
            run_dir / "cache_bounded",
            lease_ttl_seconds=cache_lease_ttl_seconds,
            max_bytes=cache_max_bytes,
        )
        if cache_policy == "bounded" else None
    )
    scheduler = (
        TaskScheduler(
            run_dir / "scheduler.sqlite",
            artifact_dir=run_dir / "scheduler_results",
            lease_seconds=scheduler_lease_seconds,
        )
        if durable_scheduler else None
    )
    scheduler_worker_id = f"runner-pid-{os.getpid()}"
    if scheduler is not None:
        scheduler.reconcile()
        # A new process on this single-host run may replace a coordinator
        # whose lease is still live.  Reclaim only foreign worker leases after
        # reconciling any result artifact that was already durably published.
        scheduler.reclaim_foreign_leases(scheduler_worker_id)
    prior_cache_audit = (existing_manifest or {}).get("cache_audit", {})
    if cache_audit and prior_cache_audit.get("enabled"):
        cache_audit_state: dict[str, Any] = {
            "enabled": True,
            "artifacts": dict(prior_cache_audit.get("artifacts", {})),
            "policy": prior_cache_audit.get(
                "policy",
                "one build, all compatible consumers, delete after durable terminal consumers and no active leases",
            ),
        }
    else:
        cache_audit_state = (
            {
                "enabled": True,
                "artifacts": {},
                "policy": "one build, all compatible consumers, delete after durable terminal consumers and no active leases",
            }
            if cache_audit else {"enabled": False}
        )
    base_manifest = {
        "run_id": run_id, "protocol_version": PROTOCOL_VERSION,
        "created_utc": (existing_manifest or {}).get(
            "created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
        "code_commit": current_git_commit(Path.cwd()),
        "code_fingerprint": source_fingerprint,
        "configuration": config_data, "configuration_fingerprint": config_fingerprint,
        "runtime": runtime_info,
        "runtime_fingerprint": runtime_fingerprint,
        "primary_conditions": ["clean", "gaussian_noise_*", "missing_values_*", "label_noise_*"],
        "experiment_scope": experiment_scope,
        "split_policy": split_policy,
        "cache_policy": cache_policy,
        "durable_scheduler": durable_scheduler,
        "cache_audit": cache_audit_state,
        "partition_uses_held_out_features": False,
        "datasets": list((existing_manifest or {}).get("datasets", [])),
    }
    outcomes: list[dict[str, Any]] = list((existing_manifest or {}).get("tasks", []))
    if existing_manifest is None:
        _save_run_manifest(run_dir, base_manifest, outcomes)

    # A bounded process pool is used only for the expensive model-fit phase.
    # Feature preparation, cache leases, scheduler claims, and durable writes
    # remain coordinator-owned.  This makes the main runner itself concurrent
    # while preserving one authoritative task ledger.
    executor: ProcessPoolExecutor | None = None
    pending_futures: list[tuple[Future, dict[str, Any]]] = []
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        )

    def _record_phase2(meta: dict[str, Any], fit: dict[str, Any]) -> None:
        """Publish one worker result or durable failure in the coordinator."""
        task_manifest = meta["task_manifest"]
        task = meta["task"]
        task_key = meta["task_key"]
        task_fp = meta["task_fp"]
        scheduler_lease = meta.get("scheduler_lease")
        dataset_name = meta["dataset_name"]
        seed = meta["seed"]
        fold = meta["fold"]
        condition = meta["condition"]
        pipeline_name = meta["pipeline_name"]
        model_name = meta["model_name"]
        audit_record = meta.get("audit_record")
        try:
            if fit.get("status") != "success":
                raise RuntimeError(str(fit.get("error", "worker returned no result")))
            x_train_fe = meta["x_train_fe"]
            x_test_fe = meta["x_test_fe"]
            x_test_clean = meta["x_test_clean"]
            y_pred = np.asarray(fit["y_pred"])
            test_metrics = fit["test_metrics"]
            result = {
                **task, "run_id": run_id, "task_key": task_key,
                "status": "success", "n_train": int(meta["n_train"]), "n_test": int(meta["n_test"]),
                "split_policy": split_policy,
                "model_backend": fit.get("model_backend", "cpu"),
                "experiment_scope": experiment_scope,
                "split_status": meta["task_split_status"],
                "n_original": int(x_test_clean.shape[1]),
                "train_time_s": float(fit.get("train_time_s", 0.0)),
                "infer_time_s": float(fit.get("infer_time_s", 0.0)),
                "worker_pid": int(fit.get("pid", -1)),
                "worker_started_unix": float(fit.get("worker_started_unix", 0.0)),
                "worker_finished_unix": float(fit.get("worker_finished_unix", 0.0)),
                "worker_elapsed_s": float(fit.get("worker_elapsed_s", 0.0)),
                "autofe_gen_time_s": float(meta["fe_meta"].get("generation_time_s", 0.0)),
                "autofe_cache_hit": bool(meta["cache_was_present"]),
                "n_generated": int(meta["fe_meta"].get("n_generated", 0)),
                "n_retained": int(meta["fe_meta"].get("n_retained", x_train_fe.shape[1])),
                "ram_used_mb": float(meta["fe_meta"].get("ram_used_mb", 0.0)),
                "operator_counts": meta["fe_meta"].get("operator_counts", {}),
                "operator_candidate_counts": meta["fe_meta"].get("operator_candidate_counts", {}),
                "operator_configuration": meta["fe_meta"].get("operator_configuration", {
                    "enabled_operators": list(PIPELINE_CONFIGS[pipeline_name].trans_primitives)
                    if PIPELINE_CONFIGS[pipeline_name].enable_dfs else [],
                    "excluded_operators": [],
                }),
                "candidate_history": meta["fe_meta"].get("candidate_history", {
                    "enabled": False, "records_written": 0, "score_scope": "train",
                }),
                "wasserstein": None, "ks_stat": None,
                "cache_fingerprint": meta["feature_cache_fp"],
                "task_fingerprint": task_fp,
                "train_matrix_sha256": frame_sha256(x_train_fe),
                "test_matrix_sha256": frame_sha256(x_test_fe),
                "prediction_sha256": stable_digest(np.asarray(y_pred).tolist()),
                "train_auc": fit.get("train_auc"),
                **test_metrics,
            }
            _safe_result_write(run_dir / "results.jsonl", result, index_path=result_index_path)
            if scheduler is not None and scheduler_lease is not None:
                scheduler.publish_result(scheduler_lease, _scheduler_payload(result))
                if failure_hook:
                    failure_hook("after_scheduler_publish", task)
            record_task(
                db_path, run_id=run_id, task_key=task_key, phase="phase2", status="success",
                dataset=dataset_name, seed=seed, fold=fold, condition=condition,
                pipeline=pipeline_name, model=model_name, manifest_fingerprint=task_fp,
            )
            _upsert_outcome(outcomes, dict(task_manifest, status="success"))
        except Exception as exc:
            if scheduler is not None and scheduler_lease is not None:
                try:
                    scheduler.record_failure(scheduler_lease, type(exc).__name__, str(exc))
                except LeaseLost:
                    # The result may already have been durably published (for
                    # example, a coordinator manifest write failed after the
                    # scheduler transition).  Preserve the local failure
                    # evidence and let reconciliation repair the checkpoint.
                    pass
            failure = _record_failure(
                run_dir=run_dir, db_path=db_path, run_id=run_id, task_key=task_key,
                phase="phase2", dataset=dataset_name, seed=seed, fold=fold,
                condition=condition, pipeline=pipeline_name, model=model_name,
                fingerprint=task_fp, exc=exc, result_index_path=result_index_path,
            )
            _upsert_outcome(outcomes, dict(task_manifest, **failure))
        if audit_record is not None and task_key not in audit_record["terminal_consumer_task_keys"]:
            audit_record["terminal_consumer_task_keys"].append(task_key)
        _save_run_manifest(run_dir, base_manifest, outcomes)

    def _drain_one() -> None:
        def heartbeat_pending() -> None:
            if scheduler is None:
                return
            for _, pending_meta in pending_futures:
                lease = pending_meta.get("scheduler_lease")
                if lease is None:
                    continue
                try:
                    pending_meta["scheduler_lease"] = scheduler.heartbeat(lease)
                except LeaseLost:
                    # Publication will classify the task as a durable lease
                    # failure; do not let one stale receipt stop unrelated
                    # futures from being drained.
                    continue

        future, meta = pending_futures[0]
        while True:
            heartbeat_pending()
            try:
                # Keep coordinator-owned scheduler leases alive while a
                # child process is fitting.  This also bounds how long the
                # parent can remain unresponsive to recovery signals.
                fit = future.result(timeout=min(30.0, max(1.0, scheduler_lease_seconds / 3.0)))
                break
            except FutureTimeoutError:
                continue
            except Exception as exc:
                fit = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                break
        pending_futures.pop(0)
        _record_phase2(meta, fit)

    has_transductive_partition = experiment_scope == "transductive_domain_partition"
    base_manifest["partition_uses_held_out_features"] = has_transductive_partition
    if experiment_scope != "primary_training_corruption":
        base_manifest["primary_conditions"] = []
    if has_transductive_partition:
        base_manifest["domain_partition_note"] = (
            "PCA/K-means partition geometry uses the complete feature matrix, including held-out rows; "
            "these conditions are a separate transductive domain-partition experiment."
        )
    for dataset_name, csv_path in normalized_data_paths.items():
        source_csv_checksum = source_checksums[dataset_name]
        sidecar_path = sidecar_paths[dataset_name]
        dataset_identity, sidecar = _dataset_identity(dataset_name, csv_path, sidecar_path)
        frame = load_csv_dataset(csv_path)
        target_column = _resolve_target_column(frame, sidecar.get("target_column"))
        X = frame.drop(columns=[target_column])
        if target_column in X.columns:
            raise AssertionError("Target column reached splitter input")
        y = frame[target_column]
        proxy_audit = inspect_target_proxy_candidates(
            X, y, target_column=target_column,
            source_target_name=sidecar.get("source_target_name"),
        )
        # Exact-feature groups are formed once from the unperturbed raw
        # predictors.  Their IDs and summary are persisted for both tracks so
        # the row-level overlap diagnostic is inspectable and the group-aware
        # track can assert zero shared groups without a silent fallback.
        group_ids = canonical_feature_group_ids(X)
        group_summary = _compact_group_status(summarize_groups(group_ids, y))
        group_status = _compact_group_status(assess_group_fold_feasibility(group_ids, y, n_splits, 42))
        dataset_manifest = {
            "dataset": dataset_name, "path": csv_path.as_posix(),
            "dataset_identity": dataset_identity, "source_csv_sha256": source_csv_checksum,
            "source_metadata_sha256": sidecar_checksums[dataset_name],
            "target_column": target_column, "schema_columns": list(frame.columns),
            "target_proxy_review": proxy_audit,
            "split_policy": split_policy,
            "canonicalization": {
                "target_excluded_raw_predictors": True,
                "schema_and_column_order_included": True,
                "missing_values_unified": True,
                "numeric_1_equals_numeric_1_0": True,
                "text_1_distinct_from_numeric_1": True,
                "group_id_sha256": stable_digest(group_ids.tolist()),
            },
            "group_summary": group_summary,
            "group_fold_status_seed_42": group_status,
            "group_fold_status_by_seed": {},
        }
        base_manifest["datasets"] = [
            row for row in base_manifest.get("datasets", []) if row.get("dataset") != dataset_name
        ] + [dataset_manifest]
        for seed in seeds:
            if split_policy == "group_aware":
                # Primary classification metrics require all classes in every
                # train/test fold.  Infeasibility is explicit and never
                # substituted with row-level folds.
                try:
                    ordinary_splits, split_status = get_group_stratified_splits(
                        X, y, n_splits, seed, target_column=target_column,
                        require_class_support=True, require_auc=True, return_status=True,
                    )
                except GroupSplitInfeasibleError as exc:
                    dataset_manifest["group_fold_status_by_seed"][str(seed)] = _compact_group_status(exc.status)
                    base_manifest["datasets"] = [
                        row for row in base_manifest.get("datasets", []) if row.get("dataset") != dataset_name
                    ] + [dataset_manifest]
                    base_manifest["status"] = "blocked_group_split_infeasible"
                    base_manifest["split_policy_blocker"] = {
                        "dataset": dataset_name, "seed": seed,
                        "reason": str(exc), "status": _compact_group_status(exc.status),
                    }
                    _save_run_manifest(run_dir, base_manifest, outcomes)
                    raise
                assert_group_fold_integrity(ordinary_splits, group_ids)
                split_status = _compact_group_status(split_status)
                split_status["split_policy"] = "group_aware"
                dataset_manifest["group_fold_status_by_seed"][str(seed)] = split_status
            else:
                ordinary_splits = get_stratified_splits(X, y, n_splits, seed, target_column=target_column)
                assert_fold_integrity(ordinary_splits, len(frame))
                split_status = {"split_policy": "row_level", "split_feasible": True}
            domain_splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
            domain_metadata: dict[str, dict[str, Any]] = {}
            for family, severity in conditions:
                if family == "covariate_partition":
                    if family not in domain_splits:
                        splits, partition_meta = get_covariate_splits(
                            X, n_splits, seed, target_column=target_column, return_metadata=True,
                        )
                        domain_splits[family] = splits
                        domain_metadata[family] = {
                            **partition_meta, **_domain_fold_status(splits, y),
                        }
                    splits = domain_splits[family]
                elif family == "population_partition":
                    if family not in domain_splits:
                        splits, partition_meta = get_population_splits(
                            X, n_splits, seed, target_column=target_column, return_metadata=True,
                        )
                        domain_splits[family] = splits
                        domain_metadata[family] = {
                            **partition_meta, **_domain_fold_status(splits, y),
                        }
                    splits = domain_splits[family]
                else:
                    splits = ordinary_splits
                assert_fold_integrity(splits, len(frame))
                for fold in folds:
                    if fold < 1 or fold > len(splits):
                        raise ValueError(f"Fold {fold} outside 1..{len(splits)}")
                    train_idx, test_idx = (np.asarray(idx, dtype=np.int64) for idx in splits[fold - 1])
                    x_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
                    x_test, y_test = X.iloc[test_idx].copy(), y.iloc[test_idx].copy()
                    task_checksum = _task_dataset_checksum(x_train, y_train, x_test, train_idx, test_idx)
                    condition = _condition_name(family, severity)
                    perturb_seed = _stable_perturbation_seed(
                        dataset_identity, x_train, y_train, seed, fold, condition,
                    )
                    feature_key = stable_digest({
                        "dataset": dataset_name, "seed": seed, "fold": fold,
                        "condition": condition, "dataset_checksum": task_checksum,
                    })[:20]
                    for pipeline_name in pipelines:
                        feature_consumer_task_keys: list[str] = []
                        for model_position, model_name in enumerate(models):
                            task = {
                                "dataset": dataset_name, "seed": seed, "fold": fold,
                                "condition": condition, "pipeline": pipeline_name, "model": model_name,
                                "split_policy": split_policy,
                            }
                            task_key = stable_digest(dict(task, run_id=run_id))
                            task_split_status = dict(split_status)
                            if family in domain_metadata:
                                task_split_status["domain_partition"] = domain_metadata[family]
                            task_manifest = {
                                **task, "task_key": task_key, "feature_task_key": feature_key,
                                "dataset_identity": dataset_identity,
                                "source_csv_sha256": source_csv_checksum,
                                "dataset_checksum": task_checksum,
                                "train_indices_sha256": index_sha256(train_idx),
                                "test_indices_sha256": index_sha256(test_idx),
                                "stable_perturbation_seed": perturb_seed,
                                "configuration_fingerprint": config_fingerprint,
                                "code_fingerprint": source_fingerprint,
                                "runtime_fingerprint": runtime_fingerprint,
                                "target_column": target_column,
                                "split_status": task_split_status,
                                "status": "running",
                            }
                            feature_manifest = {
                                "dataset_checksum": task_checksum,
                                "train_indices_sha256": task_manifest["train_indices_sha256"],
                                "test_indices_sha256": task_manifest["test_indices_sha256"],
                                "stable_perturbation_seed": perturb_seed,
                                "configuration_fingerprint": config_fingerprint,
                                "code_fingerprint": source_fingerprint,
                                "runtime_fingerprint": runtime_fingerprint,
                                "protocol_version": PROTOCOL_VERSION,
                                "source_csv_sha256": source_csv_checksum,
                                "dataset_identity": dataset_identity,
                                "condition": condition,
                                "pipeline": pipeline_name,
                                "split_policy": split_policy,
                            }
                            task_fp = cache_fingerprint(task_manifest)
                            feature_cache_fp = cache_fingerprint(feature_manifest)
                            task_manifest["task_fingerprint"] = task_fp
                            task_manifest["cache_fingerprint"] = feature_cache_fp
                            feature_consumer_task_keys.append(task_key)
                            cache_audit_key = _bounded_cache_key(feature_key, pipeline_name, feature_cache_fp)
                            if cache_audit:
                                cache_audit_artifacts = base_manifest["cache_audit"]["artifacts"]
                                cache_audit_artifacts.setdefault(cache_audit_key, {
                                    "cache_key": cache_audit_key,
                                    "feature_task_key": feature_key,
                                    "pipeline": pipeline_name,
                                    "expected_consumers": len(models),
                                    "consumer_task_keys": [],
                                    "terminal_consumer_task_keys": [],
                                    "pending_or_retryable_consumer_task_keys": [],
                                    "build_count": 0,
                                    "hit_count": 0,
                                    "regeneration_count": 0,
                                    "active_reader_history": [],
                                    "deletion_time_utc": None,
                                    "deletion_observed": False,
                                })
                                audit_record = cache_audit_artifacts[cache_audit_key]
                                if task_key not in audit_record["consumer_task_keys"]:
                                    audit_record["consumer_task_keys"].append(task_key)

                                def cache_audit_event(event: str, details: dict[str, Any]) -> None:
                                    if event == "build":
                                        audit_record["build_count"] += 1
                                        audit_record["regeneration_count"] = max(0, audit_record["build_count"] - 1)
                                    elif event == "hit":
                                        audit_record["hit_count"] += 1
                                    elif event == "reader_acquired":
                                        audit_record["active_reader_history"].append({
                                            "task_key": task_key,
                                            "active_readers": int(details.get("active_readers", 0)),
                                            "observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        })
                            else:
                                audit_record = None
                                cache_audit_event = None
                            domain_auc_status = (
                                domain_metadata.get(family, {}).get("auc_status")
                                if family in domain_metadata else None
                            )
                            if family in domain_metadata and domain_auc_status != "supported":
                                skip_reason = domain_metadata[family]["auc_reason"]
                                skipped = dict(
                                    task_manifest,
                                    status="skipped",
                                    skip_reason=skip_reason,
                                    metric_scope="roc_auc",
                                )
                                record_task(
                                    db_path, run_id=run_id, task_key=task_key,
                                    phase="phase1", status="skipped", dataset=dataset_name,
                                    seed=seed, fold=fold, condition=condition,
                                    pipeline=pipeline_name, model=model_name,
                                    manifest_fingerprint=task_fp,
                                    error_summary=skip_reason,
                                )
                                record_task(
                                    db_path, run_id=run_id, task_key=task_key,
                                    phase="phase2", status="skipped", dataset=dataset_name,
                                    seed=seed, fold=fold, condition=condition,
                                    pipeline=pipeline_name, model=model_name,
                                    manifest_fingerprint=task_fp,
                                    error_summary=skip_reason,
                                )
                                _safe_result_write(run_dir / "results.jsonl", skipped, index_path=result_index_path)
                                _upsert_outcome(outcomes, skipped)
                                if audit_record is not None and task_key not in audit_record["terminal_consumer_task_keys"]:
                                    audit_record["terminal_consumer_task_keys"].append(task_key)
                                _save_run_manifest(run_dir, base_manifest, outcomes)
                                continue
                            scheduler_lease: Lease | None = None
                            if scheduler is not None:
                                scheduler.register_tasks([
                                    TaskSpec(
                                        task_key=task_key,
                                        run_id=run_id,
                                        payload=task_manifest,
                                        max_attempts=scheduler_max_attempts,
                                    )
                                ])
                                scheduler_state = scheduler.task_state(task_key)
                                if scheduler_state and scheduler_state.get("status") == "success":
                                    if has_success(db_path, run_id, task_key, "phase2", task_fp):
                                        _upsert_outcome(outcomes, dict(task_manifest, status="success", resumed=True))
                                        if audit_record is not None and task_key not in audit_record["terminal_consumer_task_keys"]:
                                            audit_record["terminal_consumer_task_keys"].append(task_key)
                                        _save_run_manifest(run_dir, base_manifest, outcomes)
                                        continue
                                    recovered = _recover_scheduler_payload(scheduler, task_key)
                                    if recovered is None:
                                        raise RuntimeError(
                                            f"Scheduler marks {task_key} successful but checkpoint is incomplete and its result artifact is unavailable"
                                        )
                                    # Repair the checkpoint after a crash in
                                    # the publication boundary.  The
                                    # scheduler result is already immutable
                                    # and checksum-validated, so no model is
                                    # refit or result row appended twice.
                                    record_task(
                                        db_path, run_id=run_id, task_key=task_key,
                                        phase="phase1", status="success", dataset=dataset_name,
                                        seed=seed, fold=fold, condition=condition,
                                        pipeline=pipeline_name, model=model_name,
                                        manifest_fingerprint=task_fp,
                                    )
                                    record_task(
                                        db_path, run_id=run_id, task_key=task_key,
                                        phase="phase2", status="success", dataset=dataset_name,
                                        seed=seed, fold=fold, condition=condition,
                                        pipeline=pipeline_name, model=model_name,
                                        manifest_fingerprint=task_fp,
                                    )
                                    _upsert_outcome(outcomes, dict(recovered, status="success", resumed=True))
                                    if audit_record is not None and task_key not in audit_record["terminal_consumer_task_keys"]:
                                        audit_record["terminal_consumer_task_keys"].append(task_key)
                                    _save_run_manifest(run_dir, base_manifest, outcomes)
                                    continue
                                if scheduler_state and scheduler_state.get("status") in {"failed", "timed_out"}:
                                    terminal_state = scheduler_state["status"]
                                    prior = next((row for row in outcomes if row.get("task_key") == task_key), None)
                                    terminal_row = dict(prior or task_manifest, status=terminal_state, resumed=True)
                                    _upsert_outcome(outcomes, terminal_row)
                                    if audit_record is not None and task_key not in audit_record["terminal_consumer_task_keys"]:
                                        audit_record["terminal_consumer_task_keys"].append(task_key)
                                    _save_run_manifest(run_dir, base_manifest, outcomes)
                                    continue
                            if (has_success(db_path, run_id, task_key, "phase2", task_fp)
                                    and has_success(db_path, run_id, task_key, "phase1", task_fp)):
                                _upsert_outcome(outcomes, dict(task_manifest, status="success", resumed=True))
                                if audit_record is not None and task_key not in audit_record["terminal_consumer_task_keys"]:
                                    audit_record["terminal_consumer_task_keys"].append(task_key)
                                _save_run_manifest(run_dir, base_manifest, outcomes)
                                continue
                            if scheduler is not None:
                                scheduler_lease = scheduler.claim_task(scheduler_worker_id, task_key=task_key)
                                if scheduler_lease is None:
                                    raise RuntimeError(f"Unable to claim durable scheduler task {task_key}")
                            try:
                                if scheduler is not None and scheduler_lease is not None:
                                    scheduler_lease = scheduler.heartbeat(scheduler_lease)
                                if failure_hook:
                                    failure_hook("phase1", task)
                                if cache_manager is None:
                                    cache_path, cache_manifest_path = _cache_paths(run_dir, feature_key, pipeline_name)
                                    cache_was_present = cache_path.exists() and cache_manifest_path.exists()
                                    payload = _load_or_create_feature_cache(
                                        run_dir, feature_key, pipeline_name, feature_manifest,
                                        lambda: _prepare_matrices(
                                            x_train, y_train, x_test,
                                            family=family, severity=severity,
                                            perturbation_seed=perturb_seed,
                                            pipeline_name=pipeline_name,
                                            config=PIPELINE_CONFIGS[pipeline_name],
                                        ),
                                    )
                                else:
                                    payload, cache_was_present = _load_or_create_bounded_feature_cache(
                                        cache_manager,
                                        feature_task_key=feature_key,
                                        pipeline_name=pipeline_name,
                                        expected_manifest=feature_manifest,
                                        make_payload=lambda: _prepare_matrices(
                                            x_train, y_train, x_test,
                                            family=family, severity=severity,
                                            perturbation_seed=perturb_seed,
                                            pipeline_name=pipeline_name,
                                            config=PIPELINE_CONFIGS[pipeline_name],
                                        ),
                                        audit_callback=cache_audit_event,
                                    )
                                (x_train_fe, x_test_fe, y_train_enc, encoder,
                                 fe_meta, x_test_clean) = payload
                                record_task(
                                    db_path, run_id=run_id, task_key=task_key,
                                    phase="phase1", status="success", dataset=dataset_name,
                                    seed=seed, fold=fold, condition=condition,
                                    pipeline=pipeline_name, model=model_name,
                                    manifest_fingerprint=task_fp,
                                )
                                if scheduler is not None and scheduler_lease is not None:
                                    scheduler_lease = scheduler.heartbeat(scheduler_lease)
                            except Exception as exc:
                                if scheduler is not None and scheduler_lease is not None:
                                    try:
                                        scheduler.record_failure(scheduler_lease, type(exc).__name__, str(exc))
                                    except LeaseLost:
                                        pass
                                failure = _record_failure(
                                    run_dir=run_dir, db_path=db_path, run_id=run_id,
                                    task_key=task_key, phase="phase1", dataset=dataset_name,
                                    seed=seed, fold=fold, condition=condition,
                                    pipeline=pipeline_name, model=model_name,
                                    fingerprint=task_fp, exc=exc, result_index_path=result_index_path,
                                )
                                _upsert_outcome(outcomes, dict(task_manifest, **failure))
                                _save_run_manifest(run_dir, base_manifest, outcomes)
                                continue
                            try:
                                if scheduler is not None and scheduler_lease is not None:
                                    scheduler_lease = scheduler.heartbeat(scheduler_lease)
                                if failure_hook:
                                    failure_hook("phase2", task)
                                xtr = np.nan_to_num(x_train_fe.to_numpy(dtype=np.float32), nan=0.0,
                                                    posinf=1e10, neginf=-1e10)
                                xte = np.nan_to_num(x_test_fe.to_numpy(dtype=np.float32), nan=0.0,
                                                    posinf=1e10, neginf=-1e10)
                                try:
                                    y_test_enc = encoder.transform(y_test.astype(str))
                                except ValueError as exc:
                                    raise ValueError("Held-out labels contain a class absent from training labels") from exc
                                phase2_meta = {
                                    "task_manifest": task_manifest, "task": task, "task_key": task_key,
                                    "task_fp": task_fp, "feature_cache_fp": feature_cache_fp,
                                    "scheduler_lease": scheduler_lease, "audit_record": audit_record,
                                    "dataset_name": dataset_name, "seed": seed, "fold": fold,
                                    "condition": condition, "pipeline_name": pipeline_name,
                                    "model_name": model_name, "task_split_status": task_split_status,
                                    "x_train_fe": x_train_fe, "x_test_fe": x_test_fe,
                                    "x_test_clean": x_test_clean, "fe_meta": fe_meta,
                                    "cache_was_present": cache_was_present,
                                    "n_train": len(xtr), "n_test": len(xte),
                                }
                                fit_payload = {
                                    "model": model_name, "seed": seed, "use_gpu": use_gpu,
                                    "xtr": xtr, "xte": xte, "y_train_enc": np.asarray(y_train_enc),
                                    "y_test_enc": np.asarray(y_test_enc), "classes": np.asarray(encoder.classes_),
                                }
                                if executor is not None:
                                    pending_futures.append((
                                        executor.submit(_fit_and_score_worker, fit_payload),
                                        phase2_meta,
                                    ))
                                    # Keep serialized matrices bounded in memory.
                                    if len(pending_futures) >= max(1, workers * 2):
                                        _drain_one()
                                else:
                                    _record_phase2(phase2_meta, _fit_and_score_worker(fit_payload))
                            except Exception as exc:
                                if scheduler is not None and scheduler_lease is not None:
                                    try:
                                        scheduler.record_failure(scheduler_lease, type(exc).__name__, str(exc))
                                    except LeaseLost:
                                        pass
                                failure = _record_failure(
                                    run_dir=run_dir, db_path=db_path, run_id=run_id, task_key=task_key,
                                    phase="phase2", dataset=dataset_name, seed=seed, fold=fold,
                                    condition=condition, pipeline=pipeline_name, model=model_name,
                                    fingerprint=task_fp, exc=exc, result_index_path=result_index_path,
                                )
                                _upsert_outcome(outcomes, dict(task_manifest, **failure))
                                if audit_record is not None and task_key not in audit_record["terminal_consumer_task_keys"]:
                                    audit_record["terminal_consumer_task_keys"].append(task_key)
                                _save_run_manifest(run_dir, base_manifest, outcomes)
                            if cache_manager is not None and model_position == len(models) - 1:
                                # All compatible classifiers for this
                                # feature-task have reached a terminal phase.
                                # Lease-aware cleanup can now reclaim the
                                # regenerated artifact without affecting
                                # resumption or any active reader.
                                terminal_statuses = {"success", "failed", "skipped", "timed_out"}
                                terminal = True
                                for consumer_key in feature_consumer_task_keys:
                                    if scheduler is not None:
                                        state = scheduler.task_state(consumer_key)
                                        consumer_status = state.get("status") if state else None
                                    else:
                                        row = next((item for item in outcomes if item.get("task_key") == consumer_key), None)
                                        consumer_status = row.get("status") if row else None
                                    if consumer_status not in terminal_statuses:
                                        terminal = False
                                        break
                                if terminal:
                                    cleanup_report = cache_manager.cleanup(max_age_seconds=0, dry_run=False)
                                    if cache_audit:
                                        deleted = set(cleanup_report.get("deleted_digests", []))
                                        for record in base_manifest["cache_audit"]["artifacts"].values():
                                            if CacheManager._digest(record["cache_key"]) in deleted:
                                                record["deletion_observed"] = True
                                                record["deletion_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                                elif cache_audit and audit_record is not None:
                                    audit_record["pending_or_retryable_consumer_task_keys"] = [
                                        key for key in feature_consumer_task_keys
                                        if key not in audit_record["terminal_consumer_task_keys"]
                                    ]
    # Finish all queued worker fits before final cache reconciliation.  This is
    # also the durable boundary for the bounded feature fan-out: no artifact
    # may be deleted while an outstanding classifier future can still read it.
    while pending_futures:
        _drain_one()
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=False)

    if cache_manager is not None:
        cache_reconciled = cache_manager.reconcile()
        scheduler_pending = False
        if scheduler is not None:
            with scheduler._connection() as connection:
                scheduler_pending = bool(connection.execute(
                    "SELECT 1 FROM scheduler_tasks WHERE status IN ('pending','running') LIMIT 1"
                ).fetchone())
        cache_cleanup = (
            {"deferred": True, "reason": "durable scheduler has pending or running consumers"}
            if scheduler_pending else cache_manager.cleanup(max_age_seconds=0, dry_run=False)
        )
        if cache_audit:
            deleted = set(cache_cleanup.get("deleted_digests", []))
            for record in base_manifest["cache_audit"]["artifacts"].values():
                if CacheManager._digest(record["cache_key"]) in deleted:
                    record["deletion_observed"] = True
                    record["deletion_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        base_manifest["cache_storage"] = {
            "policy": "bounded_regenerable",
            "root": str(cache_manager.root),
            "reconciled_before_final_cleanup": cache_reconciled,
            "final_cleanup": cache_cleanup,
            "high_water": cache_cleanup.get("high_water"),
            "durable_cache_policy": "feature artifacts may be regenerated from frozen identities after all current consumers terminate",
        }
    _save_run_manifest(run_dir, base_manifest, outcomes)
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a provenance-tracked corrected AutoFE-ShiftBench experiment")
    parser.add_argument("--run-id", default=time.strftime("corrected-%Y%m%dT%H%M%SZ", time.gmtime()))
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-root", default="corrected_runs")
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--max-conditions", type=int)
    parser.add_argument("--pipelines", nargs="+", default=["Raw", "AutoFE_Baseline"])
    parser.add_argument("--models", nargs="+", default=["logistic_regression"])
    parser.add_argument("--split-policy", choices=["row_level", "group_aware"], default="row_level")
    parser.add_argument("--cache-policy", choices=["retain", "bounded"], default="retain")
    parser.add_argument("--cache-max-gib", type=float)
    parser.add_argument("--durable-scheduler", action="store_true")
    parser.add_argument("--scheduler-lease-seconds", type=float, default=3600.0)
    parser.add_argument("--scheduler-max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="bounded outer worker processes for model fits")
    parser.add_argument(
        "--cache-audit", action="store_true",
        help="Persist per-feature cache build/hit/consumer/reader/deletion evidence (bounded runs only)",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA for XGBoost/CatBoost cells when the host/backend supports it")
    args = parser.parse_args(argv)

    names = load_dataset_names("config/dataset_list.yaml")
    if args.max_datasets:
        names = names[:args.max_datasets]
    data_paths = {name: Path(args.data_dir) / f"{name}.csv" for name in names}
    missing = [str(path) for path in data_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Configured dataset CSVs are missing; no partial run started: {missing}")
    conditions = PRIMARY_CONDITIONS[:args.max_conditions] if args.max_conditions else PRIMARY_CONDITIONS
    manifest = run_experiment(
        data_paths, run_id=args.run_id, output_root=args.output_root,
        seeds=[42, 123, 456, 789, 2025][:args.max_seeds] if args.max_seeds else None,
        folds=list(range(1, args.max_folds + 1)) if args.max_folds else None,
        conditions=conditions, pipelines=tuple(args.pipelines), models=tuple(args.models),
        split_policy=args.split_policy,
        cache_policy=args.cache_policy,
        cache_max_bytes=(int(args.cache_max_gib * 1024**3) if args.cache_max_gib is not None else None),
        durable_scheduler=args.durable_scheduler,
        scheduler_lease_seconds=args.scheduler_lease_seconds,
        scheduler_max_attempts=args.scheduler_max_attempts,
        cache_audit=args.cache_audit,
        use_gpu=args.use_gpu,
        workers=args.workers,
    )
    print(json.dumps({"run_id": manifest["run_id"], "status": manifest["status"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
