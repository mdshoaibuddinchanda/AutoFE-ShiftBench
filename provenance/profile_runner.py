"""Bounded, read-only performance profiling for the corrected benchmark.

This module deliberately does not call ``run_experiment`` and does not edit
the benchmark runner.  It profiles one fold of the named local datasets, a
complete pipeline/model task mix on one small dataset, and a temporary feature
cache.  It is intended for the existing Conda ``p12`` environment.

Example::

    set OMP_NUM_THREADS=1
    set MKL_NUM_THREADS=1
    set OPENBLAS_NUM_THREADS=1
    set NUMEXPR_NUM_THREADS=1
    python -m provenance.profile_runner

The command prints one JSON document to stdout.  No profile artifacts are
written unless the caller redirects stdout explicitly.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import load_csv_dataset
from src.evaluation import compute_classification_metrics
from src.model import build_model
from src.pipeline_runner import (
    PIPELINE_CONFIGS,
    _load_or_create_feature_cache,
    _prepare_matrices,
    _resolve_target_column,
)
from src.preprocessing import _to_dense_array
from src.provenance import (
    PROTOCOL_VERSION,
    code_fingerprint,
    current_git_commit,
    file_sha256,
    frame_sha256,
    index_sha256,
    stable_digest,
    stable_seed,
    vector_sha256,
)
from src.splitters import get_stratified_splits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42
DEFAULT_FOLD = 1
DEFAULT_SPLITS = 5
DEFAULT_CONDITION = "clean"
DEFAULT_PIPELINES = (
    "Raw",
    "Raw_Variance",
    "Raw_MI",
    "Raw_CapMatched",
    "AutoFE_Baseline",
    "AutoFE_MI",
    "AutoFE_Random",
    "AutoFE_NoMultiply",
)
DEFAULT_MODELS = (
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "linear_svm",
    "knn",
    "gaussian_nb",
    "mlp",
    "lightgbm",
    "xgboost",
    "catboost",
)


def _now() -> float:
    return time.perf_counter()


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _safe(value: Any) -> Any:
    """Convert numpy values and non-finite floats to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(child) for child in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _package_versions() -> dict[str, str]:
    names = (
        "numpy", "pandas", "scikit-learn", "scipy", "featuretools", "woodwork",
        "xgboost", "lightgbm", "catboost", "psutil", "threadpoolctl",
    )
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def _thread_inventory() -> dict[str, Any]:
    names = (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    )
    environment = {name: os.environ.get(name) for name in names}
    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
    except Exception as exc:
        pools = {"error": f"{type(exc).__name__}: {exc}"}
    return {"environment": environment, "threadpoolctl": pools}


def _gpu_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False,
        )
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError:
        return {"available": False, "reason": "nvidia-smi_not_found"}
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "nvidia-smi_timeout"}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _cpu_topology_inventory() -> dict[str, Any]:
    """Best-effort Windows hybrid-core inventory; unknown fields stay explicit."""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,ThreadCount | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        raw = completed.stdout.strip()
    except Exception as exc:
        return {"status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return {
        "status": "unknown",
        "p_cores": None,
        "e_cores": None,
        "logical_processors": os.cpu_count(),
        "windows_processor_record": value,
        "reason": "Windows WMI record does not expose reliable heterogeneous-core counts",
    }


def environment_inventory() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {
        "python": __import__("platform").python_version(),
        "platform": __import__("platform").platform(),
        "machine": __import__("platform").machine(),
        "processor": __import__("platform").processor(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_topology": _cpu_topology_inventory(),
        "packages": _package_versions(),
        "threads": _thread_inventory(),
        "gpu": _gpu_inventory(),
        "disk": {
            "root": str(ROOT),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "used_bytes": usage.used,
        },
        "git_commit": current_git_commit(ROOT),
        "code_fingerprint": code_fingerprint(ROOT),
    }


def _dataset_context(dataset_name: str, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    path = ROOT / "data" / "raw" / f"{dataset_name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = load_csv_dataset(path)
    sidecar_path = path.with_name(f"{path.stem}_meta.json")
    sidecar: dict[str, Any] = {}
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    target_column = _resolve_target_column(frame, sidecar.get("target_column"))
    X = frame.drop(columns=[target_column])
    y = frame[target_column]
    splits = get_stratified_splits(X, y, DEFAULT_SPLITS, seed, target_column=target_column)
    train_idx, test_idx = splits[DEFAULT_FOLD - 1]
    return {
        "name": dataset_name,
        "path": str(path),
        "frame": frame,
        "target_column": target_column,
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
        "source_sha256": file_sha256(path),
        "frame_shape": [int(frame.shape[0]), int(frame.shape[1])],
    }


def _task_payload(
    context: dict[str, Any], pipeline: str, model: str,
    *, condition: str = DEFAULT_CONDITION,
    severity: float = 0.0,
    use_gpu: bool = False,
) -> dict[str, Any]:
    return {
        "dataset": context["name"],
        "frame": context["frame"],
        "target_column": context["target_column"],
        "train_idx": context["train_idx"],
        "test_idx": context["test_idx"],
        "pipeline": pipeline,
        "model": model,
        "condition": condition,
        "severity": severity,
        "seed": DEFAULT_SEED,
        "use_gpu": bool(use_gpu),
    }


def _execute_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one bounded task and return timing, memory, and parity hashes."""
    started = _now()
    rss_before = _rss_mb()
    try:
        prep_started = _now()
        x_train_fe, x_test_fe, y_train_enc, encoder, fe_meta, _ = _prepare_matrices(
            payload["frame"].drop(columns=[payload["target_column"]]).iloc[payload["train_idx"]].copy(),
            payload["frame"][payload["target_column"]].iloc[payload["train_idx"]].copy(),
            payload["frame"].drop(columns=[payload["target_column"]]).iloc[payload["test_idx"]].copy(),
            family=payload["condition"], severity=payload["severity"],
            perturbation_seed=stable_seed(
                {"dataset": payload["dataset"]}, payload["seed"], DEFAULT_FOLD, payload["condition"],
            ),
            pipeline_name=payload["pipeline"],
            config=PIPELINE_CONFIGS[payload["pipeline"]],
        )
        preparation_s = _now() - prep_started
        xtr = np.nan_to_num(x_train_fe.to_numpy(dtype=np.float32), nan=0.0, posinf=1e10, neginf=-1e10)
        xte = np.nan_to_num(x_test_fe.to_numpy(dtype=np.float32), nan=0.0, posinf=1e10, neginf=-1e10)
        model = build_model(payload["model"], random_state=payload["seed"], use_gpu=payload.get("use_gpu", False))
        fit_started = _now()
        model.fit(xtr, y_train_enc)
        fit_s = _now() - fit_started
        predict_started = _now()
        y_pred = model.predict(xte)
        y_proba = model.predict_proba(xte) if hasattr(model, "predict_proba") else np.zeros(
            (len(xte), len(encoder.classes_)), dtype=float,
        )
        predict_s = _now() - predict_started
        y_test_enc = encoder.transform(payload["frame"][payload["target_column"]].iloc[payload["test_idx"]].astype(str))
        metrics = compute_classification_metrics(y_test_enc, y_pred, y_proba, encoder.classes_)
        return _safe({
            "status": "success",
            "dataset": payload["dataset"], "pipeline": payload["pipeline"], "model": payload["model"],
            "condition": payload["condition"], "seed": payload["seed"], "fold": DEFAULT_FOLD,
            "preparation_s": preparation_s, "fit_s": fit_s, "predict_s": predict_s,
            "total_s": _now() - started, "rss_before_mb": rss_before, "rss_after_mb": _rss_mb(),
            "n_train": len(xtr), "n_test": len(xte),
            "n_features_train": xtr.shape[1], "n_features_test": xte.shape[1],
            "n_generated": fe_meta.get("n_generated", 0), "n_retained": fe_meta.get("n_retained", xtr.shape[1]),
            "train_matrix_sha256": frame_sha256(x_train_fe),
            "test_matrix_sha256": frame_sha256(x_test_fe),
            "prediction_sha256": stable_digest(np.asarray(y_pred).tolist()),
            "roc_auc": metrics.get("roc_auc"), "f1": metrics.get("f1"),
        })
    except Exception as exc:
        return _safe({
            "status": "failed", "dataset": payload["dataset"], "pipeline": payload["pipeline"],
            "model": payload["model"], "condition": payload["condition"], "seed": payload["seed"],
            "fold": DEFAULT_FOLD, "total_s": _now() - started, "rss_before_mb": rss_before,
            "rss_after_mb": _rss_mb(), "exception_type": type(exc).__name__,
            "error": " ".join(str(exc).split())[:1000],
        })


def stage_profile(dataset_name: str, pipelines: tuple[str, ...]) -> dict[str, Any]:
    context_started = _now()
    context = _dataset_context(dataset_name)
    load_and_split_s = _now() - context_started
    rows: list[dict[str, Any]] = []
    for pipeline in pipelines:
        result = _execute_task(_task_payload(context, pipeline, "logistic_regression"))
        result["load_and_split_s"] = load_and_split_s
        result["source_sha256"] = context["source_sha256"]
        result["frame_shape"] = context["frame_shape"]
        rows.append(result)
    return {
        "dataset": dataset_name,
        "source_sha256": context["source_sha256"],
        "frame_shape": context["frame_shape"],
        "target_column": context["target_column"],
        "seed": DEFAULT_SEED, "fold": DEFAULT_FOLD, "n_splits": DEFAULT_SPLITS,
        "rows": rows,
    }


def _run_pool(payloads: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        return list(executor.map(_execute_task, payloads))


def worker_benchmark(dataset_name: str, workers: tuple[int, ...], *, use_gpu: bool = False) -> dict[str, Any]:
    context = _dataset_context(dataset_name)
    pipelines = tuple(name for name in DEFAULT_PIPELINES if name in PIPELINE_CONFIGS)
    models = tuple(name for name in DEFAULT_MODELS)
    payloads = [_task_payload(context, pipeline, model, use_gpu=use_gpu) for pipeline in pipelines for model in models]
    runs: list[dict[str, Any]] = []
    digest_by_workers: dict[str, dict[str, str]] = {}
    for worker_count in workers:
        started = _now()
        results = _run_pool(payloads, worker_count)
        elapsed = _now() - started
        success = [row for row in results if row.get("status") == "success"]
        failures = [row for row in results if row.get("status") != "success"]
        digest_by_workers[str(worker_count)] = {
            f"{row['pipeline']}::{row['model']}": str(row.get("prediction_sha256"))
            for row in success
        }
        runs.append(_safe({
            "workers": worker_count,
            "elapsed_s": elapsed,
            "tasks_expected": len(payloads),
            "tasks_success": len(success),
            "tasks_failed": len(failures),
            "throughput_tasks_per_s": len(payloads) / elapsed if elapsed else None,
            "median_task_total_s": float(np.median([row["total_s"] for row in success])) if success else None,
            "p95_task_total_s": float(np.percentile([row["total_s"] for row in success], 95)) if success else None,
            "max_rss_after_mb": max((row.get("rss_after_mb") or 0.0) for row in results),
            "failure_types": {
                str(key): int(value) for key, value in pd.Series(
                    [row.get("exception_type", "unknown") for row in failures], dtype="string",
                ).value_counts().items()
            },
        }))
    parity: dict[str, Any] = {}
    if "1" in digest_by_workers and "4" in digest_by_workers:
        one = digest_by_workers["1"]
        four = digest_by_workers["4"]
        common = sorted(set(one).intersection(four))
        parity = {
            "compared_tasks": len(common),
            "hash_mismatches": [key for key in common if one[key] != four[key]],
            "missing_from_1": sorted(set(four).difference(one)),
            "missing_from_4": sorted(set(one).difference(four)),
        }
    return {
        "dataset": dataset_name,
        "seed": DEFAULT_SEED, "fold": DEFAULT_FOLD, "condition": DEFAULT_CONDITION,
        "pipelines": list(pipelines), "models": list(models),
        "tasks_expected": len(payloads), "runs": runs, "numerical_parity_1_vs_4": parity,
    }


def cache_profile(dataset_name: str, pipeline: str) -> dict[str, Any]:
    context = _dataset_context(dataset_name)
    result: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="autofe_profile_cache_") as temporary:
        run_dir = Path(temporary) / "run"
        manifest = {
            "dataset": dataset_name, "pipeline": pipeline, "condition": DEFAULT_CONDITION,
            "dataset_checksum": stable_digest({"source_sha256": context["source_sha256"]}),
            "train_indices_sha256": index_sha256(context["train_idx"]),
            "test_indices_sha256": index_sha256(context["test_idx"]),
            "stable_perturbation_seed": DEFAULT_SEED,
            "configuration_fingerprint": stable_digest({"pipeline": pipeline}),
            "code_fingerprint": code_fingerprint(ROOT),
            "runtime_fingerprint": stable_digest(_package_versions()),
            "protocol_version": PROTOCOL_VERSION,
            "source_csv_sha256": context["source_sha256"],
            "dataset_identity": {"dataset": dataset_name},
        }
        payload = _task_payload(context, pipeline, "logistic_regression")
        def make_payload():
            return _prepare_matrices(
                context["frame"].drop(columns=[context["target_column"]]).iloc[context["train_idx"]].copy(),
                context["frame"][context["target_column"]].iloc[context["train_idx"]].copy(),
                context["frame"].drop(columns=[context["target_column"]]).iloc[context["test_idx"]].copy(),
                family=DEFAULT_CONDITION, severity=0.0, perturbation_seed=DEFAULT_SEED,
                pipeline_name=pipeline, config=PIPELINE_CONFIGS[pipeline],
            )
        first_started = _now()
        _load_or_create_feature_cache(run_dir, "profile-task", pipeline, manifest, make_payload)
        first_s = _now() - first_started
        cache_files = list((run_dir / "cache").glob("*"))
        bytes_after_first = sum(path.stat().st_size for path in cache_files if path.is_file())
        second_started = _now()
        _load_or_create_feature_cache(run_dir, "profile-task", pipeline, manifest, make_payload)
        second_s = _now() - second_started
        bytes_after_second = sum(path.stat().st_size for path in cache_files if path.is_file())
        result = _safe({
            "dataset": dataset_name, "pipeline": pipeline,
            "first_call_s": first_s, "cache_hit_call_s": second_s,
            "cache_bytes_first": bytes_after_first, "cache_bytes_second": bytes_after_second,
            "cache_speedup": first_s / second_s if second_s else None,
            "cache_file_count": len(cache_files),
            "temporary_cache_removed": False,
        })
    result["temporary_cache_removed"] = not Path(temporary).exists()
    return result


def parity_check(dataset_name: str) -> dict[str, Any]:
    context = _dataset_context(dataset_name)
    payload = _task_payload(context, "Raw", "logistic_regression")
    first = _execute_task(payload)
    second = _execute_task(payload)
    worker = _run_pool([payload], 1)[0]
    fields = ("train_matrix_sha256", "test_matrix_sha256", "prediction_sha256")
    return {
        "dataset": dataset_name,
        "direct_repeat_equal": all(first.get(field) == second.get(field) for field in fields),
        "direct_vs_worker_equal": all(first.get(field) == worker.get(field) for field in fields),
        "compared_fields": list(fields),
        "direct_first": {field: first.get(field) for field in fields},
        "direct_second": {field: second.get(field) for field in fields},
        "worker": {field: worker.get(field) for field in fields},
        "metric_deltas": {
            field: (first.get(field) - second.get(field))
            if isinstance(first.get(field), (int, float)) and isinstance(second.get(field), (int, float))
            else None
            for field in ("roc_auc", "f1")
        },
    }


def run_profile(
    stage_datasets: tuple[str, ...] = ("sonar", "airlines"),
    mix_dataset: str = "sonar",
    workers: tuple[int, ...] = (1, 2, 3, 4),
    use_gpu: bool = False,
) -> dict[str, Any]:
    return {
        "artifact_type": "bounded_performance_profile",
        "profile_protocol": "one seed=42, fold=1 of 5, clean condition; complete 8-pipeline x 10-model mix on one dataset",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_inventory(),
        "stage_profiles": [stage_profile(name, ("Raw", "AutoFE_Baseline")) for name in stage_datasets],
        "worker_benchmark": worker_benchmark(mix_dataset, workers, use_gpu=use_gpu),
        "gpu_mode": bool(use_gpu),
        "scheduling_modes": [
            {"mode": "windows_default", "status": "measured", "worker_counts": list(workers)},
            {"mode": "all_available", "status": "measured_only_for_requested_workers", "worker_counts": list(workers)},
            {"mode": "p_core_preferred", "status": "pending_friend_pc", "reason": "No affinity applied until hybrid topology is measured"},
            {"mode": "os_headroom", "status": "pending_friend_pc", "reason": "Worker/RAM reservation must be selected from host probe"},
        ],
        "cache_profile": cache_profile("sonar", "AutoFE_Baseline"),
        "numerical_parity": parity_check("sonar"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-datasets", nargs="+", default=["sonar", "airlines"])
    parser.add_argument("--mix-dataset", default="sonar")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--use-gpu", action="store_true", help="Enable CUDA routing for XGBoost/CatBoost during this bounded profile")
    args = parser.parse_args(argv)
    if any(worker < 1 or worker > (os.cpu_count() or 1) for worker in args.workers):
        raise SystemExit(f"--workers values must be between 1 and {os.cpu_count() or 1} for bounded profiling")
    result = run_profile(tuple(args.stage_datasets), args.mix_dataset, tuple(args.workers), args.use_gpu)
    print(json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
