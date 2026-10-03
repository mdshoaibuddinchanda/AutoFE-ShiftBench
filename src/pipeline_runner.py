"""Benchmark orchestrator with human-readable caching and full parallelization.

Cache Layout (human-readable):
    data/cache/{protocol}/{seed_scheme}/{dataset}/splits_s{seed}_{split_policy}.pkl
    data/cache/{protocol}/{seed_scheme}/{dataset}/{pipeline}_s{seed}_f{fold}_{condition}_*.{pkl,json}

Progress Tracking:
    Each dataset gets a subdirectory under the versioned data/cache namespace.
    To check progress:  python -m src.check_progress
    Or simply:          dir /b data\\cache\\<dataset>\\*_train.pkl | find /c /v ""
"""

import argparse
import gc
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

# ---------------------------------------------------------------------------
# Environment setup (inherited by child processes on Windows via spawn)
# ---------------------------------------------------------------------------
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() - 1)))
warnings.filterwarnings("ignore", category=UserWarning, module="woodwork")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=FutureWarning)

from src.checkpoint import init_db, has_run, log_run
from src.artifact_integrity import (CACHE_SCHEMA_VERSION, PREPROCESSING_SEMANTICS_VERSION,
    array_identity, artifact_lock, atomic_json, atomic_pickle, dataset_identity,
    file_sha256, fingerprint, frame_identity, validate_feature_cache)
from src.data_loader import load_csv_dataset, load_dataset_names
from src.evaluation import (
    compute_classification_metrics,
    compute_distribution_distance,
    compute_jaccard_similarity,
)
from src.feature_engineering import (
    BASE_FEATURE_POLICY_VERSION,
    CAP_POLICY_VERSION,
    expand_features_with_dfs,
    DFSConfig,
)
from src.fsva import (
    DEFAULT_PERTURBATION_MAGNITUDES,
    FSVA_SCHEMA_VERSION,
    compute_empirical_amplification,
    compute_jacobian_diagnostic,
    validate_jacobian_finite_difference,
)
from src.operator_registry import OPERATOR_REGISTRY_VERSION, OPERATOR_SEMANTICS_VERSION, raw_expression
from src.feature_selection import FeatureSelectionConfig, select_top_features
from src.model import build_model
from src.preprocessing import _build_preprocessor, _to_dense_array
from src.protocol import EVALUATION_PROTOCOL_VERSION, cache_root, results_ledger_path
from src.provenance import collect_code_identity
from src.seeding import (
    SEED_SCHEME_VERSION,
    corruption_seed,
    distance_sample_seed,
    estimator_seed,
    feature_selection_seed,
    split_seed,
    stable_seed,
)
from src.shift_generator import apply_perturbation
from src.splitters import SplitInfeasibleError, assert_fold_integrity, get_splits
from src.shap_explainer import compute_shap_values
from src.task_manifest import (
    ExecutionConfig,
    ManifestError,
    ManifestConflictError,
    ManifestStore,
    build_task_records,
    run_id_for,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CPU_WORKERS = max(1, os.cpu_count() - 1)
N_GPU_WORKERS = 1  # XGBoost + CatBoost share VRAM; sequential is correct
RAM_LIMIT_PERCENT = 85  # Pause spawning if RAM exceeds this %
MAX_TASKS_PER_CHILD = 50  # Reduce process respawn overhead

PIPELINE_CONFIGS = {
    # Historical Raw retained its old 20-column training variance pre-cap.
    "Raw": DFSConfig(enable_dfs=False, selection_method="none", max_features=None,
                     max_base_features=20, operator_set_id="none_v1",
                     baseline_kind="historical_raw", display_identity="Raw (historical 20-column control)"),
    "Raw_Full": DFSConfig(enable_dfs=False, selection_method="none", max_features=None,
                           max_base_features=None, operator_set_id="none_v1",
                           baseline_kind="full_dimensional_raw", display_identity="Raw_Full (all eligible base features)"),
    "Raw_Capped": DFSConfig(enable_dfs=False, selection_method="variance", max_features=100,
                             max_base_features=None, operator_set_id="none_v1",
                             baseline_kind="cap_matched_raw", display_identity="Raw_Capped (post-candidate cap matched)"),
    "Raw_Variance": DFSConfig(enable_dfs=False, selection_method="variance", max_features=100,
                               max_base_features=20, operator_set_id="none_v1",
                               baseline_kind="historical_raw_variant", display_identity="Raw_Variance"),
    "Raw_MI": DFSConfig(enable_dfs=False, selection_method="mi", max_features=100,
                         max_base_features=20, operator_set_id="none_v1",
                         baseline_kind="historical_raw_variant", display_identity="Raw_MI"),
    "AutoFE_Baseline": DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1,
                                  operator_set_id="full_arithmetic_v1", display_identity="AutoFE_Baseline (full arithmetic)"),
    "AutoFE_MI": DFSConfig(enable_dfs=True, selection_method="mi", max_features=100, depth=1,
                            operator_set_id="full_arithmetic_v1", display_identity="AutoFE_MI (full arithmetic)"),
    "AutoFE_Random": DFSConfig(enable_dfs=True, selection_method="random", max_features=100, depth=1,
                                operator_set_id="full_arithmetic_v1", display_identity="AutoFE_Random (full arithmetic)"),
    # Historical name retained as an explicit compatibility alias for {add, sub}.
    "AutoFE_NoMultiply": DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1,
                                    operator_set_id="add_sub_v1", display_identity="AutoFE_NoMultiply (legacy add_sub alias)"),
    "AutoFE_AddSub": DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1,
                                operator_set_id="add_sub_v1", display_identity="AutoFE_AddSub ({add, sub})"),
    "AutoFE_AddSubDiv": DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1,
                                   operator_set_id="add_sub_div_v1", display_identity="AutoFE_AddSubDiv ({add, sub, div})"),
    "AutoFE_NoDivision": DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1,
                                    operator_set_id="add_sub_mul_v1", display_identity="AutoFE_NoDivision ({add, sub, mul})"),
    "AutoFE_MultiplyOnly": DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1,
                                      operator_set_id="multiply_only_v1", display_identity="AutoFE_MultiplyOnly ({mul})"),
    "AutoFE_DivideOnly": DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1,
                                    operator_set_id="divide_only_v1", display_identity="AutoFE_DivideOnly ({div})"),
}

PIPELINE_NAMES = list(PIPELINE_CONFIGS.keys())

CPU_MODELS = [
    "logistic_regression", "random_forest", "extra_trees",
    "linear_svm", "knn", "gaussian_nb", "mlp", "lightgbm",
]
GPU_MODELS = ["xgboost", "catboost"]

SHIFT_FAMILIES = [
    ("clean", 0.0),
    ("gaussian_noise", 0.01), ("gaussian_noise", 0.05), ("gaussian_noise", 0.10),
    ("missing_values", 0.05), ("missing_values", 0.10), ("missing_values", 0.20),
    ("label_noise", 0.05), ("label_noise", 0.10), ("label_noise", 0.20),
    ("covariate_shift", 0.0),
    ("feature_removal", 0.20),
    ("population_shift", 0.0),
    ("class_prior_shift", 0.0),
]

_writer_queue = None
_stop_requested = False


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and nonfinite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_file: str | Path) -> logging.Logger:
    logger = logging.getLogger("AutoFE_Benchmark")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Human-readable cache helpers
# ---------------------------------------------------------------------------

def _cache_dir_for_dataset(dataset_name: str) -> Path:
    """Return the current protocol's dataset cache directory."""
    d = cache_root() / dataset_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _split_cache_path(dataset_name: str, seed: int, split_policy: str, dependency: str = "") -> Path:
    """Return a split cache path keyed by dataset, replicate, and split policy."""
    suffix = "__"+dependency if dependency else ""
    return _cache_dir_for_dataset(dataset_name) / f"splits_s{seed}_{split_policy}{suffix}.pkl"


def split_predictors_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    """Separate one declared target column from the predictor matrix."""
    target_col = "target_label" if "target_label" in frame.columns else "target"
    if target_col not in frame.columns:
        raise KeyError("Dataset must contain either 'target_label' or 'target'")
    y = frame[target_col].copy()
    X = frame.drop(columns=[target_col]).copy()
    if target_col in X.columns:
        raise AssertionError("The target column entered the predictor frame")
    return X, y, target_col


def split_policy_for_condition(shift_family: str) -> str:
    """Resolve the partition policy from a benchmark condition family."""
    if shift_family in {"covariate_shift", "population_shift"}:
        return shift_family
    return "stratified"


def _pipeline_cache_paths(dataset_name: str, pipeline_name: str,
                          seed: int, fold: int, condition: str, dependency: str = ""):
    """Return (train_pkl, test_pkl, meta_json) with human-readable names.

    Example:
        data/cache/{protocol}/{seed_scheme}/adult/AutoFE_MI_s42_f1_gaussian_noise_0.05_train.pkl
        data/cache/{protocol}/{seed_scheme}/adult/AutoFE_MI_s42_f1_gaussian_noise_0.05_test.pkl
        data/cache/{protocol}/{seed_scheme}/adult/AutoFE_MI_s42_f1_gaussian_noise_0.05_meta.json
    """
    d = _cache_dir_for_dataset(dataset_name)
    if dependency:
        d = d / dependency
        d.mkdir(parents=True,exist_ok=True)
    cfg = PIPELINE_CONFIGS[pipeline_name]
    token = pipeline_identity_token(pipeline_name)
    base = f"{token}_s{seed}_f{fold}_{condition}"
    return (
        d / f"{base}_train.pkl",
        d / f"{base}_test.pkl",
        d / f"{base}_meta.json",
    )


def pipeline_identity_token(pipeline_name: str) -> str:
    """Return a collision-resistant cache/checkpoint identity for a pipeline spec."""
    cfg = PIPELINE_CONFIGS[pipeline_name]
    return "__".join(
        [
            pipeline_name,
            OPERATOR_REGISTRY_VERSION,
            OPERATOR_SEMANTICS_VERSION,
            cfg.operator_set_id,
            cfg.baseline_kind or "autofe",
            CAP_POLICY_VERSION,
            BASE_FEATURE_POLICY_VERSION,
            f"cap{cfg.max_features if cfg.max_features is not None else 'all'}",
            f"base{cfg.max_base_features if cfg.max_base_features is not None else 'all'}",
            f"sel{cfg.selection_method}",
            f"depth{cfg.depth}",
        ]
    ).replace(" ", "_")


def _diagnostic_paths(dataset_name: str, pipeline_name: str, seed: int, fold: int, condition: str, dependency: str = ""):
    train_cache, test_cache, meta_cache = _pipeline_cache_paths(
        dataset_name, pipeline_name, seed, fold, condition, dependency,
    )
    stem = meta_cache.with_suffix("")
    return train_cache, test_cache, meta_cache, stem.with_name(stem.name + "_history.jsonl"), stem.with_name(stem.name + "_fsva.json")


# ---------------------------------------------------------------------------
# Pipeline generation (Phase 1 core)
# ---------------------------------------------------------------------------

def _run_pipeline_generation(x_train, x_test, y_train,
                             dataset_name, split_policy, seed, fold, condition,
                             diagnostics_enabled: bool = False,
                             diagnostic_config: dict[str, Any] | None = None,
                             data_identity: dict[str, Any] | None = None):
    """Generate all configured raw controls and arithmetic variants for one unit.

    Returns:
        res_pipelines: dict[str, (DataFrame, DataFrame)]
        res_meta: dict[str, dict]
    """
    res_pipelines = {}
    res_meta = {}

    for p_name, cfg in PIPELINE_CONFIGS.items():
        # Set seed on configs that need it
        selection_random_state = feature_selection_seed(
            dataset_name, split_policy, seed, fold, condition, p_name,
        )
        cfg_copy = DFSConfig(
            enable_dfs=cfg.enable_dfs,
            depth=cfg.depth,
            max_features=cfg.max_features,
            max_base_features=cfg.max_base_features,
            selection_method=cfg.selection_method,
            trans_primitives=list(cfg.trans_primitives),
            monitor_ram=cfg.monitor_ram,
            random_seed=selection_random_state,
            operator_set_id=cfg.operator_set_id,
            baseline_kind=cfg.baseline_kind,
            display_identity=cfg.display_identity,
        )

        dependency = fingerprint({"train": frame_identity(x_train), "test": frame_identity(x_test),
            "labels": array_identity(y_train), "data_identity": data_identity,
            "pipeline_spec": asdict(cfg_copy), "protocol": EVALUATION_PROTOCOL_VERSION,
            "seed_scheme": SEED_SCHEME_VERSION, "preprocessing": PREPROCESSING_SEMANTICS_VERSION})
        train_cache, test_cache, meta_cache, history_path, fsva_path = _diagnostic_paths(
            dataset_name, p_name, seed, fold, condition, dependency
        )

        verified_cache = validate_feature_cache(train_cache,test_cache,meta_cache,dependency)
        cache_compatible = verified_cache is not None
        if diagnostics_enabled:
            cache_compatible = cache_compatible and history_path.exists() and fsva_path.exists()
        if cache_compatible:
            x_train_fe,x_test_fe,meta = verified_cache
            if diagnostics_enabled:
                with history_path.open(encoding="utf-8") as history_file:
                    meta["selection_history"] = [json.loads(line) for line in history_file if line.strip()]
            meta["dfs_cache_hit"] = True
        else:
            t0 = time.time()
            x_train_fe, x_test_fe, dfs_meta = expand_features_with_dfs(
                x_train, x_test, y_train, config=cfg_copy,
            )
            gen_time = time.time() - t0

            meta = {
                **dfs_meta,
                "num_original": x_train.shape[1],
                "num_generated": dfs_meta.get("n_generated", 0),
                "num_selected": dfs_meta.get("n_retained", x_train_fe.shape[1]),
                "generation_time_s": gen_time,
                "ram_used_mb": dfs_meta.get("ram_used_mb", 0),
                "feature_metadata": dfs_meta.get("feature_metadata", []),
                "dfs_cache_hit": False,
            }
            with artifact_lock(meta_cache.with_suffix(".lock")):
                atomic_pickle(train_cache,x_train_fe)
                atomic_pickle(test_cache,x_test_fe)
                meta.update({"cache_schema": CACHE_SCHEMA_VERSION, "dependency_signature": dependency,
                    "data_identity": data_identity, "preprocessing_semantics": PREPROCESSING_SEMANTICS_VERSION,
                    "artifacts": {role: {"path": str(path), "sha256": file_sha256(path), "identity": frame_identity(frame)}
                        for role,path,frame in (("train",train_cache,x_train_fe),("test",test_cache,x_test_fe))}})
                atomic_json(meta_cache,{key:value for key,value in meta.items() if key != "selection_history"})

        meta["pipeline_identity"] = pipeline_identity_token(p_name)
        meta["operator_registry_version"] = OPERATOR_REGISTRY_VERSION
        meta["operator_semantics_version"] = OPERATOR_SEMANTICS_VERSION
        meta["operator_set_id"] = cfg.operator_set_id
        meta["operator_set"] = list(cfg.trans_primitives)
        meta["cap_policy_version"] = meta.get("cap_policy_version", CAP_POLICY_VERSION if cfg.max_features is not None else "none_v1")
        meta["selection_seed"] = selection_random_state

        history = meta.get("selection_history", [])
        if diagnostics_enabled and not history:
            # Reconstruct candidate metadata from the frozen selected matrices only
            # is intentionally disallowed; a diagnostic-enabled cache must have
            # been generated with real candidate events.
            raise RuntimeError("Diagnostic history is missing from the generated candidate event stream")
        if diagnostics_enabled:
            task_context = {
                "dataset": dataset_name,
                "split_policy": split_policy,
                "seed": seed,
                "fold": fold,
                "condition": condition,
                "pipeline": p_name,
                "pipeline_identity": pipeline_identity_token(p_name),
                "operator_set_id": cfg.operator_set_id,
                "cap_policy_version": meta.get("cap_policy_version"),
                "requested_cap": meta.get("requested_cap"),
                "requested_base_cap": meta.get("requested_base_cap"),
                "candidate_count": meta.get("candidate_count"),
                "selection_stage": meta.get("selector_identity"),
            }
            with history_path.open("w", encoding="utf-8") as history_file:
                for event in history:
                    history_file.write(json.dumps({**task_context, **event}, sort_keys=True) + "\n")
            diagnostic_cfg = diagnostic_config or {}
            max_rows = int(diagnostic_cfg.get("max_rows", 128))
            diag_seed = int(diagnostic_cfg.get("random_state", seed))
            selected_expressions = [item["expression"] for item in meta.get("selected_feature_expressions", [])]
            # Expression dictionaries are converted by the helper below.
            from src.fsva import expression_from_dict
            selected_exprs = [expression_from_dict(item) for item in selected_expressions]
            raw_exprs = [raw_expression(column) for column in x_train.columns]
            settings = {
                "max_rows": max_rows,
                "random_state": diag_seed,
                "magnitudes": list(diagnostic_cfg.get("magnitudes", DEFAULT_PERTURBATION_MAGNITUDES)),
            }
            try:
                jac = compute_jacobian_diagnostic(
                    x_train, selected_exprs, raw_control_expressions=raw_exprs,
                    max_rows=max_rows, random_state=diag_seed,
                )
                amp = compute_empirical_amplification(
                    x_train, selected_exprs, raw_control_expressions=raw_exprs,
                    magnitudes=settings["magnitudes"],
                    max_rows=max_rows, random_state=diag_seed,
                )
                derivative_validation = validate_jacobian_finite_difference(
                    x_train, selected_exprs, max_rows=min(max_rows, 32), random_state=diag_seed,
                )
                diagnostic_status = "diagnostic_complete"
                diagnostics = {
                    "schema_version": FSVA_SCHEMA_VERSION,
                    "diagnostic_status": diagnostic_status,
                    "task": task_context,
                    "settings": settings,
                    "jacobian": jac,
                    "empirical_amplification": amp,
                    "derivative_validation": derivative_validation,
                }
            except Exception as diagnostic_error:
                # Feature caches and benchmark results remain usable, but the
                # artifact is explicitly marked incomplete and cannot be
                # mistaken for diagnostic-complete evidence.
                diagnostic_status = "diagnostic_failed"
                diagnostics = {
                    "schema_version": FSVA_SCHEMA_VERSION,
                    "diagnostic_status": diagnostic_status,
                    "task": task_context,
                    "settings": settings,
                    "error_type": type(diagnostic_error).__name__,
                    "error_message": str(diagnostic_error),
                }
            fsva_path.write_text(json.dumps(diagnostics, sort_keys=True), encoding="utf-8")
            meta["diagnostic_schema_version"] = FSVA_SCHEMA_VERSION
            meta["diagnostic_status"] = diagnostic_status
            meta["diagnostic_path"] = str(fsva_path)
            meta["history_path"] = str(history_path)
        else:
            meta["diagnostic_schema_version"] = None
            meta["diagnostic_status"] = "diagnostic_disabled"

        res_pipelines[p_name] = (x_train_fe, x_test_fe)
        res_meta[p_name] = meta

    return res_pipelines, res_meta


# ---------------------------------------------------------------------------
# Data splitting + perturbation
# ---------------------------------------------------------------------------

def get_data_splits(data_path, dataset_name, seed, fold, condition,
                    shift_family, severity, diagnostics_enabled: bool = False,
                    diagnostic_config: dict[str, Any] | None = None):
    """Load data and keep labels out of feature-based split geometry.

    Covariate/population fold definitions intentionally use all predictor rows
    to establish an unsupervised stress-test geometry. All learned model-input
    transformations still fit only on the corrupted training partition.
    """
    df = load_csv_dataset(data_path)
    data_signature = dataset_identity(data_path,frame=df)
    X, y, _target_col = split_predictors_and_target(df)

    split_policy = split_policy_for_condition(shift_family)

    split_dependency = fingerprint({"data": data_signature, "split_policy": split_policy,
        "split_seed": split_seed(dataset_name,split_policy,seed,n_splits=5), "protocol": EVALUATION_PROTOCOL_VERSION})
    split_cache = _split_cache_path(dataset_name, seed, split_policy, split_dependency)
    split_meta = split_cache.with_suffix(".json")
    with artifact_lock(split_cache.with_suffix(".lock")):
        try:
            metadata = json.loads(split_meta.read_text(encoding="utf-8"))
            if metadata["dependency_signature"] != split_dependency or metadata["sha256"] != file_sha256(split_cache):
                raise ValueError("Incompatible split cache")
            splits = pd.read_pickle(split_cache)
        except (OSError,ValueError,KeyError,EOFError):
            derived_split_seed = split_seed(dataset_name, split_policy, seed, n_splits=5)
            splits = get_splits(X, y, split_policy, n_splits=5, seed=derived_split_seed)
            atomic_pickle(split_cache,splits)
            atomic_json(split_meta,{"dependency_signature": split_dependency, "sha256": file_sha256(split_cache), "data_identity": data_signature})

    assert_fold_integrity(splits, len(X))
    if fold < 1 or fold > len(splits):
        raise SplitInfeasibleError(
            f"Requested fold {fold} is outside the available 1..{len(splits)} range"
        )
    train_idx, test_idx = splits[fold - 1]

    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    x_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    derived_corruption_seed = corruption_seed(
        dataset_name, split_policy, seed, fold, condition,
    )
    x_train_cond, y_train_cond, x_test_cond, y_test_cond = apply_training_condition(
        x_train, y_train, shift_family=shift_family,
        x_test=x_test, y_test=y_test, severity=severity,
        random_state=derived_corruption_seed,
    )

    train_classes = set(y_train_cond.dropna().astype(str))
    test_classes = set(y_test_cond.dropna().astype(str))
    missing_train_classes = test_classes.difference(train_classes)
    if missing_train_classes:
        raise SplitInfeasibleError(
            "Requested split leaves held-out target classes absent from training: "
            f"{sorted(missing_train_classes)}"
        )

    from sklearn.preprocessing import LabelEncoder
    label_enc = LabelEncoder()
    y_train_enc = label_enc.fit_transform(y_train_cond.astype(str))
    y_test_enc = label_enc.transform(y_test_cond.astype(str))

    preprocessor = _build_preprocessor(x_train_cond, encoding="onehot", scale_numeric=True)
    x_train_prep = pd.DataFrame(
        _to_dense_array(preprocessor.fit_transform(x_train_cond)),
        columns=preprocessor.get_feature_names_out(),
    )
    x_test_prep = pd.DataFrame(
        _to_dense_array(preprocessor.transform(x_test_cond)),
        columns=preprocessor.get_feature_names_out(),
    )

    # Clean test set for Wasserstein distances
    x_test_clean_prep = pd.DataFrame(
        _to_dense_array(preprocessor.transform(x_test)),
        columns=preprocessor.get_feature_names_out(),
    )

    res_pipelines, res_meta = _run_pipeline_generation(
        x_train_prep, x_test_prep, y_train_enc,
        dataset_name, split_policy, seed, fold, condition,
        diagnostics_enabled=diagnostics_enabled,
        diagnostic_config=diagnostic_config,
        data_identity=data_signature,
    )
    for pipeline_name, meta in res_meta.items():
        meta["split_seed"] = split_seed(dataset_name, split_policy, seed, n_splits=5)
        meta["corruption_seed"] = derived_corruption_seed
        meta["seed_scheme_version"] = SEED_SCHEME_VERSION

    return res_pipelines, y_train_enc, y_test_enc, label_enc, res_meta, x_test_clean_prep


def apply_training_condition(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    shift_family: str,
    severity: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Corrupt copies of training data and return untouched held-out copies."""
    x_train_cond, y_train_cond = apply_perturbation(
        x_train.copy(), y_train.copy(), shift_family=shift_family,
        severity=severity, random_state=random_state,
    )
    return x_train_cond, y_train_cond, x_test.copy(), y_test.copy()


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def _manifest_for_task(task: dict[str, Any]) -> ManifestStore | None:
    path = task.get("manifest_db")
    return ManifestStore(path) if path else None


def _claim_manifest_task(task: dict[str, Any], *, worker_id: str) -> str | None:
    store = _manifest_for_task(task)
    if store is None or not task.get("scientific_task_id") or not task.get("run_id"):
        return None
    attempt_id = task.get("attempt_id")
    if attempt_id:
        return str(attempt_id)
    return store.claim_task(
        str(task["run_id"]),
        str(task["scientific_task_id"]),
        worker_id=worker_id,
        timeout_seconds=task.get("task_timeout_seconds"),
    )


def _record_manifest_failure(task: dict[str, Any], attempt_id: str | None, *, failure_class: str, exception: BaseException | None = None, retry: bool = False) -> None:
    store = _manifest_for_task(task)
    if store is None or not attempt_id or not task.get("scientific_task_id") or not task.get("run_id"):
        return
    try:
        next_state = store.record_failure(
            str(task["run_id"]), str(task["scientific_task_id"]), str(attempt_id),
            failure_class=failure_class, exception=exception,
            timeout_seconds=task.get("task_timeout_seconds"), retry=retry,
        )
        if failure_class == "precompute_failure" and next_state in {"failed", "timeout"}:
            store.propagate_dependency_failure(str(task["run_id"]), str(task["scientific_task_id"]))
    except ManifestConflictError:
        # A newer retry may already own the task.  Preserve that authoritative
        # state instead of allowing a stale worker to overwrite it.
        return


def _manifest_result_fields(task: dict[str, Any], attempt_id: str | None) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if task.get("run_id"):
        fields["run_id"] = task["run_id"]
    if task.get("scientific_task_id"):
        fields["scientific_task_id"] = task["scientific_task_id"]
    if attempt_id:
        fields["attempt_id"] = attempt_id
    return fields


def precompute_unit(kwargs):
    """Phase 1 worker: generate splits and every configured pipeline cache."""
    attempt_id = _claim_manifest_task(kwargs, worker_id=f"precompute:{os.getpid()}")
    if kwargs.get("manifest_db") and kwargs.get("scientific_task_id") and attempt_id is None:
        return kwargs.get("dataset_name")
    try:
        kwargs_copy = kwargs.copy()
        for key in ("size", "run_id", "manifest_db", "manifest_path", "scientific_task_id", "task_timeout_seconds", "retry", "attempt_id"):
            kwargs_copy.pop(key, None)
        get_data_splits(**kwargs_copy)
        store = _manifest_for_task(kwargs)
        if store is not None and attempt_id:
            store.commit_result(
                str(kwargs["run_id"]), str(kwargs["scientific_task_id"]), str(attempt_id),
                {**_manifest_result_fields(kwargs, attempt_id), "stage": "precompute", "status": "completed"},
                result_ref=str(cache_root() / kwargs["dataset_name"]),
            )
        gc.collect()
        return kwargs_copy["dataset_name"]
    except Exception as exc:
        _record_manifest_failure(kwargs, attempt_id, failure_class="precompute_failure", exception=exc, retry=kwargs.get("retry", False))
        with open("reports/worker_logs/phase1_error.log", "a") as f:
            f.write(f"Precompute error on {kwargs}: {traceback.format_exc()}\n")
        return None


def train_unit(kwargs):
    """Phase 2 worker: train one (pipeline, model) combo and write results."""
    attempt_id = _claim_manifest_task(kwargs, worker_id=f"train:{os.getpid()}")
    if kwargs.get("manifest_db") and kwargs.get("scientific_task_id") and attempt_id is None:
        return
    try:
        dataset_name = kwargs["dataset_name"]
        seed = kwargs["seed"]
        fold = kwargs["fold"]
        condition = kwargs["condition"]
        pipeline_name = kwargs["pipeline"]
        model_type = kwargs["model"]
        split_policy = split_policy_for_condition(kwargs["shift_family"])
        pipeline_identity = pipeline_identity_token(pipeline_name)

        if not kwargs.get("manifest_db") and has_run(
            dataset_name, seed, fold, condition, pipeline_name, model_type, split_policy,
            pipeline_identity,
        ):
            return

        pipelines, y_train_enc, y_test_enc, label_enc, res_meta, x_test_clean = (
            get_data_splits(
                kwargs["data_path"], dataset_name, seed, fold, condition,
                kwargs["shift_family"], kwargs["severity"],
                kwargs.get("diagnostics_enabled", False), kwargs.get("diagnostic_config"),
            )
        )

        X_tr, X_te = pipelines[pipeline_name]
        # Replace infs with large finite values so models don't crash
        X_tr = np.nan_to_num(X_tr.astype(np.float32), nan=np.nan, posinf=1e10, neginf=-1e10)
        X_te = np.nan_to_num(X_te.astype(np.float32), nan=np.nan, posinf=1e10, neginf=-1e10)
        meta = res_meta[pipeline_name]

        use_gpu = model_type in GPU_MODELS
        t0 = time.time()
        model_seed = estimator_seed(
            dataset_name,
            split_policy,
            seed,
            fold,
            condition,
            model_type,
        )
        model = build_model(model_type, random_state=model_seed, use_gpu=use_gpu)
        model.fit(X_tr, y_train_enc)
        train_time = time.time() - t0

        t1 = time.time()
        y_pred = model.predict(X_te)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_te)
        else:
            y_proba = None
        infer_time = time.time() - t1

        y_pred_train = model.predict(X_tr)
        if hasattr(model, "predict_proba"):
            y_proba_train = model.predict_proba(X_tr)
        else:
            y_proba_train = None

        encoded_classes = np.arange(len(label_enc.classes_))
        probability_classes = np.asarray(model.classes_)
        metrics_test = compute_classification_metrics(y_test_enc, y_pred, y_proba, encoded_classes, probability_classes=probability_classes)
        metrics_train = compute_classification_metrics(y_train_enc, y_pred_train, y_proba_train, encoded_classes, probability_classes=probability_classes)

        distance_seed = distance_sample_seed(
            dataset_name, split_policy, seed, fold, condition,
        )
        dist_metrics = compute_distribution_distance(
            x_test_clean,
            pipelines[pipeline_name][1],
            random_state=distance_seed,
        )
        if condition == "clean":
            dist_metrics["wasserstein"] = 0.0
            dist_metrics["ks_stat"] = 0.0

        res = {
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "seed_scheme_version": SEED_SCHEME_VERSION,
            "dataset": dataset_name,
            "split_policy": split_policy,
            "seed": seed,
            "fold": fold,
            "condition": condition,
            "pipeline": pipeline_name,
            "pipeline_identity": pipeline_identity,
            "model": model_type,
            "status": "success",
            **_manifest_result_fields(kwargs, attempt_id),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "n_original": x_test_clean.shape[1],
            "train_time_s": train_time,
            "infer_time_s": infer_time,
            "autofe_gen_time_s": meta.get("generation_time_s", 0),
            "split_seed": meta.get("split_seed"),
            "corruption_seed": meta.get("corruption_seed"),
            "selection_seed": meta.get("selection_seed"),
            "model_seed": model_seed,
            "distance_sample_seed": distance_seed,
            "operator_registry_version": meta.get("operator_registry_version"),
            "operator_semantics_version": meta.get("operator_semantics_version"),
            "operator_set_id": meta.get("operator_set_id"),
            "operator_set": meta.get("operator_set", []),
            "operator_set_manifest": meta.get("operator_set_manifest"),
            "baseline_kind": meta.get("baseline_kind"),
            "cap_policy_version": meta.get("cap_policy_version"),
            "requested_cap": meta.get("requested_cap"),
            "requested_base_cap": meta.get("requested_base_cap"),
            "eligible_base_feature_count": meta.get("eligible_base_feature_count"),
            "candidate_count": meta.get("candidate_count"),
            "generated_candidate_count": meta.get("num_generated"),
            "raw_candidate_count": meta.get("num_original"),
            "retained_raw_count": meta.get("retained_raw_count"),
            "retained_generated_count": meta.get("retained_generated_count"),
            "retained_feature_count": meta.get("num_selected"),
            "actual_estimator_input_dimension": meta.get("actual_estimator_input_dimension"),
            "selector_identity": meta.get("selector_identity"),
            "selected_feature_identities": meta.get("selected_feature_identities", []),
            "diagnostic_schema_version": meta.get("diagnostic_schema_version"),
            "diagnostic_status": meta.get("diagnostic_status", "diagnostic_disabled"),
            "diagnostic_path": meta.get("diagnostic_path"),
            "history_path": meta.get("history_path"),
            "autofe_cache_hit": meta.get("dfs_cache_hit", False),
            "n_generated": meta.get("num_generated", 0),
            "n_retained": meta.get("num_selected", 0),
            "ram_used_mb": meta.get("ram_used_mb", 0),
            "wasserstein": dist_metrics["wasserstein"],
            "ks_stat": dist_metrics["ks_stat"],
            "train_auc": metrics_train.get("roc_auc", np.nan),
            "test_auc": metrics_test.get("roc_auc", np.nan),
            **metrics_test,
        }

        _writer_queue.put(res)

        del model, X_tr, X_te, pipelines, x_test_clean
        gc.collect()

    except Exception as exc:
        _record_manifest_failure(kwargs, attempt_id, failure_class="worker_exception", exception=exc, retry=kwargs.get("retry", False))
        with open("reports/worker_logs/phase2_error.log", "a") as f:
            f.write(f"Train error {kwargs}: {traceback.format_exc()}\n")


# ---------------------------------------------------------------------------
# Writer process (sequential disk I/O)
# ---------------------------------------------------------------------------

def _train_process_entry(task: dict[str, Any], result_queue) -> None:
    global _writer_queue
    _writer_queue = result_queue
    train_unit(task)


def run_bounded_train_task(task: dict[str, Any], result_queue, *, timeout_seconds: float) -> str:
    """Run one training worker in a killable process for explicit timeouts."""
    store = _manifest_for_task(task)
    if store is not None and task.get("run_id") and task.get("scientific_task_id") and not task.get("attempt_id"):
        task["attempt_id"] = store.claim_task(
            str(task["run_id"]), str(task["scientific_task_id"]),
            worker_id=f"bounded-parent:{os.getpid()}", timeout_seconds=timeout_seconds,
        )
    if task.get("manifest_db") and not task.get("attempt_id"):
        return "skipped"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_train_process_entry, args=(task, result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        _record_manifest_failure(task, task.get("attempt_id"), failure_class="timeout", retry=task.get("retry", False))
        return "timeout"
    if process.exitcode != 0:
        _record_manifest_failure(task, task.get("attempt_id"), failure_class="worker_exception", retry=task.get("retry", False))
        return "failed"
    return "completed"


def dispatch_training_tasks(cpu_tasks: list[dict[str, Any]], gpu_tasks: list[dict[str, Any]], queue, execution_config: ExecutionConfig) -> None:
    """Dispatch one bounded attempt batch under the declared execution policy."""
    if not (cpu_tasks or gpu_tasks):
        return
    if execution_config.task_timeout_seconds is not None:
        for bounded_task in cpu_tasks + gpu_tasks:
            if _stop_requested:
                break
            run_bounded_train_task(
                bounded_task, queue,
                timeout_seconds=execution_config.task_timeout_seconds,
            )
        return
    cpu_pool = multiprocessing.Pool(
        execution_config.max_workers, initializer=init_worker, initargs=(queue,),
        maxtasksperchild=MAX_TASKS_PER_CHILD,
    )
    gpu_pool = multiprocessing.Pool(
        N_GPU_WORKERS, initializer=init_worker, initargs=(queue,),
        maxtasksperchild=MAX_TASKS_PER_CHILD,
    )
    try:
        cpu_res = cpu_pool.map_async(train_unit, cpu_tasks)
        gpu_res = gpu_pool.map_async(train_unit, gpu_tasks)
        cpu_res.wait()
        gpu_res.wait()
    finally:
        cpu_pool.close()
        cpu_pool.join()
        gpu_pool.close()
        gpu_pool.join()


def _wait_for_manifest_tasks(
    store: ManifestStore,
    run_id: str,
    task_ids: list[str],
    *,
    timeout_seconds: float = 120.0,
) -> None:
    """Wait for the writer to fence/commit an attempt batch before retrying."""
    if not task_ids:
        return
    deadline = time.monotonic() + timeout_seconds
    while True:
        states = [store.get_task(run_id, task_id) for task_id in task_ids]
        if not any(task is not None and task.get("state") == "running" for task in states):
            return
        if time.monotonic() >= deadline:
            # Leave still-running attempts visible for coordinator recovery;
            # never dispatch a second attempt while ownership is unresolved.
            return
        time.sleep(0.05)

def writer_process(queue, results_path, manifest_db: str | Path | None = None, run_id: str | None = None):
    """Write durable result records, then commit their manifest completion."""
    init_db()
    store = ManifestStore(manifest_db) if manifest_db and run_id else None
    with open(results_path, "a", encoding="utf-8", newline="\n") as f:
        while True:
            res = queue.get()
            if res == "DONE":
                break
            res = _json_safe(res)
            if store is not None and res.get("run_id") and res.get("scientific_task_id") and res.get("attempt_id"):
                task = store.get_task(str(run_id), str(res["scientific_task_id"]))
                if task is None:
                    raise ManifestError(f"result references unknown manifest task {res['scientific_task_id']}")
                if task["state"] == "completed":
                    # Idempotent duplicate delivery: the authoritative result
                    # is already durable, so do not append a second analysis row.
                    continue
                if task["state"] != "running" or task["active_attempt_id"] != res["attempt_id"]:
                    raise ManifestConflictError("result writer received a stale task attempt")
            f.write(json.dumps(res, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if store is not None:
                store.commit_result(
                    str(run_id), str(res["scientific_task_id"]), str(res["attempt_id"]),
                    res, result_ref=str(results_path),
                )
            log_run(
                res["dataset"], res["seed"], res["fold"],
                res["condition"], res["pipeline"], res["model"],
                res["split_policy"],
                res.get("pipeline_identity", ""),
            )


def init_worker(q):
    """Pool initializer: share the writer queue with child processes."""
    global _writer_queue
    _writer_queue = q


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_hardware():
    """Print a summary of the available hardware."""
    cpu_count = os.cpu_count() or 1
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    print("=" * 60)
    print("  AutoFE-ShiftBench — Hardware Detection")
    print("=" * 60)
    print(f"  CPU cores:       {cpu_count}")
    print(f"  Workers (CPU):   {N_CPU_WORKERS}")
    print(f"  Workers (GPU):   {N_GPU_WORKERS}")
    print(f"  RAM:             {ram_gb:.1f} GB")

    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            print(f"  GPU:             {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        else:
            print("  GPU:             Not detected (CUDA unavailable)")
    except Exception:
        # Try xgboost device detection instead
        try:
            from xgboost import XGBClassifier
            m = XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
            print("  GPU:             Available (XGBoost CUDA)")
        except Exception:
            print("  GPU:             Not detected")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    global _stop_requested
    parser = argparse.ArgumentParser(description="AutoFE-ShiftBench runner")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-conditions", type=int, default=None)
    parser.add_argument("--condition-start", type=int, default=0)
    parser.add_argument("--pipelines", type=str, default=None, help="Optional comma-separated pipeline subset for a bounded run")
    parser.add_argument("--models", type=str, default=None, help="Optional comma-separated estimator subset for a bounded run")
    parser.add_argument("--enable-fsva-diagnostics", action="store_true")
    parser.add_argument("--fsva-max-rows", type=int, default=128)
    parser.add_argument("--dry-run-manifest", action="store_true", help="Create and validate the complete manifest without dispatching work")
    parser.add_argument("--manifest-db", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--task-timeout-seconds", type=float, default=None)
    parser.add_argument("--run-wall-time-seconds", type=float, default=None)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--stop-after-tasks", type=int, default=None)
    parser.add_argument("--stale-after-seconds", type=float, default=3600.0, help="Recover running attempts older than this on resume")
    args = parser.parse_args()

    def _request_stop(_signal_number, _frame):
        global _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    # Ensure directories exist
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/worker_logs").mkdir(parents=True, exist_ok=True)

    init_db()
    if not args.dry_run_manifest:
        detect_hardware()
    logger = setup_logger("reports/terminal.log")
    results_path = results_ledger_path()

    # ---- Load experiment grid ----
    datasets = load_dataset_names("config/dataset_list.yaml")
    if args.max_datasets:
        datasets = datasets[:args.max_datasets]

    seeds = [42, 123, 456, 789, 2025]
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]

    folds = list(range(1, 6))
    if args.max_folds:
        folds = folds[:args.max_folds]

    families = list(SHIFT_FAMILIES)
    if args.condition_start < 0:
        raise ValueError("condition-start must be non-negative")
    families = families[args.condition_start:]
    if args.max_conditions:
        families = families[:args.max_conditions]

    requested_pipelines = PIPELINE_NAMES if args.pipelines is None else [value.strip() for value in args.pipelines.split(",") if value.strip()]
    unknown_pipelines = sorted(set(requested_pipelines) - set(PIPELINE_NAMES))
    if unknown_pipelines or not requested_pipelines:
        raise ValueError(f"Unknown or empty pipeline subset: {unknown_pipelines or args.pipelines}")
    selected_pipelines = [name for name in PIPELINE_NAMES if name in set(requested_pipelines)]
    requested_models = (CPU_MODELS + GPU_MODELS) if args.models is None else [value.strip() for value in args.models.split(",") if value.strip()]
    unknown_models = sorted(set(requested_models) - set(CPU_MODELS + GPU_MODELS))
    if unknown_models or not requested_models:
        raise ValueError(f"Unknown or empty estimator subset: {unknown_models or args.models}")
    selected_cpu_models = [name for name in CPU_MODELS if name in set(requested_models)]
    selected_gpu_models = [name for name in GPU_MODELS if name in set(requested_models)]
    selected_models = selected_cpu_models + selected_gpu_models

    # ---- Build precompute task list ----
    precompute_tasks = []
    for d in datasets:
        dp = Path(f"data/raw/{d}.csv")
        if not dp.exists():
            logger.warning(f"Dataset CSV not found, skipping: {dp}")
            continue
        size = dp.stat().st_size
        for s in seeds:
            for f in folds:
                for fam, sev in families:
                    cond_name = fam if sev == 0.0 else f"{fam}_{sev}"
                    precompute_tasks.append({
                        "dataset_name": d, "data_path": dp, "seed": s, "fold": f,
                        "shift_family": fam, "severity": sev, "condition": cond_name,
                        "diagnostics_enabled": args.enable_fsva_diagnostics,
                        "diagnostic_config": {
                            "max_rows": args.fsva_max_rows,
                            "random_state": stable_seed("fsva_diagnostic", {
                                "dataset": d, "split_policy": split_policy_for_condition(fam),
                                "seed": s, "fold": f, "condition": cond_name,
                            }),
                            "magnitudes": list(DEFAULT_PERTURBATION_MAGNITUDES),
                        },
                        "size": size,
                    })

    # Construct the complete scientific task manifest before any expensive
    # precompute or model dispatch.  Missing datasets remain explicit skipped
    # tasks so intended and executable counts cannot be conflated.
    execution_config = ExecutionConfig(
        max_workers=args.max_workers or min(N_CPU_WORKERS, 4),
        task_timeout_seconds=args.task_timeout_seconds,
        run_wall_time_seconds=args.run_wall_time_seconds,
        max_attempts=args.max_attempts,
        stop_after_tasks=args.stop_after_tasks,
        stale_after_seconds=args.stale_after_seconds,
    )
    data_paths = {dataset: Path(f"data/raw/{dataset}.csv") for dataset in datasets}
    data_identities = {name: dataset_identity(path) for name,path in data_paths.items()}
    manifest_config = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "seed_scheme_version": SEED_SCHEME_VERSION,
        "code_identity": collect_code_identity(Path.cwd()),
        "datasets": datasets,
        "dataset_identities": data_identities,
        "seeds": seeds,
        "folds": folds,
        "conditions": [{"shift_family": fam, "severity": sev} for fam, sev in families],
        "pipelines": selected_pipelines,
        "models": selected_models,
        "diagnostics_enabled": args.enable_fsva_diagnostics,
        "diagnostic_max_rows": args.fsva_max_rows,
        "execution": execution_config.to_dict(),
    }
    run_id = args.run_id or run_id_for(manifest_config)
    manifest_db = args.manifest_db or Path("reports/manifests/task_manifest.db")
    manifest_path = args.manifest_path or Path("reports/manifests") / f"{run_id}.jsonl"
    manifest_records = build_task_records(
        datasets, seeds, folds, families, selected_pipelines, selected_models,
        pipeline_identity=pipeline_identity_token,
        pipeline_metadata=lambda name: {
            "operator_set_id": PIPELINE_CONFIGS[name].operator_set_id,
            "cap_policy_version": CAP_POLICY_VERSION if PIPELINE_CONFIGS[name].max_features is not None else "none_v1",
        },
        data_paths=data_paths,
        data_identities=data_identities,
    )
    manifest_store = ManifestStore(manifest_db, manifest_path=manifest_path)
    expected_manifest_count = manifest_store.create_run(run_id, manifest_config, manifest_records, manifest_path=manifest_path)
    recovered_attempts = manifest_store.recover_stale_attempts(
        run_id,
        stale_after_seconds=execution_config.stale_after_seconds,
        retry=True,
    )
    if recovered_attempts:
        logger.warning(f"Recovered {recovered_attempts} stale running attempts for run {run_id}")
    precompute_ids = {
        (record["dataset"], record["seed"], record["fold"], record["condition"]): record["scientific_task_id"]
        for record in manifest_records if record["task_kind"] == "precompute"
    }
    model_ids = {
        (record["dataset"], record["seed"], record["fold"], record["condition"], record["pipeline"], record["model"]): record["scientific_task_id"]
        for record in manifest_records if record["task_kind"] == "model"
    }
    for task in precompute_tasks:
        task.update({
            "run_id": run_id,
            "manifest_db": str(manifest_db),
            "manifest_path": str(manifest_path),
            "scientific_task_id": precompute_ids[(task["dataset_name"], task["seed"], task["fold"], task["condition"])],
            "task_timeout_seconds": args.task_timeout_seconds,
            "retry": args.max_attempts > 1,
        })
    logger.info(f"Manifest {run_id}: {expected_manifest_count} intended tasks ({manifest_store.state_counts(run_id)})")
    if args.dry_run_manifest:
        print(json.dumps({"run_id": run_id, "manifest_db": str(manifest_db), "manifest_path": str(manifest_path), "state_counts": manifest_store.state_counts(run_id)}, sort_keys=True))
        return

    # Sort datasets so we still process the smallest ones first for fast feedback
    # Calculate dataset sizes
    dataset_sizes = {}
    for pt in precompute_tasks:
        dataset_sizes[pt["dataset_name"]] = pt["size"]
    
    sorted_datasets = sorted([d for d in datasets if d in dataset_sizes], key=lambda x: dataset_sizes[x])

    # ---- Setup Writer ----
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    writer = multiprocessing.Process(target=writer_process, args=(queue, results_path, str(manifest_db), run_id))
    writer.start()

    phase1_workers = min(args.max_workers or N_CPU_WORKERS, 4)  # Capped at 4 to prevent OOM
    
    # Process each dataset completely to allow cache cleanup
    run_started = time.monotonic()
    dispatched_model_tasks = 0
    for d in sorted_datasets:
        if _stop_requested or (execution_config.stop_after_tasks is not None and dispatched_model_tasks >= execution_config.stop_after_tasks) or (execution_config.run_wall_time_seconds is not None and time.monotonic() - run_started >= execution_config.run_wall_time_seconds):
            manifest_store.request_stop(run_id, reason="run_wall_time_or_signal")
            break
        logger.info(f"--- Processing dataset: {d} ---")
        dataset_tasks = [t for t in precompute_tasks if t["dataset_name"] == d]
        
        if not dataset_tasks:
            continue
            
        # ---- Phase 1: Precompute splits + AutoFE caches for this dataset ----
        logger.info(f"Phase 1 [{d}]: {len(dataset_tasks)} units using {phase1_workers} workers...")
        total = len(dataset_tasks)
        for precompute_round in range(execution_config.max_attempts):
            pending_precompute: list[dict[str, Any]] = []
            for task in dataset_tasks:
                task_id = str(task["scientific_task_id"])
                manifest_task = manifest_store.get_task(run_id, task_id)
                if manifest_task is None or manifest_task["state"] != "pending":
                    continue
                task["retry"] = precompute_round + 1 < execution_config.max_attempts
                pending_precompute.append(task)
            if not pending_precompute:
                break
            completed = total - len(pending_precompute)
            logger.info(f"Phase 1 [{d}] attempt {precompute_round + 1}/{execution_config.max_attempts}: {len(pending_precompute)} pending units")
            with multiprocessing.Pool(phase1_workers, maxtasksperchild=1) as pool:
                for _result in pool.imap_unordered(precompute_unit, pending_precompute):
                    completed += 1
                    if completed % 50 == 0 or completed == total:
                        ram_pct = psutil.virtual_memory().percent
                        logger.info(f"Phase 1 [{d}]: {completed}/{total} ({100*completed/total:.1f}%) | RAM: {ram_pct:.0f}%")
            if _stop_requested or (execution_config.run_wall_time_seconds is not None and time.monotonic() - run_started >= execution_config.run_wall_time_seconds):
                break

        if not all(
            (manifest_store.get_task(run_id, str(task["scientific_task_id"])) or {}).get("state") == "completed"
            for task in dataset_tasks
        ):
            # Model tasks remain pending/skipped until every precompute
            # dependency is authoritative; a cache or a running lease alone
            # is never treated as estimator completion.
            continue

        # ---- Phase 2: Train models for this dataset with bounded retries ----
        for attempt_round in range(execution_config.max_attempts):
            if _stop_requested:
                break
            if execution_config.run_wall_time_seconds is not None and time.monotonic() - run_started >= execution_config.run_wall_time_seconds:
                break
            cpu_tasks: list[dict[str, Any]] = []
            gpu_tasks: list[dict[str, Any]] = []
            for pt in dataset_tasks:
                for p in selected_pipelines:
                    for m in selected_cpu_models:
                        model_task_id = model_ids[(pt["dataset_name"], pt["seed"], pt["fold"], pt["condition"], p, m)]
                        manifest_task = manifest_store.get_task(run_id, model_task_id)
                        if manifest_task is None or manifest_task["state"] != "pending":
                            continue
                        t = pt.copy()
                        t["pipeline"] = p
                        t["model"] = m
                        t["model_task_id"] = model_task_id
                        t["scientific_task_id"] = model_task_id
                        t["task_timeout_seconds"] = args.task_timeout_seconds
                        t["retry"] = attempt_round + 1 < execution_config.max_attempts
                        cpu_tasks.append(t)
                    for m in selected_gpu_models:
                        model_task_id = model_ids[(pt["dataset_name"], pt["seed"], pt["fold"], pt["condition"], p, m)]
                        manifest_task = manifest_store.get_task(run_id, model_task_id)
                        if manifest_task is None or manifest_task["state"] != "pending":
                            continue
                        t = pt.copy()
                        t["pipeline"] = p
                        t["model"] = m
                        t["model_task_id"] = model_task_id
                        t["scientific_task_id"] = model_task_id
                        t["task_timeout_seconds"] = args.task_timeout_seconds
                        t["retry"] = attempt_round + 1 < execution_config.max_attempts
                        gpu_tasks.append(t)

            logger.info(f"Phase 2 [{d}] attempt {attempt_round + 1}/{execution_config.max_attempts}: evaluating {len(cpu_tasks)} CPU and {len(gpu_tasks)} GPU tasks...")
            if execution_config.stop_after_tasks is not None:
                remaining = max(0, execution_config.stop_after_tasks - dispatched_model_tasks)
                cpu_tasks = cpu_tasks[:remaining]
                remaining = max(0, execution_config.stop_after_tasks - dispatched_model_tasks - len(cpu_tasks))
                gpu_tasks = gpu_tasks[:remaining]
            dispatched_model_tasks += len(cpu_tasks) + len(gpu_tasks)
            dispatch_training_tasks(cpu_tasks, gpu_tasks, queue, execution_config)
            _wait_for_manifest_tasks(
                manifest_store,
                run_id,
                [str(task["scientific_task_id"]) for task in cpu_tasks + gpu_tasks],
                timeout_seconds=max(30.0, (args.task_timeout_seconds or 0.0) * 2.0),
            )
            if not (cpu_tasks or gpu_tasks) or (execution_config.stop_after_tasks is not None and dispatched_model_tasks >= execution_config.stop_after_tasks):
                break

        # ---- Phase 3: Cleanup cache to prevent 600GB disk usage ----
        import shutil
        cache_dir = cache_root() / d
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info(f"Phase 3 [{d}]: Deleted cache directory {cache_dir}")
            
    queue.put("DONE")
    writer.join()
    if writer.exitcode != 0:
        manifest_store.set_run_status(run_id, "failed")
        logger.error(f"Result writer exited with code {writer.exitcode}; reconciliation is required")
    elif _stop_requested or (execution_config.stop_after_tasks is not None and dispatched_model_tasks >= execution_config.stop_after_tasks) or (execution_config.run_wall_time_seconds is not None and time.monotonic() - run_started >= execution_config.run_wall_time_seconds):
        manifest_store.request_stop(run_id, reason="declared_stop_limit")
        manifest_store.set_run_status(run_id, "stopped")
    else:
        manifest_store.set_run_status(run_id, "completed")
    logger.info(f"Benchmark finished. Manifest counts: {manifest_store.state_counts(run_id)}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
