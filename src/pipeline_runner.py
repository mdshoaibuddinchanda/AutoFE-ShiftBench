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
import sys
import time
import traceback
import warnings
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
from src.data_loader import load_csv_dataset, load_dataset_names
from src.evaluation import (
    compute_classification_metrics,
    compute_distribution_distance,
    compute_jaccard_similarity,
)
from src.feature_engineering import expand_features_with_dfs, DFSConfig
from src.feature_selection import FeatureSelectionConfig, select_top_features
from src.model import build_model
from src.preprocessing import _build_preprocessor, _to_dense_array
from src.protocol import EVALUATION_PROTOCOL_VERSION, cache_root, results_ledger_path
from src.seeding import (
    SEED_SCHEME_VERSION,
    corruption_seed,
    distance_sample_seed,
    estimator_seed,
    feature_selection_seed,
    split_seed,
)
from src.shift_generator import apply_perturbation
from src.splitters import SplitInfeasibleError, assert_fold_integrity, get_splits
from src.shap_explainer import compute_shap_values

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CPU_WORKERS = max(1, os.cpu_count() - 1)
N_GPU_WORKERS = 1  # XGBoost + CatBoost share VRAM; sequential is correct
RAM_LIMIT_PERCENT = 85  # Pause spawning if RAM exceeds this %
MAX_TASKS_PER_CHILD = 50  # Reduce process respawn overhead

PIPELINE_CONFIGS = {
    "Raw": DFSConfig(enable_dfs=False, selection_method="none"),
    "Raw_Variance": DFSConfig(enable_dfs=False, selection_method="variance", max_features=100),
    "Raw_MI": DFSConfig(enable_dfs=False, selection_method="mi", max_features=100),
    "AutoFE_Baseline": DFSConfig(enable_dfs=True, selection_method="variance", max_features=100, depth=1),
    "AutoFE_MI": DFSConfig(enable_dfs=True, selection_method="mi", max_features=100, depth=1),
    "AutoFE_Random": DFSConfig(enable_dfs=True, selection_method="random", max_features=100, depth=1),
    "AutoFE_NoMultiply": DFSConfig(
        enable_dfs=True, selection_method="variance", max_features=100, depth=1,
        trans_primitives=["add_numeric", "subtract_numeric"],
    ),
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


def _split_cache_path(dataset_name: str, seed: int, split_policy: str) -> Path:
    """Return a split cache path keyed by dataset, replicate, and split policy."""
    return _cache_dir_for_dataset(dataset_name) / f"splits_s{seed}_{split_policy}.pkl"


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
                          seed: int, fold: int, condition: str):
    """Return (train_pkl, test_pkl, meta_json) with human-readable names.

    Example:
        data/cache/{protocol}/{seed_scheme}/adult/AutoFE_MI_s42_f1_gaussian_noise_0.05_train.pkl
        data/cache/{protocol}/{seed_scheme}/adult/AutoFE_MI_s42_f1_gaussian_noise_0.05_test.pkl
        data/cache/{protocol}/{seed_scheme}/adult/AutoFE_MI_s42_f1_gaussian_noise_0.05_meta.json
    """
    d = _cache_dir_for_dataset(dataset_name)
    base = f"{pipeline_name}_s{seed}_f{fold}_{condition}"
    return (
        d / f"{base}_train.pkl",
        d / f"{base}_test.pkl",
        d / f"{base}_meta.json",
    )


# ---------------------------------------------------------------------------
# Pipeline generation (Phase 1 core)
# ---------------------------------------------------------------------------

def _run_pipeline_generation(x_train, x_test, y_train,
                             dataset_name, split_policy, seed, fold, condition):
    """Generate all 7 pipeline variants for one experimental unit.

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
        )

        train_cache, test_cache, meta_cache = _pipeline_cache_paths(
            dataset_name, p_name, seed, fold, condition
        )

        if train_cache.exists() and test_cache.exists() and meta_cache.exists():
            x_train_fe = pd.read_pickle(train_cache)
            x_test_fe = pd.read_pickle(test_cache)
            with open(meta_cache) as f:
                meta = json.load(f)
            meta["dfs_cache_hit"] = True
        else:
            t0 = time.time()
            x_train_fe, x_test_fe, dfs_meta = expand_features_with_dfs(
                x_train, x_test, y_train, config=cfg_copy,
            )
            gen_time = time.time() - t0

            meta = {
                "num_original": x_train.shape[1],
                "num_generated": dfs_meta.get("n_generated", 0),
                "num_selected": dfs_meta.get("n_retained", x_train_fe.shape[1]),
                "generation_time_s": gen_time,
                "ram_used_mb": dfs_meta.get("ram_used_mb", 0),
                "feature_metadata": dfs_meta.get("feature_metadata", []),
                "dfs_cache_hit": False,
            }
            x_train_fe.to_pickle(train_cache)
            x_test_fe.to_pickle(test_cache)
            with open(meta_cache, "w") as f:
                json.dump(meta, f)

        res_pipelines[p_name] = (x_train_fe, x_test_fe)
        meta["selection_seed"] = selection_random_state
        res_meta[p_name] = meta

    return res_pipelines, res_meta


# ---------------------------------------------------------------------------
# Data splitting + perturbation
# ---------------------------------------------------------------------------

def get_data_splits(data_path, dataset_name, seed, fold, condition,
                    shift_family, severity):
    """Load data and keep labels out of feature-based split geometry.

    Covariate/population fold definitions intentionally use all predictor rows
    to establish an unsupervised stress-test geometry. All learned model-input
    transformations still fit only on the corrupted training partition.
    """
    df = load_csv_dataset(data_path)
    X, y, _target_col = split_predictors_and_target(df)

    split_policy = split_policy_for_condition(shift_family)

    split_cache = _split_cache_path(dataset_name, seed, split_policy)
    if split_cache.exists():
        splits = pd.read_pickle(split_cache)
    else:
        derived_split_seed = split_seed(dataset_name, split_policy, seed, n_splits=5)
        splits = get_splits(X, y, split_policy, n_splits=5, seed=derived_split_seed)
        split_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(splits, split_cache)

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

def precompute_unit(kwargs):
    """Phase 1 worker: generate splits + all 7 pipeline caches for one unit."""
    try:
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("size", None)
        get_data_splits(**kwargs_copy)
        gc.collect()
        return kwargs_copy["dataset_name"]
    except Exception:
        with open("reports/worker_logs/phase1_error.log", "a") as f:
            f.write(f"Precompute error on {kwargs}: {traceback.format_exc()}\n")
        return None


def train_unit(kwargs):
    """Phase 2 worker: train one (pipeline, model) combo and write results."""
    try:
        dataset_name = kwargs["dataset_name"]
        seed = kwargs["seed"]
        fold = kwargs["fold"]
        condition = kwargs["condition"]
        pipeline_name = kwargs["pipeline"]
        model_type = kwargs["model"]
        split_policy = split_policy_for_condition(kwargs["shift_family"])

        if has_run(
            dataset_name, seed, fold, condition, pipeline_name, model_type, split_policy,
        ):
            return

        pipelines, y_train_enc, y_test_enc, label_enc, res_meta, x_test_clean = (
            get_data_splits(
                kwargs["data_path"], dataset_name, seed, fold, condition,
                kwargs["shift_family"], kwargs["severity"],
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
            y_proba = np.zeros((len(y_test_enc), len(label_enc.classes_)))
        infer_time = time.time() - t1

        y_pred_train = model.predict(X_tr)
        if hasattr(model, "predict_proba"):
            y_proba_train = model.predict_proba(X_tr)
        else:
            y_proba_train = np.zeros((len(y_train_enc), len(label_enc.classes_)))

        metrics_test = compute_classification_metrics(y_test_enc, y_pred, y_proba, label_enc.classes_)
        metrics_train = compute_classification_metrics(y_train_enc, y_pred_train, y_proba_train, label_enc.classes_)

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
            "model": model_type,
            "status": "success",
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

    except Exception:
        with open("reports/worker_logs/phase2_error.log", "a") as f:
            f.write(f"Train error {kwargs}: {traceback.format_exc()}\n")


# ---------------------------------------------------------------------------
# Writer process (sequential disk I/O)
# ---------------------------------------------------------------------------

def writer_process(queue, results_path):
    """Dedicated process that writes results to JSONL and logs to checkpoint DB."""
    init_db()
    with open(results_path, "a") as f:
        while True:
            res = queue.get()
            if res == "DONE":
                break
            f.write(json.dumps(res) + "\n")
            f.flush()
            log_run(
                res["dataset"], res["seed"], res["fold"],
                res["condition"], res["pipeline"], res["model"],
                res["split_policy"],
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
    parser = argparse.ArgumentParser(description="AutoFE-ShiftBench runner")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-conditions", type=int, default=None)
    args = parser.parse_args()

    # Ensure directories exist
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/worker_logs").mkdir(parents=True, exist_ok=True)

    init_db()
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
    if args.max_conditions:
        families = families[:args.max_conditions]

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
                        "size": size,
                    })

    # Sort datasets so we still process the smallest ones first for fast feedback
    # Calculate dataset sizes
    dataset_sizes = {}
    for pt in precompute_tasks:
        dataset_sizes[pt["dataset_name"]] = pt["size"]
    
    sorted_datasets = sorted([d for d in datasets if d in dataset_sizes], key=lambda x: dataset_sizes[x])

    # ---- Setup Writer ----
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    writer = multiprocessing.Process(target=writer_process, args=(queue, results_path))
    writer.start()

    phase1_workers = min(N_CPU_WORKERS, 4)  # Capped at 4 to prevent OOM
    
    # Process each dataset completely to allow cache cleanup
    for d in sorted_datasets:
        logger.info(f"--- Processing dataset: {d} ---")
        dataset_tasks = [t for t in precompute_tasks if t["dataset_name"] == d]
        
        if not dataset_tasks:
            continue
            
        # ---- Phase 1: Precompute splits + AutoFE caches for this dataset ----
        logger.info(f"Phase 1 [{d}]: {len(dataset_tasks)} units using {phase1_workers} workers...")
        completed = 0
        total = len(dataset_tasks)
        with multiprocessing.Pool(phase1_workers, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(precompute_unit, dataset_tasks):
                completed += 1
                if completed % 50 == 0 or completed == total:
                    ram_pct = psutil.virtual_memory().percent
                    logger.info(f"Phase 1 [{d}]: {completed}/{total} ({100*completed/total:.1f}%) | RAM: {ram_pct:.0f}%")
        
        # ---- Phase 2: Train models for this dataset ----
        cpu_tasks = []
        gpu_tasks = []
        for pt in dataset_tasks:
            for p in PIPELINE_NAMES:
                for m in CPU_MODELS:
                    if not has_run(
                        pt["dataset_name"], pt["seed"], pt["fold"], pt["condition"], p, m,
                        split_policy_for_condition(pt["shift_family"]),
                    ):
                        t = pt.copy()
                        t["pipeline"] = p
                        t["model"] = m
                        cpu_tasks.append(t)
                for m in GPU_MODELS:
                    if not has_run(
                        pt["dataset_name"], pt["seed"], pt["fold"], pt["condition"], p, m,
                        split_policy_for_condition(pt["shift_family"]),
                    ):
                        t = pt.copy()
                        t["pipeline"] = p
                        t["model"] = m
                        gpu_tasks.append(t)
                        
        logger.info(f"Phase 2 [{d}]: Evaluating {len(cpu_tasks)} CPU and {len(gpu_tasks)} GPU tasks...")
        
        if cpu_tasks or gpu_tasks:
            cpu_pool = multiprocessing.Pool(
                N_CPU_WORKERS, initializer=init_worker, initargs=(queue,),
                maxtasksperchild=MAX_TASKS_PER_CHILD,
            )
            gpu_pool = multiprocessing.Pool(
                N_GPU_WORKERS, initializer=init_worker, initargs=(queue,),
                maxtasksperchild=MAX_TASKS_PER_CHILD,
            )

            cpu_res = cpu_pool.map_async(train_unit, cpu_tasks)
            gpu_res = gpu_pool.map_async(train_unit, gpu_tasks)

            cpu_res.wait()
            gpu_res.wait()

            cpu_pool.close()
            cpu_pool.join()
            gpu_pool.close()
            gpu_pool.join()

        # ---- Phase 3: Cleanup cache to prevent 600GB disk usage ----
        import shutil
        cache_dir = cache_root() / d
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info(f"Phase 3 [{d}]: Deleted cache directory {cache_dir}")
            
    queue.put("DONE")
    writer.join()
    logger.info("Benchmark finished! All caches cleaned up.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
