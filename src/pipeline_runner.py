"""Unified Robustness Benchmark Pipeline Runner."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Internal imports
from src.data_loader import load_dataset_names, load_csv_dataset
from src.evaluation import compute_classification_metrics
from src.model import build_model
from src.preprocessing import _build_preprocessor, _to_dense_array, _get_stratify_target
from src.shift_generator import apply_perturbation
from src.shap_explainer import compute_shap_values

# We will dynamically import Featuretools and Selection to keep process lightweight
# until actually executing the worker.


def setup_logger(log_file: str | Path) -> logging.Logger:
    """Setup a dedicated logger that writes to terminal.log."""
    logger = logging.getLogger("ShiftBench")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def _run_autofe(
    x_train: pd.DataFrame, 
    x_test: pd.DataFrame, 
    y_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run DFS and Feature Selection inside the worker."""
    from src.feature_engineering import expand_features_with_dfs, DFSConfig
    from src.feature_selection import select_top_features, FeatureSelectionConfig
    
    dfs_cfg = DFSConfig(depth=2, max_features=100, max_base_features=40, monitor_ram=False)
    
    # 1. Generate
    t0 = time.time()
    x_train_fe, x_test_fe, dfs_meta = expand_features_with_dfs(x_train, x_test, config=dfs_cfg)
    gen_time = time.time() - t0
    
    # 2. Select
    t1 = time.time()
    fs_cfg = FeatureSelectionConfig(max_features=40, l1_strength=0.1, task="classification")
    x_train_sel, x_test_sel, sel_meta = select_top_features(x_train_fe, x_test_fe, y_train, config=fs_cfg)
    sel_time = time.time() - t1
    
    meta = {
        "num_original": x_train.shape[1],
        "num_generated": x_train_fe.shape[1],
        "num_selected": x_train_sel.shape[1],
        "generation_time_s": gen_time,
        "selection_time_s": sel_time
    }
    
    return x_train_sel, x_test_sel, meta


def evaluate_unit(
    dataset_name: str,
    seed: int,
    fold: int,
    condition: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    data_path: str,
    model_types: list[str],
) -> list[dict[str, Any]]:
    """Worker function to evaluate one condition for all models on a given fold."""
    try:
        # 1. Load Data
        df = load_csv_dataset(data_path)
        target_col = "target_label" if "target_label" in df.columns else "target"
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        x_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        x_test, y_test = X.iloc[test_idx].copy(), y.iloc[test_idx].copy()
        
        # 2. Apply Perturbation
        x_train_cond, y_train_cond = apply_perturbation(x_train, y_train, condition, random_state=seed + fold)
        x_test_cond, y_test_cond = apply_perturbation(x_test, y_test, condition, random_state=seed + fold + 100)
        
        # 3. Preprocess (Impute & Encode)
        # We assume classification for this benchmark
        from sklearn.preprocessing import LabelEncoder
        label_enc = LabelEncoder()
        y_train_enc = label_enc.fit_transform(y_train_cond.astype(str))
        y_test_enc = label_enc.transform(y_test_cond.astype(str))
        
        preprocessor = _build_preprocessor(x_train_cond, encoding="onehot", scale_numeric=True)
        
        x_train_prep = pd.DataFrame(
            _to_dense_array(preprocessor.fit_transform(x_train_cond)),
            columns=preprocessor.get_feature_names_out()
        )
        x_test_prep = pd.DataFrame(
            _to_dense_array(preprocessor.transform(x_test_cond)),
            columns=preprocessor.get_feature_names_out()
        )
        
        # 4. Pipeline A (Raw) & Pipeline B (AutoFE)
        # AutoFE
        try:
            x_train_autofe, x_test_autofe, autofe_meta = _run_autofe(
                x_train_prep, x_test_prep, pd.Series(y_train_enc)
            )
        except Exception as e:
            # If AutoFE fails (e.g., all cols zero variance), fallback to raw
            x_train_autofe, x_test_autofe = x_train_prep, x_test_prep
            autofe_meta = {"generation_time_s": 0, "selection_time_s": 0, "error": str(e)}
        
        pipelines = {
            "Raw": (x_train_prep, x_test_prep),
            "AutoFE": (x_train_autofe, x_test_autofe)
        }
        
        results = []
        for pipeline_name, (X_tr, X_te) in pipelines.items():
            
            # Sub-sample data to smaller dtypes to save memory inside worker
            X_tr = X_tr.astype(np.float32)
            X_te = X_te.astype(np.float32)
            
            for model_type in model_types:
                res = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "fold": fold,
                    "condition": condition,
                    "pipeline": pipeline_name,
                    "model": model_type,
                }
                
                # Training
                t0 = time.time()
                # Explicitly set n_jobs=1 for models so parallel workers don't thrash CPU
                model = build_model(model_type, random_state=seed, n_jobs_override=1)
                
                try:
                    model.fit(X_tr, y_train_enc)
                    train_time = time.time() - t0
                    
                    # Inference
                    t1 = time.time()
                    y_pred = model.predict(X_te)
                    
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_te)
                    else:
                        y_proba = np.zeros((len(y_test_enc), len(label_enc.classes_)))
                        
                    infer_time = time.time() - t1
                    
                    # Metrics
                    metrics = compute_classification_metrics(y_test_enc, y_pred, y_proba)
                    
                    # SHAP Computation (Only once per dataset/pipeline/model to save time)
                    shap_results = {}
                    if fold == 1 and condition == "clean":
                        try:
                            shap_vals = compute_shap_values(model, X_tr, X_te, model_type)
                            shap_results = {"shap_importance": shap_vals}
                        except Exception:
                            pass
                            
                    res.update({
                        "status": "success",
                        "train_time_s": train_time,
                        "infer_time_s": infer_time,
                        "autofe_gen_time_s": autofe_meta.get("generation_time_s", 0) if pipeline_name == "AutoFE" else 0,
                        "autofe_sel_time_s": autofe_meta.get("selection_time_s", 0) if pipeline_name == "AutoFE" else 0,
                        **metrics,
                        **shap_results
                    })
                except Exception as e:
                    res.update({
                        "status": "error",
                        "error_message": str(e)
                    })
                    
                results.append(res)
                
        return results
        
    except Exception as e:
        # Catch-all to prevent worker from crashing silently
        return [{
            "dataset": dataset_name,
            "seed": seed,
            "fold": fold,
            "condition": condition,
            "status": "fatal_error",
            "error_message": traceback.format_exc()
        }]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=1, help="Number of concurrent processes")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-conditions", type=int, default=None)
    args = parser.parse_args()

    # 1. Setup Logging & Output streaming
    logger = setup_logger("reports/terminal.log")
    json_path = Path("reports/tables/results_stream.jsonl")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Config
    datasets = load_dataset_names("config/dataset_list.yaml")
    if args.max_datasets:
        datasets = datasets[:args.max_datasets]
        
    seeds = [42, 123, 456, 789, 2025]
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]
        
    conditions = [
        "clean",
        "gaussian_0.01", "gaussian_0.05", "gaussian_0.10",
        "label_0.05", "label_0.10", "label_0.20",
        "missing_0.05", "missing_0.10", "missing_0.20"
    ]
    if args.max_conditions:
        conditions = conditions[:args.max_conditions]
    
    model_types = [
        "logistic_regression", "random_forest", "extra_trees",
        "xgboost", "lightgbm", "catboost", "svm", "knn", "gaussian_nb", "mlp"
    ]
    
    # 3. Generate Tasks
    tasks = []
    logger.info("Generating tasks for cross-validation...")
    for dataset in datasets:
        data_path = Path(f"data/raw/{dataset}.csv")
        if not data_path.exists():
            logger.warning(f"Dataset {dataset} missing in data/raw/, skipping.")
            continue
            
        df = pd.read_csv(data_path)
        target_col = "target_label" if "target_label" in df.columns else "target"
        if target_col not in df.columns:
            logger.warning(f"Target column missing in {dataset}, skipping.")
            continue
            
        y = df[target_col]
        
        for seed in seeds:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            stratify_y = _get_stratify_target(y)
            # Fallback to unstratified if classes are too small
            if stratify_y is None:
                from sklearn.model_selection import KFold
                skf = KFold(n_splits=5, shuffle=True, random_state=seed)
                splits = skf.split(df)
            else:
                splits = skf.split(df, stratify_y)
                
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                if args.max_folds and fold > args.max_folds:
                    break
                for condition in conditions:
                    tasks.append({
                        "dataset_name": dataset,
                        "seed": seed,
                        "fold": fold,
                        "condition": condition,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "data_path": str(data_path),
                        "model_types": model_types
                    })
                    
    total_tasks = len(tasks)
    logger.info(f"Total CV units to evaluate: {total_tasks}")
    
    # 4. Job Sequencing
    # Ensure at least 1 core is free by bounding max_workers
    available_cores = multiprocessing.cpu_count()
    if args.n_workers > 1 and args.n_workers >= available_cores:
        n_workers = max(1, available_cores - 1)
        logger.info(f"Capping workers to {n_workers} to leave 1 background core free.")
    else:
        n_workers = args.n_workers
        
    logger.info(f"Starting ProcessPoolExecutor with {n_workers} workers...")
    
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {
            executor.submit(evaluate_unit, **kwargs): kwargs 
            for kwargs in tasks
        }
        
        with json_path.open("a", encoding="utf-8") as f:
            for future in as_completed(future_to_task):
                task_kwargs = future_to_task[future]
                completed += 1
                try:
                    results = future.result()
                    for res in results:
                        f.write(json.dumps(res) + "\n")
                    f.flush() # Stream to disk
                    
                    if completed % 10 == 0 or completed == total_tasks:
                        logger.info(
                            f"[{completed}/{total_tasks}] "
                            f"{task_kwargs['dataset_name']} (seed={task_kwargs['seed']}, "
                            f"fold={task_kwargs['fold']}, cond={task_kwargs['condition']}) completed."
                        )
                except Exception as exc:
                    logger.error(
                        f"Task generated an exception: {exc}\n"
                        f"Dataset: {task_kwargs['dataset_name']}, Cond: {task_kwargs['condition']}"
                    )
                    
    logger.info("Benchmark pipeline fully completed.")

if __name__ == "__main__":
    main()
