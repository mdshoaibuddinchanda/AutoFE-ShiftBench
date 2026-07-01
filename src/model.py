"""Model factory for the 10 benchmark models."""

from __future__ import annotations

import multiprocessing
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_core_count() -> int:
    """Get the number of CPU cores to avoid thread thrashing."""
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 4


def build_model(
    model_type: str,
    random_state: int = 42,
    n_jobs_override: int | None = None,
) -> BaseEstimator:
    """
    Build one of the 10 benchmark models with sensible defaults.
    
    If n_jobs_override is provided, we use that for parallel models (like RF, XGB)
    to prevent thread thrashing when using multiprocessing.
    """
    normalized = model_type.strip().lower()
    
    # Threading logic: if we are sequencing jobs outside, we want models to use 1 thread.
    # If not specified, we can use a safe default like 4 threads, or all cores.
    n_jobs = n_jobs_override if n_jobs_override is not None else get_core_count()

    if normalized == "logistic_regression":
        return LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            n_jobs=n_jobs,
        )
        
    elif normalized == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        
    elif normalized == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        
    elif normalized == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="logloss",
        )
        
    elif normalized == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
        
    elif normalized == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=100,
            random_state=random_state,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
        )
        
    elif normalized == "svm":
        return SVC(
            kernel="rbf",
            probability=True,  # Required for ROC/PR AUC
            random_state=random_state,
            max_iter=5000, # Bound runtime
        )
        
    elif normalized == "knn":
        return KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=n_jobs,
        )
        
    elif normalized == "gaussian_nb":
        return GaussianNB()
        
    elif normalized == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
