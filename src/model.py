"""Model factory for the 10 benchmark models.

All sklearn models use n_jobs=1 because we parallelize at the task level
(multiprocessing Pool). Setting n_jobs>1 here would cause thread thrashing.

GPU models (XGBoost, CatBoost) are configured to use CUDA automatically.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def build_model(
    model_type: str,
    random_state: int = 42,
    use_gpu: bool = True,
) -> BaseEstimator:
    """
    Build one of the 10 benchmark models with sensible defaults.

    All models use n_jobs=1 to prevent CPU thread thrashing when used
    inside a multiprocessing Pool. GPU models (XGBoost, CatBoost) are
    configured to use CUDA when use_gpu=True.
    """
    normalized = model_type.strip().lower()

    if normalized == "logistic_regression":
        return LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            n_jobs=1,
        )

    elif normalized == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        )

    elif normalized == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        )

    elif normalized == "xgboost":
        from xgboost import XGBClassifier
        params = dict(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )
        if use_gpu:
            params["device"] = "cuda"
        return XGBClassifier(**params)

    elif normalized == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )

    elif normalized == "catboost":
        from catboost import CatBoostClassifier
        params = dict(
            iterations=100,
            random_state=random_state,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        if use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"
        return CatBoostClassifier(**params)

    elif normalized == "linear_svm":
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base_svm = LinearSVC(
            random_state=random_state,
            max_iter=1000,
            dual="auto",
        )
        return CalibratedClassifierCV(base_svm, cv=3)

    elif normalized == "knn":
        return KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=1,
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
