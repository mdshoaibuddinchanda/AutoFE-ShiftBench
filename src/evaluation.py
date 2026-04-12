"""Evaluation and aggregation utilities for benchmark experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_pred, task: str = "classification") -> dict[str, float]:
    """Evaluate predictions with task-specific metrics."""
    normalized_task = task.strip().lower()
    if normalized_task == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
    if normalized_task == "regression":
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred)),
        }
    raise ValueError("Task must be 'classification' or 'regression'")


def compute_roc_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Compute ROC-AUC for binary or multiclass outputs."""
    if y_proba.ndim != 2:
        raise ValueError("y_proba must be a 2D array with class probabilities")

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return float("nan")

    if y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))

    return float(
        roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro",
        )
    )


def aggregate_final_results(
    final_results_path: str | Path = "reports/tables/final_results.csv",
    output_path: str | Path = "reports/tables/aggregated_results.csv",
) -> pd.DataFrame:
    """Aggregate experiment outcomes into dataset-level severity summaries."""
    input_path = Path(final_results_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Final results file not found: {input_path}")

    results = pd.read_csv(input_path)
    required_cols = {"dataset", "seed", "shift_type", "severity", "pipeline", "roc_auc"}
    missing_cols = required_cols.difference(results.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns in final results: {sorted(missing_cols)}")

    severity_stats = (
        results.groupby(["dataset", "pipeline", "severity"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
        )
        .sort_values(["dataset", "pipeline", "severity"])
    )

    dataset_stats = (
        results.groupby(["dataset", "pipeline"], as_index=False)
        .agg(
            dataset_mean_roc_auc=("roc_auc", "mean"),
            dataset_std_roc_auc=("roc_auc", "std"),
        )
    )

    baseline = (
        severity_stats.sort_values("severity")
        .groupby(["dataset", "pipeline"], as_index=False)
        .first()[["dataset", "pipeline", "mean_roc_auc"]]
        .rename(columns={"mean_roc_auc": "baseline_roc_auc_s020"})
    )

    aggregated = severity_stats.merge(
        dataset_stats,
        on=["dataset", "pipeline"],
        how="left",
    ).merge(
        baseline,
        on=["dataset", "pipeline"],
        how="left",
    )
    aggregated["avg_degradation_from_s020"] = (
        aggregated["baseline_roc_auc_s020"] - aggregated["mean_roc_auc"]
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(out_path, index=False)
    return aggregated
