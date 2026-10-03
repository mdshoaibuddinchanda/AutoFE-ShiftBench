"""Comprehensive evaluation metrics for benchmark experiments."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

METRIC_SEMANTICS_VERSION = "encoded_class_probability_metrics_v2"
METRIC_RANGES = {name: (0.0, 1.0) for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc", "pr_auc", "brier_score")}
METRIC_RANGES.update({"mcc": (-1.0, 1.0), "log_loss": (0.0, np.inf)})


class MetricInputError(ValueError):
    """An input violates the declared target/probability coordinate contract."""


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray | None = None,
    *,
    probability_classes: np.ndarray | None = None,
) -> dict:
    """
    Compute 10 classification metrics for the benchmark.
    
    Supports both Binary and Multiclass tasks automatically.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    declared = np.asarray(classes if classes is not None else np.unique(y_true))
    if declared.ndim != 1 or len(set(declared.tolist())) != len(declared):
        raise MetricInputError("class coordinates must be unique and one-dimensional")
    if len(y_true) != len(y_pred):
        raise MetricInputError("target and prediction lengths differ")
    allowed = set(declared.tolist())
    if not set(y_true.tolist()).issubset(allowed) or not set(y_pred.tolist()).issubset(allowed):
        raise MetricInputError("target/prediction class is absent from declared class coordinates")
    # sklearn log_loss orders label coordinates; align to that order explicitly.
    labels = np.unique(declared)
    binary = len(labels) == 2
    names = list(METRIC_RANGES)
    metrics = {name: np.nan for name in names}
    statuses = {name: "undefined_empty_target" for name in names}
    metrics.update({"metric_semantics_version": METRIC_SEMANTICS_VERSION,
        "metric_status": statuses, "f1_averaging": "binary" if binary else "macro",
        "brier_normalization": "positive_class_mean_squared_error" if binary else "mean_over_samples_and_declared_classes"})
    if not len(y_true):
        return metrics
    metrics.update(accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)))
    kwargs = {"average": "binary", "pos_label": labels[1]} if binary else {"average": "macro", "labels": labels}
    metrics["precision"] = float(precision_score(y_true,y_pred,zero_division=0,**kwargs))
    metrics["recall"] = float(recall_score(y_true,y_pred,zero_division=0,**kwargs))
    metrics["f1"] = float(f1_score(y_true,y_pred,zero_division=0,**kwargs))
    metrics["f1_macro"] = float(f1_score(y_true,y_pred,average="macro",labels=labels,zero_division=0))
    for name in ("accuracy","balanced_accuracy","mcc","precision","recall","f1","f1_macro"):
        statuses[name] = "complete"
    for name in ("roc_auc","pr_auc","log_loss","brier_score"):
        statuses[name] = "undefined_no_probabilities" if y_proba is None else "undefined_single_declared_class"
    if y_proba is None:
        return metrics
    probabilities = np.asarray(y_proba)
    columns = np.asarray(probability_classes if probability_classes is not None else declared)
    if probabilities.ndim != 2 or probabilities.shape != (len(y_true), len(columns)):
        raise MetricInputError("probability dimensions do not match rows and class columns")
    if columns.ndim != 1 or len(set(columns.tolist())) != len(columns) or not set(columns.tolist()).issubset(allowed):
        raise MetricInputError("probability class coordinates are incompatible")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise MetricInputError("probabilities must be finite and within [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-7):
        raise MetricInputError("probability rows must sum to one")
    aligned = np.zeros((len(y_true),len(labels)), dtype=probabilities.dtype)
    positions = {label: i for i,label in enumerate(labels.tolist())}
    for i,label in enumerate(columns.tolist()):
        aligned[:,positions[label]] = probabilities[:,i]
    if len(labels) < 2:
        return metrics
    metrics["log_loss"] = float(log_loss(y_true, aligned, labels=labels))
    if binary:
        metrics["brier_score"] = float(brier_score_loss(y_true == labels[1], aligned[:,1]))
    else:
        metrics["brier_score"] = float(np.mean([brier_score_loss(y_true == label, aligned[:,i]) for i,label in enumerate(labels)]))
    statuses["log_loss"] = statuses["brier_score"] = "complete"
    if len(np.unique(y_true)) != len(labels):
        statuses["roc_auc"] = statuses["pr_auc"] = "undefined_missing_held_out_class"
        return metrics
    if binary:
        metrics["roc_auc"] = float(roc_auc_score(y_true == labels[1],aligned[:,1]))
        metrics["pr_auc"] = float(average_precision_score(y_true == labels[1],aligned[:,1]))
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true,aligned,labels=labels,multi_class="ovr",average="macro"))
        metrics["pr_auc"] = float(np.mean([average_precision_score(y_true == label,aligned[:,i]) for i,label in enumerate(labels)]))
    statuses["roc_auc"] = statuses["pr_auc"] = "complete"
    return metrics

from scipy.stats import wasserstein_distance, ks_2samp

DISTRIBUTION_SEMANTICS_VERSION='frozen_mapping_training_corruption_distance_v2'

def compute_distribution_distance(
    x_clean: pd.DataFrame,
    x_shifted: pd.DataFrame,
    max_samples: int = 5000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compare identical numeric coordinates under one frozen fitted mapping.

    Callers declare training corruption or held-out input scope. Independently
    selected/generated coordinate systems are unsupported.
    """
    status={'distribution_semantics_version':DISTRIBUTION_SEMANTICS_VERSION}
    if x_clean.empty or x_shifted.empty:
        return {**status,'distance_status':'unsupported_empty_inputs','wasserstein': None, 'ks_stat': None}
    if list(x_clean.columns) != list(x_shifted.columns):
        return {**status,'distance_status':'unsupported_coordinate_mismatch','wasserstein':None,'ks_stat':None}

    # Sample rows to keep compute feasible
    if len(x_clean) > max_samples:
        x_clean = x_clean.sample(n=max_samples, random_state=random_state)
    if len(x_shifted) > max_samples:
        x_shifted = x_shifted.sample(n=max_samples, random_state=random_state)

    # Ensure we only compare common numeric columns
    cols = [c for c in x_clean.columns if c in x_shifted.columns and pd.api.types.is_numeric_dtype(x_clean[c])]
    if not cols:
        return {**status,'distance_status':'unsupported_no_numeric_coordinates','wasserstein':None,'ks_stat':None}

    w_dists = []
    ks_dists = []
    for c in cols:
        # Drop NaNs for stability
        c_clean = x_clean[c].dropna().values
        c_shifted = x_shifted[c].dropna().values
        if len(c_clean) > 0 and len(c_shifted) > 0:
            try:
                w_dists.append(wasserstein_distance(c_clean, c_shifted))
                ks_dists.append(ks_2samp(c_clean, c_shifted).statistic)
            except Exception:
                pass

    return {
        **status,'distance_status':'complete' if len(w_dists) == len(cols) else 'incomplete_numeric_coordinates',
        'wasserstein': float(np.mean(w_dists)) if w_dists else np.nan,
        'ks_stat': float(np.mean(ks_dists)) if ks_dists else np.nan,
    }

def compute_jaccard_similarity(list_a: list[str], list_b: list[str]) -> float:
    """Compute Jaccard similarity between two lists of feature names."""
    set_a = set(list_a)
    set_b = set(list_b)
    if not set_a and not set_b:
        return 1.0
    return float(len(set_a.intersection(set_b)) / len(set_a.union(set_b)))
