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


def _probability_label_indices(y_true, unique_classes):
    """Use the same original-label/encoded-label mapping in both AUC paths."""
    y_values = np.asarray(y_true)
    class_lookup = {value: index for index, value in enumerate(unique_classes.tolist())}
    try:
        return np.asarray([class_lookup[value] for value in y_values], dtype=int)
    except (KeyError, TypeError):
        if np.issubdtype(y_values.dtype, np.integer) and np.all((y_values >= 0) & (y_values < len(unique_classes))):
            return y_values.astype(int)
        raise ValueError("y_true labels do not match the supplied probability class order")


def _auc_metrics(y_indices, y_proba, unique_classes):
    """Preserve the historical joint ROC/PR exception boundary."""
    labels = np.arange(len(unique_classes), dtype=int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(unique_classes) == 2:
                roc = float(roc_auc_score(y_indices, y_proba[:, 1]))
                pr = float(average_precision_score(y_indices, y_proba[:, 1]))
            else:
                roc = float(roc_auc_score(y_indices, y_proba, labels=labels, multi_class="ovr", average="macro"))
                pr_scores = [average_precision_score((y_indices == i).astype(int), y_proba[:, i])
                             for i in labels if (y_indices == i).sum() > 0]
                pr = float(np.mean(pr_scores)) if pr_scores else np.nan
        return {"roc_auc": roc, "pr_auc": pr}
    except Exception:
        return {"roc_auc": np.nan, "pr_auc": np.nan}


def compute_training_roc_auc(y_true, y_proba, classes=None):
    """Compute only the retained training metric, without hard predictions.

    PR-AUC remains part of the shared exception guard so an exceptional input
    cannot turn a formerly undefined training ROC-AUC into a finite value.
    No unused classification, log-loss or Brier computations are performed.
    """
    unique_classes = np.asarray(classes) if classes is not None else np.unique(y_true)
    if len(unique_classes) < 2 or y_proba is None or y_proba.size == 0:
        return np.nan
    indices = _probability_label_indices(y_true, unique_classes)
    return _auc_metrics(indices, y_proba, unique_classes)["roc_auc"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute 10 classification metrics for the benchmark.
    
    Supports both Binary and Multiclass tasks automatically.
    """
    if classes is not None:
        unique_classes = np.asarray(classes)
    else:
        unique_classes = np.unique(y_true)
        
    is_binary = len(unique_classes) == 2
    
    metrics = {}
    
    # Standard metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    
    # Averages for multi-class support
    avg_type = "binary" if is_binary else "macro"
    
    metrics["precision"] = float(precision_score(y_true, y_pred, average=avg_type, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=avg_type, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average=avg_type, zero_division=0))
    
    # Map labels to probability-column positions. The runner passes integer
    # encoded y with the LabelEncoder's original string classes; accept either
    # representation without mixing them in log-loss/Brier calculations.
    if len(unique_classes) < 2 or y_proba is None or y_proba.size == 0:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["log_loss"] = np.nan
        metrics["brier_score"] = np.nan
        return metrics

    y_indices = _probability_label_indices(y_true, unique_classes)
    probability_labels = np.arange(len(unique_classes), dtype=int)

    # Log Loss
    try:
        metrics["log_loss"] = float(log_loss(y_indices, y_proba, labels=probability_labels))
    except Exception:
        metrics["log_loss"] = np.nan
        
    # Brier Score (only standard for binary, but we can compute average Brier for multiclass)
    try:
        one_hot = np.eye(len(unique_classes), dtype=float)[y_indices]
        if is_binary:
            metrics["brier_score"] = float(brier_score_loss(y_indices, y_proba[:, 1]))
        else:
            metrics["brier_score"] = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))
    except Exception:
        metrics["brier_score"] = np.nan
        
    metrics.update(_auc_metrics(y_indices, y_proba, unique_classes))

    return metrics

from scipy.stats import wasserstein_distance, ks_2samp

def compute_distribution_distance(x_clean: pd.DataFrame, x_shifted: pd.DataFrame, max_samples: int = 5000) -> dict[str, float]:
    """Compute average Wasserstein and KS distance between clean and shifted test sets."""
    if x_clean.empty or x_shifted.empty:
        return {'wasserstein': np.nan, 'ks_stat': np.nan}

    # Sample rows to keep compute feasible
    if len(x_clean) > max_samples:
        x_clean = x_clean.sample(n=max_samples, random_state=42)
        x_shifted = x_shifted.sample(n=max_samples, random_state=42)

    # Ensure we only compare common numeric columns
    cols = [c for c in x_clean.columns if c in x_shifted.columns and pd.api.types.is_numeric_dtype(x_clean[c])]
    if not cols:
        return {'wasserstein': np.nan, 'ks_stat': np.nan}

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
