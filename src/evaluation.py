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


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """
    Compute 10 classification metrics for the benchmark.
    
    Supports both Binary and Multiclass tasks automatically.
    """
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
    
    # Probabilistic metrics
    if len(unique_classes) < 2 or y_proba is None or y_proba.size == 0:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["log_loss"] = np.nan
        metrics["brier_score"] = np.nan
        return metrics

    # Log Loss
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=unique_classes))
    except Exception:
        metrics["log_loss"] = np.nan
        
    # Brier Score (only standard for binary, but we can compute average Brier for multiclass)
    try:
        if is_binary:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_proba[:, 1]))
        else:
            # Multiclass Brier Score approximation (Brier Score per class, averaged)
            brier_scores = []
            for i, cls in enumerate(unique_classes):
                y_true_binary = (y_true == cls).astype(int)
                brier_scores.append(brier_score_loss(y_true_binary, y_proba[:, i]))
            metrics["brier_score"] = float(np.mean(brier_scores))
    except Exception:
        metrics["brier_score"] = np.nan
        
    # ROC-AUC and PR-AUC
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if is_binary:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
                # PR-AUC multiclass is not natively "macro" in sklearn average_precision_score for labels.
                # Compute OVR PR-AUC manually
                pr_scores = []
                for i, cls in enumerate(unique_classes):
                    y_true_binary = (y_true == cls).astype(int)
                    if y_true_binary.sum() > 0:
                        pr_scores.append(average_precision_score(y_true_binary, y_proba[:, i]))
                metrics["pr_auc"] = float(np.mean(pr_scores)) if pr_scores else np.nan
    except Exception:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

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

