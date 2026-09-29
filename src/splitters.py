"""Split construction with explicit target-free domain partition inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _assert_target_absent(X: pd.DataFrame, target_column: str) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Splitters require an explicit pandas feature matrix X")
    if not target_column:
        raise ValueError("The resolved target column name is required for split validation")
    if target_column in X.columns:
        raise AssertionError(f"Target column {target_column!r} must be removed before split construction")


def assert_fold_integrity(splits, n_samples: int, *, held_out_once: bool = True) -> None:
    """Assert train/test disjointness, valid bounds, and optional CV coverage."""
    test_counts = np.zeros(n_samples, dtype=np.int64)
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        if np.intersect1d(train_idx, test_idx).size:
            raise AssertionError(f"Train/test indices overlap in fold {fold_no}")
        if ((train_idx < 0) | (train_idx >= n_samples)).any() or ((test_idx < 0) | (test_idx >= n_samples)).any():
            raise AssertionError(f"Out-of-range row index in fold {fold_no}")
        test_counts[test_idx] += 1
    if held_out_once and not np.all(test_counts == 1):
        raise AssertionError("Each sample must be held out exactly once in this split repetition")


def get_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    seed: int,
    target_column: str,
):
    """Make ordinary outer folds; labels are used only for stratification."""
    _assert_target_absent(X, target_column)
    if len(X) != len(y):
        raise ValueError("X and y must have equal row counts for splitting")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("X and y indices must match for stratified splitting")
    value_counts = y.value_counts(dropna=False)
    if len(value_counts) < 2 or value_counts.min() < n_splits:
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X))
    else:
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
    assert_fold_integrity(splits, len(X))
    return splits


def get_covariate_splits(
    X: pd.DataFrame,
    n_splits: int,
    seed: int,
    target_column: str,
):
    """Target-free, transductive covariate partitions based on global feature PCA.

    The held-out feature rows contribute to scaling and PCA because the complete
    X table defines the domain-partition stress test. This is not a training-only
    corruption experiment or an estimate for an unseen deployment domain.
    """
    _assert_target_absent(X, target_column)
    numeric = X.select_dtypes(include=["number", "bool"]).fillna(0)
    if numeric.shape[1] == 0:
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X))
    else:
        scaled = StandardScaler().fit_transform(numeric)
        pc1 = PCA(n_components=1, random_state=seed).fit_transform(scaled).ravel()
        ordered = np.argsort(pc1, kind="mergesort")
        bins = np.array_split(ordered, n_splits)
        splits = [
            (np.concatenate([part for j, part in enumerate(bins) if j != i]), bins[i])
            for i in range(n_splits)
        ]
    assert_fold_integrity(splits, len(X))
    return splits


def get_population_splits(
    X: pd.DataFrame,
    n_splits: int,
    seed: int,
    target_column: str,
):
    """Target-free, transductive feature-cluster partitions for a separate study."""
    _assert_target_absent(X, target_column)
    numeric = X.select_dtypes(include=["number", "bool"]).fillna(0)
    if numeric.shape[1] == 0:
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X))
    else:
        scaled = StandardScaler().fit_transform(numeric)
        cluster_ids = KMeans(n_clusters=n_splits, random_state=seed, n_init=10).fit_predict(scaled)
        if len(np.unique(cluster_ids)) != n_splits:
            # Keep the held-out-once invariant if duplicate feature rows collapse a cluster.
            splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X))
        else:
            splits = [
                (np.flatnonzero(cluster_ids != cluster), np.flatnonzero(cluster_ids == cluster))
                for cluster in range(n_splits)
            ]
    assert_fold_integrity(splits, len(X))
    return splits
