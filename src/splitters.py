"""Fold construction with explicit target and evaluation boundaries.

Stratified folds use ``y`` only to balance class counts. Covariate and
population folds use the complete predictor matrix ``X`` to define an
unsupervised, dataset-wide geometry before assigning folds. That transductive
use of predictor values is part of these stress-test definitions; labels never
enter their PCA or clustering geometry. These conditions are not estimates of
deployment performance on an unseen population.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class SplitInfeasibleError(ValueError):
    """Raised when the requested split policy cannot be satisfied as stated."""


def _validate_split_dimensions(n_rows: int, n_splits: int) -> None:
    if n_splits < 2:
        raise SplitInfeasibleError("n_splits must be at least 2")
    if n_rows < n_splits:
        raise SplitInfeasibleError(
            f"Cannot create {n_splits} folds from {n_rows} rows"
        )


def assert_fold_integrity(
    splits: Sequence[tuple[Sequence[int], Sequence[int]]],
    n_rows: int,
    groups: Sequence[object] | None = None,
) -> None:
    """Validate row coverage, train/test disjointness, and optional group isolation."""
    if not splits:
        raise SplitInfeasibleError("The split policy produced no folds")
    _validate_split_dimensions(n_rows, len(splits))

    row_ids = np.arange(n_rows)
    test_parts: list[np.ndarray] = []
    group_values = None if groups is None else np.asarray(groups, dtype=object)
    if group_values is not None and len(group_values) != n_rows:
        raise SplitInfeasibleError("groups length does not match the number of rows")

    for fold_number, (train_indices, test_indices) in enumerate(splits, start=1):
        train = np.asarray(train_indices, dtype=int)
        test = np.asarray(test_indices, dtype=int)
        if train.ndim != 1 or test.ndim != 1:
            raise SplitInfeasibleError(f"Fold {fold_number} indices must be one-dimensional")
        if train.size == 0 or test.size == 0:
            raise SplitInfeasibleError(f"Fold {fold_number} has an empty train or test partition")
        if np.unique(train).size != train.size or np.unique(test).size != test.size:
            raise SplitInfeasibleError(f"Fold {fold_number} contains duplicate row indices")
        if np.any(train < 0) or np.any(test < 0) or np.any(train >= n_rows) or np.any(test >= n_rows):
            raise SplitInfeasibleError(f"Fold {fold_number} contains an out-of-range row index")
        if np.intersect1d(train, test).size:
            raise SplitInfeasibleError(f"Fold {fold_number} has overlapping train and test rows")
        if np.union1d(train, test).size != n_rows:
            raise SplitInfeasibleError(f"Fold {fold_number} does not partition all rows")
        if group_values is not None:
            train_groups = set(group_values[train].tolist())
            test_groups = set(group_values[test].tolist())
            if train_groups.intersection(test_groups):
                raise SplitInfeasibleError(
                    f"Fold {fold_number} places the same group in train and test"
                )
        test_parts.append(test)

    all_test = np.concatenate(test_parts)
    if np.unique(all_test).size != n_rows or not np.array_equal(np.sort(all_test), row_ids):
        raise SplitInfeasibleError("Test folds must cover every row exactly once")


def get_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create shuffled stratified folds, failing explicitly if class counts are too small."""
    _validate_split_dimensions(len(X), n_splits)
    if len(y) != len(X):
        raise SplitInfeasibleError("X and y lengths differ")
    counts = pd.Series(y).value_counts(dropna=False)
    if counts.empty or int(counts.min()) < n_splits:
        smallest = int(counts.min()) if not counts.empty else 0
        raise SplitInfeasibleError(
            f"Stratified {n_splits}-fold CV requires at least {n_splits} rows per class; "
            f"the smallest class has {smallest}. No random-fold fallback was applied."
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y))
    assert_fold_integrity(splits, len(X))
    return splits


def _scaled_numeric_predictors(X: pd.DataFrame) -> np.ndarray:
    """Return numeric predictor geometry, rejecting nonnumeric-only inputs."""
    numeric = X.select_dtypes(include=["number"]).fillna(0)
    if numeric.shape[1] == 0:
        raise SplitInfeasibleError(
            "Feature-based partitioning requires at least one numeric predictor; "
            "no alternate random split was substituted."
        )
    return StandardScaler().fit_transform(numeric)


def get_covariate_splits(
    X: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition by contiguous bins of the first PC of predictor-only geometry.

    PCA and scaling use all rows of ``X`` to define this stress-test partition.
    The function accepts no target argument, so labels cannot affect its folds.
    """
    _validate_split_dimensions(len(X), n_splits)
    scaled = _scaled_numeric_predictors(X)
    pc1 = PCA(n_components=1, random_state=seed).fit_transform(scaled).ravel()
    sorted_indices = np.argsort(pc1, kind="mergesort")
    bins = np.array_split(sorted_indices, n_splits)
    splits = [
        (
            np.concatenate([bins[j] for j in range(n_splits) if j != fold]),
            bins[fold],
        )
        for fold in range(n_splits)
    ]
    assert_fold_integrity(splits, len(X))
    return splits


def get_population_splits(
    X: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out one predictor-only K-means cluster per fold.

    Clustering uses all rows of ``X`` to define this unsupervised stress-test
    partition. Empty clusters are an explicit infeasibility; no random K-fold
    replacement is allowed.
    """
    _validate_split_dimensions(len(X), n_splits)
    scaled = _scaled_numeric_predictors(X)
    labels = KMeans(n_clusters=n_splits, random_state=seed, n_init="auto").fit_predict(scaled)
    cluster_sizes = np.bincount(labels, minlength=n_splits)
    if np.any(cluster_sizes == 0):
        raise SplitInfeasibleError(
            f"K-means produced an empty cluster for {n_splits} population folds"
        )

    row_ids = np.arange(len(X))
    splits = [
        (row_ids[labels != cluster], row_ids[labels == cluster])
        for cluster in range(n_splits)
    ]
    assert_fold_integrity(splits, len(X))
    return splits


def get_splits(
    X: pd.DataFrame,
    y: pd.Series,
    split_policy: str,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dispatch an explicit split policy using target labels only for stratification."""
    if split_policy == "stratified":
        return get_stratified_splits(X, y, n_splits, seed)
    if split_policy == "covariate_shift":
        return get_covariate_splits(X, n_splits, seed)
    if split_policy == "population_shift":
        return get_population_splits(X, n_splits, seed)
    raise ValueError(f"Unknown split policy: {split_policy}")
