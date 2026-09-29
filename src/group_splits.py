"""Deterministic exact-feature groups and group-aware stratified folds.

This module is deliberately separate from :mod:`src.splitters`.  The existing
row-level splitters remain available for legacy-comparable results; the
functions here define the second, group-aware evaluation track.  Groups are
formed from the unperturbed raw predictor table after the target column has
been removed.

The row key is a canonical, typed JSON representation.  Numeric values use a
normalised decimal representation, missing values share one sentinel, and
text/category values keep their text/type distinction.  Group IDs are SHA-256
digests of those keys.  A digest collision is checked against the complete
canonical keys before IDs are used for splitting.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


_MISSING = "<MISSING>"


class GroupSplitInfeasibleError(ValueError):
    """Raised when a requested group-aware fold plan cannot be supported."""

    def __init__(self, message: str, status: Mapping[str, Any]):
        super().__init__(message)
        self.status = dict(status)


def _dtype_family(dtype: Any) -> str:
    """Return a stable logical type family for a dataframe column."""
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(dtype, pd.DatetimeTZDtype):
        return "datetime"
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if pd.api.types.is_string_dtype(dtype):
        return "text"
    return f"dtype:{str(dtype)}"


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not isinstance(result, (list, tuple, np.ndarray, pd.Series)) else False
    except (TypeError, ValueError):
        return False


def _normalise_number(value: Any) -> str:
    """Normalise integer/float/decimal values while preserving special values."""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if np.isnan(value):
            return _MISSING
        if np.isposinf(value):
            return "+inf"
        if np.isneginf(value):
            return "-inf"
    try:
        # Decimal(str(...)) treats 1, 1.0, and scientific notation as the same
        # numeric value while retaining meaningful precision from the source.
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def _canonical_cell(value: Any, family: str) -> list[str]:
    """Encode one scalar as a JSON-safe typed token."""
    if _is_missing(value):
        return ["missing"]
    if family == "number" or isinstance(value, (int, float, np.integer, np.floating, Decimal)) and not isinstance(value, (bool, np.bool_)):
        return ["number", _normalise_number(value)]
    if family == "bool" or isinstance(value, (bool, np.bool_)):
        return ["bool", "1" if bool(value) else "0"]
    if family == "datetime" or isinstance(value, (pd.Timestamp, np.datetime64)):
        return ["datetime", pd.Timestamp(value).isoformat()]
    if family in {"text", "category"} or isinstance(value, str):
        return [family if family in {"text", "category"} else "text", str(value)]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    # Tabular benchmark values should be scalar.  The type-qualified repr is a
    # deterministic fallback for unusual object columns and avoids silently
    # treating e.g. a tuple as equivalent to a string.
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    return [value_type, repr(value)]


def _schema_token(features: pd.DataFrame) -> list[list[str]]:
    return [[str(column), _dtype_family(dtype)] for column, dtype in zip(features.columns, features.dtypes)]


def canonical_feature_rows(features: pd.DataFrame) -> list[str]:
    """Return deterministic canonical keys for target-excluded predictor rows.

    Column order, column names, and logical dtype families are part of the key.
    Missing scalar values are equal across pandas missing representations.  In
    numeric columns, ``1`` and ``1.0`` represent the same numeric value, while
    a text value ``"1"`` remains distinct from numeric one.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    schema = _schema_token(features)
    families = [_dtype_family(dtype) for dtype in features.dtypes]
    keys: list[str] = []
    for row in features.itertuples(index=False, name=None):
        encoded = [_canonical_cell(value, family) for value, family in zip(row, families)]
        keys.append(json.dumps({"schema": schema, "row": encoded}, ensure_ascii=False, separators=(",", ":")))
    return keys


def canonical_feature_group_ids(features: pd.DataFrame) -> np.ndarray:
    """Return SHA-256 group IDs for canonical predictor rows."""
    keys = canonical_feature_rows(features)
    group_ids = np.asarray([hashlib.sha256(key.encode("utf-8")).hexdigest() for key in keys], dtype=object)
    assert_no_group_hash_collisions(group_ids, keys)
    return group_ids


def assert_no_group_hash_collisions(group_ids: Sequence[Any], canonical_keys: Sequence[str]) -> None:
    """Ensure no digest maps to two different complete canonical row keys."""
    if len(group_ids) != len(canonical_keys):
        raise ValueError("group_ids and canonical_keys must have equal lengths")
    seen: dict[str, str] = {}
    for group_id, key in zip(group_ids, canonical_keys):
        digest = str(group_id)
        previous = seen.get(digest)
        if previous is not None and previous != key:
            raise AssertionError(f"Canonical group hash collision detected for digest {digest!r}")
        seen[digest] = key


def _validate_group_inputs(group_ids: Sequence[Any], y: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = np.asarray(group_ids, dtype=object)
    labels = np.asarray(y, dtype=object)
    if groups.ndim != 1 or labels.ndim != 1:
        raise ValueError("group_ids and y must be one-dimensional")
    if len(groups) != len(labels):
        raise ValueError("group_ids and y must have equal lengths")
    label_tokens = [json.dumps(_canonical_cell(value, "object"), separators=(",", ":")) for value in labels]
    return groups, labels, label_tokens


def summarize_groups(group_ids: Sequence[Any], y: Sequence[Any]) -> dict[str, Any]:
    """Summarise group sizes, class membership, and conflicting-label groups."""
    groups, labels, label_tokens = _validate_group_inputs(group_ids, y)
    members: dict[str, list[int]] = defaultdict(list)
    group_labels: dict[str, set[str]] = defaultdict(set)
    for index, (group, token) in enumerate(zip(groups, label_tokens)):
        group_key = str(group)
        members[group_key].append(index)
        group_labels[group_key].add(token)
    sizes = [len(indices) for indices in members.values()]
    conflicts = sorted(group for group, tokens in group_labels.items() if len(tokens) > 1)
    class_counts = Counter(label_tokens)
    class_group_counts = Counter()
    for group, tokens in group_labels.items():
        for token in tokens:
            class_group_counts[token] += 1
    return {
        "n_rows": int(len(groups)),
        "n_groups": int(len(members)),
        "group_size_min": int(min(sizes)) if sizes else 0,
        "group_size_max": int(max(sizes)) if sizes else 0,
        "group_size_mean": float(np.mean(sizes)) if sizes else 0.0,
        "group_size_median": float(np.median(sizes)) if sizes else 0.0,
        "group_size_histogram": {str(size): int(sizes.count(size)) for size in sorted(set(sizes))},
        "conflicting_label_group_count": int(len(conflicts)),
        "conflicting_label_group_ids": conflicts,
        "class_row_counts": {token: int(count) for token, count in sorted(class_counts.items())},
        "class_group_counts": {token: int(count) for token, count in sorted(class_group_counts.items())},
        "class_count": int(len(class_counts)),
    }


def _candidate_splits(groups: np.ndarray, labels: np.ndarray, n_splits: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Use a simple index matrix so pandas index labels cannot affect fold IDs.
    indices = np.arange(len(labels), dtype=np.int64)
    # Encode labels because sklearn treats an object array of numpy scalar
    # values as ``unknown`` even when it is semantically binary/multiclass.
    encoded_labels, _ = pd.factorize(pd.Series(labels), sort=False)
    return [(train.astype(np.int64), test.astype(np.int64)) for train, test in splitter.split(indices, encoded_labels, groups)]


def assess_group_fold_feasibility(
    group_ids: Sequence[Any],
    y: Sequence[Any],
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    """Return explicit split, class-support, and ROC-AUC feasibility status.

    A group plan is split-feasible when it has enough groups and the installed
    sklearn splitter can construct disjoint folds.  Class support is assessed
    fold by fold.  ROC-AUC is marked supported only when every train and test
    fold contains every global class (and at least two classes); callers can
    still use the folds for metrics that do not require that condition.
    """
    groups, labels, label_tokens = _validate_group_inputs(group_ids, y)
    summary = summarize_groups(groups, labels)
    structural_reasons: list[str] = []
    metric_reasons: list[str] = []
    if n_splits < 2:
        structural_reasons.append("n_splits_must_be_at_least_two")
    if len(groups) == 0:
        structural_reasons.append("empty_dataset")
    if summary["n_groups"] < n_splits:
        structural_reasons.append("fewer_groups_than_splits")
    if any(_is_missing(value) for value in labels):
        structural_reasons.append("missing_target_labels")
    if summary["class_count"] < 2:
        metric_reasons.append("fewer_than_two_target_classes")
    class_group_counts = summary["class_group_counts"]
    if class_group_counts and min(class_group_counts.values()) < n_splits:
        metric_reasons.append("a_target_class_occurs_in_fewer_groups_than_splits")
    class_row_counts = summary["class_row_counts"]
    if class_row_counts and min(class_row_counts.values()) < n_splits:
        metric_reasons.append("a_target_class_has_fewer_rows_than_splits")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    if not structural_reasons:
        try:
            splits = _candidate_splits(groups, labels, n_splits, seed)
        except Exception as exc:
            structural_reasons.append(f"sklearn_splitter_error:{type(exc).__name__}:{exc}")

    global_classes = set(label_tokens)
    fold_class_support: list[dict[str, Any]] = []
    if splits:
        for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
            train_classes = set(label_tokens[index] for index in train_idx)
            test_classes = set(label_tokens[index] for index in test_idx)
            fold_class_support.append({
                "fold": fold_no,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_class_count": int(len(train_classes)),
                "test_class_count": int(len(test_classes)),
                "train_has_all_classes": bool(train_classes == global_classes),
                "test_has_all_classes": bool(test_classes == global_classes),
                "test_has_at_least_two_classes": bool(len(test_classes) >= 2),
            })
        if any(not item["test_has_all_classes"] for item in fold_class_support):
            metric_reasons.append("one_or_more_test_folds_missing_a_target_class")
        class_support_status = "supported" if all(
            item["train_has_all_classes"] and item["test_has_all_classes"] for item in fold_class_support
        ) else "infeasible"
        auc_status = "supported" if class_support_status == "supported" and all(
            item["test_has_at_least_two_classes"] for item in fold_class_support
        ) else "infeasible"
    else:
        class_support_status = "infeasible"
        auc_status = "infeasible"

    return {
        **summary,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "split_feasible": bool(splits) and not structural_reasons,
        "class_support_status": class_support_status,
        "auc_status": auc_status,
        "auc_reason": "all_train_and_test_folds_contain_all_classes" if auc_status == "supported" else "at_least_one_fold_cannot_support_multiclass_or_binary_auc",
        "reasons": sorted(set(structural_reasons + metric_reasons)),
        "structural_reasons": sorted(set(structural_reasons)),
        "metric_reasons": sorted(set(metric_reasons)),
        "fold_class_support": fold_class_support,
    }


def assert_zero_shared_groups(
    splits: Iterable[tuple[Sequence[int], Sequence[int]]],
    group_ids: Sequence[Any],
) -> None:
    """Assert no exact-feature group appears in both train and test of a fold."""
    groups = np.asarray(group_ids, dtype=object)
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = {str(groups[int(index)]) for index in train_idx}
        test_groups = {str(groups[int(index)]) for index in test_idx}
        shared = train_groups.intersection(test_groups)
        if shared:
            raise AssertionError(f"Shared exact-feature groups in fold {fold_no}: {sorted(shared)[:3]}")


def assert_group_fold_integrity(
    splits: Iterable[tuple[Sequence[int], Sequence[int]]],
    group_ids: Sequence[Any],
    *,
    held_out_once: bool = True,
) -> None:
    """Assert bounds, row disjointness, held-out coverage, and group disjointness."""
    groups = np.asarray(group_ids, dtype=object)
    normalised = _normalise_splits(splits)
    test_counts = np.zeros(len(groups), dtype=np.int64)
    for fold_no, (train_idx, test_idx) in enumerate(normalised, start=1):
        train = np.asarray(train_idx, dtype=np.int64)
        test = np.asarray(test_idx, dtype=np.int64)
        if np.intersect1d(train, test).size:
            raise AssertionError(f"Train/test row indices overlap in fold {fold_no}")
        if ((train < 0) | (train >= len(groups))).any() or ((test < 0) | (test >= len(groups))).any():
            raise AssertionError(f"Out-of-range row index in fold {fold_no}")
        test_counts[test] += 1
    if held_out_once and not np.all(test_counts == 1):
        raise AssertionError("Each row must be held out exactly once")
    assert_zero_shared_groups(normalised, groups)


def _normalise_splits(splits: Iterable[tuple[Sequence[int], Sequence[int]]]):
    return [(np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)) for train, test in splits]


def get_group_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series | Sequence[Any],
    n_splits: int,
    seed: int,
    target_column: str,
    *,
    require_class_support: bool = True,
    require_auc: bool = False,
    return_status: bool = False,
):
    """Construct deterministic ``StratifiedGroupKFold`` folds from raw X.

    ``X`` must already exclude ``target_column``.  There is no silent fallback
    to row-level KFold.  Set ``return_status=True`` to receive the audit status
    alongside folds; otherwise an infeasible request raises
    :class:`GroupSplitInfeasibleError` with the same status on ``.status``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Group splitters require an explicit pandas feature DataFrame X")
    if not target_column:
        raise ValueError("The resolved target column name is required for split validation")
    if target_column in X.columns:
        raise AssertionError(f"Target column {target_column!r} must be removed before group split construction")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("X and y indices must match for group stratified splitting")
    if len(X) != len(y):
        raise ValueError("X and y must have equal row counts for group splitting")

    group_ids = canonical_feature_group_ids(X)
    status = assess_group_fold_feasibility(group_ids, y, n_splits, seed)
    failed = not status["split_feasible"]
    if require_class_support and status["class_support_status"] != "supported":
        failed = True
    if require_auc and status["auc_status"] != "supported":
        failed = True
    if failed:
        raise GroupSplitInfeasibleError(
            f"Group-aware split is infeasible: {', '.join(status['reasons']) or 'metric support requirement failed'}",
            status,
        )
    splits = _candidate_splits(group_ids, np.asarray(y, dtype=object), n_splits, seed)
    assert_group_fold_integrity(splits, group_ids)
    return (splits, status) if return_status else splits


def audit_raw_dataset(
    csv_path: str | Path,
    *,
    target_column: str = "target",
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Audit one saved raw CSV for exact groups and group-fold feasibility."""
    path = Path(csv_path)
    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        if "target_label" in frame.columns:
            target_column = "target_label"
        else:
            raise ValueError(f"Target column {target_column!r} is absent from {path}")
    X = frame.drop(columns=[target_column])
    y = frame[target_column]
    group_ids = canonical_feature_group_ids(X)
    summary = summarize_groups(group_ids, y)
    status = assess_group_fold_feasibility(group_ids, y, n_splits, seed)
    return {
        "dataset": path.stem,
        "path": str(path),
        "target_column": target_column,
        "feature_count": int(X.shape[1]),
        "summary": summary,
        "fold_status": status,
    }
