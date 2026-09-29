from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.group_splits import (
    GroupSplitInfeasibleError,
    assess_group_fold_feasibility,
    assert_group_fold_integrity,
    assert_no_group_hash_collisions,
    canonical_feature_group_ids,
    canonical_feature_rows,
    get_group_stratified_splits,
    summarize_groups,
)


def _grouped_frame() -> tuple[pd.DataFrame, pd.Series]:
    # Eight repeated groups, with two rows per group.  Each class occurs in at
    # least four groups, making a deterministic four-fold plan feasible.
    features = pd.DataFrame({
        "number": [1, 1.0, 2, 2.0, 3, 3.0, 4, 4.0, 5, 5.0, 6, 6.0, 7, 7.0, 8, 8.0],
        "text": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f", "g", "g", "h", "h"],
        "missing": [np.nan, None, "x", "x", np.nan, None, "y", "y", np.nan, None, "z", "z", np.nan, None, "w", "w"],
    })
    target = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], index=features.index)
    return features, target


def test_canonical_rows_normalise_numeric_and_missing_but_preserve_text_type():
    first = pd.DataFrame({
        "number": pd.Series([1, 2], dtype="int64"),
        "text": pd.Series(["1", "x"], dtype="string"),
        "missing": pd.Series([np.nan, "ok"], dtype="object"),
    })
    second = pd.DataFrame({
        "number": pd.Series([1.0, 2.0], dtype="float64"),
        "text": pd.Series(["1", "x"], dtype="string"),
        "missing": pd.Series([None, "ok"], dtype="object"),
    })
    assert canonical_feature_rows(first) == canonical_feature_rows(second)

    numeric_one = pd.DataFrame({"value": [1]})
    text_one = pd.DataFrame({"value": pd.Series(["1"], dtype="string")})
    assert canonical_feature_rows(numeric_one) != canonical_feature_rows(text_one)


def test_hash_collision_check_rejects_digest_reuse_for_different_keys():
    with pytest.raises(AssertionError, match="collision"):
        assert_no_group_hash_collisions(["same", "same"], ["key-a", "key-b"])


def test_group_summary_reports_sizes_and_conflicting_labels():
    group_ids = np.array(["a", "a", "b", "c", "c"])
    labels = np.array([0, 1, 1, 0, 0])
    summary = summarize_groups(group_ids, labels)
    assert summary["n_groups"] == 3
    assert summary["group_size_min"] == 1
    assert summary["group_size_max"] == 2
    assert summary["conflicting_label_group_count"] == 1
    assert summary["conflicting_label_group_ids"] == ["a"]


def test_group_folds_are_deterministic_and_have_zero_shared_groups():
    features, target = _grouped_frame()
    first = get_group_stratified_splits(
        features, target, 4, 42, target_column="target", require_class_support=False
    )
    second = get_group_stratified_splits(
        features, target, 4, 42, target_column="target", require_class_support=False
    )
    assert all(np.array_equal(a, c) and np.array_equal(b, d) for (a, b), (c, d) in zip(first, second))
    assert_group_fold_integrity(first, canonical_feature_group_ids(features))

    for train_idx, test_idx in first:
        train_groups = set(canonical_feature_group_ids(features)[train_idx])
        test_groups = set(canonical_feature_group_ids(features)[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_target_column_must_be_excluded():
    features, target = _grouped_frame()
    with pytest.raises(AssertionError, match="must be removed"):
        get_group_stratified_splits(
            features.assign(target=target), target, 4, 42, target_column="target"
        )


def test_status_exposes_class_support_and_auc_infeasibility():
    features = pd.DataFrame({"value": [0, 0, 1, 2, 3, 4]})
    target = pd.Series([0, 0, 0, 0, 1, 1])
    groups = canonical_feature_group_ids(features)
    status = assess_group_fold_feasibility(groups, target, 3, 42)
    assert status["split_feasible"] is True
    assert status["class_support_status"] == "infeasible"
    assert status["auc_status"] == "infeasible"
    assert "a_target_class_occurs_in_fewer_groups_than_splits" in status["reasons"]
    with pytest.raises(GroupSplitInfeasibleError) as error:
        get_group_stratified_splits(features, target, 3, 42, target_column="target")
    assert error.value.status["auc_status"] == "infeasible"


def test_conflicting_labels_are_reported_but_do_not_break_group_split():
    features = pd.DataFrame({"value": [1, 1, 2, 2, 3, 3, 4, 4]})
    target = pd.Series([0, 1, 0, 0, 1, 1, 0, 0])
    groups = canonical_feature_group_ids(features)
    status = assess_group_fold_feasibility(groups, target, 4, 9)
    assert status["conflicting_label_group_count"] == 1
    assert status["split_feasible"] is True
    # Conflicts are a data property; the group itself remains indivisible.
    splits = get_group_stratified_splits(features, target, 4, 9, target_column="target", require_class_support=False)
    assert_group_fold_integrity(splits, groups)
