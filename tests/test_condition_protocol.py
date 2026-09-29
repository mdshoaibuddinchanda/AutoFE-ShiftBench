from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.group_splits import (
    assert_group_fold_integrity,
    canonical_feature_group_ids,
    get_group_stratified_splits,
)
from src.pipeline_runner import (
    DOMAIN_PARTITION_CONDITIONS,
    PRIMARY_CONDITIONS,
    SEPARATE_EXPERIMENT_CONDITIONS,
    _domain_fold_status,
    prepare_task,
    run_experiment,
)
from src.shift_generator import apply_perturbation
from src.splitters import get_covariate_splits, get_population_splits


def _condition_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(20260930)
    signal = rng.normal(size=n)
    return pd.DataFrame({
        "a": signal,
        "b": rng.normal(size=n),
        "c": np.arange(n, dtype=float),
        "target": (signal > 0).astype(int),
    })


def test_condition_scopes_account_for_primary_and_four_separate_conditions():
    assert len(PRIMARY_CONDITIONS) == 10
    assert len(DOMAIN_PARTITION_CONDITIONS) == 2
    assert len(SEPARATE_EXPERIMENT_CONDITIONS) == 2
    assert {family for family, _ in PRIMARY_CONDITIONS} == {
        "clean", "gaussian_noise", "missing_values", "label_noise",
    }
    assert {family for family, _ in DOMAIN_PARTITION_CONDITIONS} == {
        "covariate_partition", "population_partition",
    }
    assert {family for family, _ in SEPARATE_EXPERIMENT_CONDITIONS} == {
        "feature_availability_ablation", "majority_label_relabeling",
    }
    with pytest.raises(ValueError, match="exactly one scope"):
        run_experiment(
            {}, run_id="mixed-condition-scope", conditions=(
                ("clean", 0.0), ("covariate_partition", 0.0),
            ), pipelines=("Raw",), models=("logistic_regression",),
        )


def test_transductive_partitions_are_target_free_and_stable_to_label_permutation():
    frame = _condition_frame()
    X = frame.drop(columns=["target"])
    first_y = frame["target"]
    permuted_y = first_y.sample(frac=1.0, random_state=11).reset_index(drop=True)
    for splitter in (get_covariate_splits, get_population_splits):
        first = splitter(X, 5, 17, target_column="target")
        second = splitter(X.copy(), 5, 17, target_column="target")
        assert all(np.array_equal(a, c) and np.array_equal(b, d)
                   for (a, b), (c, d) in zip(first, second))
        assert not first_y.equals(permuted_y)
        with pytest.raises(AssertionError, match="must be removed"):
            splitter(frame, 5, 17, target_column="target")
    categorical = pd.DataFrame({"category": ["a", "b", "c", "a", "b"] * 3})
    _, covariate_meta = get_covariate_splits(
        categorical, 5, 17, target_column="target", return_metadata=True,
    )
    _, population_meta = get_population_splits(
        categorical, 5, 17, target_column="target", return_metadata=True,
    )
    assert covariate_meta["partition_method"] == "row_level_kfold_fallback"
    assert population_meta["partition_method"] == "row_level_kfold_fallback"
    assert covariate_meta["fallback_reason"] == "no_numeric_predictors"
    assert population_meta["fallback_reason"] == "no_numeric_predictors"


def test_group_aware_transductive_scope_fails_closed_instead_of_bypassing_groups():
    with pytest.raises(ValueError, match="supports row_level only"):
        run_experiment(
            {}, run_id="blocked-transductive-group", split_policy="group_aware",
            conditions=(("covariate_partition", 0.0),), pipelines=("Raw",),
            models=("logistic_regression",),
        )


def test_separate_training_conditions_do_not_use_held_out_labels_and_group_folds_are_disjoint():
    frame = _condition_frame()
    X = frame.drop(columns=["target"])
    y = frame["target"]
    splits = get_group_stratified_splits(
        X, y, 3, 23, target_column="target", require_class_support=False,
    )
    assert_group_fold_integrity(splits, canonical_feature_group_ids(X))
    train_idx, test_idx = splits[0]
    poisoned = frame.copy()
    poisoned.loc[test_idx, "target"] = 1 - poisoned.loc[test_idx, "target"]
    for family, severity in SEPARATE_EXPERIMENT_CONDITIONS:
        clean = prepare_task(
            frame, target_column="target", train_indices=train_idx, test_indices=test_idx,
            family=family, severity=severity, perturbation_seed=991, pipeline_name="Raw",
        )
        changed = prepare_task(
            poisoned, target_column="target", train_indices=train_idx, test_indices=test_idx,
            family=family, severity=severity, perturbation_seed=991, pipeline_name="Raw",
        )
        np.testing.assert_allclose(clean[0].to_numpy(), changed[0].to_numpy())
        np.testing.assert_allclose(clean[1].to_numpy(), changed[1].to_numpy())
        np.testing.assert_array_equal(clean[2], changed[2])
        assert clean[5]["feature_metadata"] == changed[5]["feature_metadata"]


def test_historical_class_prior_and_feature_removal_names_are_not_silently_reused():
    frame = _condition_frame()
    X = frame.drop(columns=["target"])
    y = frame["target"]
    x_relabel, y_relabel = apply_perturbation(
        X, y, "majority_label_relabeling", random_state=5,
    )
    assert x_relabel.equals(X)
    assert len(y_relabel) == len(y)
    x_removed, y_removed = apply_perturbation(
        X, y, "feature_availability_ablation", 0.5, random_state=5,
    )
    assert len(x_removed.columns) < len(X.columns)
    assert y_removed.equals(y)


def test_domain_fold_status_makes_auc_infeasibility_explicit():
    y = pd.Series([0, 0, 0, 1, 1, 1])
    splits = [
        (np.array([0, 1, 3, 4]), np.array([2, 5])),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([], dtype=int)),
    ]
    status = _domain_fold_status(splits, y)
    assert status["auc_status"] == "infeasible"
    assert status["auc_reason"] == "one_or_more_domain_folds_lacks_class_support_for_auc"


def test_all_four_out_of_grid_conditions_have_bounded_scope_smokes(tmp_path):
    frame = _condition_frame(90)
    data_path = tmp_path / "conditions.csv"
    frame.to_csv(data_path, index=False)
    data_path.with_name("conditions_meta.json").write_text(
        '{"target_column":"target","dataset_identity":{"provider":"condition-test"}}',
        encoding="utf-8",
    )
    conditions = (
        ("covariate_partition", 0.0),
        ("population_partition", 0.0),
        ("feature_availability_ablation", 0.20),
        ("majority_label_relabeling", 0.0),
    )
    for number, condition in enumerate(conditions):
        manifest = run_experiment(
            {"conditions": data_path}, run_id=f"condition-smoke-{number}",
            output_root=tmp_path / "runs", seeds=[31], folds=[1],
            conditions=(condition,), pipelines=("Raw",),
            models=("logistic_regression",), n_splits=3, cache_policy="bounded",
            cache_max_bytes=4 * 1024 * 1024,
        )
        assert manifest["configuration"]["conditions"] == [list(condition)]
        rows = [json.loads(line) for line in (
            tmp_path / "runs" / f"condition-smoke-{number}" / "results.jsonl"
        ).read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["experiment_scope"] in {
            "transductive_domain_partition", "feature_availability_ablation", "majority_label_relabeling",
        }
        if condition[0] in {"covariate_partition", "population_partition"}:
            assert rows[0]["status"] in {"success", "skipped"}
            assert "domain_partition" in rows[0].get("split_status", {})
