from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.checkpoint import compute_hash, has_run, init_db, log_run
from src.feature_engineering import DFSConfig, expand_features_with_dfs
from src.pipeline_runner import (
    _cache_dir_for_dataset,
    apply_training_condition,
    split_predictors_and_target,
)
from src.preprocessing import _build_preprocessor, _to_dense_array
from src.protocol import EVALUATION_PROTOCOL_VERSION, results_ledger_path
from src.splitters import (
    SplitInfeasibleError,
    assert_fold_integrity,
    get_splits,
    get_stratified_splits,
)


class LeakageBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(81)
        self.X = pd.DataFrame(
            {
                "sensor_a": rng.normal(size=60),
                "sensor_b": rng.normal(loc=1.0, scale=2.0, size=60),
            }
        )
        self.y = pd.Series(np.tile(["low", "high"], 30), name="target_label")

    def test_feature_splitters_receive_predictors_and_ignore_target_values(self) -> None:
        y_changed = pd.Series(np.tile(["north", "south", "east"], 20), name="target_label")
        frame_a = self.X.assign(target_label=self.y)
        frame_b = self.X.assign(target_label=y_changed)
        X_a, y_a, target_a = split_predictors_and_target(frame_a)
        X_b, y_b, target_b = split_predictors_and_target(frame_b)

        self.assertEqual(target_a, "target_label")
        self.assertEqual(target_b, "target_label")
        self.assertNotIn("target_label", X_a.columns)
        self.assertNotIn("target_label", X_b.columns)
        assert_frame_equal(X_a, X_b)

        for policy in ("covariate_shift", "population_shift"):
            with self.subTest(policy=policy):
                splits_a = get_splits(X_a, y_a, policy, n_splits=5, seed=13)
                splits_b = get_splits(X_b, y_b, policy, n_splits=5, seed=13)
                for (train_a, test_a), (train_b, test_b) in zip(splits_a, splits_b):
                    np.testing.assert_array_equal(train_a, train_b)
                    np.testing.assert_array_equal(test_a, test_b)

    def test_stratification_still_uses_training_labels(self) -> None:
        blocked_y = pd.Series([0] * 30 + [1] * 30)
        alternating_y = pd.Series(np.tile([0, 1], 30))
        blocked = get_stratified_splits(self.X, blocked_y, n_splits=5, seed=19)
        alternating = get_stratified_splits(self.X, alternating_y, n_splits=5, seed=19)

        self.assertFalse(
            all(np.array_equal(a[1], b[1]) for a, b in zip(blocked, alternating)),
            "stratified folds should respond to the labels used for stratification",
        )
        for _, test_indices in blocked:
            self.assertEqual(blocked_y.iloc[test_indices].value_counts().to_dict(), {0: 6, 1: 6})

    def test_stratified_infeasibility_is_not_silently_replaced(self) -> None:
        labels = pd.Series([0] * 30 + [1] * 29 + [2])
        with self.assertRaisesRegex(SplitInfeasibleError, "No random-fold fallback"):
            get_stratified_splits(self.X, labels, n_splits=5, seed=4)

    def test_group_and_row_separation_are_checked(self) -> None:
        groups = np.repeat(np.arange(5), 2)
        all_rows = np.arange(10)
        group_splits = [
            (all_rows[groups != group], all_rows[groups == group])
            for group in range(5)
        ]
        assert_fold_integrity(group_splits, len(all_rows), groups=groups)

        overlapping = group_splits.copy()
        overlapping[0] = (np.array([0, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([0, 1]))
        with self.assertRaisesRegex(SplitInfeasibleError, "overlapping train and test"):
            assert_fold_integrity(overlapping, len(all_rows), groups=groups)

        test_blocks = [
            np.array([0, 2]),
            np.array([1, 3]),
            np.array([4, 5]),
            np.array([6, 7]),
            np.array([8, 9]),
        ]
        group_overlap = [(np.setdiff1d(all_rows, test), test) for test in test_blocks]
        with self.assertRaisesRegex(SplitInfeasibleError, "same group"):
            assert_fold_integrity(group_overlap, len(all_rows), groups=groups)

    def test_feature_splitter_without_numeric_predictors_fails_explicitly(self) -> None:
        categorical = pd.DataFrame({"category": ["a", "b", "a", "b", "a"]})
        with self.assertRaisesRegex(SplitInfeasibleError, "no alternate random split"):
            get_splits(categorical, pd.Series([0, 0, 1, 1, 0]), "covariate_shift", 2, 7)

    def test_preprocessor_fit_state_uses_training_rows_only(self) -> None:
        train = pd.DataFrame({"value": [1.0, 3.0, 5.0], "kind": ["a", "a", "b"]})
        heldout_a = pd.DataFrame({"value": [1000.0], "kind": ["heldout-only"]})
        heldout_b = pd.DataFrame({"value": [-1e8], "kind": ["another-new-level"]})

        pre_a = _build_preprocessor(train, encoding="onehot", scale_numeric=True)
        train_a = _to_dense_array(pre_a.fit_transform(train))
        test_a = _to_dense_array(pre_a.transform(heldout_a))
        pre_b = _build_preprocessor(train, encoding="onehot", scale_numeric=True)
        train_b = _to_dense_array(pre_b.fit_transform(train))
        test_b = _to_dense_array(pre_b.transform(heldout_b))

        np.testing.assert_allclose(train_a, train_b)
        np.testing.assert_allclose(pre_a.named_transformers_["num"].named_steps["scaler"].mean_, [3.0])
        self.assertEqual(pre_a.named_transformers_["cat"].named_steps["encoder"].categories_[0].tolist(), ["a", "b"])
        self.assertEqual(test_a.shape[1], train_a.shape[1])
        self.assertEqual(test_b.shape[1], train_b.shape[1])

    def test_heldout_values_do_not_change_synthesis_state(self) -> None:
        train = pd.DataFrame(
            {"a": [1.0, 2.0, 4.0, 8.0], "b": [3.0, 2.0, 1.0, 0.0]}
        )
        test_a = pd.DataFrame({"a": [5.0], "b": [6.0]})
        test_b = pd.DataFrame({"a": [5e9], "b": [-9e8]})
        config = DFSConfig(
            enable_dfs=True,
            depth=1,
            max_features=None,
            max_base_features=None,
            selection_method="none",
            trans_primitives=["add_numeric"],
            monitor_ram=False,
            random_seed=5,
        )

        train_a, _, _ = expand_features_with_dfs(train, test_a, config=config)
        train_b, _, _ = expand_features_with_dfs(train, test_b, config=config)
        assert_frame_equal(train_a, train_b)

    def test_training_corruption_preserves_heldout_data_and_labels(self) -> None:
        x_train = pd.DataFrame({"sensor": np.arange(20, dtype=float)})
        y_train = pd.Series(np.tile(["a", "b"], 10))
        x_test = pd.DataFrame({"sensor": [500.0, 600.0]})
        y_test = pd.Series(["a", "b"])
        original_train = x_train.copy(deep=True)
        original_train_y = y_train.copy(deep=True)
        original_test = x_test.copy(deep=True)
        original_test_y = y_test.copy(deep=True)

        x_train_cond, y_train_cond, x_test_cond, y_test_cond = apply_training_condition(
            x_train,
            y_train,
            x_test=x_test,
            y_test=y_test,
            shift_family="gaussian_noise",
            severity=0.2,
            random_state=77,
        )

        assert_frame_equal(x_train, original_train)
        assert_series_equal(y_train, original_train_y)
        assert_frame_equal(x_test_cond, original_test)
        assert_series_equal(y_test_cond, original_test_y)
        self.assertFalse(np.allclose(x_train_cond["sensor"], original_train["sensor"]))
        assert_series_equal(y_train_cond, original_train_y)

    def test_old_cache_and_checkpoint_identity_are_not_reused(self) -> None:
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_dir:
            try:
                os.chdir(temporary_dir)
                init_db()
                legacy_key = "demo_11_1_clean_Raw_logistic_regression"
                connection = sqlite3.connect("reports/cache.db")
                try:
                    connection.execute(
                        "INSERT INTO completed_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (legacy_key, "demo", 11, 1, "clean", "Raw", "logistic_regression"),
                    )
                    connection.commit()
                finally:
                    connection.close()

                self.assertNotEqual(
                    compute_hash(
                        "demo", 11, 1, "clean", "Raw", "logistic_regression", "stratified",
                    ),
                    legacy_key,
                )
                self.assertFalse(
                    has_run("demo", 11, 1, "clean", "Raw", "logistic_regression", "stratified")
                )
                self.assertNotEqual(_cache_dir_for_dataset("demo"), Path("data/cache/demo"))
                self.assertNotEqual(results_ledger_path(), Path("reports/tables/results_stream.jsonl"))

                log_run("demo", 11, 1, "clean", "Raw", "logistic_regression", "stratified")
                self.assertTrue(
                    has_run("demo", 11, 1, "clean", "Raw", "logistic_regression", "stratified")
                )
                self.assertTrue(
                    EVALUATION_PROTOCOL_VERSION in compute_hash(
                        "demo", 11, 1, "clean", "Raw", "logistic_regression", "stratified",
                    )
                )
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
