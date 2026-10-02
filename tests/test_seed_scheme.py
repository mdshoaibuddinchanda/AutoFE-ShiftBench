from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.checkpoint import compute_hash, has_run, init_db
from src.pipeline_runner import _cache_dir_for_dataset
from src.protocol import EVALUATION_PROTOCOL_VERSION, cache_root, results_ledger_path
from src.seeding import (
    SEED_MAX,
    SEED_SCHEME_VERSION,
    corruption_seed,
    distance_sample_seed,
    estimator_seed,
    feature_selection_seed,
    split_seed,
    stable_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class StableSeedTests(unittest.TestCase):
    def test_canonical_identity_ignores_dictionary_insertion_order(self) -> None:
        first = {
            "dataset": "credit-g",
            "task": {"condition": "missing_values_0.10", "fold": 3, "replicate": 42},
        }
        second = {
            "task": {"replicate": 42, "fold": 3, "condition": "missing_values_0.10"},
            "dataset": "credit-g",
        }
        self.assertEqual(stable_seed("training_corruption", first), stable_seed("training_corruption", second))

    def test_seed_scheme_is_validated_and_domain_separated(self) -> None:
        identity = {
            "dataset": "adult",
            "split_policy": "stratified",
            "repetition_seed": 42,
            "fold": 2,
            "condition": "gaussian_noise_0.05",
        }
        seed = stable_seed("training_corruption", identity)
        self.assertGreaterEqual(seed, 0)
        self.assertLessEqual(seed, SEED_MAX)
        self.assertNotEqual(seed, stable_seed("feature_selection", identity))
        self.assertEqual(SEED_SCHEME_VERSION, "sha256_canonical_json_u32_v1")

    def test_task_fields_change_seeds_and_purpose_streams_are_paired(self) -> None:
        base = {
            "dataset": "adult",
            "split_policy": "stratified",
            "repetition_seed": 42,
            "fold": 2,
            "condition": "missing_values_0.10",
        }
        baseline = stable_seed("training_corruption", base)
        for field, changed in (
            ("dataset", "diabetes"),
            ("split_policy", "population_shift"),
            ("repetition_seed", 43),
            ("fold", 3),
            ("condition", "missing_values_0.20"),
        ):
            with self.subTest(field=field):
                candidate = {**base, field: changed}
                self.assertNotEqual(baseline, stable_seed("training_corruption", candidate))

        compared_pipelines = ("Raw", "AutoFE_MI")
        split_states = {
            pipeline: split_seed("adult", "stratified", 42, 5)
            for pipeline in compared_pipelines
        }
        corruption_states = {
            pipeline: corruption_seed(
                "adult", "stratified", 42, 2, "missing_values_0.10",
            )
            for pipeline in compared_pipelines
        }
        estimator_states = {
            pipeline: estimator_seed(
                "adult", "stratified", 42, 2, "clean", "random_forest",
            )
            for pipeline in compared_pipelines
        }
        self.assertEqual(len(set(split_states.values())), 1)
        self.assertEqual(len(set(corruption_states.values())), 1)
        self.assertEqual(len(set(estimator_states.values())), 1)
        self.assertNotEqual(
            feature_selection_seed("adult", "stratified", 42, 2, "clean", "AutoFE_MI"),
            feature_selection_seed("adult", "stratified", 42, 2, "clean", "AutoFE_Random"),
        )
        self.assertNotEqual(
            distance_sample_seed("adult", "stratified", 42, 2, "clean"),
            corruption_states["Raw"],
        )

    def test_separate_processes_ignore_python_hash_salt_and_reconstruct_task_inputs(self) -> None:
        script = r'''
import json
import numpy as np
import pandas as pd
from src.seeding import split_seed, corruption_seed
from src.splitters import get_splits
from src.shift_generator import apply_perturbation

X = pd.DataFrame({"a": np.arange(60, dtype=float), "b": np.sin(np.arange(60, dtype=float))})
y = pd.Series(np.tile([0, 1], 30))
split_state = split_seed("fixture", "stratified", 123, 5)
splits = get_splits(X, y, "stratified", 5, split_state)
train, test = splits[2]
condition = "gaussian_noise_0.10"
noise_state = corruption_seed("fixture", "stratified", 123, 3, condition)
x_corrupt, y_corrupt = apply_perturbation(
    X.iloc[train], y.iloc[train], "gaussian_noise", 0.10, random_state=noise_state
)
print(json.dumps({
    "split_seed": split_state,
    "corruption_seed": noise_state,
    "train": train.tolist(),
    "test": test.tolist(),
    "x_corrupt": x_corrupt.round(12).values.tolist(),
    "y_corrupt": y_corrupt.tolist(),
}, separators=(",", ":")))
'''
        outputs = []
        for hash_seed in ("1", "9127"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = hash_seed
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                ).strip()
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(json.loads(outputs[0])["split_seed"], split_seed("fixture", "stratified", 123, 5))

    def test_scheduling_order_does_not_change_task_seeds(self) -> None:
        tasks = [
            ("adult", "stratified", 42, 1, "clean"),
            ("adult", "stratified", 42, 1, "label_noise_0.05"),
            ("sonar", "covariate_shift", 123, 4, "covariate_shift"),
        ]

        def derive(items: list[tuple[str, str, int, int, str]]) -> dict[tuple, int]:
            return {
                task: corruption_seed(task[0], task[1], task[2], task[3], task[4])
                for task in items
            }

        self.assertEqual(derive(tasks), derive(list(reversed(tasks))))

    def test_old_seed_scheme_cache_checkpoint_and_ledger_are_not_reused(self) -> None:
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_dir:
            try:
                os.chdir(temporary_dir)
                self.assertEqual(cache_root().parts[-1], SEED_SCHEME_VERSION)
                self.assertEqual(_cache_dir_for_dataset("demo").parent, cache_root())
                self.assertIn(EVALUATION_PROTOCOL_VERSION, results_ledger_path().name)
                self.assertIn(SEED_SCHEME_VERSION, results_ledger_path().name)

                init_db()
                # Identity produced by the leakage-only protocol before this seed version.
                old_seed_key = f"{EVALUATION_PROTOCOL_VERSION}|demo|11|1|clean|Raw|logistic_regression"
                connection = __import__("sqlite3").connect("reports/cache.db")
                try:
                    connection.execute(
                        "INSERT INTO completed_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (old_seed_key, "demo", 11, 1, "clean", "Raw", "logistic_regression"),
                    )
                    connection.commit()
                finally:
                    connection.close()
                self.assertNotEqual(
                    compute_hash("demo", 11, 1, "clean", "Raw", "logistic_regression", "stratified"),
                    old_seed_key,
                )
                self.assertFalse(
                    has_run("demo", 11, 1, "clean", "Raw", "logistic_regression", "stratified")
                )
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
