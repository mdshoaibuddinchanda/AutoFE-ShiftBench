from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src import pipeline_runner
from src.checkpoint import has_run
from src.protocol import results_ledger_path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ResultCollector:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def put(self, value: dict) -> None:
        self.items.append(value)


def _worker_entry(task: dict, result_queue) -> None:
    pipeline_runner._writer_queue = result_queue
    pipeline_runner.train_unit(task)


class ProductionResumeTests(unittest.TestCase):
    def test_worker_execution_and_resume_reconstruct_identical_task_inputs(self) -> None:
        previous_cwd = Path.cwd()
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        with tempfile.TemporaryDirectory() as temporary_dir:
            try:
                os.chdir(temporary_dir)
                root = Path(temporary_dir)
                data_path = root / "data" / "raw" / "smoke.csv"
                data_path.parent.mkdir(parents=True)
                row_count = 60
                data = pd.DataFrame(
                    {
                        "measurement_a": np.linspace(-2.0, 3.0, row_count),
                        "measurement_b": np.sin(np.arange(row_count, dtype=float)),
                        "target": np.tile(["case", "control"], row_count // 2),
                    }
                )
                data.to_csv(data_path, index=False)
                Path("reports/tables").mkdir(parents=True)
                Path("reports/worker_logs").mkdir(parents=True)

                task = {
                    "dataset_name": "smoke",
                    "data_path": data_path,
                    "seed": 17,
                    "fold": 1,
                    "condition": "gaussian_noise_0.05",
                    "shift_family": "gaussian_noise",
                    "severity": 0.05,
                    "pipeline": "Raw",
                    "model": "logistic_regression",
                    "diagnostics_enabled": True,
                    "diagnostic_config": {"max_rows": 24, "random_state": 909, "magnitudes": [1e-3, 1e-2]},
                }

                precompute_task = {
                    key: task[key]
                    for key in (
                        "dataset_name",
                        "data_path",
                        "seed",
                        "fold",
                        "condition",
                        "shift_family",
                        "severity",
                        "diagnostics_enabled",
                        "diagnostic_config",
                    )
                }
                clean_precompute_task = dict(precompute_task)
                clean_precompute_task.update({"condition": "clean", "shift_family": "clean", "severity": 0.0})
                self.assertEqual(pipeline_runner.precompute_unit(clean_precompute_task), "smoke")
                self.assertEqual(pipeline_runner.precompute_unit(precompute_task), "smoke")
                first = pipeline_runner.get_data_splits(
                    data_path,
                    "smoke",
                    17,
                    1,
                    task["condition"],
                    task["shift_family"],
                    task["severity"],
                    task["diagnostics_enabled"],
                    task["diagnostic_config"],
                )
                clean_inputs = pipeline_runner.get_data_splits(
                    data_path,
                    "smoke",
                    17,
                    1,
                    "clean",
                    "clean",
                    0.0,
                    task["diagnostics_enabled"],
                    task["diagnostic_config"],
                )
                for pipeline_name in pipeline_runner.PIPELINE_NAMES:
                    self.assertEqual(clean_inputs[4][pipeline_name]["diagnostic_status"], "diagnostic_complete")
                resumed_inputs = pipeline_runner.get_data_splits(
                    data_path,
                    "smoke",
                    17,
                    1,
                    task["condition"],
                    task["shift_family"],
                    task["severity"],
                    task["diagnostics_enabled"],
                    task["diagnostic_config"],
                )
                for pipeline_name in pipeline_runner.PIPELINE_NAMES:
                    assert_frame_equal(
                        first[0][pipeline_name][0],
                        resumed_inputs[0][pipeline_name][0],
                    )
                    assert_frame_equal(
                        first[0][pipeline_name][1],
                        resumed_inputs[0][pipeline_name][1],
                    )
                    self.assertEqual(
                        first[4][pipeline_name]["split_seed"],
                        resumed_inputs[4][pipeline_name]["split_seed"],
                    )
                    self.assertEqual(
                        first[4][pipeline_name]["corruption_seed"],
                        resumed_inputs[4][pipeline_name]["corruption_seed"],
                    )
                    self.assertEqual(
                        first[4][pipeline_name]["selection_seed"],
                        resumed_inputs[4][pipeline_name]["selection_seed"],
                    )
                    self.assertEqual(first[4][pipeline_name]["operator_set_id"], resumed_inputs[4][pipeline_name]["operator_set_id"])
                    self.assertEqual(first[4][pipeline_name]["diagnostic_status"], "diagnostic_complete")
                    history_path = Path(first[4][pipeline_name]["history_path"])
                    diagnostic_path = Path(first[4][pipeline_name]["diagnostic_path"])
                    self.assertTrue(history_path.exists())
                    self.assertTrue(diagnostic_path.exists())
                    with history_path.open(encoding="utf-8") as history_file:
                        history_event = json.loads(next(history_file))
                    self.assertEqual(history_event["dataset"], "smoke")
                    self.assertEqual(history_event["pipeline"], pipeline_name)
                    self.assertIn("requested_cap", history_event)
                    self.assertIn("rejection_reason", history_event)
                    with diagnostic_path.open(encoding="utf-8") as diagnostic_file:
                        diagnostic = json.load(diagnostic_file)
                    self.assertEqual(diagnostic["diagnostic_status"], "diagnostic_complete")
                    self.assertEqual(diagnostic["task"]["pipeline"], pipeline_name)

                context = mp.get_context("spawn")
                result_queue = context.Queue()
                ledger_path = results_ledger_path()
                writer = context.Process(
                    target=pipeline_runner.writer_process,
                    args=(result_queue, str(ledger_path)),
                )
                writer.start()
                worker = context.Process(target=_worker_entry, args=(task, result_queue))
                worker.start()
                worker.join(timeout=60)
                self.assertFalse(worker.is_alive(), "worker did not finish within 60 seconds")
                self.assertEqual(worker.exitcode, 0)

                result_queue.put("DONE")
                writer.join(timeout=30)
                self.assertFalse(writer.is_alive(), "result writer did not finish within 30 seconds")
                self.assertEqual(writer.exitcode, 0)

                with ledger_path.open(encoding="utf-8") as result_file:
                    records = [json.loads(line) for line in result_file if line.strip()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["seed_scheme_version"], pipeline_runner.SEED_SCHEME_VERSION)
                self.assertEqual(records[0]["diagnostic_status"], "diagnostic_complete")
                self.assertTrue(
                    has_run(
                        "smoke", 17, 1, task["condition"], "Raw", "logistic_regression",
                        "stratified", pipeline_runner.pipeline_identity_token("Raw"),
                    )
                )

                # A resumed worker must skip the already committed task.
                collector = ResultCollector()
                pipeline_runner._writer_queue = collector
                pipeline_runner.train_unit(task)
                self.assertEqual(collector.items, [])
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
