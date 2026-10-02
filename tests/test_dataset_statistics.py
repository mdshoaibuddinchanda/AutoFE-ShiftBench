from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset_statistics import AnalysisConfig, AnalysisInputError, _holm, _sign_flip_test, analyze_ledger
from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION
from src.stats_analysis import run_wilcoxon_analysis


def _record(dataset: str, seed: int, pipeline: str, metric: float, *, status: str = "success", condition: str = "clean") -> dict:
    return {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "seed_scheme_version": SEED_SCHEME_VERSION,
        "dataset": dataset,
        "split_policy": "stratified",
        "seed": seed,
        "fold": 1,
        "condition": condition,
        "severity": 0.0,
        "pipeline": pipeline,
        "pipeline_identity": pipeline + "__v1",
        "operator_set_id": "none_v1" if pipeline == "Raw" else "full_arithmetic_v1",
        "cap_policy_version": "none_v1",
        "model": "logistic_regression",
        "status": status,
        "roc_auc": metric,
    }


class DatasetStatisticsTests(unittest.TestCase):
    def _write(self, rows: list[dict]) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "results.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_dataset_equal_pairing_is_order_invariant_and_reproducible(self) -> None:
        rows = []
        for dataset, offset in (("d1", 0.1), ("d2", 0.2), ("d3", 0.3)):
            for seed in (1, 2):
                rows.extend([_record(dataset, seed, "Raw", 0.6), _record(dataset, seed, "AutoFE_Baseline", 0.6 + offset)])
        shuffled = list(reversed(rows))
        path = self._write(shuffled)
        config = AnalysisConfig(bootstrap_resamples=100, permutation_resamples=100, random_state=44)
        first = analyze_ledger(path, config)
        second = analyze_ledger(path, config)
        self.assertAlmostEqual(float(first.summaries.iloc[0]["estimate_b_minus_a"]), 0.2, places=10)
        self.assertEqual(int(first.summaries.iloc[0]["n_datasets"]), 3)
        self.assertEqual(first.summaries["p_value"].tolist(), second.summaries["p_value"].tolist())
        self.assertEqual(first.summaries["ci_lower"].tolist(), second.summaries["ci_lower"].tolist())
        self.assertEqual(len(first.task_pairs), 6)

    def test_missing_partner_and_failed_status_are_explicit(self) -> None:
        rows = [_record("d1", 1, "Raw", 0.5), _record("d1", 1, "AutoFE_Baseline", 0.7)]
        failed = _record("d2", 1, "AutoFE_Baseline", np.nan, status="failed")
        rows.extend([_record("d2", 1, "Raw", 0.5), failed])
        bundle = analyze_ledger(self._write(rows), AnalysisConfig(bootstrap_resamples=0, permutation_resamples=0))
        self.assertEqual(int((bundle.task_pairs["pair_status"] == "paired").sum()), 1)
        reasons = set(bundle.exclusions.get("reason", []))
        self.assertIn("status_failed", reasons)
        self.assertIn("missing_valid_partner", reasons)
        self.assertEqual(bundle.summaries.iloc[0]["status"], "too_few_datasets")

    def test_duplicate_success_and_protocol_mix_are_rejected(self) -> None:
        row = _record("d1", 1, "Raw", 0.5)
        rows = [row, dict(row), _record("d1", 1, "AutoFE_Baseline", 0.7)]
        with self.assertRaises(AnalysisInputError):
            analyze_ledger(self._write(rows))
        incompatible = [_record("d1", 1, "Raw", 0.5), _record("d1", 1, "AutoFE_Baseline", 0.7)]
        incompatible[0]["evaluation_protocol_version"] = "old_protocol"
        with self.assertRaises(AnalysisInputError):
            analyze_ledger(self._write(incompatible))

    def test_holm_correction_handles_ties(self) -> None:
        adjusted, rejected = _holm([0.01, 0.01, 0.2], 0.05)
        self.assertEqual(adjusted, [0.03, 0.03, 0.2])
        self.assertEqual(rejected, [True, True, False])

    def test_all_zero_and_too_few_dataset_outcomes_are_explicit(self) -> None:
        zero_rows = []
        for dataset in ("d1", "d2"):
            zero_rows.extend([_record(dataset, 1, "Raw", 0.5), _record(dataset, 1, "AutoFE_Baseline", 0.5)])
        zero = analyze_ledger(self._write(zero_rows), AnalysisConfig(bootstrap_resamples=10, permutation_resamples=10))
        self.assertEqual(zero.summaries.iloc[0]["status"], "all_zero_effect")
        one = analyze_ledger(self._write([_record("d1", 1, "Raw", 0.5), _record("d1", 1, "AutoFE_Baseline", 0.7)]), AnalysisConfig(bootstrap_resamples=10, permutation_resamples=10))
        self.assertEqual(one.summaries.iloc[0]["status"], "too_few_datasets")

    def test_sign_flip_keeps_zero_datasets_in_mean_denominator(self) -> None:
        result = _sign_flip_test(
            np.asarray([0.0, 1.0, 2.0]),
            AnalysisConfig(bootstrap_resamples=0, permutation_resamples=0),
            stratum_label="test",
            fingerprint="fixture",
        )
        self.assertEqual(result["method"], "sign_flip_exact")
        self.assertAlmostEqual(result["observed_mean"], 1.0)
        self.assertAlmostEqual(result["p_value"], 0.5)

    def test_summary_reports_valid_and_unpaired_counts(self) -> None:
        rows = [_record("d1", 1, "Raw", 0.5), _record("d1", 1, "AutoFE_Baseline", 0.7)]
        rows.append(_record("d1", 2, "Raw", 0.6))
        summary = analyze_ledger(self._write(rows), AnalysisConfig(bootstrap_resamples=0, permutation_resamples=0)).summaries.iloc[0]
        self.assertEqual(int(summary["n_intended_tasks"]), 2)
        self.assertEqual(int(summary["n_valid_tasks_a"]), 2)
        self.assertEqual(int(summary["n_valid_tasks_b"]), 1)
        self.assertEqual(int(summary["n_unpaired_tasks"]), 1)

    def test_active_stats_reader_uses_dataset_level_outputs(self) -> None:
        rows = []
        for dataset in ("d1", "d2"):
            rows.extend([_record(dataset, 1, "Raw", 0.5), _record(dataset, 1, "AutoFE_Baseline", 0.7)])
        ledger = self._write(rows)
        output = ledger.parent / "statistical_results.csv"
        summary = run_wilcoxon_analysis(ledger, output, alpha=0.05)
        self.assertIn("n_datasets", summary.columns)
        self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
