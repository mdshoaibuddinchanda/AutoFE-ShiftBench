from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION
from src.sensitivity_analysis import SensitivityConfig, analyze_sensitivity
from src.task_manifest import ManifestStore, build_task_records


class SensitivityAnalysisTests(unittest.TestCase):
    def _fixture(self, *, omit: tuple[str, int, str] | None = None) -> tuple[Path, Path, str]:
        directory = Path(tempfile.mkdtemp())
        datasets = {name: directory / f"{name}.csv" for name in ("d1", "d2")}
        for path in datasets.values():
            path.write_text("feature,target_label\n0,0\n1,1\n", encoding="utf-8")
        pipelines = ["Raw", "Raw_Capped", "Raw_Full", "AutoFE_Baseline"]
        metadata = {
            "Raw": {"operator_set_id": "none_v1", "cap_policy_version": "none_v1"},
            "Raw_Capped": {"operator_set_id": "none_v1", "cap_policy_version": "cap_v1"},
            "Raw_Full": {"operator_set_id": "none_v1", "cap_policy_version": "none_v1"},
            "AutoFE_Baseline": {"operator_set_id": "full_arithmetic_v1", "cap_policy_version": "cap_v1"},
        }
        records = build_task_records(
            datasets,
            [1, 2],
            [0],
            [("clean", 0.0, "clean")],
            pipelines,
            ["logistic_regression"],
            data_paths=datasets,
            include_precompute=False,
            pipeline_metadata=lambda pipeline: metadata[pipeline],
        )
        db = directory / "manifest.db"
        store = ManifestStore(db)
        run_id = "sensitivity_test"
        store.create_run(run_id, {"protocol_version": EVALUATION_PROTOCOL_VERSION, "seed_scheme_version": SEED_SCHEME_VERSION}, records)
        ledger = directory / "results.jsonl"
        with ledger.open("w", encoding="utf-8") as handle:
            for record in records:
                key = (record["dataset"], record["seed"], record["pipeline"])
                attempt = store.claim_task(run_id, record["scientific_task_id"], worker_id="test")
                if omit == key:
                    store.record_failure(run_id, record["scientific_task_id"], attempt, failure_class="worker_exception")
                    continue
                value = {"Raw": 0.60, "Raw_Capped": 0.70, "Raw_Full": 0.68, "AutoFE_Baseline": 0.75}[record["pipeline"]]
                payload = {"status": "success", "roc_auc": value, "diagnostic_status": "not_requested"}
                store.commit_result(run_id, record["scientific_task_id"], attempt, payload)
                handle.write(json.dumps({"run_id": run_id, "scientific_task_id": record["scientific_task_id"], **payload}) + "\n")
        return db, ledger, run_id

    def test_predeclared_regimes_and_fair_contrasts_are_reported(self) -> None:
        db, ledger, run_id = self._fixture(omit=("d2", 2, "Raw_Capped"))
        bundle = analyze_sensitivity(db, ledger, run_id, config=SensitivityConfig(bootstrap_resamples=0, permutation_resamples=0))
        regimes = set(bundle.summaries["regime"])
        self.assertIn("primary_observed", regimes)
        self.assertIn("common_eligible_task_set", regimes)
        self.assertIn("coverage_threshold_0.75", regimes)
        self.assertIn("leave_one_dataset_out", set(bundle.leave_one_out.get("regime", [])))
        self.assertEqual(set(bundle.summaries["contrast_id"]), {"historical_raw", "fair_cap_control", "fair_full_control"})
        fair = bundle.summaries[(bundle.summaries["regime"] == "primary_observed") & (bundle.summaries["contrast_id"] == "fair_cap_control")].iloc[0]
        self.assertEqual(int(fair["n_intended_tasks"]), 4)
        self.assertEqual(int(fair["n_valid_pairs"]), 3)
        self.assertEqual(fair["bound_status"], "complete")
        self.assertIsNotNone(bundle.exclusions)

    def test_unresolved_pairs_produce_identification_bounds_and_snapshot_hash(self) -> None:
        db, ledger, run_id = self._fixture(omit=("d1", 1, "Raw_Full"))
        bundle = analyze_sensitivity(db, ledger, run_id, config=SensitivityConfig(bootstrap_resamples=0, permutation_resamples=0))
        pairs = bundle.pair_blocks[(bundle.pair_blocks["contrast_id"] == "fair_full_control") & (bundle.pair_blocks["dataset"] == "d1")]
        unresolved = pairs[pairs["bound_status"] == "one_sided_missing_a"]
        self.assertEqual(len(unresolved), 1)
        self.assertIsNotNone(unresolved.iloc[0]["bound_lower"])
        self.assertIsNotNone(bundle.snapshot["ledger_sha256"])


if __name__ == "__main__":
    unittest.main()
