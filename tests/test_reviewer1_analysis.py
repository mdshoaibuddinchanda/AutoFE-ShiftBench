from __future__ import annotations

import json

from src.reviewer1_analysis import build_corrected_result_note, paired_dataset_bootstrap


def test_bootstrap_is_seeded_and_dataset_level():
    first = paired_dataset_bootstrap([0.1, -0.2, 0.3], seed=7, replicates=200)
    second = paired_dataset_bootstrap([0.1, -0.2, 0.3], seed=7, replicates=200)
    assert first == second
    assert first[0] <= first[1]


def test_result_note_exposes_incomplete_cells_and_holm_family(tmp_path):
    rows = []
    for dataset in ("a", "b"):
        for pipeline, value in (("Raw", 0.5), ("AutoFE_Baseline", 0.6), ("AutoFE_MI", 0.55), ("AutoFE_Random", 0.52), ("AutoFE_NoMultiply", 0.51)):
            rows.append({
                "run_id": "corrected-test", "dataset": dataset, "seed": 1, "fold": 1,
                "condition": "clean", "pipeline": pipeline, "model": "logistic_regression",
                "split_policy": "row_level", "status": "success", "roc_auc": value,
            })
    rows.pop()  # one candidate cell is absent: it must remain incomplete
    ledger = tmp_path / "results.jsonl"
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "run_id": "corrected-test", "protocol_version": "p",
        "configuration": {
            "datasets": ["a", "b"], "seeds": [1], "folds": [1],
            "conditions": [["clean", 0.0]],
            "pipelines": ["Raw", "AutoFE_Baseline", "AutoFE_MI", "AutoFE_Random", "AutoFE_NoMultiply"],
            "models": ["logistic_regression"], "n_splits": 1,
        }, "expected_tasks": 10,
    }), encoding="utf-8")
    note = build_corrected_result_note(ledger, manifest_path=manifest, bootstrap_replicates=100)
    assert note["task_accounting"]["counts_by_status"]["pending"] == 1
    baseline = [row for row in note["contrasts"] if row["pipeline_b"] == "AutoFE_Baseline"][0]
    assert baseline["family_id"] == "F1_primary_roc_auc"
    assert baseline["ci_method"] == "paired_dataset_bootstrap_percentile"

