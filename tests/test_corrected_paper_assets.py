"""Small synthetic contract check for corrected paper tables and images."""

from __future__ import annotations

import csv
import json
import sqlite3

import pytest

from provenance.generate_corrected_assets import OPERATORS, _enabled, add_mechanism, add_sensitivities, generate


PIPELINES = (
    "Raw", "AutoFE_Baseline", "Raw_CapMatched", "AutoFE_MI", "AutoFE_Random",
    "AutoFE_NoMultiply", "AutoFE_Isolate_Add", "AutoFE_Isolate_Subtract",
    "AutoFE_Isolate_Multiply", "AutoFE_Isolate_Divide", "AutoFE_LeaveOut_Add",
    "AutoFE_LeaveOut_Subtract", "AutoFE_LeaveOut_Multiply", "AutoFE_LeaveOut_Divide",
)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _fixture(tmp_path):
    scope = {
        "artifact_type": "reporting_diagnostic_scope",
        "code_fingerprint": "synthetic-source",
        "datasets": [{"name": "synthetic-data"}], "seeds": [42], "folds": [1],
        "primary_conditions": ["clean"],
        "models": ["synthetic-model"], "pipelines": list(PIPELINES),
        "runs": [
            {"run_id": "r1-v2-primary-row", "scope": "primary", "split_policy": "row_level",
             "intended_cells": 14, "conditions": 1},
            {"run_id": "r1-v2-primary-group", "scope": "primary", "split_policy": "group_aware",
             "intended_cells": 14, "conditions": 1},
        ],
    }
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    directories = []
    for policy, suffix in (("row_level", "row"), ("group_aware", "group")):
        run_id = f"r1-v2-primary-{suffix}"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        manifest = {
            "status": "complete", "run_id": run_id,
            "code_fingerprint": "synthetic-source", "expected_tasks": 14,
            "configuration": {"split_policy": policy},
            "counts_by_status": {"success": 14, "failed": 0, "skipped": 0, "timed_out": 0},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with sqlite3.connect(run_dir / "manifest_outcomes.sqlite") as connection:
            connection.execute("CREATE TABLE outcomes (task_key TEXT PRIMARY KEY, status TEXT, phase TEXT)")
            with (run_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
                for pipeline in PIPELINES:
                    key = f"{policy}:{pipeline}"
                    connection.execute("INSERT INTO outcomes VALUES (?, 'success', 'phase2')", (key,))
                    enabled = _enabled(pipeline)
                    counts = {
                        operator: {"generated": 7 if operator in enabled else 0,
                                   "rejected": 1 if operator in enabled else 0,
                                   "duplicates": 1 if operator in enabled else 0,
                                   "eligible": 5 if operator in enabled else 0,
                                   "selected": 2 if operator in enabled else 0}
                        for operator in OPERATORS
                    }
                    row = {
                        "task_key": key, "status": "success", "run_id": run_id,
                        "split_policy": policy, "dataset": "synthetic-data", "seed": 42,
                        "fold": 1, "condition": "clean", "pipeline": pipeline,
                        "model": "synthetic-model",
                        "roc_auc": 0.81 if pipeline == "AutoFE_LeaveOut_Multiply" else
                                   (0.77 if pipeline == "AutoFE_Isolate_Divide" else 0.70),
                        "operator_configuration": {"enabled_operators": list(enabled)},
                        "operator_candidate_counts": counts,
                    }
                    stream.write(json.dumps(row) + "\n")
        directories.append(run_dir)
    return scope_path, directories


def test_corrected_values_reach_tables_and_figures(tmp_path):
    scope_path, (row_dir, group_dir) = _fixture(tmp_path)
    output = tmp_path / "paper_assets"
    manifest = generate(row_dir, group_dir, output, scope_path=scope_path)
    assert manifest["artifact_type"] == "reviewer1_reporting_diagnostic_assets"
    summary = _rows(output / "primary_pipeline_auc_summary.csv")
    multiply = next(row for row in summary if row["split_policy"] == "row_level"
                    and row["pipeline"] == "AutoFE_LeaveOut_Multiply")
    assert float(multiply["mean_dataset_roc_auc"]) == pytest.approx(0.81)
    assert float(multiply["mean_paired_auc_delta_vs_full_arithmetic"]) == pytest.approx(0.11)
    counts = _rows(output / "primary_operator_candidate_counts.csv")
    divide = next(row for row in counts if row["split_policy"] == "row_level"
                  and row["pipeline"] == "AutoFE_LeaveOut_Multiply"
                  and row["operator"] == "divide_numeric")
    absent = next(row for row in counts if row["split_policy"] == "row_level"
                  and row["pipeline"] == "AutoFE_LeaveOut_Multiply"
                  and row["operator"] == "multiply_numeric")
    assert divide["enabled"] == "True"
    assert (divide["generated"], divide["rejected"], divide["eligible"],
            divide["selected"], divide["observed_feature_tasks"]) == ("7", "1", "5", "2", "1")
    assert absent["enabled"] == "False" and absent["generated"] == "0"
    assert len(_rows(output / "primary_dataset_auc_coverage.csv")) == 28
    assert len(_rows(output / "primary_condition_auc_summary.csv")) == 28
    assert len(manifest["figures"]) == 8
    assert all((output / name).stat().st_size > 1000 for name in manifest["figures"])


def test_incomplete_run_cannot_be_presented_as_corrected_result(tmp_path):
    scope_path, (row_dir, group_dir) = _fixture(tmp_path)
    path = group_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        generate(row_dir, group_dir, tmp_path / "assets", scope_path=scope_path)


def test_disabled_operator_count_is_rejected_before_reporting(tmp_path):
    scope_path, (row_dir, group_dir) = _fixture(tmp_path)
    path = row_dir / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    target = next(row for row in rows if row["pipeline"] == "AutoFE_LeaveOut_Multiply")
    target["operator_candidate_counts"]["multiply_numeric"]["generated"] = 1
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Disabled"):
        generate(row_dir, group_dir, tmp_path / "assets", scope_path=scope_path)


def test_jacobian_association_values_reach_table_and_image(tmp_path):
    note = {
        "artifact_type": "reviewer1_exploratory_mechanism_association",
        "associations": [
            {"split_policy": policy, "n_complete_datasets": 1, "rho": 0.42,
             "permutation_p": 0.03, "holm_adjusted_permutation_p": 0.06}
            for policy in ("row_level", "group_aware")
        ],
        "dataset_records": [
            {"split_policy": policy, "dataset": "synthetic-data",
             "included_in_association": True,
             "mechanism": {"mean_fold_median_selected_scaled_jacobian_norm": 3.25,
                           "n_finite_folds": 1},
             "performance": {"mean_auc_delta": 0.11, "n_paired_clean_cells": 1}}
            for policy in ("row_level", "group_aware")
        ],
    }
    path = tmp_path / "synthetic_association.json"
    path.write_text(json.dumps(note), encoding="utf-8")
    output = tmp_path / "figures"
    output.mkdir()
    figures, _ = add_mechanism(path, output)
    associations = _rows(output / "mechanism_associations.csv")
    datasets = _rows(output / "mechanism_dataset_records.csv")
    assert len(associations) == 2 and associations[0]["rho"] == "0.42"
    assert datasets[0]["mean_selected_scaled_jacobian_norm"] == "3.25"
    assert len(figures) == 2 and all((output / name).stat().st_size > 1000 for name in figures)


def test_separate_sensitivity_numbers_reach_tables_and_image(tmp_path):
    scope_path, _ = _fixture(tmp_path)
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    specs = (
        ("r1-v2-domain-row", "transductive_domain_partition", "row_level",
         ["covariate_partition", "population_partition"]),
        ("r1-v2-availability-row", "feature_availability_ablation", "row_level",
         ["feature_availability_ablation_0.20"]),
        ("r1-v2-availability-group", "feature_availability_ablation", "group_aware",
         ["feature_availability_ablation_0.20"]),
        ("r1-v2-relabel-row", "majority_label_relabeling", "row_level",
         ["majority_label_relabeling"]),
        ("r1-v2-relabel-group", "majority_label_relabeling", "group_aware",
         ["majority_label_relabeling"]),
    )
    for run_id, experiment_scope, policy, conditions in specs:
        scope["runs"].append({"run_id": run_id, "scope": experiment_scope,
                              "split_policy": policy, "condition_names": conditions,
                              "conditions": len(conditions), "intended_cells": 14 * len(conditions)})
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({
            "status": "complete", "run_id": run_id,
            "code_fingerprint": "synthetic-source", "expected_tasks": 14 * len(conditions),
            "configuration": {"split_policy": policy},
        }), encoding="utf-8")
        with sqlite3.connect(run_dir / "manifest_outcomes.sqlite") as connection:
            connection.execute("CREATE TABLE outcomes (task_key TEXT PRIMARY KEY, status TEXT, phase TEXT)")
            with (run_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
                for condition in conditions:
                    for pipeline in PIPELINES:
                        key = f"{run_id}:{condition}:{pipeline}"
                        connection.execute("INSERT INTO outcomes VALUES (?, 'success', 'phase2')", (key,))
                        enabled = _enabled(pipeline)
                        stream.write(json.dumps({
                            "task_key": key, "status": "success", "run_id": run_id,
                            "split_policy": policy, "dataset": "synthetic-data", "seed": 42,
                            "fold": 1, "condition": condition, "pipeline": pipeline,
                            "model": "synthetic-model", "roc_auc": 0.8 if pipeline == "AutoFE_Baseline" else 0.6,
                            "operator_configuration": {"enabled_operators": list(enabled)},
                            "operator_candidate_counts": {operator: {
                                "generated": 1 if operator in enabled else 0,
                                "rejected": 0, "duplicates": 0,
                                "eligible": 1 if operator in enabled else 0,
                                "selected": 1 if operator in enabled else 0,
                            } for operator in OPERATORS},
                        }) + "\n")
    output = tmp_path / "sensitivity_assets"
    output.mkdir()
    figures, manifest = add_sensitivities(tmp_path, scope, output)
    rows = _rows(output / "sensitivity_pipeline_auc_summary.csv")
    row = next(item for item in rows if item["run_id"] == "r1-v2-domain-row"
               and item["condition"] == "covariate_partition"
               and item["pipeline"] == "AutoFE_Baseline")
    assert float(row["mean_paired_auc_delta_vs_raw"]) == pytest.approx(0.2)
    assert len(rows) == 6 * 14
    assert len(_rows(output / "sensitivity_operator_candidate_counts.csv")) == 5 * 14 * 4
    assert manifest["pooled_with_primary"] is False
    assert len(figures) == 2 and all((output / name).stat().st_size > 1000 for name in figures)
