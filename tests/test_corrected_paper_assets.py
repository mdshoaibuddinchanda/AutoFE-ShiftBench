"""Small synthetic contract check for corrected paper tables and images."""

from __future__ import annotations

import csv
import json
import sqlite3

import pytest

from provenance.generate_corrected_assets import (
    OPERATORS, _enabled, add_mechanism, add_sensitivities, collect_primary, dataset_contrasts,
    generate, missingness_sensitivity, row_group_comparisons,
)


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
                        "n_original": 20, "preprocessing_time_s": 1.25,
                        "autofe_gen_time_s": 2.5, "preparation_time_s": 3.75,
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
    assert float(multiply["mean_dataset_n_original"]) == 20
    assert float(multiply["mean_dataset_preprocessing_time_s"]) == 1.25
    assert float(multiply["mean_dataset_preparation_time_s"]) == 3.75
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
    assert len(manifest["figures"]) == 16
    assert len(manifest['tables']) == 15
    resources = _rows(output/'primary_resource_policy_and_use.csv')
    assert resources[0]['ram_budget_gib'] == ''
    assert len(_rows(output / "primary_f1_contrasts.csv")) == 8
    assert len(_rows(output / "row_group_dataset_auc.csv")) == 14
    assert len(_rows(output / "primary_missingness_auc_bounds.csv")) == 28
    comparison = next(row for row in _rows(output / "row_group_pipeline_auc_summary.csv")
                      if row["pipeline"] == "AutoFE_LeaveOut_Multiply")
    assert float(comparison["mean_delta"]) == 0.0
    assert manifest["scientific_complete"] is False  # custom scope is a pilot diagnostic
    record = (output / manifest["result_record"]).read_text(encoding="utf-8")
    assert "pilot code verification; not corrected performance" in record
    assert "Frozen F1 dataset contrasts" in record and "Missing-outcome sensitivity" in record
    assert all((output / name).stat().st_size > 1000 for name in manifest["figures"])


def test_resource_budget_and_observed_numbers_reach_resource_table_and_figure(tmp_path):
    from provenance.generate_corrected_assets import resource_summary, plot_resource_summary
    source=dict(run_id='synthetic', configured_workers=6,
                resource_plan=dict(policy='adaptive-v1', ram_budget_bytes=int(32*1024**3*.8),
                    ram_reserve_bytes=int(32*1024**3*.2),cache_max_bytes=12*1024**3,
                    settings=dict(reserve_cpus=2,vram_target_fraction=.8),
                    hardware=dict(allowed_cpu_ids=list(range(8)),ram_total_bytes=32*1024**3,
                                  gpu_devices=[dict(total_bytes=4*1024**3)])),
                resource_usage=dict(observed_process_tree_rss_bytes=20*1024**3),
                maximum_worker_sampled_rss_bytes=2*1024**3,
                model_backend_counts=dict(cpu=80,gpu=20))
    rows=resource_summary([source,source])
    assert rows[0]['ram_budget_gib']==pytest.approx(25.6)
    assert rows[0]['observed_process_tree_rss_gib']==20
    assert rows[0]['configured_worker_ceiling']==6 and rows[0]['gpu_success_cells']==20
    assert rows[0]['vram_admission_budget_gib']==pytest.approx(3.2)
    files=plot_resource_summary(rows,tmp_path,diagnostic=True,missing_outcomes=False)
    assert len(files)==2 and all((tmp_path/p).stat().st_size>1000 for p in files)


def _dataset_summary_fixture():
    scope = {"datasets": [{"name": "a"}, {"name": "b"},
                           {"name": "c", "group_auc_infeasible_seeds": [42]}],
             "pipelines": list(PIPELINES)}
    rows = []
    for policy in ("row_level", "group_aware"):
        for dataset in ("a", "b", "c"):
            for pipeline in PIPELINES:
                raw = {"a": 0.6, "b": 0.5, "c": 0.8}[dataset]
                baseline = {"a": 0.7, "b": 0.4, "c": 0.85}[dataset]
                if policy == "group_aware":
                    raw += 0.05 if dataset == "a" else -0.05
                    baseline += 0.1
                value = raw if pipeline == "Raw" else baseline
                ineligible = policy == "group_aware" and dataset == "c"
                rows.append({
                    "run_id": policy, "split_policy": policy, "dataset": dataset, "pipeline": pipeline,
                    "expected_cells": 2, "success_cells": 0 if ineligible else 2,
                    "skipped_cells": 2 if ineligible else 0, "failed_cells": 0, "timed_out_cells": 0,
                    "mean_roc_auc": None if ineligible else value, "complete_auc": not ineligible,
                    "auc_eligible": not ineligible, "outcome_reasons": "class unsupported" if ineligible else "",
                    "auc_ineligibility_reason": "all_seed_group_auc_undefined" if ineligible else None,
                })
    return scope, rows


def test_dataset_f1_and_policy_comparisons_use_common_complete_dataset_units():
    scope, rows = _dataset_summary_fixture()
    _, summaries = dataset_contrasts(rows, scope)
    f1 = [row for row in summaries if row["family_id"] == "F1_primary_roc_auc"]
    assert len(f1) == 8 and all(row["family_size"] == 4 for row in f1)
    group = next(row for row in f1 if row["split_policy"] == "group_aware"
                 and row["pipeline_b"] == "AutoFE_Baseline")
    assert group["configured_datasets"] == 3 and group["auc_eligible_datasets"] == 2
    assert group["n_complete_datasets"] == 2
    assert group["mean_delta"] == pytest.approx(0.1)
    assert group["p_value"] is not None and group["holm_adjusted_p_value"] >= group["p_value"]
    assert group["bootstrap_replicates"] == 10000 and group["bootstrap_seed"] == 20260929
    assert group["ci_low"] == pytest.approx(0.05) and group["ci_high"] == pytest.approx(0.15)
    assert dataset_contrasts(rows, scope)[1] == summaries
    _, policies, contrasts = row_group_comparisons(rows, scope)
    baseline = next(row for row in policies if row["pipeline"] == "AutoFE_Baseline")
    assert baseline["row_complete_datasets"] == 3 and baseline["group_complete_datasets"] == 2
    assert baseline["common_complete_dataset_names"] == "a;b"
    assert baseline["mean_row_auc_common_datasets"] == pytest.approx(0.55)
    assert baseline["mean_group_auc_common_datasets"] == pytest.approx(0.65)
    assert baseline["mean_delta"] == pytest.approx(0.1)
    contrast = next(row for row in contrasts if row["pipeline_b"] == "AutoFE_Baseline")
    assert contrast["mean_row_delta_common_datasets"] == pytest.approx(0)
    assert contrast["mean_group_delta_common_datasets"] == pytest.approx(0.1)
    assert contrast["mean_delta"] == pytest.approx(0.1)
    assert all(row["p_value"] is None for row in summaries if row["family_id"] != "F1_primary_roc_auc")


def test_missing_auc_bounds_include_unknown_outcomes_without_bounding_undefined_group_auc():
    scope, rows = _dataset_summary_fixture()
    target = next(row for row in rows if row["split_policy"] == "group_aware"
                  and row["dataset"] == "a" and row["pipeline"] == "AutoFE_Baseline")
    target.update(success_cells=1, failed_cells=1, complete_auc=False, outcome_reasons="failed: MemoryError")
    score_bounds, contrast_bounds, summaries = missingness_sensitivity(rows, scope)
    score = next(row for row in score_bounds if row["split_policy"] == "group_aware"
                 and row["dataset"] == "a" and row["pipeline"] == "AutoFE_Baseline")
    assert score["unknown_eligible_cells"] == 1
    assert score["mean_auc_lower_bound"] == pytest.approx(0.4)
    assert score["mean_auc_upper_bound"] == pytest.approx(0.9)
    undefined = next(row for row in score_bounds if row["split_policy"] == "group_aware"
                     and row["dataset"] == "c" and row["pipeline"] == "AutoFE_Baseline")
    assert undefined["expected_eligible_cells"] == 0 and undefined["unknown_eligible_cells"] == 0
    assert undefined["mean_auc_lower_bound"] is None and undefined["mean_auc_upper_bound"] is None
    contrast = next(row for row in contrast_bounds if row["split_policy"] == "group_aware"
                    and row["dataset"] == "a" and row["contrast_id"] == "roc_auc_raw_baseline")
    assert contrast["delta_lower_bound"] == pytest.approx(-0.25)
    assert contrast["delta_upper_bound"] == pytest.approx(0.25)
    summary = next(row for row in summaries if row["split_policy"] == "group_aware"
                   and row["contrast_id"] == "roc_auc_raw_baseline")
    assert summary["auc_eligible_datasets"] == 2 and summary["datasets_with_unknown_outcomes"] == 1
    assert summary["mean_dataset_delta_lower_bound"] == pytest.approx(-0.1)
    assert summary["mean_dataset_delta_upper_bound"] == pytest.approx(0.15)
    _, contrasts = dataset_contrasts(rows, scope)
    complete = next(row for row in contrasts if row["split_policy"] == "group_aware"
                    and row["contrast_id"] == "roc_auc_raw_baseline")
    assert complete["n_complete_datasets"] == 1
    assert complete["p_value"] is None and complete["ci_low"] is None


def test_terminal_failures_are_visibly_reported_with_bounds(tmp_path):
    scope_path, (row_dir, group_dir) = _fixture(tmp_path)
    path = group_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "completed_with_failures"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    path = group_dir / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    target = next(row for row in rows if row["pipeline"] == "AutoFE_Baseline")
    target.update(status="failed", roc_auc=None, exception_type="MemoryError")
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with sqlite3.connect(group_dir / "manifest_outcomes.sqlite") as connection:
        connection.execute("UPDATE outcomes SET status='failed' WHERE task_key=?", (target["task_key"],))
    output = tmp_path / "failure_assets"
    manifest = generate(row_dir, group_dir, output, scope_path=scope_path)
    assert manifest["has_terminal_missing_outcomes"] and not manifest["scientific_complete"]
    assert manifest["source_runs"][1]["counts_by_status"]["failed"] == 1
    bounds = next(row for row in _rows(output / "primary_missingness_auc_bounds.csv")
                  if row["split_policy"] == "group_aware" and row["pipeline"] == "AutoFE_Baseline")
    assert bounds["mean_auc_lower_bound"] == "0.0" and bounds["mean_auc_upper_bound"] == "1.0"
    assert "MemoryError" in bounds["outcome_reasons"]


def test_operator_counts_deduplicate_classifier_copies_and_missing_resources_stay_missing(tmp_path):
    scope_path, (row_dir, _group_dir) = _fixture(tmp_path)
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["models"].append("second-model")
    scope["runs"][0]["intended_cells"] = 28
    path = row_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["expected_tasks"] = 28
    path.write_text(json.dumps(manifest), encoding="utf-8")
    path = row_dir / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    with sqlite3.connect(row_dir / "manifest_outcomes.sqlite") as connection:
        for row in rows[:]:
            copy = {**row, "task_key": row["task_key"] + ":second", "model": "second-model"}
            rows.append(copy)
            connection.execute("INSERT INTO outcomes VALUES (?, 'success', 'phase2')", (copy["task_key"],))
    for row in rows:
        row.pop("preparation_time_s")
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    datasets, _conditions, operators, source = collect_primary(row_dir, scope, "row_level")
    reference = next(row for row in operators if row["pipeline"] == "AutoFE_Baseline"
                     and row["operator"] == "multiply_numeric")
    assert reference["generated"] == 7 and reference["observed_feature_tasks"] == 1
    assert source["terminal_rows"] == 28
    assert all(row["mean_preparation_time_s"] is None and row["n_preparation_time_s_cells"] == 0
               for row in datasets)


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
