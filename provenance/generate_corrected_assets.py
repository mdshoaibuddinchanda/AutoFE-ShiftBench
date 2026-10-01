"""Stream frozen primary results into paper tables and standalone figures.

Run only after both primary manifests are terminal. Historical paper assets are
never overwritten. The optional mechanism association is a separate, clean-only
exploratory analysis and retains its own denominator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from src.pipeline_runner import PIPELINE_CONFIGS
from src.provenance import atomic_write_json, file_sha256, stable_digest
from src.reviewer1_analysis import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, paired_dataset_bootstrap
from src.stats_analysis import PRIMARY_CONDITIONS, PRESPECIFIED_PIPELINES, _holm_adjust


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ("add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric")
COUNT_FIELDS = ("generated", "rejected", "duplicates", "eligible", "selected")
POLICIES = ("row_level", "group_aware")
SECONDARY_METRICS = (
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
    "pr_auc", "log_loss", "brier_score", "train_auc", "train_time_s",
    "infer_time_s", "autofe_gen_time_s", "ram_used_mb", "n_generated",
    "n_retained", "n_original", "preprocessing_time_s", "preparation_time_s",
    'worker_peak_rss_bytes', 'gpu_ram_part', 'train_infer_time_s', 'scoring_time_s', 'input_open_time_s', 'worker_elapsed_s',
)


def _auc_ineligibility(scope: dict, policy: str, dataset: str) -> str | None:
    """Prespecified undefined AUC cells have no missing-outcome bounds."""
    if policy == "group_aware":
        item = next(item for item in scope["datasets"] if item["name"] == dataset)
        seeds = item.get("group_auc_infeasible_seeds", [])
        if seeds:
            return "all_configured_seed_group_auc_infeasibility; seeds=" + ",".join(map(str, seeds))
    return None


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _enabled(pipeline: str) -> tuple[str, ...]:
    config = PIPELINE_CONFIGS[pipeline]
    return tuple(config.trans_primitives) if config.enable_dfs else ()


def _terminal_lookup(run_dir: Path, manifest: dict):
    ledger = run_dir / "manifest_outcomes.sqlite"
    if ledger.exists():
        connection = sqlite3.connect(f"file:{ledger.as_posix()}?mode=ro", uri=True)
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='outcomes'"
        ).fetchone()
        if table is None:
            connection.close()
            raise ValueError(f"Compact outcome ledger lacks outcomes table: {ledger}")

        def lookup(key: str) -> str | None:
            row = connection.execute("SELECT status FROM outcomes WHERE task_key=?", (key,)).fetchone()
            return row[0] if row else None

        return lookup, connection
    tasks = {row["task_key"]: row["status"] for row in manifest.get("tasks", [])}
    if not tasks:
        raise ValueError(f"No terminal outcome ledger in {run_dir}")
    return tasks.get, None


def collect_primary(run_dir: Path, scope: dict, policy: str,
                    run_record: dict | None = None) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Keep dataset summaries and one count record per feature task."""
    manifest_path = run_dir / "manifest.json"
    result_path = run_dir / "results.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = run_record or next(item for item in scope["runs"]
                                if item["scope"] == "primary" and item["split_policy"] == policy)
    expected_id = target["run_id"]
    if (manifest.get("status") not in {"complete", "completed_with_failures"} or manifest.get("run_id") != expected_id
            or manifest.get("code_fingerprint") != scope["code_fingerprint"]
            or manifest.get("configuration", {}).get("split_policy") != policy):
        raise ValueError(f"Primary run is incomplete or has a different frozen identity: {run_dir}")
    if manifest.get("expected_tasks") != target["intended_cells"]:
        raise ValueError(f"Primary task denominator changed: {run_dir}")
    if scope.get('resource_plan') and manifest.get('configuration', {}).get('resource_plan') != scope['resource_plan']:
        raise ValueError('Primary run differs from the frozen adaptive resource plan')
    if scope.get('_resource_profile_sha256') and manifest.get('configuration', {}).get('resource_profile_sha256') != scope['_resource_profile_sha256']:
        raise ValueError('Primary run used a different frozen resource profile')
    if scope.get('execution_reuse'):
        configuration=manifest.get('configuration',{})
        frozen=scope['execution_reuse']
        policy_settings=configuration.get('array_transport_policy',{})
        if (configuration.get('reuse_preprocessing')!=frozen['preprocessing_reuse']
                or configuration.get('array_transport')!=frozen['array_transport']
                or policy_settings.get('budget_bytes')!=frozen['array_transport_budget_bytes']
                or policy_settings.get('threshold_bytes')!=frozen['array_transport_threshold_bytes']):
            raise ValueError('Primary run differs from the frozen execution reuse policy')
    lookup, connection = _terminal_lookup(run_dir, manifest)
    aggregate = defaultdict(lambda: {"success": 0, "skipped": 0, "failed": 0,
                                     "timed_out": 0, "auc_sum": 0.0, "reasons": set(),
                                     "outcome_reasons": set(),
                                     "metric_sum": defaultdict(float),
                                     "metric_count": defaultdict(int)})
    conditions = defaultdict(lambda: {"success": 0, "auc_sum": 0.0})
    feature_seen = {}
    counts = defaultdict(lambda: {field: 0 for field in COUNT_FIELDS})
    feature_count = defaultdict(int)
    included_rows = 0
    parameter_catalog = {}
    backend_counts = defaultdict(int)
    transport_counts = defaultdict(int)
    execution_timing = {k: {"sum":0.0,"count":0} for k in ("train_infer_time_s","scoring_time_s","input_open_time_s")}
    preprocessing_reused_features = 0
    preprocessing_recorded_features = 0
    comparison_backends = {}
    fallback_cells = 0
    worker_peak = None
    dataset_names = {item["name"] for item in scope["datasets"]}
    pipeline_names = set(scope["pipelines"])
    enabled_by_pipeline = {name: set(_enabled(name)) for name in scope["pipelines"]}
    run_conditions = target.get("condition_names", scope["primary_conditions"])
    condition_names = set(run_conditions)
    try:
        with result_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row["task_key"])
                status = str(row["status"])
                if lookup(key) != status:
                    continue  # prior failed attempt, later recovered
                if row.get("split_policy") != policy or row.get("run_id") != expected_id:
                    raise ValueError(f"Result identity differs from primary manifest: {key}")
                dataset, pipeline = str(row["dataset"]), str(row["pipeline"])
                if dataset not in dataset_names or pipeline not in pipeline_names:
                    raise ValueError(f"Unexpected dataset or pipeline: {dataset}, {pipeline}")
                if str(row["condition"]) not in condition_names:
                    raise ValueError(f"Unexpected primary condition: {row['condition']}")
                bucket = aggregate[(dataset, pipeline)]
                if status not in {"success", "skipped", "failed", "timed_out"}:
                    raise ValueError(f"Unexpected terminal status: {status}")
                bucket[status] += 1
                included_rows += 1
                if status == "skipped":
                    bucket["reasons"].add(str(row.get("skip_reason", "unspecified")))
                if status != "success":
                    bucket["outcome_reasons"].add(status + ": " + str(
                        row.get("skip_reason") or row.get("exception_type")
                        or row.get("error_summary") or "unspecified"))
                    continue
                auc = row.get("roc_auc")
                if auc is None or not math.isfinite(float(auc)) or not 0 <= float(auc) <= 1:
                    raise ValueError(f"Successful cell lacks finite ROC-AUC: {key}")
                bucket["auc_sum"] += float(auc)
                backend_counts[str(row.get('model_backend', 'unrecorded'))] += 1
                transport_counts[str(row.get('array_transport', 'unrecorded'))] += 1
                for field, summary in execution_timing.items():
                    value = row.get(field)
                    if value is not None and math.isfinite(float(value)):
                        summary['sum'] += float(value)
                        summary['count'] += 1
                comparison = tuple(row.get(field) for field in ('dataset', 'seed', 'fold', 'condition', 'model'))
                backend = row.get('model_backend', 'unrecorded')
                if comparison_backends.setdefault(comparison, backend) != backend:
                    raise ValueError('Classifiers use different backends within a pipeline comparison')
                fallback_cells += row.get('model_backend_reason') == 'static matrix estimate exceeds GPU VRAM budget'
                peak = row.get('worker_peak_rss_bytes')
                if peak is not None:
                    worker_peak = max(worker_peak or 0, int(peak))
                fingerprint = row.get('model_parameters_fingerprint')
                if fingerprint and fingerprint not in parameter_catalog:
                    catalog_path = (run_dir / row['model_parameters_path']).resolve()
                    if not catalog_path.is_relative_to((run_dir / 'model_parameters').resolve()):
                        raise ValueError('Model parameter catalog path escapes its run directory')
                    parameters = json.loads(catalog_path.read_text(encoding='utf-8'))
                    if stable_digest(parameters) != fingerprint or parameters.get('model') != row['model']:
                        raise ValueError('Model parameter catalog identity changed')
                    parameter_catalog[fingerprint] = file_sha256(catalog_path)
                if manifest.get('configuration', {}).get('resource_plan') and not fingerprint:
                    raise ValueError('Adaptive result lacks its resolved model parameter catalog')
                condition_bucket = conditions[(dataset, pipeline, str(row["condition"]))]
                condition_bucket["success"] += 1
                condition_bucket["auc_sum"] += float(auc)
                for metric in SECONDARY_METRICS:
                    value = row.get(metric)
                    if value is not None and math.isfinite(float(value)):
                        bucket["metric_sum"][metric] += float(value)
                        bucket["metric_count"][metric] += 1
                configured = tuple(row.get("operator_configuration", {}).get("enabled_operators", []))
                if set(configured) != enabled_by_pipeline[pipeline]:
                    raise ValueError(f"Operator metadata conflicts with configuration: {key}")
                feature_key = (dataset, int(row["seed"]), int(row["fold"]),
                               str(row["condition"]), pipeline)
                candidate = row.get("operator_candidate_counts")
                if not isinstance(candidate, dict):
                    raise ValueError(f"Missing per-operator candidate counts: {key}")
                signature = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
                if feature_key in feature_seen:
                    if feature_seen[feature_key] != signature:
                        raise ValueError(f"Candidate counts differ across classifiers: {feature_key}")
                    continue
                feature_seen[feature_key] = signature
                if "preprocessing_reused" in row:
                    preprocessing_recorded_features += 1
                    preprocessing_reused_features += bool(row["preprocessing_reused"])
                feature_count[pipeline] += 1
                for operator in OPERATORS:
                    values = candidate.get(operator, {})
                    for field in COUNT_FIELDS:
                        value = int(values.get(field, 0) or 0)
                        if value < 0 or (operator not in enabled_by_pipeline[pipeline] and value):
                            raise ValueError(f"Disabled or negative operator count: {feature_key}, {operator}, {field}")
                        counts[(pipeline, operator)][field] += value
    finally:
        if connection is not None:
            connection.close()
    if included_rows != manifest["expected_tasks"]:
        raise ValueError(f"Terminal result coverage differs from manifest: {included_rows} != {manifest['expected_tasks']}")
    expected_cells = len(scope["seeds"]) * len(scope["folds"]) * target["conditions"] * len(scope["models"])
    expected_features = len(scope["seeds"]) * len(scope["folds"]) * target["conditions"]
    dataset_rows, condition_rows = [], []
    for item in scope["datasets"]:
        dataset = item["name"]
        for pipeline in scope["pipelines"]:
            bucket = aggregate[(dataset, pipeline)]
            success = bucket["success"]
            if sum(bucket[status] for status in ("success", "skipped", "failed", "timed_out")) != expected_cells:
                raise ValueError(f"Dataset/pipeline coverage differs from frozen grid: {policy}, {dataset}, {pipeline}")
            dataset_rows.append({
                "run_id": expected_id, "experiment_scope": target["scope"] if "scope" in target else "primary",
                "split_policy": policy, "dataset": dataset, "pipeline": pipeline,
                "expected_cells": expected_cells, "success_cells": success,
                "skipped_cells": bucket["skipped"], "failed_cells": bucket["failed"],
                "timed_out_cells": bucket["timed_out"],
                "coverage_fraction": success / expected_cells,
                "mean_roc_auc": bucket["auc_sum"] / success if success else None,
                "complete_auc": success == expected_cells,
                "skip_reasons": "; ".join(sorted(bucket["reasons"])),
                "outcome_reasons": "; ".join(sorted(bucket["outcome_reasons"])),
                "auc_eligible": _auc_ineligibility(scope, policy, dataset) is None,
                "auc_ineligibility_reason": _auc_ineligibility(scope, policy, dataset),
                **{f"mean_{metric}": bucket["metric_sum"][metric] / bucket["metric_count"][metric]
                   if bucket["metric_count"][metric] else None for metric in SECONDARY_METRICS},
                **{f"n_{metric}_cells": bucket["metric_count"][metric]
                   for metric in SECONDARY_METRICS},
            })
            for condition in run_conditions:
                part = conditions[(dataset, pipeline, condition)]
                condition_rows.append({
                    "run_id": expected_id, "experiment_scope": target["scope"] if "scope" in target else "primary",
                    "split_policy": policy, "dataset": dataset, "pipeline": pipeline,
                    "condition": condition,
                    "expected_cells": len(scope["seeds"]) * len(scope["folds"]) * len(scope["models"]),
                    "success_cells": part["success"],
                    "mean_roc_auc": part["auc_sum"] / part["success"] if part["success"] else None,
                    "complete_auc": part["success"] == len(scope["seeds"]) * len(scope["folds"]) * len(scope["models"]),
                })
    operator_rows = []
    for pipeline in scope["pipelines"]:
        for operator in OPERATORS:
            values = counts[(pipeline, operator)]
            observed = feature_count[pipeline]
            operator_rows.append({
                "run_id": expected_id, "experiment_scope": target["scope"] if "scope" in target else "primary",
                "split_policy": policy, "pipeline": pipeline, "operator": operator,
                "enabled": operator in _enabled(pipeline),
                "expected_feature_tasks": expected_features * len(scope["datasets"]),
                "observed_feature_tasks": observed,
                **values,
                "generated_per_observed_feature_task": values["generated"] / observed if observed else None,
                "selected_per_observed_feature_task": values["selected"] / observed if observed else None,
            })
    return dataset_rows, condition_rows, operator_rows, {
        "run_id": expected_id, "manifest_sha256": file_sha256(manifest_path),
        "manifest_path": str(manifest_path), "results_path": str(result_path),
        "code_commit": manifest.get("code_commit"),
        "configuration_fingerprint": manifest.get("configuration_fingerprint"),
        "runtime_fingerprint": manifest.get("runtime_fingerprint"),
        "results_sha256": file_sha256(result_path), "terminal_rows": included_rows,
        "counts_by_status": {status: sum(bucket[status] for bucket in aggregate.values())
                             for status in ("success", "failed", "skipped", "timed_out")},
        "terminal_manifest_status": manifest["status"],
        'resource_plan': manifest.get('configuration', {}).get('resource_plan'),
        'resource_usage': manifest.get('resource_usage'),
        'array_transport_usage': manifest.get('array_transport_usage'),
        'transport_counts': dict(transport_counts), 'execution_timing': execution_timing,
        'preprocessing_reused_features': preprocessing_reused_features if preprocessing_recorded_features else None,
        'preprocessing_recorded_features': preprocessing_recorded_features or None,
        'configured_workers': manifest.get('configuration', {}).get('workers'),
        'model_backend_counts': dict(backend_counts),
        'capacity_cpu_fallback_cells': fallback_cells,
        'maximum_worker_sampled_rss_bytes': worker_peak,
        'model_parameter_catalog_sha256': parameter_catalog,
    }


def resource_summary(sources: list[dict]) -> list[dict]:
    rows = []
    for policy, source in zip(POLICIES, sources):
        plan = source.get('resource_plan') or {}
        usage = source.get('resource_usage') or {}
        transport = source.get('array_transport_usage') or {}
        timing = source.get('execution_timing') or {}
        settings = plan.get('settings', {})
        hardware = plan.get('hardware', {})
        devices = hardware.get('gpu_devices', [])
        rows.append(dict(
            run_id=source['run_id'], split_policy=policy,
            resource_policy=plan.get('policy', 'unrecorded'),
            logical_cpu_slots=len(hardware['allowed_cpu_ids']) if 'allowed_cpu_ids' in hardware else None,
            reserved_cpu_slots=settings.get('reserve_cpus'),
            configured_worker_ceiling=source.get('configured_workers'),
            ram_total_gib=hardware.get('ram_total_bytes', 0)/1024**3 if hardware else None,
            ram_budget_gib=plan['ram_budget_bytes']/1024**3 if plan else None,
            ram_reserve_gib=plan['ram_reserve_bytes']/1024**3 if plan else None,
            observed_process_tree_rss_gib=usage['observed_process_tree_rss_bytes']/1024**3 if usage else None,
            maximum_worker_sampled_rss_gib=source['maximum_worker_sampled_rss_bytes']/1024**3
                if source.get('maximum_worker_sampled_rss_bytes') is not None else None,
            vram_total_gib=sum(d['total_bytes'] for d in devices)/1024**3 if plan else None,
            vram_target_fraction=settings.get('vram_target_fraction'),
            vram_admission_budget_gib=sum(d['total_bytes'] for d in devices)*settings.get('vram_target_fraction', 0)/1024**3 if plan else None,
            cache_cap_gib=plan['cache_max_bytes']/1024**3 if plan else None,
            cpu_success_cells=source.get('model_backend_counts', {}).get('cpu', 0),
            gpu_success_cells=source.get('model_backend_counts', {}).get('gpu', 0),
            unrecorded_backend_cells=source.get('model_backend_counts', {}).get('unrecorded', 0),
            capacity_cpu_fallback_cells=source.get('capacity_cpu_fallback_cells', 0),
            ram_admission_waits=usage.get('ram_waits'),
            gpu_admission_waits=usage.get('gpu_waits'),
            preparation_waits=usage.get('preparation_waits'),
            distinct_parameter_records=len(source.get('model_parameter_catalog_sha256', {})),
            mapped_success_cells=source.get('transport_counts', {}).get('mapped',0),
            pickle_success_cells=source.get('transport_counts', {}).get('pickle',0),
            private_copy_success_cells=source.get('transport_counts', {}).get('private_copy',0),
            unrecorded_transport_cells=source.get('transport_counts', {}).get('unrecorded',0),
            array_transport_peak_disk_gib=transport.get('peak_disk_bytes',0)/1024**3 if transport else None,
            array_transport_budget_gib=transport.get('budget_bytes',0)/1024**3 if transport else None,
            array_transport_threshold_mib=transport.get('threshold_bytes',0)/1024**2 if transport else None,
            preprocessing_reused_feature_tasks=source.get('preprocessing_reused_features'),
            preprocessing_recorded_feature_tasks=source.get('preprocessing_recorded_features'),
            **{f'sum_{field}': timing[field]['sum'] if timing.get(field, {}).get('count') else None
               for field in ('train_infer_time_s','scoring_time_s','input_open_time_s')},
            measurement='RAM: sampled RSS; GPU budget: estimate, not observed allocation; native allocation is not forcibly capped',
        ))
    return rows


def plot_resource_summary(rows: list[dict], output_dir: Path, **options) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes=axes.flat
    labels = [r['split_policy'].replace('_', ' ') for r in rows]
    x = np.arange(len(rows))
    for ax, fields, title, unit in (
        (axes[0], ('ram_budget_gib', 'observed_process_tree_rss_gib'), 'RAM budget and sampled use', 'GiB'),
        (axes[1], ('cpu_success_cells', 'gpu_success_cells', 'unrecorded_backend_cells'), 'Actual successful fit backends', 'Cells'),
        (axes[2], ('mapped_success_cells','pickle_success_cells','private_copy_success_cells','unrecorded_transport_cells'), 'Actual array transport per successful fit', 'Cells'),
        (axes[3], ('sum_train_infer_time_s','sum_scoring_time_s','sum_input_open_time_s'), 'Sum of worker phase times (not wall time)', 'Seconds'),
    ):
        for index, field in enumerate(fields):
            values = [r.get(field) for r in rows]
            width = .8/len(fields)
            ax.bar(x+(index-(len(fields)-1)/2)*width, [v if v is not None else np.nan for v in values], width,
                   label=field.replace('_', ' '))
        if all(r.get(fields[0]) is None for r in rows):
            ax.text(.5, .5, 'Resource policy not recorded', ha='center', transform=ax.transAxes)
        ax.set_xticks(x, labels); ax.set_ylabel(unit); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(rect=(0,0,1,.89))
    return _save_figure(fig, output_dir/'primary_resource_budget_and_use', **options)


def pipeline_summary(dataset_rows: list[dict], scope: dict) -> list[dict]:
    by_key = {(row["split_policy"], row["dataset"], row["pipeline"]): row
              for row in dataset_rows}
    output = []
    for policy in POLICIES:
        for pipeline in scope["pipelines"]:
            scores, paired_delta = [], []
            for item in scope["datasets"]:
                dataset = item["name"]
                candidate = by_key[(policy, dataset, pipeline)]
                baseline = by_key[(policy, dataset, "AutoFE_Baseline")]
                if candidate["complete_auc"]:
                    scores.append(candidate["mean_roc_auc"])
                    if baseline["complete_auc"]:
                        paired_delta.append(candidate["mean_roc_auc"] - baseline["mean_roc_auc"])
            complete_rows = [by_key[(policy, item["name"], pipeline)] for item in scope["datasets"]
                             if by_key[(policy, item["name"], pipeline)]["complete_auc"]]
            output.append({
                "split_policy": policy, "pipeline": pipeline,
                "configured_datasets": len(scope["datasets"]),
                "complete_auc_datasets": len(scores),
                "paired_to_baseline_datasets": len(paired_delta),
                "mean_dataset_roc_auc": float(np.mean(scores)) if scores else None,
                "mean_paired_auc_delta_vs_full_arithmetic": float(np.mean(paired_delta)) if paired_delta else None,
                **{f"mean_dataset_{metric}": float(np.mean(values)) if values else None
                   for metric in SECONDARY_METRICS
                   for values in [[row[f"mean_{metric}"] for row in complete_rows
                                   if row[f"mean_{metric}"] is not None]]},
                "interpretation": "descriptive; corrected primary run",
            })
    return output


def condition_summary(rows: list[dict], scope: dict) -> list[dict]:
    output = []
    for policy in POLICIES:
        for condition in scope.get("primary_conditions", []):
            for pipeline in scope["pipelines"]:
                complete = [row["mean_roc_auc"] for row in rows
                            if row["split_policy"] == policy and row["condition"] == condition
                            and row["pipeline"] == pipeline and row["complete_auc"]]
                output.append({
                    "split_policy": policy, "condition": condition,
                    "pipeline": pipeline, "configured_datasets": len(scope["datasets"]),
                    "complete_auc_datasets": len(complete),
                    "mean_dataset_roc_auc": float(np.mean(complete)) if complete else None,
                })
    return output


def _dataset_statistics(differences: list[float]) -> dict:
    """Each input is one complete dataset, never a fold or prediction pair."""
    values = np.asarray(differences, dtype=float)
    interval = paired_dataset_bootstrap(values) if len(values) >= 2 else (None, None)
    return {
        "n_complete_datasets": len(values),
        "mean_delta": float(values.mean()) if len(values) else None,
        "median_delta": float(np.median(values)) if len(values) else None,
        "win_count": int((values > 0).sum()), "tie_count": int((values == 0).sum()),
        "loss_count": int((values < 0).sum()), "ci_low": interval[0], "ci_high": interval[1],
        "ci_method": "paired_dataset_bootstrap_percentile", "ci_level": 0.95,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "independent_unit": "dataset", "tie_tolerance": 0.0,
    }


def _contrast_definitions(scope: dict) -> list[tuple[str, str, str, str]]:
    definitions = [(f"roc_auc_raw_{pipeline.removeprefix('AutoFE_').lower()}",
                    "Raw", pipeline, "F1_primary_roc_auc")
                   for pipeline in PRESPECIFIED_PIPELINES]
    definitions.extend([
        ("roc_auc_raw_capmatched", "Raw", "Raw_CapMatched", "descriptive_matched_cap"),
        ("roc_auc_capmatched_baseline", "Raw_CapMatched", "AutoFE_Baseline", "descriptive_matched_cap"),
    ])
    definitions.extend((f"roc_auc_full_arithmetic_{pipeline.removeprefix('AutoFE_').lower()}",
                        "AutoFE_Baseline", pipeline, "descriptive_operator_ablation")
                       for pipeline in scope["pipelines"]
                       if pipeline.startswith(("AutoFE_Isolate_", "AutoFE_LeaveOut_"))
                       or pipeline == "AutoFE_NoMultiply")
    return definitions


def dataset_contrasts(dataset_rows: list[dict], scope: dict) -> tuple[list[dict], list[dict]]:
    """F1 remains the frozen four-contrast family within each split policy."""
    indexed = {(row["split_policy"], row["dataset"], row["pipeline"]): row for row in dataset_rows}
    per_dataset, summaries = [], []
    definitions = _contrast_definitions(scope)
    for policy in POLICIES:
        local_f1 = []
        for contrast_id, reference, candidate, family in definitions:
            differences = []
            eligible = 0
            for item in scope["datasets"]:
                name = item["name"]
                a, b = indexed[(policy, name, reference)], indexed[(policy, name, candidate)]
                eligible += int(a["auc_eligible"] and b["auc_eligible"])
                complete = a["complete_auc"] and b["complete_auc"] and a["auc_eligible"] and b["auc_eligible"]
                delta = b["mean_roc_auc"] - a["mean_roc_auc"] if complete else None
                if complete:
                    differences.append(delta)
                per_dataset.append({
                    "split_policy": policy, "dataset": name, "contrast_id": contrast_id,
                    "family_id": family, "pipeline_a": reference, "pipeline_b": candidate,
                    "expected_cells_per_pipeline": a["expected_cells"],
                    "success_cells_a": a["success_cells"], "success_cells_b": b["success_cells"],
                    "mean_auc_a": a["mean_roc_auc"], "mean_auc_b": b["mean_roc_auc"],
                    "complete_dataset_pair": complete, "delta_b_minus_a": delta,
                    "auc_eligible": a["auc_eligible"] and b["auc_eligible"],
                    "outcome_reasons_a": a["outcome_reasons"], "outcome_reasons_b": b["outcome_reasons"],
                    "auc_ineligibility_reason": a["auc_ineligibility_reason"] or b["auc_ineligibility_reason"],
                })
            row = {
                "split_policy": policy, "contrast_id": contrast_id, "family_id": family,
                "pipeline_a": reference, "pipeline_b": candidate,
                "configured_datasets": len(scope["datasets"]), "auc_eligible_datasets": eligible,
                **_dataset_statistics(differences), "p_value": None,
                "holm_adjusted_p_value": None,
                "family_size": len(PRESPECIFIED_PIPELINES) if family == "F1_primary_roc_auc" else None,
                "interpretation": "confirmatory complete-dataset contrast" if family == "F1_primary_roc_auc"
                                  else "descriptive complete-dataset contrast; no multiplicity family",
            }
            if family == "F1_primary_roc_auc":
                if len(differences) >= 2:
                    row["p_value"] = 1.0 if np.all(np.asarray(differences) == 0) else float(
                        wilcoxon(differences, alternative="two-sided").pvalue)
                local_f1.append(row)
            summaries.append(row)
        adjusted = _holm_adjust([row["p_value"] if row["p_value"] is not None else float("nan")
                                 for row in local_f1])
        for row, value in zip(local_f1, adjusted):
            row["holm_adjusted_p_value"] = float(value) if math.isfinite(value) else None
    return per_dataset, summaries


def row_group_comparisons(dataset_rows: list[dict], scope: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Compare dataset summaries on the common complete set across policies."""
    indexed = {(row["split_policy"], row["dataset"], row["pipeline"]): row for row in dataset_rows}
    dataset_pairs, pipeline_pairs, contrast_pairs = [], [], []
    for pipeline in scope["pipelines"]:
        differences, row_values, group_values = [], [], []
        for item in scope["datasets"]:
            name = item["name"]
            a, b = indexed[("row_level", name, pipeline)], indexed[("group_aware", name, pipeline)]
            comparable = a["complete_auc"] and b["complete_auc"] and a["auc_eligible"] and b["auc_eligible"]
            difference = b["mean_roc_auc"] - a["mean_roc_auc"] if comparable else None
            if comparable:
                row_values.append(a["mean_roc_auc"])
                group_values.append(b["mean_roc_auc"])
                differences.append(difference)
            dataset_pairs.append({
                "dataset": name, "pipeline": pipeline, "row_run_id": a["run_id"], "group_run_id": b["run_id"],
                "row_mean_auc": a["mean_roc_auc"], "group_mean_auc": b["mean_roc_auc"],
                "common_complete_dataset": comparable, "group_minus_row": difference,
                **{f"{prefix}_{field}": record[field] for prefix, record in (("row", a), ("group", b))
                   for field in ("expected_cells", "success_cells", "skipped_cells", "failed_cells",
                                 "timed_out_cells", "complete_auc", "auc_eligible", "outcome_reasons")},
                "auc_ineligibility_reason": a["auc_ineligibility_reason"] or b["auc_ineligibility_reason"],
            })
        pipeline_pairs.append({
            "pipeline": pipeline, "configured_datasets": len(scope["datasets"]),
            "row_complete_datasets": sum(indexed[("row_level", item["name"], pipeline)]["complete_auc"] for item in scope["datasets"]),
            "group_complete_datasets": sum(indexed[("group_aware", item["name"], pipeline)]["complete_auc"] for item in scope["datasets"]),
            "common_complete_dataset_names": ";".join(row["dataset"] for row in dataset_pairs
                                                       if row["pipeline"] == pipeline and row["common_complete_dataset"]),
            "mean_row_auc_common_datasets": float(np.mean(row_values)) if row_values else None,
            "mean_group_auc_common_datasets": float(np.mean(group_values)) if group_values else None,
            **_dataset_statistics(differences), "delta_definition": "group_mean_auc_minus_row_mean_auc",
            "interpretation": "descriptive comparison; no pairing of predictions or folds",
        })
    for pipeline in PRESPECIFIED_PIPELINES:
        row_deltas, group_deltas, differences, names = [], [], [], []
        for item in scope["datasets"]:
            name = item["name"]
            records = [indexed[(policy, name, candidate)] for policy in POLICIES for candidate in ("Raw", pipeline)]
            if not all(row["complete_auc"] and row["auc_eligible"] for row in records):
                continue
            row_delta = records[1]["mean_roc_auc"] - records[0]["mean_roc_auc"]
            group_delta = records[3]["mean_roc_auc"] - records[2]["mean_roc_auc"]
            names.append(name)
            row_deltas.append(row_delta)
            group_deltas.append(group_delta)
            differences.append(group_delta - row_delta)
        contrast_pairs.append({
            "pipeline_a": "Raw", "pipeline_b": pipeline, "configured_datasets": len(scope["datasets"]),
            "common_complete_dataset_names": ";".join(names),
            "mean_row_delta_common_datasets": float(np.mean(row_deltas)) if row_deltas else None,
            "mean_group_delta_common_datasets": float(np.mean(group_deltas)) if group_deltas else None,
            **_dataset_statistics(differences), "delta_definition": "group_candidate_minus_raw_delta_minus_row_delta",
            "interpretation": "descriptive difference of dataset contrasts; no cross-policy fold/prediction pairing",
        })
    return dataset_pairs, pipeline_pairs, contrast_pairs


def missingness_sensitivity(dataset_rows: list[dict], scope: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """AUC [0,1] partial-identification bounds; no imputation or p-values."""
    score_bounds, contrast_bounds, summaries = [], [], []
    indexed = {}
    for row in dataset_rows:
        expected, observed = row["expected_cells"], row["success_cells"]
        if observed > expected:
            raise ValueError("Observed AUC cells exceed the declared grid")
        eligible = bool(row["auc_eligible"])
        total = (row["mean_roc_auc"] or 0.0) * observed
        lower, upper = (total / expected, (total + expected - observed) / expected) if eligible and expected else (None, None)
        record = {
            "split_policy": row["split_policy"], "dataset": row["dataset"], "pipeline": row["pipeline"],
            "auc_eligible": eligible, "expected_eligible_cells": expected if eligible else 0,
            "observed_finite_auc_cells": observed, "unknown_eligible_cells": expected - observed if eligible else 0,
            "skipped_cells": row["skipped_cells"], "failed_cells": row["failed_cells"],
            "timed_out_cells": row["timed_out_cells"], "mean_auc_lower_bound": lower, "mean_auc_upper_bound": upper,
            "outcome_reasons": row["outcome_reasons"], "auc_ineligibility_reason": row["auc_ineligibility_reason"],
            "bound_method": "unknown eligible AUC cells individually range from 0 to 1; no outcome imputation",
        }
        score_bounds.append(record)
        indexed[(row["split_policy"], row["dataset"], row["pipeline"])] = record
    for policy in POLICIES:
        for contrast_id, reference, candidate, family in _contrast_definitions(scope):
            eligible_records = []
            for item in scope["datasets"]:
                name = item["name"]
                a, b = indexed[(policy, name, reference)], indexed[(policy, name, candidate)]
                eligible = a["auc_eligible"] and b["auc_eligible"]
                lower = b["mean_auc_lower_bound"] - a["mean_auc_upper_bound"] if eligible else None
                upper = b["mean_auc_upper_bound"] - a["mean_auc_lower_bound"] if eligible else None
                record = {
                    "split_policy": policy, "dataset": name, "contrast_id": contrast_id,
                    "family_id": family, "pipeline_a": reference, "pipeline_b": candidate,
                    "auc_eligible": eligible, "delta_lower_bound": lower, "delta_upper_bound": upper,
                    "unknown_cells_a": a["unknown_eligible_cells"], "unknown_cells_b": b["unknown_eligible_cells"],
                    "outcome_reasons_a": a["outcome_reasons"], "outcome_reasons_b": b["outcome_reasons"],
                    "auc_ineligibility_reason": a["auc_ineligibility_reason"] or b["auc_ineligibility_reason"],
                }
                contrast_bounds.append(record)
                if eligible:
                    eligible_records.append(record)
            summaries.append({
                "split_policy": policy, "contrast_id": contrast_id, "family_id": family,
                "pipeline_a": reference, "pipeline_b": candidate, "configured_datasets": len(scope["datasets"]),
                "auc_eligible_datasets": len(eligible_records),
                "datasets_with_unknown_outcomes": sum(row["unknown_cells_a"] > 0 or row["unknown_cells_b"] > 0
                                                      for row in eligible_records),
                "unknown_eligible_cells_a": sum(row["unknown_cells_a"] for row in eligible_records),
                "unknown_eligible_cells_b": sum(row["unknown_cells_b"] for row in eligible_records),
                "mean_dataset_delta_lower_bound": float(np.mean([row["delta_lower_bound"] for row in eligible_records])) if eligible_records else None,
                "mean_dataset_delta_upper_bound": float(np.mean([row["delta_upper_bound"] for row in eligible_records])) if eligible_records else None,
                "interpretation": "equal-weight eligible-dataset bounds; undefined group AUC datasets excluded visibly; no confidence interval or p-value",
            })
    return score_bounds, contrast_bounds, summaries


def _save_figure(fig, stem: Path, *, diagnostic: bool = False, missing_outcomes: bool = False) -> list[str]:
    if diagnostic or missing_outcomes:
        prior_title = fig._suptitle.get_text() if fig._suptitle is not None else ""
        warnings = (["PILOT DIAGNOSTIC — NOT A CORRECTED RESULT"] if diagnostic else [])
        if missing_outcomes:
            warnings.append("TERMINAL MISSING OUTCOMES — COMPLETE-CASE ESTIMATES; SEE AUC BOUNDS")
        fig.suptitle("\n".join(warnings + [prior_title]),
                     color="#8b3e29", fontsize=14, weight="bold")
    paths = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path.name)
    plt.close(fig)
    return paths


def plot_primary(summary: list[dict], output_dir: Path, *, diagnostic: bool = False,
                 missing_outcomes: bool = False) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, constrained_layout=True)
    variants = [row["pipeline"] for row in summary if row["split_policy"] == "row_level"
                and row["pipeline"] != "AutoFE_Baseline"]
    for ax, policy in zip(axes, POLICIES):
        indexed = {row["pipeline"]: row for row in summary if row["split_policy"] == policy}
        for position, pipeline in enumerate(variants):
            row = indexed[pipeline]
            value = row["mean_paired_auc_delta_vs_full_arithmetic"]
            if value is None:
                ax.text(0, position, "NA", va="center", ha="center", color="#444444")
                continue
            ax.barh(position, value, height=0.7, color="#315d82" if value >= 0 else "#b67748")
            display_value = 0.0 if abs(value) < 0.0005 else value
            ax.text(value if value >= 0 else 0, position,
                    f" {display_value:+.3f} (n={row['paired_to_baseline_datasets']})",
                    va="center", fontsize=8, ha="left")
        ax.axvline(0, color="#444444", linewidth=1)
        ax.set_yticks(range(len(variants)), variants)
        ax.set_title(policy.replace("_", " ").title())
        ax.set_xlabel("Mean paired dataset ROC-AUC difference vs full arithmetic")
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Primary pipeline comparisons: complete dataset pairs only")
    return _save_figure(fig, output_dir / "primary_auc_ablation", diagnostic=diagnostic, missing_outcomes=missing_outcomes)


def plot_operator_counts(rows: list[dict], scope: dict, output_dir: Path,
                         *, diagnostic: bool = False, missing_outcomes: bool = False) -> list[str]:
    variants = [name for name in scope["pipelines"] if name.startswith("AutoFE_")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True, sharey=True)
    selected = {(row["split_policy"], row["pipeline"], row["operator"]): row for row in rows}
    finite = [row["selected_per_observed_feature_task"] for row in rows
              if row["pipeline"] in variants and row["selected_per_observed_feature_task"] is not None]
    ceiling = max(finite, default=1) or 1
    for ax, policy in zip(axes, POLICIES):
        matrix = np.full((len(variants), len(OPERATORS)), np.nan)
        for i, pipeline in enumerate(variants):
            for j, operator in enumerate(OPERATORS):
                row = selected[(policy, pipeline, operator)]
                if row["enabled"] and row["selected_per_observed_feature_task"] is not None:
                    matrix[i, j] = row["selected_per_observed_feature_task"]
                ax.text(j, i, "off" if not row["enabled"] else
                        ("NA" if row["selected_per_observed_feature_task"] is None else f"{matrix[i,j]:.1f}"),
                        ha="center", va="center", fontsize=8,
                        color="white" if row["enabled"] and matrix[i, j] > ceiling * 0.55 else "#222222")
        ax.imshow(np.ma.masked_invalid(matrix), cmap="Blues", vmin=0, vmax=ceiling, aspect="auto")
        ax.set_facecolor("#eeeeee")
        ax.set_xticks(range(4), [name.removesuffix("_numeric") for name in OPERATORS], rotation=30)
        ax.set_yticks(range(len(variants)), variants)
        ax.set_title(policy.replace("_", " ").title())
    fig.suptitle("Selected candidates per observed feature task; disabled operators marked off")
    return _save_figure(fig, output_dir / "operator_selected_candidates", diagnostic=diagnostic, missing_outcomes=missing_outcomes)


def plot_operator_funnel(rows: list[dict], output_dir: Path, *, diagnostic: bool = False,
                         missing_outcomes: bool = False) -> list[str]:
    """Show each candidate stage for the full arithmetic reference."""
    indexed = {(row["split_policy"], row["pipeline"], row["operator"]): row for row in rows}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    colors = ("#315d82", "#b67748", "#777777", "#548376", "#7b6395")
    for i, policy in enumerate(POLICIES):
        for j, operator in enumerate(OPERATORS):
            ax = axes[i, j]
            row = indexed[(policy, "AutoFE_Baseline", operator)]
            observed = row["observed_feature_tasks"]
            values = [row[field] / observed if observed else 0 for field in COUNT_FIELDS]
            ax.barh(range(len(COUNT_FIELDS)), values, color=colors)
            ax.set_yticks(range(len(COUNT_FIELDS)), COUNT_FIELDS)
            ax.invert_yaxis()
            ax.set_title(f"{policy.replace('_', ' ').title()} / {operator.removesuffix('_numeric')}")
            ax.set_xlabel(f"Count per observed feature task (n={observed})")
            ax.set_xlim(0, max(values, default=0) * 1.25 or 1)
            ax.grid(axis="x", color="#dddddd", linewidth=0.5)
            ax.set_axisbelow(True)
            for position, value in enumerate(values):
                ax.text(value, position, f" {value:.1f}", va="center", fontsize=8)
    fig.suptitle("Full arithmetic candidate counts by operator and stage")
    return _save_figure(fig, output_dir / "baseline_operator_candidate_stages", diagnostic=diagnostic, missing_outcomes=missing_outcomes)


def plot_coverage(rows: list[dict], scope: dict, output_dir: Path,
                  *, diagnostic: bool = False, missing_outcomes: bool = False) -> list[str]:
    datasets = [item["name"] for item in scope["datasets"]]
    lookup = {(row["split_policy"], row["dataset"], row["pipeline"]): row for row in rows}
    matrix = np.asarray([[lookup[(policy, dataset, "AutoFE_Baseline")]["coverage_fraction"]
                          for policy in POLICIES] for dataset in datasets])
    fig, ax = plt.subplots(figsize=(7, 9), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_xticks(range(2), ["Row level", "Group aware"])
    for i in range(len(datasets)):
        for j in range(2):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.0%}" if value in (0, 1) else f"{value:.1%}",
                    ha="center", va="center", fontsize=8,
                    color="white" if value > 0.55 else "#222222")
    fig.colorbar(image, ax=ax, label="Successful ROC-AUC cells / configured cells")
    ax.set_title("Primary AUC coverage by dataset; full arithmetic pipeline")
    return _save_figure(fig, output_dir / "primary_auc_coverage", diagnostic=diagnostic, missing_outcomes=missing_outcomes)


def plot_dataset_estimates(rows: list[dict], output_dir: Path, stem: str, title: str,
                          *, policy_panels: bool = True, bounds: bool = False,
                          diagnostic: bool = False, missing_outcomes: bool = False) -> list[str]:
    """Show dataset contrasts or identification ranges; never task-row errors."""
    panels = [(policy, [row for row in rows if row.get("split_policy") == policy])
              for policy in POLICIES] if policy_panels else [("common complete datasets", rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(15 if policy_panels else 11,
                                                    max(5, len(panels[0][1]) * 0.42)),
                             constrained_layout=True, squeeze=False)
    for ax, (policy, subset) in zip(axes[0], panels):
        labels = []
        plotted_values = []
        for position, row in enumerate(subset):
            labels.append(row.get("pipeline_b", row.get("pipeline", row.get("contrast_id"))))
            if bounds:
                low, high = row["mean_dataset_delta_lower_bound"], row["mean_dataset_delta_upper_bound"]
                if low is None:
                    ax.text(0, position, "NA", va="center")
                    continue
                ax.plot([low, high], [position, position], color="#b67748", linewidth=3)
                ax.plot([low, high], [position, position], "|", color="#b67748", markersize=10)
                plotted_values.extend((low, high))
                label = f" [{0.0 if abs(low) < 0.0005 else low:+.3f}, {0.0 if abs(high) < 0.0005 else high:+.3f}]; missing={row['datasets_with_unknown_outcomes']}"
                ax.text(high, position, label, va="center", fontsize=8)
            else:
                value, low, high = row["mean_delta"], row["ci_low"], row["ci_high"]
                if value is None:
                    ax.text(0, position, "NA (n=0)", va="center")
                    continue
                if low is not None:
                    ax.plot([low, high], [position, position], color="#315d82", linewidth=2)
                    plotted_values.extend((low, high))
                ax.plot(value, position, "o", color="#315d82", markersize=5)
                plotted_values.append(value)
                label = f" {0.0 if abs(value) < 0.0005 else value:+.3f}; n={row['n_complete_datasets']}"
                adjusted = row.get("holm_adjusted_p_value")
                if adjusted is not None:
                    label += f"; Holm p={adjusted:.3g}"
                elif low is None:
                    label += "; CI NA"
                ax.text(high if high is not None else value, position, label, va="center", fontsize=8)
        ax.set_yticks(range(len(subset)), labels)
        ax.invert_yaxis()
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_title(policy.replace("_", " ").title())
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
        scale = max(0.01, max((abs(value) for value in plotted_values), default=0))
        ax.set_xlim(-scale * 1.25, scale * 2.3)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_xlabel("Mean dataset ROC-AUC difference" + ("; AUC [0,1] bounds" if bounds else "; 95% dataset bootstrap CI"))
    fig.suptitle(title)
    return _save_figure(fig, output_dir / stem, diagnostic=diagnostic, missing_outcomes=missing_outcomes)


def add_sensitivities(run_root: Path, scope: dict, output_dir: Path) -> tuple[list[str], dict]:
    """Export five sensitivity identities without pooling them into primary AUC."""
    dataset_rows, condition_rows, operator_rows, sources = [], [], [], []
    for record in scope["runs"]:
        if record["scope"] == "primary":
            continue
        datasets, conditions, operators, source = collect_primary(
            run_root / record["run_id"], scope, record["split_policy"], record,
        )
        dataset_rows.extend(datasets)
        condition_rows.extend(conditions)
        operator_rows.extend(operators)
        sources.append(source)
    indexed = {(row["run_id"], row["dataset"], row["condition"], row["pipeline"]): row
               for row in condition_rows}
    summaries = []
    for record in scope["runs"]:
        if record["scope"] == "primary":
            continue
        run_id = record["run_id"]
        for condition in record["condition_names"]:
            for pipeline in scope["pipelines"]:
                scores, paired = [], []
                for item in scope["datasets"]:
                    name = item["name"]
                    candidate = indexed[(run_id, name, condition, pipeline)]
                    raw = indexed[(run_id, name, condition, "Raw")]
                    if candidate["complete_auc"]:
                        scores.append(candidate["mean_roc_auc"])
                        if raw["complete_auc"]:
                            paired.append(candidate["mean_roc_auc"] - raw["mean_roc_auc"])
                summaries.append({
                    "run_id": run_id, "experiment_scope": record["scope"],
                    "split_policy": record["split_policy"], "condition": condition,
                    "pipeline": pipeline, "configured_datasets": len(scope["datasets"]),
                    "complete_auc_datasets": len(scores), "paired_to_raw_datasets": len(paired),
                    "mean_dataset_roc_auc": float(np.mean(scores)) if scores else None,
                    "mean_paired_auc_delta_vs_raw": float(np.mean(paired)) if paired else None,
                    "interpretation": "separate descriptive sensitivity track",
                })
    tables = {
        "sensitivity_dataset_auc_coverage.csv": dataset_rows,
        "sensitivity_dataset_condition_auc.csv": condition_rows,
        "sensitivity_pipeline_auc_summary.csv": summaries,
        "sensitivity_operator_candidate_counts.csv": operator_rows,
    }
    for name, rows in tables.items():
        _write_csv(output_dir / name, list(rows[0]), rows)
    baseline = [row for row in summaries if row["pipeline"] == "AutoFE_Baseline"]
    labels = [f"{row['experiment_scope']} / {row['condition']} / {row['split_policy']}"
              for row in baseline]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for position, row in enumerate(baseline):
        value = row["mean_paired_auc_delta_vs_raw"]
        if value is None:
            ax.text(0, position, "NA", va="center")
            continue
        ax.barh(position, value, color="#315d82" if value >= 0 else "#b67748")
        ax.text(value if value >= 0 else 0, position,
                f" {value:+.3f} (n={row['paired_to_raw_datasets']})",
                va="center", ha="left")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=1)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Separate sensitivity tracks: full arithmetic vs Raw")
    ax.set_xlabel("Mean paired dataset ROC-AUC difference")
    figures = _save_figure(fig, output_dir / "sensitivity_baseline_vs_raw")
    return figures, {"sources": sources, "tables": list(tables), "figures": figures,
                     "pooled_with_primary": False}


def add_mechanism(association_path: Path, output_dir: Path) -> tuple[list[str], dict]:
    note = json.loads(association_path.read_text(encoding="utf-8"))
    if note.get("artifact_type") != "reviewer1_exploratory_mechanism_association":
        raise ValueError("Mechanism association artifact type differs from frozen analysis")
    associations = note["associations"]
    rows = []
    for item in note["dataset_records"]:
        rows.append({
            "split_policy": item["split_policy"], "dataset": item["dataset"],
            "included_in_association": item["included_in_association"],
            "mean_selected_scaled_jacobian_norm": item["mechanism"].get("mean_fold_median_selected_scaled_jacobian_norm"),
            "mean_clean_auc_delta_vs_raw": item["performance"].get("mean_auc_delta"),
            "finite_mechanism_folds": item["mechanism"].get("n_finite_folds"),
            "paired_clean_auc_cells": item["performance"].get("n_paired_clean_cells"),
        })
    _write_csv(output_dir / "mechanism_dataset_records.csv", list(rows[0]), rows)
    _write_csv(output_dir / "mechanism_associations.csv", list(associations[0]), associations)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, policy in zip(axes, POLICIES):
        subset = [row for row in rows if row["split_policy"] == policy and row["included_in_association"]]
        ax.scatter([row["mean_selected_scaled_jacobian_norm"] for row in subset],
                   [row["mean_clean_auc_delta_vs_raw"] for row in subset],
                   facecolors="#315d82", edgecolors="#263746", s=36)
        statistic = next(row for row in associations if row["split_policy"] == policy)
        rho = statistic.get("rho")
        adjusted = statistic.get("holm_adjusted_permutation_p")
        statistic_label = f"rho={rho:.2f}" if rho is not None else "rho=NA"
        if adjusted is not None:
            statistic_label += f", Holm p={adjusted:.3g}"
        ax.set_title(f"{policy.replace('_', ' ').title()} | n={len(subset)}, {statistic_label}")
        ax.set_xlabel("Selected candidate scaled Jacobian norm (dataset mean)")
        ax.set_ylabel("Clean paired ROC-AUC difference: Baseline minus Raw")
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.grid(color="#dddddd", linewidth=0.5)
    fig.suptitle("Exploratory dataset-level mechanism association; no causal interpretation")
    paths = _save_figure(fig, output_dir / "mechanism_jacobian_auc_association")
    return paths, {"association_sha256": file_sha256(association_path),
                   "association_table": "mechanism_associations.csv",
                   "dataset_table": "mechanism_dataset_records.csv"}


def _write_result_record(output_dir: Path, scope: dict, manifest: dict,
                         f1: list[dict], policy_rows: list[dict], contrast_rows: list[dict],
                         bounds: list[dict]) -> None:
    """Durable human-readable record from the same checked table values."""
    def table(rows: list[dict], fields: list[str]) -> list[str]:
        def value(item):
            if item is None:
                return "NA"
            if isinstance(item, float):
                return f"{item:.8g}"
            return str(item).replace("|", "\\|").replace("\n", " ")
        return ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |",
                *("| " + " | ".join(value(row.get(field)) for field in fields) + " |" for row in rows)]
    lines = ["# Corrected result record", "", "Status: **" + manifest["status"] + "**.", "",
             f"Scientific completeness flag: `{manifest['scientific_complete']}`; terminal unknown outcomes: `{manifest['has_terminal_missing_outcomes']}`.", "",
             "This record is generated from terminal, identity-checked ledgers. Historical ledgers are excluded.", "",
             f"Frozen source fingerprint: `{scope['code_fingerprint']}`.",
             f"Scope SHA-256: `{manifest['scope_sha256']}`.",
             f"Dataset hash digest: `{scope.get('dataset_hash_digest', 'diagnostic scope')}`.", "",
             "## Source identity and terminal coverage", ""]
    lines.extend(table([{**row, **row["counts_by_status"]} for row in manifest["source_runs"]],
                       ["run_id", "code_commit", "terminal_manifest_status", "terminal_rows", "success", "failed", "skipped", "timed_out",
                        "manifest_sha256", "results_sha256", "configuration_fingerprint", "runtime_fingerprint"]))
    lines.extend(["", "## Frozen F1 dataset contrasts", "",
                  "AUC difference is candidate minus Raw. Datasets are the inferential and bootstrap units. Complete expected finite-cell grids are required; intervals use 10,000 percentile bootstrap replicates with seed 20260929. Holm covers the four finite prespecified p-values within each split policy. Fewer than two complete datasets gives NA inference.", ""])
    lines.extend(table(f1, ["split_policy", "pipeline_b", "configured_datasets", "auc_eligible_datasets", "n_complete_datasets",
                            "mean_delta", "ci_low", "ci_high", "p_value", "holm_adjusted_p_value"]))
    lines.extend(["", "## Row-level and group-aware values on common complete datasets", "",
                  "Each dataset contributes one mean to each policy; predictions and fold samples are not paired across policies. Cross-policy intervals are descriptive dataset bootstrap intervals.", ""])
    lines.extend(table(policy_rows, ["pipeline", "row_complete_datasets", "group_complete_datasets", "n_complete_datasets",
                                    "mean_row_auc_common_datasets", "mean_group_auc_common_datasets", "mean_delta", "ci_low", "ci_high"]))
    lines.extend(["", "## Common-dataset changes in the Raw-versus-AutoFE contrast", ""])
    lines.extend(table(contrast_rows, ["pipeline_b", "n_complete_datasets", "mean_row_delta_common_datasets",
                                      "mean_group_delta_common_datasets", "mean_delta", "ci_low", "ci_high"]))
    lines.extend(["", "## Missing-outcome sensitivity", "",
                  "Unknown eligible AUC cells range over [0,1]. Bounds average each eligible dataset equally and are identification ranges, not confidence intervals or imputed results. Prespecified undefined group AUC datasets remain visible in coverage tables and receive no artificial bounds.", ""])
    lines.extend(table([row for row in bounds if row["family_id"] == "F1_primary_roc_auc"],
                       ["split_policy", "pipeline_b", "auc_eligible_datasets", "datasets_with_unknown_outcomes",
                        "unknown_eligible_cells_a", "unknown_eligible_cells_b", "mean_dataset_delta_lower_bound", "mean_dataset_delta_upper_bound"]))
    lines.extend(["", "## Artifact paths", "", *[f"- [{name}]({name})" for name in manifest["tables"] + manifest["figures"]], "",
                  "Per-dataset values, all status counts, exclusion/failure reasons, individual operator counts, matched-cap descriptive comparisons, and condition/resource measurements remain inspectable in the linked CSVs. Candidate counts are deduplicated across classifier copies. Optional mechanism and separate sensitivity assets retain their own source identities and denominators.", ""])
    (output_dir / "corrected_results_note.md").write_text("\n".join(lines), encoding="utf-8")


def generate(row_dir: Path, group_dir: Path, output_dir: Path,
             association_path: Path | None = None, scope_path: Path | None = None,
             sensitivity_root: Path | None = None) -> dict:
    frozen_scope = ROOT / "provenance" / "reviewer1_launch_scope_v5.json"
    scope_path = scope_path or frozen_scope
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    diagnostic = scope_path.resolve() != frozen_scope.resolve()
    if diagnostic and scope.get("artifact_type") != "reporting_diagnostic_scope":
        raise ValueError("A custom scope must be explicitly labeled reporting_diagnostic_scope")
    if not diagnostic and scope.get('resource_plan'):
        scope['_resource_profile_sha256'] = file_sha256(scope_path)
    scope.setdefault("primary_conditions", sorted(PRIMARY_CONDITIONS))
    if len(scope["pipelines"]) != 14:
        raise ValueError("Corrected paper assets require the frozen 14-pipeline scope")
    dataset_rows, condition_rows, operator_rows, sources = [], [], [], []
    for run_dir, policy in ((row_dir, "row_level"), (group_dir, "group_aware")):
        datasets, conditions, operators, source = collect_primary(run_dir, scope, policy)
        dataset_rows.extend(datasets)
        condition_rows.extend(conditions)
        operator_rows.extend(operators)
        sources.append(source)
    summary = pipeline_summary(dataset_rows, scope)
    condition_summaries = condition_summary(condition_rows, scope)
    contrasts, contrast_summaries = dataset_contrasts(dataset_rows, scope)
    f1 = [row for row in contrast_summaries if row["family_id"] == "F1_primary_roc_auc"]
    descriptive = [row for row in contrast_summaries if row["family_id"] != "F1_primary_roc_auc"]
    row_group_datasets, row_group_pipelines, row_group_contrasts = row_group_comparisons(dataset_rows, scope)
    auc_bounds, contrast_bounds, bound_summaries = missingness_sensitivity(dataset_rows, scope)
    missing_outcomes = any(row["unknown_eligible_cells"] for row in auc_bounds)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "primary_dataset_auc_coverage.csv": dataset_rows,
        'primary_resource_policy_and_use.csv': resource_summary(sources),
        "primary_pipeline_auc_summary.csv": summary,
        "primary_operator_candidate_counts.csv": operator_rows,
        "primary_dataset_contrasts.csv": contrasts,
        "primary_f1_contrasts.csv": f1,
        "primary_descriptive_contrasts.csv": descriptive,
        "row_group_dataset_auc.csv": row_group_datasets,
        "row_group_pipeline_auc_summary.csv": row_group_pipelines,
        "row_group_f1_contrast_comparison.csv": row_group_contrasts,
        "primary_missingness_auc_bounds.csv": auc_bounds,
        "primary_missingness_contrast_bounds.csv": contrast_bounds,
        "primary_missingness_contrast_summary.csv": bound_summaries,
    }
    if condition_rows:
        tables["primary_dataset_condition_auc.csv"] = condition_rows
        tables["primary_condition_auc_summary.csv"] = condition_summaries
    for name, rows in tables.items():
        _write_csv(output_dir / name, list(rows[0]), rows)
    plot_options = {"diagnostic": diagnostic, "missing_outcomes": missing_outcomes}
    figures = (plot_resource_summary(resource_summary(sources), output_dir, **plot_options)
               + plot_primary(summary, output_dir, **plot_options)
               + plot_operator_counts(operator_rows, scope, output_dir, **plot_options)
               + plot_operator_funnel(operator_rows, output_dir, **plot_options)
               + plot_coverage(dataset_rows, scope, output_dir, **plot_options)
               + plot_dataset_estimates(f1, output_dir, "primary_f1_contrasts",
                                       "Frozen F1: AutoFE minus Raw; Holm within each policy", **plot_options)
               + plot_dataset_estimates(row_group_pipelines, output_dir, "row_group_auc_difference",
                                       "Group minus row AUC on common complete datasets; descriptive",
                                       policy_panels=False, **plot_options)
               + plot_dataset_estimates([row for row in bound_summaries if row["family_id"] == "F1_primary_roc_auc"],
                                       output_dir, "primary_missingness_auc_bounds",
                                       "AUC [0,1] identification bounds; structurally undefined datasets excluded",
                                       bounds=True, **plot_options))
    mechanism = None
    if association_path is not None:
        more, mechanism = add_mechanism(association_path, output_dir)
        figures.extend(more)
    manifest = {
        "artifact_type": "reviewer1_reporting_diagnostic_assets" if diagnostic else "reviewer1_corrected_paper_assets",
        "scope_sha256": file_sha256(scope_path), "source_runs": sources,
        "tables": list(tables),
        "figures": figures, "mechanism": mechanism,
        "primary_estimand": "dataset mean over complete 5 seeds x 5 folds x 10 conditions x 10 models; row and group separate",
        "candidate_count_unit": "unique dataset x seed x fold x condition x pipeline feature task; classifier copies deduplicated",
        "scientific_complete": not diagnostic and not missing_outcomes,
        "has_terminal_missing_outcomes": missing_outcomes,
        "status": "pilot code verification; not corrected performance" if diagnostic else (
            "terminal corrected primary runs with missing outcomes; complete-case and bounded sensitivity only"
            if missing_outcomes else "corrected primary runs complete"),
        "analysis_contract": {
            "F1": {"reference": "Raw", "pipelines": list(PRESPECIFIED_PIPELINES),
                   "family": "F1_primary_roc_auc", "family_size": len(PRESPECIFIED_PIPELINES),
                   "multiplicity": "Holm within each split policy across finite prespecified p-values",
                   "test": "two-sided dataset-level Wilcoxon; exact zero differences are ties",
                   "ci_method": "paired_dataset_bootstrap_percentile", "ci_level": 0.95,
                   "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                   "minimum_complete_datasets_for_inference": 2},
            "row_group": "compare separate dataset means on common complete datasets; never pair predictions or fold samples",
            "secondary": "matched-cap and individual operator contrasts descriptive; dataset bootstrap intervals; no p-values",
            "missingness": "eligible dataset mean bounds: known finite AUC sum/N through (sum+unknown cells)/N; AUC range [0,1]; contrast [candidate lower-reference upper,candidate upper-reference lower]; average bounds equally over all eligible datasets; prespecified undefined group AUC has no artificial bound; no missing-outcome imputation, p-value, or sampling CI",
        },
    }
    if mechanism:
        manifest["tables"].extend([mechanism["association_table"], mechanism["dataset_table"]])
    if sensitivity_root is not None:
        extra_figures, sensitivity = add_sensitivities(sensitivity_root, scope, output_dir)
        manifest["sensitivity"] = sensitivity
        manifest["tables"].extend(sensitivity["tables"])
        manifest["figures"].extend(extra_figures)
    manifest["result_record"] = "corrected_results_note.md"
    _write_result_record(output_dir, scope, manifest, f1, row_group_pipelines, row_group_contrasts, bound_summaries)
    atomic_write_json(output_dir / "asset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-run-dir", type=Path, default=ROOT / "corrected_runs" / "r1-v5-primary-row")
    parser.add_argument("--group-run-dir", type=Path, default=ROOT / "corrected_runs" / "r1-v5-primary-group")
    parser.add_argument('--scope', type=Path, default=ROOT / 'provenance' / 'reviewer1_launch_scope_v5.json')
    parser.add_argument("--output-dir", type=Path, default=ROOT / "corrected_runs" / "paper_assets")
    parser.add_argument("--mechanism-association", type=Path)
    parser.add_argument("--sensitivity-root", type=Path,
                        help="Root containing all five completed frozen sensitivity run IDs")
    args = parser.parse_args()
    print(json.dumps(generate(args.row_run_dir, args.group_run_dir, args.output_dir,
                              args.mechanism_association, scope_path=args.scope,
                              sensitivity_root=args.sensitivity_root), indent=2))


if __name__ == "__main__":
    main()
