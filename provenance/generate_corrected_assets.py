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

from src.pipeline_runner import PIPELINE_CONFIGS
from src.provenance import atomic_write_json, file_sha256
from src.stats_analysis import PRIMARY_CONDITIONS


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ("add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric")
COUNT_FIELDS = ("generated", "rejected", "duplicates", "eligible", "selected")
POLICIES = ("row_level", "group_aware")
SECONDARY_METRICS = (
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
    "pr_auc", "log_loss", "brier_score", "train_auc", "train_time_s",
    "infer_time_s", "autofe_gen_time_s", "ram_used_mb", "n_generated",
    "n_retained",
)


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
    if (manifest.get("status") != "complete" or manifest.get("run_id") != expected_id
            or manifest.get("code_fingerprint") != scope["code_fingerprint"]
            or manifest.get("configuration", {}).get("split_policy") != policy):
        raise ValueError(f"Primary run is incomplete or has a different frozen identity: {run_dir}")
    if manifest.get("expected_tasks") != target["intended_cells"]:
        raise ValueError(f"Primary task denominator changed: {run_dir}")
    lookup, connection = _terminal_lookup(run_dir, manifest)
    aggregate = defaultdict(lambda: {"success": 0, "skipped": 0, "failed": 0,
                                     "timed_out": 0, "auc_sum": 0.0, "reasons": set(),
                                     "metric_sum": defaultdict(float),
                                     "metric_count": defaultdict(int)})
    conditions = defaultdict(lambda: {"success": 0, "auc_sum": 0.0})
    feature_seen = {}
    counts = defaultdict(lambda: {field: 0 for field in COUNT_FIELDS})
    feature_count = defaultdict(int)
    included_rows = 0
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
                    continue
                auc = row.get("roc_auc")
                if auc is None or not math.isfinite(float(auc)):
                    raise ValueError(f"Successful cell lacks finite ROC-AUC: {key}")
                bucket["auc_sum"] += float(auc)
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
        "results_sha256": file_sha256(result_path), "terminal_rows": included_rows,
        "counts_by_status": manifest.get("counts_by_status"),
    }


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


def _save_figure(fig, stem: Path, *, diagnostic: bool = False) -> list[str]:
    if diagnostic:
        prior_title = fig._suptitle.get_text() if fig._suptitle is not None else ""
        fig.suptitle("PILOT DIAGNOSTIC — NOT A CORRECTED RESULT\n" + prior_title,
                     color="#8b3e29", fontsize=14, weight="bold")
    paths = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path.name)
    plt.close(fig)
    return paths


def plot_primary(summary: list[dict], output_dir: Path, *, diagnostic: bool = False) -> list[str]:
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
    return _save_figure(fig, output_dir / "primary_auc_ablation", diagnostic=diagnostic)


def plot_operator_counts(rows: list[dict], scope: dict, output_dir: Path,
                         *, diagnostic: bool = False) -> list[str]:
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
    return _save_figure(fig, output_dir / "operator_selected_candidates", diagnostic=diagnostic)


def plot_operator_funnel(rows: list[dict], output_dir: Path, *, diagnostic: bool = False) -> list[str]:
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
    return _save_figure(fig, output_dir / "baseline_operator_candidate_stages", diagnostic=diagnostic)


def plot_coverage(rows: list[dict], scope: dict, output_dir: Path,
                  *, diagnostic: bool = False) -> list[str]:
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
    return _save_figure(fig, output_dir / "primary_auc_coverage", diagnostic=diagnostic)


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


def generate(row_dir: Path, group_dir: Path, output_dir: Path,
             association_path: Path | None = None, scope_path: Path | None = None,
             sensitivity_root: Path | None = None) -> dict:
    frozen_scope = ROOT / "provenance" / "reviewer1_launch_scope_v2.json"
    scope_path = scope_path or frozen_scope
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    diagnostic = scope_path.resolve() != frozen_scope.resolve()
    if diagnostic and scope.get("artifact_type") != "reporting_diagnostic_scope":
        raise ValueError("A custom scope must be explicitly labeled reporting_diagnostic_scope")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "primary_dataset_auc_coverage.csv", list(dataset_rows[0]), dataset_rows)
    _write_csv(output_dir / "primary_pipeline_auc_summary.csv", list(summary[0]), summary)
    _write_csv(output_dir / "primary_operator_candidate_counts.csv", list(operator_rows[0]), operator_rows)
    if condition_rows:
        _write_csv(output_dir / "primary_dataset_condition_auc.csv", list(condition_rows[0]), condition_rows)
        _write_csv(output_dir / "primary_condition_auc_summary.csv", list(condition_summaries[0]), condition_summaries)
    figures = (plot_primary(summary, output_dir, diagnostic=diagnostic)
               + plot_operator_counts(operator_rows, scope, output_dir, diagnostic=diagnostic)
               + plot_operator_funnel(operator_rows, output_dir, diagnostic=diagnostic)
               + plot_coverage(dataset_rows, scope, output_dir, diagnostic=diagnostic))
    mechanism = None
    if association_path is not None:
        more, mechanism = add_mechanism(association_path, output_dir)
        figures.extend(more)
    manifest = {
        "artifact_type": "reviewer1_reporting_diagnostic_assets" if diagnostic else "reviewer1_corrected_paper_assets",
        "scope_sha256": file_sha256(scope_path), "source_runs": sources,
        "tables": ["primary_dataset_auc_coverage.csv", "primary_pipeline_auc_summary.csv",
                   "primary_operator_candidate_counts.csv"],
        "figures": figures, "mechanism": mechanism,
        "primary_estimand": "dataset mean over complete 5 seeds x 5 folds x 10 conditions x 10 models; row and group separate",
        "candidate_count_unit": "unique dataset x seed x fold x condition x pipeline feature task; classifier copies deduplicated",
        "status": "pilot code verification; not corrected performance" if diagnostic else "corrected primary runs complete",
    }
    if condition_rows:
        manifest["tables"].extend(["primary_dataset_condition_auc.csv", "primary_condition_auc_summary.csv"])
    if mechanism:
        manifest["tables"].extend([mechanism["association_table"], mechanism["dataset_table"]])
    if sensitivity_root is not None:
        extra_figures, sensitivity = add_sensitivities(sensitivity_root, scope, output_dir)
        manifest["sensitivity"] = sensitivity
        manifest["tables"].extend(sensitivity["tables"])
        manifest["figures"].extend(extra_figures)
    atomic_write_json(output_dir / "asset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-run-dir", type=Path, default=ROOT / "corrected_runs" / "r1-v2-primary-row")
    parser.add_argument("--group-run-dir", type=Path, default=ROOT / "corrected_runs" / "r1-v2-primary-group")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "corrected_runs" / "paper_assets")
    parser.add_argument("--mechanism-association", type=Path)
    parser.add_argument("--sensitivity-root", type=Path,
                        help="Root containing all five completed frozen sensitivity run IDs")
    args = parser.parse_args()
    print(json.dumps(generate(args.row_run_dir, args.group_run_dir, args.output_dir,
                              args.mechanism_association,
                              sensitivity_root=args.sensitivity_root), indent=2))


if __name__ == "__main__":
    main()
