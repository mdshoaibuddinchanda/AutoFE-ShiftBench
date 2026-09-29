"""Machine-readable corrected-run accounting for Reviewer #1.

The functions in this module consume only a corrected run ledger and its
manifest.  They keep row-level and group-aware results separate, use datasets
as the inferential unit, and expose incomplete/undefined cells instead of
silently dropping them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.stats_analysis import CONDITION_ALIASES, PRIMARY_CONDITIONS, PRESPECIFIED_PIPELINES, _holm_adjust


BOOTSTRAP_SEED = 20260929
BOOTSTRAP_REPLICATES = 10_000


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def paired_dataset_bootstrap(
    differences: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float | None, float | None]:
    """Return a percentile interval for the mean over datasets."""
    values = np.asarray(list(differences), dtype=float)
    if values.size == 0:
        return None, None
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[sample_indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _expected_configuration(manifest: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    cfg = manifest.get("configuration", {})
    conditions = []
    for item in cfg.get("conditions", []):
        if isinstance(item, (list, tuple)) and item:
            family = str(item[0])
            severity = float(item[1]) if len(item) > 1 else 0.0
            condition = family if severity == 0 else f"{family}_{severity:.2f}"
        else:
            condition = str(item)
        condition = CONDITION_ALIASES.get(condition, condition)
        if condition in PRIMARY_CONDITIONS:
            conditions.append(condition)
    if not conditions:
        conditions = sorted(set(frame.get("condition", pd.Series(dtype=str)).astype(str)).intersection(PRIMARY_CONDITIONS))
    datasets = [str(item) for item in cfg.get("datasets", [])]
    if not datasets and "dataset" in frame:
        datasets = sorted(frame["dataset"].dropna().astype(str).unique())
    return {
        "datasets": datasets,
        "seeds": list(cfg.get("seeds", sorted(frame.get("seed", pd.Series(dtype=int)).dropna().unique().tolist()))),
        "folds": list(cfg.get("folds", sorted(frame.get("fold", pd.Series(dtype=int)).dropna().unique().tolist()))),
        "conditions": sorted(set(conditions)),
        "models": list(cfg.get("models", sorted(frame.get("model", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()))),
        "pipelines": list(cfg.get("pipelines", sorted(frame.get("pipeline", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()))),
    }


def _status_counts(frame: pd.DataFrame, expected: int) -> dict[str, int]:
    counts = {name: 0 for name in ("success", "failed", "skipped", "timed_out", "pending")}
    if "status" in frame:
        for status, count in frame["status"].astype(str).value_counts().items():
            if status in counts:
                counts[status] += int(count)
    counts["pending"] = max(expected - sum(counts[name] for name in ("success", "failed", "skipped", "timed_out")), 0)
    return counts


def build_corrected_result_note(
    results_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Build the Reviewer #1 result-note schema from a corrected ledger."""
    results_path = Path(results_path)
    frame = _read_jsonl(results_path)
    if frame.empty:
        raise ValueError("corrected results ledger is empty")
    required = {"dataset", "seed", "fold", "condition", "pipeline", "model", "status", "roc_auc"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"corrected ledger is missing columns: {sorted(missing)}")
    frame["condition"] = frame["condition"].astype(str).replace(CONDITION_ALIASES)
    frame["split_policy"] = frame.get("split_policy", "row_level")
    frame["metric_value"] = pd.to_numeric(frame.get("roc_auc"), errors="coerce")
    manifest: dict[str, Any] = {}
    if manifest_path is None:
        candidate = results_path.with_name("manifest.json")
        if candidate.exists():
            manifest_path = candidate
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    config = _expected_configuration(manifest, frame)
    policies = sorted(set(frame["split_policy"].astype(str)))
    if not policies:
        policies = [str(manifest.get("configuration", {}).get("split_policy", "row_level"))]
    expected_cells = len(config["seeds"]) * len(config["folds"]) * len(config["conditions"]) * len(config["models"])

    dataset_scores: list[dict[str, Any]] = []
    for policy in policies:
        scoped = frame[(frame["split_policy"].astype(str) == policy) & (frame["condition"].isin(config["conditions"]))]
        for dataset in config["datasets"]:
            for pipeline in config["pipelines"]:
                cells = scoped[(scoped["dataset"].astype(str) == dataset) & (scoped["pipeline"].astype(str) == pipeline)]
                success = cells[cells["status"].astype(str) == "success"]
                finite = success[success["metric_value"].map(_finite)]
                failures = cells[cells["status"].astype(str) != "success"]["status"].astype(str).value_counts().to_dict()
                undefined = int(len(success) - len(finite))
                score = float(finite["metric_value"].mean()) if len(finite) else None
                complete = len(finite) == expected_cells and len(success) == expected_cells
                dataset_scores.append({
                    "run_id": manifest.get("run_id"), "split_policy": policy,
                    "dataset": dataset, "metric": "roc_auc", "pipeline": pipeline,
                    "condition_scope": "primary_training_corruption",
                    "n_expected_cells": expected_cells, "n_success_cells": int(len(success)),
                    "n_finite_cells": int(len(finite)), "coverage": float(len(finite) / expected_cells) if expected_cells else 0.0,
                    "score": score, "score_status": "complete" if complete else ("partial" if len(finite) else "missing"),
                    "failure_reasons": {"status_counts": failures, "undefined_metric_cells": undefined},
                })

    contrast_rows: list[dict[str, Any]] = []
    for policy in policies:
        scores = pd.DataFrame([row for row in dataset_scores if row["split_policy"] == policy])
        if scores.empty:
            continue
        raw = scores[scores["pipeline"] == "Raw"].set_index("dataset")
        pvalues: list[float] = []
        local_rows: list[dict[str, Any]] = []
        for pipeline in PRESPECIFIED_PIPELINES:
            candidate = scores[scores["pipeline"] == pipeline].set_index("dataset")
            common = raw.index.intersection(candidate.index)
            complete = [dataset for dataset in common if raw.loc[dataset, "score_status"] == "complete" and candidate.loc[dataset, "score_status"] == "complete"]
            diffs = np.asarray([candidate.loc[d, "score"] - raw.loc[d, "score"] for d in complete], dtype=float)
            if len(diffs) >= 2 and not np.allclose(diffs, 0):
                p_value = float(wilcoxon(diffs, alternative="two-sided").pvalue)
            elif len(diffs) >= 2:
                p_value = 1.0
            else:
                p_value = float("nan")
            pvalues.append(p_value)
            ci_low, ci_high = paired_dataset_bootstrap(diffs, seed=bootstrap_seed, replicates=bootstrap_replicates)
            local_rows.append({
                "family_id": "F1_primary_roc_auc", "family_size": len(PRESPECIFIED_PIPELINES),
                "contrast_id": f"roc_auc_raw_{pipeline.removeprefix('AutoFE_').lower()}",
                "split_policy": policy, "metric": "roc_auc", "pipeline_a": "Raw", "pipeline_b": pipeline,
                "n_complete_datasets": len(diffs), "n_expected_datasets": len(config["datasets"]),
                "mean_delta": float(np.mean(diffs)) if len(diffs) else None,
                "median_delta": float(np.median(diffs)) if len(diffs) else None,
                "win_count": int(np.sum(diffs > 0)), "tie_count": int(np.sum(np.isclose(diffs, 0))),
                "loss_count": int(np.sum(diffs < 0)), "p_value": p_value,
                "ci_method": "paired_dataset_bootstrap_percentile", "ci_level": 0.95,
                "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed,
                "ci_low": ci_low, "ci_high": ci_high,
                "status": "complete" if len(diffs) == len(config["datasets"]) else "incomplete",
            })
        for row, adjusted in zip(local_rows, _holm_adjust(pvalues)):
            row["holm_adjusted_p_value"] = adjusted
            contrast_rows.append(row)

    side_by_side: list[dict[str, Any]] = []
    if {"row_level", "group_aware"}.issubset(policies):
        key = ["dataset", "seed", "fold", "condition", "pipeline", "model"]
        row_frame = frame[frame["split_policy"] == "row_level"][key + ["status", "metric_value"]].rename(
            columns={"status": "row_status", "metric_value": "row_value"}
        )
        group_frame = frame[frame["split_policy"] == "group_aware"][key + ["status", "metric_value"]].rename(
            columns={"status": "group_status", "metric_value": "group_value"}
        )
        merged = row_frame.merge(group_frame, on=key, how="outer")
        for record in merged.to_dict(orient="records"):
            row_value = record.get("row_value")
            group_value = record.get("group_value")
            pairable = _finite(row_value) and _finite(group_value)
            side_by_side.append({
                **record, "metric": "roc_auc", "row_run_id": manifest.get("row_run_id", manifest.get("run_id")),
                "group_run_id": manifest.get("group_run_id", manifest.get("run_id")),
                "row_failure_reason": None if record.get("row_status") == "success" else record.get("row_status"),
                "group_failure_reason": None if record.get("group_status") == "success" else record.get("group_status"),
                "pairable": pairable,
                "group_minus_row": float(group_value - row_value) if pairable else None,
            })

    expected_tasks = int(manifest.get("expected_tasks", len(config["datasets"]) * len(config["seeds"]) * len(config["folds"]) * len(config["conditions"]) * len(config["pipelines"]) * len(config["models"])))
    accounting_frame = frame
    if "task_key" in accounting_frame:
        # Result JSONL retains retry/phase history.  Counts are over the final
        # current-attempt task state, not over every historical line.
        accounting_frame = accounting_frame.drop_duplicates(subset=["task_key"], keep="last")
    status_counts = _status_counts(accounting_frame, expected_tasks)
    note = {
        "artifact_type": "corrected_result_note", "run_id": manifest.get("run_id"),
        "protocol_version": manifest.get("protocol_version"), "code_commit": manifest.get("code_commit"),
        "code_fingerprint": manifest.get("code_fingerprint"),
        "configuration_fingerprint": manifest.get("configuration_fingerprint"),
        "runtime_fingerprint": manifest.get("runtime_fingerprint"),
        "historical_separation": {"combined_with_corrected": False},
        "split_policies": [{"split_policy": policy, "definition": "row-level stratified folds" if policy == "row_level" else "exact raw predictor groups confined to one fold", "n_splits": manifest.get("configuration", {}).get("n_splits"), "fold_feasibility": "pending", "zero_shared_groups_asserted": policy == "group_aware"} for policy in policies],
        "task_accounting": {
            "expected_tasks": expected_tasks, "counts_by_status": status_counts,
            "counts_by_phase": accounting_frame.get("phase", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
            "coverage_denominator": "declared expected task grid",
        },
        "dataset_scores": dataset_scores, "contrasts": contrast_rows,
        "row_group_side_by_side": side_by_side,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(note, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return note


__all__ = ["build_corrected_result_note", "paired_dataset_bootstrap"]
