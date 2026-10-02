"""Incomplete-run coverage and sensitivity analysis.

This module consumes one SQLite manifest snapshot and one result-ledger
fingerprint.  It never treats retries, precompute rows, cache files, or
diagnostic artifacts as model observations.  The intended model-task universe
comes from the manifest; authoritative metric values come from durable results.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.dataset_statistics import (
    DEFAULT_TASK_FIELDS,
    AnalysisConfig,
    AnalysisInputError,
    _bootstrap_dataset_mean,
    _finite_metric,
    _holm,
    _normalise_severity,
    _sign_flip_test,
)
from src.task_manifest import ManifestError, ManifestStore


SENSITIVITY_SCHEMA_VERSION = "incomplete_run_sensitivity_v1"
SENSITIVITY_CONFIG_VERSION = "incomplete_run_regimes_v1"
DEFAULT_CONTRASTS = (
    ("historical_raw", "Raw", "AutoFE_Baseline"),
    ("fair_cap_control", "Raw_Capped", "AutoFE_Baseline"),
    ("fair_full_control", "Raw_Full", "AutoFE_Baseline"),
)
DEFAULT_STRATUM_FIELDS = ("split_policy", "condition", "model")


class SensitivityInputError(ValueError):
    """Raised when an incomplete-run analysis cannot establish its universe."""


@dataclass(frozen=True)
class SensitivityConfig:
    """Predeclared missing-outcome regimes and aggregation contract."""

    metric: str = "roc_auc"
    task_fields: tuple[str, ...] = DEFAULT_TASK_FIELDS
    stratum_fields: tuple[str, ...] = DEFAULT_STRATUM_FIELDS
    contrasts: tuple[tuple[str, str, str], ...] = DEFAULT_CONTRASTS
    coverage_thresholds: tuple[float, ...] = (0.5, 0.75, 1.0)
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2000
    permutation_resamples: int = 5000
    random_state: int = 20261003
    alpha: float = 0.05
    min_datasets: int = 2
    metric_lower: float = 0.0
    metric_upper: float = 1.0
    cutoff_fractions: tuple[float, ...] = (0.5, 0.75, 1.0)
    expected_protocol_version: str | None = None
    expected_seed_scheme_version: str | None = None
    sensitivity_config_version: str = SENSITIVITY_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.metric != "roc_auc":
            raise ValueError("ROC-AUC bounds are the only bounded metric implemented")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.bootstrap_resamples < 0 or self.permutation_resamples < 0:
            raise ValueError("resampling counts must be non-negative")
        if self.min_datasets < 1:
            raise ValueError("min_datasets must be positive")
        if self.metric_lower >= self.metric_upper:
            raise ValueError("metric bounds must be ordered")
        thresholds = tuple(float(value) for value in self.coverage_thresholds)
        cutoffs = tuple(float(value) for value in self.cutoff_fractions)
        if any(not 0.0 < value <= 1.0 for value in thresholds + cutoffs):
            raise ValueError("coverage thresholds and cutoff fractions must be in (0, 1]")
        if tuple(sorted(set(thresholds))) != thresholds:
            raise ValueError("coverage thresholds must be sorted and unique")
        if tuple(sorted(set(cutoffs))) != cutoffs:
            raise ValueError("cutoff fractions must be sorted and unique")
        if not self.task_fields or "dataset" not in self.task_fields:
            raise ValueError("task_fields must include dataset")
        for contrast_id, left, right in self.contrasts:
            if not contrast_id or left == right:
                raise ValueError("each contrast needs a unique ID and two different pipelines")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["task_fields"] = list(self.task_fields)
        value["stratum_fields"] = list(self.stratum_fields)
        value["contrasts"] = [list(item) for item in self.contrasts]
        value["coverage_thresholds"] = list(self.coverage_thresholds)
        value["cutoff_fractions"] = list(self.cutoff_fractions)
        return value


@dataclass
class SensitivityBundle:
    config: dict[str, Any]
    snapshot: dict[str, Any]
    coverage_tasks: pd.DataFrame
    coverage_summary: pd.DataFrame
    pair_blocks: pd.DataFrame
    regime_membership: pd.DataFrame
    dataset_contrasts: pd.DataFrame
    summaries: pd.DataFrame
    bounds: pd.DataFrame
    leave_one_out: pd.DataFrame
    cutoff_summaries: pd.DataFrame
    exclusions: pd.DataFrame


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_key(row: Mapping[str, Any], fields: Iterable[str]) -> tuple[Any, ...]:
    return tuple(_normalise_severity(row.get(field)) if field == "severity" else row.get(field) for field in fields)


def _stratum(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    return "|".join(f"{field}={row.get(field)}" for field in fields)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _read_ledger(path: Path, run_id: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                exclusions.append({"source_line": line_number, "reason": "malformed_json", "detail": str(exc)})
                continue
            if not isinstance(value, dict):
                exclusions.append({"source_line": line_number, "reason": "non_mapping_record"})
                continue
            if value.get("run_id") != run_id:
                if value.get("run_id") is None:
                    exclusions.append({"source_line": line_number, "reason": "unattributed_legacy_row"})
                continue
            value["source_line"] = line_number
            task_id = value.get("scientific_task_id") or value.get("task_id")
            if not task_id:
                exclusions.append({"source_line": line_number, "reason": "missing_scientific_task_id"})
                continue
            rows.setdefault(str(task_id), []).append(value)
    return rows, exclusions


def snapshot_manifest_ledger(
    manifest_db: str | Path,
    ledger_path: str | Path,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Capture a stable SQLite snapshot and verify the ledger did not change."""
    ledger = Path(ledger_path)
    before = _sha256(ledger)
    store = ManifestStore(manifest_db)
    snapshot = store.snapshot(run_id)
    ledger_rows, ledger_exclusions = _read_ledger(ledger, run_id)
    after = _sha256(ledger)
    if before != after:
        raise SensitivityInputError("Result ledger changed during the read-only snapshot")
    run_config = json.loads(snapshot["run"]["config_json"])
    snapshot_id = hashlib.sha256(_canonical({"run_id": run_id, "run_config": run_config, "ledger_sha256": after, "task_ids": [row["scientific_task_id"] for row in snapshot["tasks"]]}).encode("utf-8")).hexdigest()
    snapshot["snapshot_id"] = "snapshot_" + snapshot_id[:24]
    snapshot["ledger_path"] = str(ledger)
    snapshot["ledger_sha256"] = after
    snapshot["manifest_db"] = str(manifest_db)
    snapshot["run_config"] = run_config
    return snapshot, ledger_rows, ledger_exclusions


def _coverage_rows(snapshot: Mapping[str, Any], ledger_rows: Mapping[str, list[dict[str, Any]]], config: SensitivityConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts_by_task: dict[str, list[dict[str, Any]]] = {}
    for attempt in snapshot["attempts"]:
        attempts_by_task.setdefault(str(attempt["scientific_task_id"]), []).append(attempt)
    durable_by_task = {str(row["scientific_task_id"]): row for row in snapshot["durable_results"]}
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for task in snapshot["tasks"]:
        payload = json.loads(task["payload_json"])
        if payload.get("stage") != "model" and task.get("stage") != "model":
            continue
        task_id = str(task["scientific_task_id"])
        durable = durable_by_task.get(task_id)
        durable_payload = None if durable is None else json.loads(durable["payload_json"])
        ledger_candidates = ledger_rows.get(task_id, [])
        ledger_payload = ledger_candidates[0] if ledger_candidates else None
        ledger_conflict = len({_canonical(row) for row in ledger_candidates}) > 1
        if ledger_conflict:
            exclusions.append({"task_id": task_id, "reason": "conflicting_ledger_rows"})
        metric_value = None
        metric_reason = "missing_durable_result"
        result_payload = durable_payload
        if durable_payload is not None:
            valid, value, reason = _finite_metric(durable_payload, config.metric)
            if valid and durable_payload.get("status", "success") == "success":
                metric_value, metric_reason = value, None
            else:
                metric_reason = reason or "non_success_result"
        elif ledger_payload is not None:
            metric_reason = "ledger_only_not_manifest_durable"
        attempts = attempts_by_task.get(task_id, [])
        latest_attempt = attempts[-1] if attempts else {}
        state = str(task["state"])
        planned_skip = task.get("planned_skip_reason") or payload.get("planned_skip_reason")
        if planned_skip:
            classification = "planned_skip"
            eligible = False
        elif state == "completed" and metric_value is not None:
            classification, eligible = "completed_valid", True
        elif state == "completed":
            classification, eligible = "completed_invalid", True
        elif state in {"failed", "timeout", "pending", "running"}:
            classification, eligible = state, True
        elif state == "skipped":
            classification, eligible = "dependency_skip" if task.get("outcome_reason") == "dependency_failure" else "skip", True
        else:
            classification, eligible = "unrecognized_state", True
        row = {
            **{field: payload.get(field) for field in config.task_fields},
            "scientific_task_id": task_id,
            "pipeline": payload.get("pipeline"),
            "pipeline_identity": payload.get("pipeline_identity"),
            "operator_set_id": payload.get("operator_set_id"),
            "cap_policy_version": payload.get("cap_policy_version"),
            "state": state,
            "classification": classification,
            "eligible": eligible,
            "planned_skip_reason": planned_skip,
            "outcome_reason": task.get("outcome_reason"),
            "attempt_count": int(task.get("attempt_count", 0)),
            "latest_attempt_state": latest_attempt.get("state"),
            "latest_attempt_id": latest_attempt.get("attempt_id"),
            "latest_attempt_started_at": latest_attempt.get("started_at"),
            "latest_attempt_finished_at": latest_attempt.get("finished_at"),
            "latest_attempt_elapsed_seconds": latest_attempt.get("elapsed_seconds"),
            "metric_value": metric_value,
            "metric_reason": metric_reason,
            "diagnostic_status": None if result_payload is None else result_payload.get("diagnostic_status"),
            "result_durable_at": None if durable is None else durable.get("durable_at"),
            "result_ref": task.get("result_ref"),
            "result_payload_hash": None if durable is None else durable.get("payload_hash"),
        }
        rows.append(row)
    return rows, exclusions


def _pair_rows(coverage: pd.DataFrame, config: SensitivityConfig) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for contrast_id, left_pipeline, right_pipeline in config.contrasts:
        selected = coverage[coverage["pipeline"].isin([left_pipeline, right_pipeline])].copy()
        grouped = selected.groupby(list(config.task_fields), dropna=False, sort=True)
        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            sides = {str(pipeline): frame.iloc[0].to_dict() for pipeline, frame in group.groupby("pipeline", dropna=False)}
            left = sides.get(left_pipeline)
            right = sides.get(right_pipeline)
            source = left or right or {}
            stratum = _stratum(source, config.stratum_fields)
            eligible = bool(left and right and left["eligible"] and right["eligible"])
            if not left or not right:
                status = "missing_manifest_side"
            elif not eligible:
                status = "planned_infeasible"
            elif left["classification"] == "completed_valid" and right["classification"] == "completed_valid":
                status = "paired_valid"
            elif left["classification"] == "completed_valid" or right["classification"] == "completed_valid":
                status = "missing_valid_partner"
            else:
                status = "unresolved_outcome"
            left_value = None if left is None or pd.isna(left.get("metric_value")) else float(left.get("metric_value"))
            right_value = None if right is None or pd.isna(right.get("metric_value")) else float(right.get("metric_value"))
            diff = None if status != "paired_valid" else float(right_value - left_value)
            lower, upper, bound_status = None, None, "eligible_universe_unknown"
            if eligible:
                if left_value is not None and right_value is not None:
                    lower = upper = diff
                    bound_status = "observed_pair"
                elif left_value is not None and config.metric_lower <= left_value <= config.metric_upper:
                    lower, upper, bound_status = config.metric_lower - left_value, config.metric_upper - left_value, "one_sided_missing_b"
                elif right_value is not None and config.metric_lower <= right_value <= config.metric_upper:
                    lower, upper, bound_status = right_value - config.metric_upper, right_value - config.metric_lower, "one_sided_missing_a"
                elif left_value is None and right_value is None:
                    lower, upper, bound_status = config.metric_lower - config.metric_upper, config.metric_upper - config.metric_lower, "both_missing"
                else:
                    bound_status = "metric_outside_declared_range"
            rows.append({
                "contrast_id": contrast_id,
                "pipeline_a": left_pipeline,
                "pipeline_b": right_pipeline,
                "task_key": json.dumps(list(key), default=str),
                "task_key_tuple": key,
                "dataset": source.get("dataset"),
                "stratum": stratum,
                "pair_status": status,
                "eligible": eligible,
                "metric_a": left_value,
                "metric_b": right_value,
                "difference_b_minus_a": diff,
                "bound_lower": lower,
                "bound_upper": upper,
                "bound_status": bound_status,
                "task_id_a": None if left is None else left["scientific_task_id"],
                "task_id_b": None if right is None else right["scientific_task_id"],
                "state_a": None if left is None else left["state"],
                "state_b": None if right is None else right["state"],
                "durable_at_a": None if left is None else left.get("result_durable_at"),
                "durable_at_b": None if right is None else right.get("result_durable_at"),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["task_key_tuple"] = frame["task_key_tuple"].map(tuple)
    return frame


def _normalise_pairs_for_analysis(pairs: pd.DataFrame, regime: str, selected: set[tuple[str, str, tuple[Any, ...]]]) -> pd.DataFrame:
    if pairs.empty:
        return pairs.assign(selected=False, regime=regime)
    result = pairs.copy()
    result["selected"] = [
        (str(row.contrast_id), str(row.stratum), tuple(row.task_key_tuple)) in selected
        for row in result.itertuples()
    ]
    result["regime"] = regime
    return result


def _dataset_summary(
    selected_pairs: pd.DataFrame,
    all_pairs: pd.DataFrame,
    config: SensitivityConfig,
    *,
    regime: str,
    fingerprint: str,
    extra_label: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    if all_pairs.empty:
        return summaries, dataset_rows, bound_rows
    grouping = all_pairs.groupby(["contrast_id", "stratum"], dropna=False, sort=True)
    for (contrast_id, stratum), group in grouping:
        selected = selected_pairs[(selected_pairs["contrast_id"] == contrast_id) & (selected_pairs["stratum"] == stratum)] if not selected_pairs.empty else group.iloc[0:0]
        eligible = group[group["eligible"]].copy()
        valid = selected[selected["pair_status"] == "paired_valid"].copy()
        expected_by_dataset = {dataset: frame for dataset, frame in eligible.groupby("dataset", dropna=False)}
        valid_by_dataset = {dataset: frame for dataset, frame in valid.groupby("dataset", dropna=False)}
        for dataset, frame in valid_by_dataset.items():
            values = frame["difference_b_minus_a"].astype(float).to_numpy()
            expected = expected_by_dataset.get(dataset, frame)
            dataset_estimate = float(np.mean(values)) if len(values) else None
            lower = float(np.mean(expected["bound_lower"].astype(float))) if len(expected) and expected["bound_lower"].notna().all() else None
            upper = float(np.mean(expected["bound_upper"].astype(float))) if len(expected) and expected["bound_upper"].notna().all() else None
            dataset_rows.append({
                "regime": regime,
                "label": extra_label,
                "contrast_id": contrast_id,
                "stratum": stratum,
                "dataset": dataset,
                "estimate_b_minus_a": dataset_estimate,
                "bound_lower": lower,
                "bound_upper": upper,
                "n_intended_tasks": int(len(expected)),
                "n_valid_pairs": int(len(values)),
                "coverage_fraction": float(len(values) / len(expected)) if len(expected) else None,
                "task_ids": json.dumps(sorted(set(frame["task_id_a"].dropna()) | set(frame["task_id_b"].dropna()))),
            })
        dataset_values = [row["estimate_b_minus_a"] for row in dataset_rows if row["regime"] == regime and row["contrast_id"] == contrast_id and row["stratum"] == stratum and row["label"] == extra_label]
        dataset_values = np.asarray(dataset_values, dtype=float)
        label = f"{regime}|{contrast_id}|{stratum}|{extra_label}"
        if len(dataset_values):
            analysis_config = AnalysisConfig(
                metric=config.metric,
                pipeline_a=str(group["pipeline_a"].iloc[0]),
                pipeline_b=str(group["pipeline_b"].iloc[0]),
                confidence_level=config.confidence_level,
                bootstrap_resamples=config.bootstrap_resamples,
                permutation_resamples=config.permutation_resamples,
                random_state=config.random_state,
                alpha=config.alpha,
                min_datasets=config.min_datasets,
            )
            bootstrap = _bootstrap_dataset_mean(dataset_values, analysis_config, stratum_label=label, fingerprint=fingerprint)
            test = _sign_flip_test(dataset_values, analysis_config, stratum_label=label, fingerprint=fingerprint)
            estimate = float(np.mean(dataset_values))
            bounds = [row for row in bound_rows if row.get("regime") == regime and row.get("contrast_id") == contrast_id and row.get("stratum") == stratum and row.get("label") == extra_label]
        else:
            bootstrap = {"status": "no_valid_datasets", "ci_lower": None, "ci_upper": None, "n_datasets": 0}
            test = {"status": "no_valid_datasets", "p_value": None, "n_datasets": 0}
            estimate = None
        all_bound = eligible[eligible["bound_lower"].notna() & eligible["bound_upper"].notna()]
        bound_by_dataset = []
        for dataset, frame in all_bound.groupby("dataset", dropna=False):
            if len(frame):
                bound_by_dataset.append((float(frame["bound_lower"].mean()), float(frame["bound_upper"].mean())))
        overall_lower = float(np.mean([item[0] for item in bound_by_dataset])) if bound_by_dataset else None
        overall_upper = float(np.mean([item[1] for item in bound_by_dataset])) if bound_by_dataset else None
        bound_status = "complete" if len(bound_by_dataset) == len(expected_by_dataset) and bound_by_dataset else ("partial_eligible_datasets" if bound_by_dataset else "unsupported")
        bound_rows.append({
            "regime": regime,
            "label": extra_label,
            "contrast_id": contrast_id,
            "stratum": stratum,
            "bound_lower": overall_lower,
            "bound_upper": overall_upper,
            "bound_status": bound_status,
            "n_bound_datasets": len(bound_by_dataset),
            "n_eligible_datasets": len(expected_by_dataset),
        })
        summaries.append({
            "regime": regime,
            "label": extra_label,
            "contrast_id": contrast_id,
            "pipeline_a": group["pipeline_a"].iloc[0],
            "pipeline_b": group["pipeline_b"].iloc[0],
            "stratum": stratum,
            "estimate_b_minus_a": estimate,
            "n_intended_tasks": int(len(eligible)),
            "n_valid_pairs": int(len(valid)),
            "n_datasets": int(len(dataset_values)),
            "n_eligible_datasets": int(len(expected_by_dataset)),
            "n_excluded_datasets": int(len(expected_by_dataset) - len(dataset_values)),
            "coverage_mean_over_eligible_datasets": float(np.mean([len(valid_by_dataset.get(dataset, [])) / len(frame) for dataset, frame in expected_by_dataset.items() if len(frame)])) if expected_by_dataset else None,
            "bound_lower": overall_lower,
            "bound_upper": overall_upper,
            "bound_status": bound_status,
            "confidence_level": config.confidence_level,
            **bootstrap,
            **test,
            "analysis_schema_version": SENSITIVITY_SCHEMA_VERSION,
        })
    return summaries, dataset_rows, bound_rows


def _apply_holm(summaries: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if summaries.empty:
        return summaries
    summaries = summaries.copy()
    summaries["family_id"] = summaries["regime"].map(lambda value: f"sensitivity_{value}_v1")
    summaries["family_size"] = 0
    summaries["p_value_adjusted_holm"] = np.nan
    summaries["rejected_holm"] = False
    for family_id, indices in summaries.groupby("family_id").groups.items():
        valid = [index for index in indices if pd.notna(summaries.loc[index, "p_value"])]
        if not valid:
            continue
        adjusted, rejected = _holm([float(summaries.loc[index, "p_value"]) for index in valid], alpha)
        summaries.loc[valid, "family_size"] = len(valid)
        summaries.loc[valid, "p_value_adjusted_holm"] = adjusted
        summaries.loc[valid, "rejected_holm"] = rejected
    return summaries


def analyze_sensitivity(
    manifest_db: str | Path,
    ledger_path: str | Path,
    run_id: str,
    *,
    config: SensitivityConfig | None = None,
    output_dir: str | Path | None = None,
) -> SensitivityBundle:
    config = config or SensitivityConfig()
    snapshot, ledger_rows, ledger_exclusions = snapshot_manifest_ledger(manifest_db, ledger_path, run_id)
    run_protocol = snapshot["run"].get("protocol_version")
    run_seed = snapshot["run"].get("seed_scheme_version")
    if config.expected_protocol_version and run_protocol != config.expected_protocol_version:
        raise SensitivityInputError(f"Manifest protocol mismatch: {run_protocol}")
    if config.expected_seed_scheme_version and run_seed != config.expected_seed_scheme_version:
        raise SensitivityInputError(f"Manifest seed-scheme mismatch: {run_seed}")
    coverage_rows, coverage_exclusions = _coverage_rows(snapshot, ledger_rows, config)
    coverage = pd.DataFrame(coverage_rows)
    if coverage.empty:
        raise SensitivityInputError("Manifest contains no model tasks")
    pair_frame = _pair_rows(coverage, config)
    if pair_frame.empty:
        raise SensitivityInputError("No declared pipeline contrasts are represented in the manifest")
    exclusions = ledger_exclusions + coverage_exclusions
    for contrast_id, left, right in config.contrasts:
        if left not in set(coverage["pipeline"]) or right not in set(coverage["pipeline"]):
            exclusions.append({"contrast_id": contrast_id, "reason": "contrast_pipeline_unavailable", "pipeline_a": left, "pipeline_b": right})

    # Common eligible task keys across every declared pipeline in the family.
    family_pipelines = {pipeline for _, left, right in config.contrasts for pipeline in (left, right)}
    key_to_pipelines: dict[tuple[Any, ...], set[str]] = {}
    for _, row in coverage.iterrows():
        if row["eligible"] and row["classification"] == "completed_valid":
            key_to_pipelines.setdefault(_task_key(row, config.task_fields), set()).add(str(row["pipeline"]))
    common_keys = {key for key, pipelines in key_to_pipelines.items() if family_pipelines.issubset(pipelines)}

    selections: dict[str, set[tuple[str, str, tuple[Any, ...]]]] = {}
    primary = pair_frame[pair_frame["pair_status"] == "paired_valid"]
    selections["primary_observed"] = {(str(row.contrast_id), str(row.stratum), tuple(row.task_key_tuple)) for row in primary.itertuples()}
    selections["matched_task_blocks"] = set(selections["primary_observed"])
    selections["common_eligible_task_set"] = {
        (str(row.contrast_id), str(row.stratum), tuple(row.task_key_tuple))
        for row in primary.itertuples()
        if tuple(row.task_key_tuple) in common_keys
    }

    # Complete datasets and fixed coverage thresholds are selected using only
    # manifest-derived eligible denominators, never completed-row counts.
    for regime, threshold in [("complete_datasets", 1.0)] + [(f"coverage_threshold_{value:.2f}", value) for value in config.coverage_thresholds]:
        chosen: set[tuple[str, str, tuple[Any, ...]]] = set()
        for (contrast_id, stratum), group in pair_frame.groupby(["contrast_id", "stratum"], dropna=False):
            eligible = group[group["eligible"]]
            valid = group[group["pair_status"] == "paired_valid"]
            for dataset, expected in eligible.groupby("dataset", dropna=False):
                observed = valid[valid["dataset"] == dataset]
                fraction = len(observed) / len(expected) if len(expected) else 0.0
                if fraction >= threshold and (regime != "complete_datasets" or fraction >= 1.0):
                    chosen.update((str(contrast_id), str(stratum), tuple(row.task_key_tuple)) for row in observed.itertuples())
        selections[regime] = chosen

    cutoff_rows: list[dict[str, Any]] = []
    valid_times = sorted({parsed for value in pair_frame["durable_at_a"].dropna() for parsed in [_parse_time(value)] if parsed} | {parsed for value in pair_frame["durable_at_b"].dropna() for parsed in [_parse_time(value)] if parsed})
    if valid_times:
        for fraction in config.cutoff_fractions:
            index = max(0, min(len(valid_times) - 1, math.ceil(fraction * len(valid_times)) - 1))
            cutoff = valid_times[index]
            regime = f"stop_prefix_{fraction:.2f}"
            chosen = set()
            for row in primary.itertuples():
                left_time, right_time = _parse_time(row.durable_at_a), _parse_time(row.durable_at_b)
                if left_time and right_time and left_time <= cutoff and right_time <= cutoff:
                    chosen.add((str(row.contrast_id), str(row.stratum), tuple(row.task_key_tuple)))
            selections[regime] = chosen
            cutoff_rows.append({"regime": regime, "cutoff_fraction": fraction, "cutoff_at": cutoff.isoformat(), "n_durable_model_timestamps": len(valid_times), "status": "complete"})
    else:
        for fraction in config.cutoff_fractions:
            cutoff_rows.append({"regime": f"stop_prefix_{fraction:.2f}", "cutoff_fraction": fraction, "cutoff_at": None, "n_durable_model_timestamps": 0, "status": "unsupported_no_timestamps"})

    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    membership: list[pd.DataFrame] = []
    fingerprint = str(snapshot["ledger_sha256"])
    for regime, selected in selections.items():
        selected_frame = pair_frame.iloc[0:0] if pair_frame.empty else pair_frame[
            pair_frame.apply(
                lambda row: (str(row["contrast_id"]), str(row["stratum"]), tuple(row["task_key_tuple"])) in selected,
                axis=1,
            )
        ]
        summaries, datasets, bounds = _dataset_summary(selected_frame, pair_frame, config, regime=regime, fingerprint=fingerprint)
        summary_rows.extend(summaries)
        dataset_rows.extend(datasets)
        bound_rows.extend(bounds)
        membership.append(_normalise_pairs_for_analysis(pair_frame, regime, selected))
    summaries_frame = _apply_holm(pd.DataFrame(summary_rows), config.alpha)
    dataset_frame = pd.DataFrame(dataset_rows)
    bounds_frame = pd.DataFrame(bound_rows)
    membership_frame = pd.concat(membership, ignore_index=True) if membership else pd.DataFrame()

    loo_rows: list[dict[str, Any]] = []
    primary_dataset = dataset_frame[dataset_frame["regime"] == "primary_observed"] if not dataset_frame.empty else pd.DataFrame()
    for (contrast_id, stratum), group in primary_dataset.groupby(["contrast_id", "stratum"], dropna=False) if not primary_dataset.empty else []:
        for omitted in sorted(group["dataset"].dropna().unique(), key=str):
            remain = group[group["dataset"] != omitted]
            loo_rows.append({"regime": "leave_one_dataset_out", "contrast_id": contrast_id, "stratum": stratum, "omitted_dataset": omitted, "estimate_b_minus_a": float(remain["estimate_b_minus_a"].mean()) if len(remain) else None, "n_datasets": int(len(remain)), "status": "complete" if len(remain) >= config.min_datasets else "too_few_datasets"})
    cutoff_frame = pd.DataFrame(cutoff_rows)
    exclusions_frame = pd.DataFrame(exclusions)
    config_dict = {
        **config.to_dict(),
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "snapshot_id": snapshot["snapshot_id"],
        "manifest_run_id": run_id,
        "ledger_sha256": snapshot["ledger_sha256"],
        "aggregation": {"task_weighting": "equal_valid_task_within_dataset", "dataset_weighting": "equal_dataset"},
        "bounds": {"metric": config.metric, "lower": config.metric_lower, "upper": config.metric_upper, "interpretation": "identification_bounds_not_confidence_intervals"},
    }
    bundle = SensitivityBundle(config_dict, snapshot, coverage, coverage.groupby(["dataset", "pipeline", "model", "split_policy", "condition", "severity", "fold", "seed", "classification"], dropna=False).size().reset_index(name="n_tasks") if not coverage.empty else pd.DataFrame(), pair_frame, membership_frame, dataset_frame, summaries_frame, bounds_frame, pd.DataFrame(loo_rows), cutoff_frame, exclusions_frame)
    if output_dir is not None:
        write_sensitivity_bundle(bundle, output_dir)
    return bundle


def write_sensitivity_bundle(bundle: SensitivityBundle, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sensitivity_config.json").write_text(_canonical(bundle.config), encoding="utf-8")
    (directory / "snapshot.json").write_text(_canonical(bundle.snapshot), encoding="utf-8")
    bundle.coverage_tasks.to_csv(directory / "coverage_tasks.csv", index=False)
    bundle.coverage_summary.to_csv(directory / "coverage_summary.csv", index=False)
    bundle.pair_blocks.to_csv(directory / "pair_blocks.csv", index=False)
    bundle.regime_membership.to_csv(directory / "regime_membership.csv", index=False)
    bundle.dataset_contrasts.to_csv(directory / "dataset_contrasts.csv", index=False)
    bundle.summaries.to_csv(directory / "sensitivity_summaries.csv", index=False)
    bundle.bounds.to_csv(directory / "bounds.csv", index=False)
    bundle.leave_one_out.to_csv(directory / "leave_one_dataset_out.csv", index=False)
    bundle.cutoff_summaries.to_csv(directory / "cutoff_summaries.csv", index=False)
    bundle.exclusions.to_json(directory / "exclusions.jsonl", orient="records", lines=True)
    return directory


def run_sensitivity_analysis(
    manifest_db: str | Path,
    ledger_path: str | Path,
    run_id: str,
    *,
    output_dir: str | Path = "reports/analysis/sensitivity",
    config: SensitivityConfig | None = None,
) -> SensitivityBundle:
    return analyze_sensitivity(manifest_db, ledger_path, run_id, config=config, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze incomplete-run sensitivity from a manifest and ledger snapshot")
    parser.add_argument("--manifest-db", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="reports/analysis/sensitivity")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutation-resamples", type=int, default=5000)
    args = parser.parse_args()
    result = run_sensitivity_analysis(args.manifest_db, args.ledger, args.run_id, output_dir=args.output_dir, config=SensitivityConfig(bootstrap_resamples=args.bootstrap_resamples, permutation_resamples=args.permutation_resamples))
    print(json.dumps({"output_dir": str(args.output_dir), "snapshot_id": result.snapshot["snapshot_id"], "summary_rows": len(result.summaries), "coverage_rows": len(result.coverage_tasks), "exclusions": len(result.exclusions)}, sort_keys=True))
