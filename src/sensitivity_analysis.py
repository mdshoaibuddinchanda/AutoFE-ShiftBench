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
from src.historical_snapshot import HistoricalSnapshotUnsupported,at_event,completed_model_events


SENSITIVITY_SCHEMA_VERSION = "incomplete_run_information_boundary_v2"
SENSITIVITY_CONFIG_VERSION = "incomplete_run_logical_prefix_regimes_v2"
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
    store = ManifestStore(manifest_db,read_only=True)
    snapshot = store.snapshot(run_id)
    ledger_rows, ledger_exclusions = _read_ledger(ledger, run_id)
    after = _sha256(ledger)
    if before != after:
        raise SensitivityInputError("Result ledger changed during the read-only snapshot")
    run_config = json.loads(snapshot["run"]["config_json"])
    snapshot_id = hashlib.sha256(_canonical({"run_id": run_id, "ledger_sha256": after,
        "authoritative_contents":{key:snapshot[key] for key in ("run","tasks","attempts","durable_results","task_events","superseded_results")}}).encode("utf-8")).hexdigest()
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
        ledger_conflict = len({_canonical({key:value for key,value in row.items() if key != "source_line"}) for row in ledger_candidates}) > 1
        if ledger_conflict:
            exclusions.append({"task_id": task_id, "reason": "conflicting_ledger_rows"})
        metric_value = None
        metric_reason = "missing_durable_result"
        result_payload = durable_payload
        if durable_payload is not None:
            valid, value, reason = _finite_metric(durable_payload, config.metric)
            if hashlib.sha256(_canonical(durable_payload).encode()).hexdigest() != durable["payload_hash"]:
                valid,value,reason=False,None,"durable_payload_hash_conflict"
            if valid and durable_payload.get("status", "success") == "success":
                metric_value, metric_reason = value, None
            else:
                metric_reason = reason or "non_success_result"
        elif ledger_payload is not None:
            metric_reason = "ledger_only_not_manifest_durable"
        attempts = attempts_by_task.get(task_id, [])
        latest_attempt = max(attempts,key=lambda row:(int(row.get("attempt_number",0)),str(row.get("started_at","")))) if attempts else {}
        state = str(task["state"])
        planned_skip = task.get("planned_skip_reason") or payload.get("planned_skip_reason")
        eligibility_status="eligible"
        if planned_skip:
            classification = "planned_skip"
            known_design_skip=planned_skip in {"planned_design_infeasible","split_policy_infeasible","declared_task_infeasible"}
            eligible = False
            eligibility_status="planned_infeasible" if known_design_skip else "unknown"
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
            "eligibility_status":eligibility_status,
            "dataset_fingerprint":payload.get("data_identity",{}).get("fingerprint"),
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
            eligibility_status="eligible" if eligible else ("unknown" if not left or not right or any(side.get("eligibility_status") == "unknown" for side in (left,right) if side) else "planned_infeasible")
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
            if eligibility_status == "planned_infeasible":
                bound_status="planned_infeasible"
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
                "eligibility_status":eligibility_status,
                "dataset_fingerprint":source.get("dataset_fingerprint"),
                "pipeline_identity_a":None if left is None else left.get("pipeline_identity"),
                "pipeline_identity_b":None if right is None else right.get("pipeline_identity"),
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


def _pair_fingerprint(pairs):
    """Resampling depends on scientific values, not timing or future exports."""
    fields=("contrast_id","dataset","stratum","task_key","pair_status","eligibility_status","dataset_fingerprint",
        "pipeline_identity_a","pipeline_identity_b","metric_a","metric_b","bound_lower","bound_upper")
    rows=[]
    for row in pairs.to_dict("records"):
        values={key:(None if pd.isna(row.get(key)) else row.get(key)) for key in fields}
        rows.append(values)
    return hashlib.sha256(_canonical(sorted(rows,key=_canonical)).encode()).hexdigest()


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
        all_tasks_bounded=len(all_bound) == len(eligible) and len(eligible)>0
        unknown_count=int((group["eligibility_status"] == "unknown").sum()) if "eligibility_status" in group else 0
        all_tasks_bounded=all_tasks_bounded and unknown_count == 0
        bound_status = "complete" if all_tasks_bounded else ("incomplete_eligible_tasks" if len(eligible) else "unsupported")
        if not all_tasks_bounded:
            overall_lower=overall_upper=None
        if unknown_count:
            bound_status="unsupported_unknown_eligibility"
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
            "n_bound_tasks":len(all_bound),"n_eligible_tasks":len(eligible),"n_unknown_eligibility":unknown_count,
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
            "n_manifest_pair_tasks":len(group),"n_unknown_eligibility":unknown_count,
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
        if row["eligible"]:
            key_to_pipelines.setdefault(_task_key(row, config.task_fields), set()).add(str(row["pipeline"]))
    common_keys = {key for key, pipelines in key_to_pipelines.items() if family_pipelines.issubset(pipelines)}

    selections: dict[str, set[tuple[str, str, tuple[Any, ...]]]] = {}
    primary = pair_frame[pair_frame["pair_status"] == "paired_valid"]
    selections["primary_observed"] = {(str(row.contrast_id), str(row.stratum), tuple(row.task_key_tuple)) for row in primary.itertuples()}
    selections["matched_task_blocks"] = set(selections["primary_observed"])
    selections["common_eligible_task_set"] = {
        (str(row.contrast_id), str(row.stratum), tuple(row.task_key_tuple))
        for row in pair_frame.itertuples()
        if tuple(row.task_key_tuple) in common_keys
    }
    universes={regime:pair_frame for regime in selections}
    universes["common_eligible_task_set"]=pair_frame[pair_frame["task_key_tuple"].isin(common_keys)]

    # Complete datasets and fixed coverage thresholds are selected using only
    # manifest-derived eligible denominators, never completed-row counts.
    for regime, threshold in [("complete_datasets", 1.0)] + [(f"coverage_threshold_{value:.2f}", value) for value in config.coverage_thresholds]:
        chosen: set[tuple[str, str, tuple[Any, ...]]] = set()
        selected_datasets=set()
        for (contrast_id, stratum), group in pair_frame.groupby(["contrast_id", "stratum"], dropna=False):
            eligible = group[group["eligible"]]
            valid = group[group["pair_status"] == "paired_valid"]
            for dataset, expected in eligible.groupby("dataset", dropna=False):
                observed = valid[valid["dataset"] == dataset]
                if (group.loc[group["dataset"] == dataset,"eligibility_status"] == "unknown").any():
                    continue
                fraction = len(observed) / len(expected) if len(expected) else 0.0
                if fraction >= threshold and (regime != "complete_datasets" or fraction >= 1.0):
                    chosen.update((str(contrast_id), str(stratum), tuple(row.task_key_tuple)) for row in observed.itertuples())
                    selected_datasets.add((contrast_id,stratum,dataset))
        selections[regime] = chosen
        universes[regime]=pair_frame[[ (row.contrast_id,row.stratum,row.dataset) in selected_datasets for row in pair_frame.itertuples() ]]

    cutoff_rows: list[dict[str, Any]] = []
    try:
        model_events=completed_model_events(snapshot)
    except HistoricalSnapshotUnsupported:
        model_events=[]
    if model_events:
        for fraction in config.cutoff_fractions:
            index = max(0, min(len(model_events) - 1, math.ceil(fraction * len(model_events)) - 1))
            event=model_events[index]
            regime = f"stop_prefix_{fraction:.2f}"
            try:
                historical=at_event(snapshot,event["event_order"])
            except HistoricalSnapshotUnsupported as exc:
                cutoff_rows.append({"regime":regime,"cutoff_fraction":fraction,"status":"unsupported_history","reason":str(exc)})
                continue
            visible={row["scientific_task_id"] for row in historical["durable_results"]}
            historical_coverage,_=_coverage_rows(historical,{key:value for key,value in ledger_rows.items() if key in visible},config)
            historical_pairs=_pair_rows(pd.DataFrame(historical_coverage),config)
            chosen={(str(row.contrast_id),str(row.stratum),tuple(row.task_key_tuple)) for row in historical_pairs.itertuples() if row.pair_status == "paired_valid"}
            selections[regime] = chosen
            universes[regime]=historical_pairs
            cutoff_rows.append({"regime":regime,"cutoff_fraction":fraction,"cutoff_at":event["recorded_at"],"cutoff_event_order":event["event_order"],
                "n_durable_model_events":len(model_events),"status":"complete","scope":"transactional_logical_commit_prefix; timestamps are descriptive",
                "prefix_scientific_fingerprint":_pair_fingerprint(historical_pairs),"n_visible_model_results":sum(row["classification"] == "completed_valid" for row in historical_coverage)})
    else:
        for fraction in config.cutoff_fractions:
            cutoff_rows.append({"regime": f"stop_prefix_{fraction:.2f}", "cutoff_fraction": fraction, "cutoff_at": None, "status": "unsupported_no_complete_logical_history"})

    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    membership: list[pd.DataFrame] = []
    for regime, selected in selections.items():
        universe=universes[regime]
        selected_frame = universe.iloc[0:0] if universe.empty else universe[
            universe.apply(
                lambda row: (str(row["contrast_id"]), str(row["stratum"]), tuple(row["task_key_tuple"])) in selected,
                axis=1,
            )
        ]
        summaries, datasets, bounds = _dataset_summary(selected_frame, universe, config, regime=regime, fingerprint=_pair_fingerprint(universe))
        summary_rows.extend(summaries)
        dataset_rows.extend(datasets)
        bound_rows.extend(bounds)
        membership.append(_normalise_pairs_for_analysis(universe, regime, selected))
    summaries_frame = _apply_holm(pd.DataFrame(summary_rows), config.alpha)
    dataset_frame = pd.DataFrame(dataset_rows)
    bounds_frame = pd.DataFrame(bound_rows)
    membership_frame = pd.concat(membership, ignore_index=True) if membership else pd.DataFrame()

    loo_rows: list[dict[str, Any]] = []
    primary_dataset = dataset_frame[dataset_frame["regime"] == "primary_observed"] if not dataset_frame.empty else pd.DataFrame()
    for (contrast_id, stratum), group in primary_dataset.groupby(["contrast_id", "stratum"], dropna=False) if not primary_dataset.empty else []:
        for omitted in sorted(group["dataset"].dropna().unique(), key=str):
            remain = group[group["dataset"] != omitted]
            loo_rows.append({"regime": "leave_one_dataset_out", "inference_role":"descriptive_only", "contrast_id": contrast_id, "stratum": stratum, "omitted_dataset": omitted, "estimate_b_minus_a": float(remain["estimate_b_minus_a"].mean()) if len(remain) else None, "n_datasets": int(len(remain)), "status": "complete" if len(remain) >= config.min_datasets else "too_few_datasets"})
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
        "regime_definitions":{"matched_task_blocks":"explicit alias of primary observed paired blocks; no independent robustness claim",
            "common_eligible_task_set":"intersection of declared family eligibility, including unresolved outcomes",
            "complete_datasets":"complete contrast/stratum dataset blocks, not entire-run completion",
            "coverage_thresholds":"selected contrast/stratum datasets; bounds retain all their intended eligible tasks",
            "stop_prefix":"logical transactional task/result history; no inferred historical snapshot from wall timestamps alone",
            "leave_one_dataset_out":"descriptive dataset omission; no confirmatory interval or test"},
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
