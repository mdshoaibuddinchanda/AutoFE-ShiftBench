"""Dataset-level paired analysis for corrected benchmark ledgers.

The benchmark produces many rows per dataset (folds, repetitions, models and
conditions).  This module keeps those rows paired within a dataset and treats
the dataset as the independent unit.  It deliberately emits exclusions and
task-pair records so a summary cannot hide missing or failed work.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION, stable_seed
from src.evaluation import METRIC_RANGES, METRIC_SEMANTICS_VERSION


ANALYSIS_SCHEMA_VERSION = "dataset_cluster_analysis_verified_v2"
ANALYSIS_CONFIG_VERSION = "dataset_equal_paired_run_v2"
DEFAULT_TASK_FIELDS = (
    "dataset",
    "split_policy",
    "seed",
    "fold",
    "condition",
    "severity",
    "model",
)
COMPATIBILITY_FIELDS = (
    "evaluation_protocol_version",
    "seed_scheme_version",
    "pipeline_identity",
    "operator_registry_version",
    "operator_semantics_version",
    "operator_set_id",
    "cap_policy_version",
    "metric_semantics_version",
    "source_code_fingerprint",
    "environment_fingerprint",
)


class AnalysisInputError(ValueError):
    """Raised when a ledger cannot support the declared analysis contract."""


@dataclass(frozen=True)
class AnalysisConfig:
    """Versioned estimand and resampling contract.

    Every split policy, condition and estimator is analyzed as its own stratum
    by default.  The resulting dataset contrasts are then averaged with equal
    dataset weight; repeated task rows never become independent datasets.
    """

    metric: str = "roc_auc"
    pipeline_a: str = "Raw"
    pipeline_b: str = "AutoFE_Baseline"
    task_fields: tuple[str, ...] = DEFAULT_TASK_FIELDS
    stratum_fields: tuple[str, ...] = ("split_policy", "condition", "model")
    task_weighting: str = "equal_valid_task_within_dataset"
    dataset_weighting: str = "equal_dataset"
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2000
    permutation_resamples: int = 5000
    random_state: int = 20261002
    alpha: float = 0.05
    family_id: str = "primary_dataset_contrasts_v1"
    analysis_config_version: str = ANALYSIS_CONFIG_VERSION
    expected_protocol_version: str = EVALUATION_PROTOCOL_VERSION
    expected_seed_scheme_version: str = SEED_SCHEME_VERSION
    min_datasets: int = 2
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.bootstrap_resamples < 0 or self.permutation_resamples < 0:
            raise ValueError("resampling counts must be non-negative")
        if self.task_weighting != "equal_valid_task_within_dataset":
            raise ValueError("Only equal_valid_task_within_dataset is supported")
        if self.dataset_weighting != "equal_dataset":
            raise ValueError("Only equal_dataset weighting is supported")
        if not self.task_fields or "dataset" not in self.task_fields:
            raise ValueError("task_fields must include dataset")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["task_fields"] = list(self.task_fields)
        value["stratum_fields"] = list(self.stratum_fields)
        return value


@dataclass
class AnalysisBundle:
    """Machine-readable outputs from one ledger analysis."""

    config: dict[str, Any]
    input_fingerprint: str
    task_pairs: pd.DataFrame
    dataset_contrasts: pd.DataFrame
    summaries: pd.DataFrame
    exclusions: pd.DataFrame
    multiplicity: pd.DataFrame


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_id(row: Mapping[str, Any], config: AnalysisConfig) -> str:
    identity = {
        field: _normalise_severity(row.get(field)) if field == "severity" else row.get(field)
        for field in config.task_fields
    }
    identity["evaluation_protocol_version"] = row.get("evaluation_protocol_version")
    identity["seed_scheme_version"] = row.get("seed_scheme_version")
    encoded = _canonical_json(identity).encode("utf-8")
    return "task_" + hashlib.sha256(encoded).hexdigest()[:24]


def _read_ledger(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
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
            value["source_line"] = line_number
            records.append(value)
    return records, exclusions


def _validate_protocol(records: Iterable[Mapping[str, Any]], config: AnalysisConfig) -> None:
    selected = [
        row for row in records
        if row.get("pipeline") in {config.pipeline_a, config.pipeline_b}
    ]
    protocols = {row.get("evaluation_protocol_version") for row in selected if row.get("evaluation_protocol_version") is not None}
    seeds = {row.get("seed_scheme_version") for row in selected if row.get("seed_scheme_version") is not None}
    if any(row.get("evaluation_protocol_version") is None for row in selected):
        raise AnalysisInputError("Selected rows are missing evaluation_protocol_version")
    if any(row.get("seed_scheme_version") is None for row in selected):
        raise AnalysisInputError("Selected rows are missing seed_scheme_version")
    if protocols and protocols != {config.expected_protocol_version}:
        raise AnalysisInputError(f"Incompatible evaluation protocol mix: {sorted(protocols)}")
    if seeds and seeds != {config.expected_seed_scheme_version}:
        raise AnalysisInputError(f"Incompatible seed-scheme mix: {sorted(seeds)}")


def _normalise_severity(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value) if isinstance(value, (float, int, np.number)) else value


def _base_key(row: Mapping[str, Any], config: AnalysisConfig) -> tuple[Any, ...]:
    return tuple(
        _normalise_severity(row.get(field)) if field == "severity" else row.get(field)
        for field in config.task_fields
    )


def _stratum_key(row: Mapping[str, Any], config: AnalysisConfig) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in config.stratum_fields)


def _stratum_label(values: tuple[Any, ...], fields: tuple[str, ...]) -> str:
    return "|".join(f"{field}={value}" for field, value in zip(fields, values))


def _finite_metric(row: Mapping[str, Any], metric: str) -> tuple[bool, float | None, str | None]:
    value = row.get(metric)
    if value is None:
        return False, None, "missing_metric"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False, None, "non_numeric_metric"
    if not np.isfinite(numeric):
        return False, None, "nonfinite_metric"
    bounds=METRIC_RANGES.get(metric)
    if bounds is None:
        return False,None,"unsupported_metric_definition"
    if not bounds[0] <= numeric <= bounds[1]:
        return False,None,"metric_outside_declared_range"
    return True, numeric, None


def _check_duplicate_successes(records: list[dict[str, Any]], config: AnalysisConfig) -> None:
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in records:
        if row.get("status", "success") != "success" or row.get("pipeline") not in {config.pipeline_a, config.pipeline_b}:
            continue
        key = _base_key(row, config) + (row.get("pipeline"),)
        if key in seen:
            prior = seen[key]
            raise AnalysisInputError(
                "Duplicate successful scientific task: "
                f"{_task_id(prior, config)} lines {prior.get('source_line')} and {row.get('source_line')}"
            )
        seen[key] = row


def _validate_pipeline_semantics(records: list[dict[str, Any]], config: AnalysisConfig) -> None:
    for pipeline in (config.pipeline_a, config.pipeline_b):
        values: set[tuple[Any, ...]] = set()
        for row in records:
            if row.get("pipeline") != pipeline:
                continue
            semantics=tuple(row.get(field) for field in COMPATIBILITY_FIELDS)
            if any(value is None for value in semantics):
                raise AnalysisInputError(f"Pipeline {pipeline} has unverified scientific semantics")
            values.add(semantics)
        if len(values) > 1:
            raise AnalysisInputError(f"Pipeline {pipeline} has incompatible scientific semantics in one run")
    by_dataset={}
    global_fields={}
    for row in records:
        if row.get("pipeline") not in {config.pipeline_a,config.pipeline_b}:
            continue
        identity=row.get("dataset_fingerprint")
        if not identity:
            raise AnalysisInputError("Dataset identity is unverified")
        by_dataset.setdefault(row.get("dataset"),set()).add(identity)
        for field in ("source_code_fingerprint","environment_fingerprint","metric_semantics_version"):
            global_fields.setdefault(field,set()).add(row.get(field))
    if any(len(values) != 1 for values in by_dataset.values()) or any(len(values) != 1 for values in global_fields.values()):
        raise AnalysisInputError("Paired tasks have incompatible data/code/environment/metric semantics")


def _holm(p_values: list[float], alpha: float) -> tuple[list[float], list[bool]]:
    if not p_values:
        return [], []
    order = np.argsort(np.asarray(p_values, dtype=float), kind="stable")
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(p_values[index]) * (m - rank)))
        adjusted[index] = running
    return adjusted.tolist(), [value <= alpha for value in adjusted]


def _bootstrap_dataset_mean(values: np.ndarray, config: AnalysisConfig, *, stratum_label: str, fingerprint: str) -> dict[str, Any]:
    n = len(values)
    if n < config.min_datasets:
        return {"status": "too_few_datasets", "n_datasets": n, "ci_lower": None, "ci_upper": None, "bootstrap_resamples": 0}
    if config.bootstrap_resamples <= 0:
        return {"status": "bootstrap_disabled", "n_datasets": n, "ci_lower": None, "ci_upper": None, "bootstrap_resamples": 0}
    random_config=config.to_dict()
    random_config.pop("run_id",None)
    seed = stable_seed("analysis_dataset_bootstrap", {"config": random_config, "stratum": stratum_label, "input_fingerprint": fingerprint})
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(config.bootstrap_resamples, n))
    means = values[indices].mean(axis=1)
    tail = (1.0 - config.confidence_level) / 2.0
    return {
        "status": "complete",
        "n_datasets": n,
        "ci_lower": float(np.quantile(means, tail)),
        "ci_upper": float(np.quantile(means, 1.0 - tail)),
        "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_seed": seed,
        "bootstrap_unit": "whole_dataset_within_dataset_pairs_intact",
    }


def _sign_flip_test(values: np.ndarray, config: AnalysisConfig, *, stratum_label: str, fingerprint: str) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    nonzero = finite[np.abs(finite) > 0.0]
    if len(finite) < config.min_datasets:
        return {"status": "too_few_datasets", "p_value": None, "n_datasets": len(finite), "zero_differences": int(len(finite) - len(nonzero))}
    observed = float(np.mean(finite))
    if len(nonzero) == 0:
        return {"status": "all_zero_effect", "p_value": 1.0, "n_datasets": len(finite), "zero_differences": int(len(finite)), "observed_mean": observed, "method": "sign_flip_exact"}
    values_for_test = nonzero
    n = len(values_for_test)
    denominator = len(finite)
    if n <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=float)
        # Keep zero differences in the dataset-level mean denominator.  They
        # are omitted from the sign enumeration because their sign is
        # immaterial, but they remain part of the independent dataset sample.
        null_means = signs @ values_for_test / denominator
        p_value = float(np.mean(np.abs(null_means) >= abs(observed)))
        return {"status": "complete", "p_value": p_value, "n_datasets": len(finite), "zero_differences": int(len(finite) - n), "observed_mean": observed, "method": "sign_flip_exact", "resamples": int(len(null_means)), "exchangeability_assumption": "symmetric_dataset_level_differences"}
    if config.permutation_resamples <= 0:
        return {"status": "permutation_disabled", "p_value": None, "n_datasets": len(finite), "zero_differences": int(len(finite) - n), "observed_mean": observed, "method": "sign_flip_monte_carlo", "resamples": 0}
    random_config=config.to_dict()
    random_config.pop("run_id",None)
    seed = stable_seed("analysis_dataset_sign_flip", {"config": random_config, "stratum": stratum_label, "input_fingerprint": fingerprint})
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(config.permutation_resamples, n))
    null_means = signs @ values_for_test / denominator
    extreme = int(np.count_nonzero(np.abs(null_means) >= abs(observed)))
    p_value = float((extreme + 1) / (len(null_means) + 1))
    return {"status": "complete", "p_value": p_value, "n_datasets": len(finite), "zero_differences": int(len(finite) - n), "observed_mean": observed, "method": "sign_flip_monte_carlo", "resamples": config.permutation_resamples, "seed": seed, "exchangeability_assumption": "symmetric_dataset_level_differences", "finite_resampling_convention": "(extreme+1)/(resamples+1)"}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")


def analyze_ledger(
    ledger_path: str | Path,
    config: AnalysisConfig | None = None,
    *,
    output_dir: str | Path | None = None,
) -> AnalysisBundle:
    """Analyze a result ledger under the dataset-level comparison contract."""
    config = config or AnalysisConfig()
    path = Path(ledger_path)
    fingerprint = _fingerprint(path)
    records, parse_exclusions = _read_ledger(path)
    run_ids={row.get("run_id") for row in records if row.get("pipeline") in {config.pipeline_a,config.pipeline_b}}
    if config.run_id is None and run_ids:
        if len(run_ids) != 1 or None in run_ids:
            raise AnalysisInputError("Select exactly one declared run_id; mixed/unattributed runs are unsupported")
        config=replace(config,run_id=str(next(iter(run_ids))))
    if config.run_id is not None:
        records=[row for row in records if row.get("run_id") == config.run_id]
    if not records:
        parse_exclusions.append({"reason":"empty_selected_run_ledger","detail":"Intended task universe is unavailable without a manifest; no zero outcome is imputed"})
    _validate_protocol(records, config)
    _check_duplicate_successes(records, config)
    _validate_pipeline_semantics(records, config)
    scientific_rows=[{**{field:row.get(field) for field in config.task_fields},
        **{field:row.get(field) for field in ("pipeline","pipeline_identity","dataset_fingerprint","metric_semantics_version","status")},
        "metric":_finite_metric(row,config.metric)[1]} for row in records if row.get("pipeline") in {config.pipeline_a,config.pipeline_b}]
    resampling_fingerprint=hashlib.sha256(_canonical_json(sorted(scientific_rows,key=_canonical_json)).encode()).hexdigest()

    exclusions: list[dict[str, Any]] = list(parse_exclusions)
    selected_rows: list[dict[str, Any]] = []
    for row in records:
        if row.get("pipeline") not in {config.pipeline_a, config.pipeline_b}:
            continue
        row = dict(row)
        row["task_id"] = _task_id(row, config)
        row["task_key"] = list(_base_key(row, config))
        row["stratum_key"] = list(_stratum_key(row, config))
        if row.get("status", "success") != "success":
            exclusions.append({"task_id": row["task_id"], "pipeline": row.get("pipeline"), "reason": f"status_{row.get('status', 'unknown')}", "source_line": row.get("source_line")})
        else:
            valid, value, reason = _finite_metric(row, config.metric)
            if valid:
                row["metric_value"] = value
                selected_rows.append(row)
            else:
                exclusions.append({"task_id": row["task_id"], "pipeline": row.get("pipeline"), "reason": reason, "source_line": row.get("source_line")})

    # Group all selected rows by the scientific task, retaining failed rows in
    # the intended count through the full ledger records below.
    all_by_key: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
    for row in records:
        if row.get("pipeline") not in {config.pipeline_a, config.pipeline_b}:
            continue
        key = _base_key(row, config)
        all_by_key.setdefault(key, {config.pipeline_a: [], config.pipeline_b: []})[row.get("pipeline")].append(dict(row))

    valid_by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in selected_rows:
        valid_by_key.setdefault(tuple(row["task_key"]), {})[row["pipeline"]] = row

    task_pair_rows: list[dict[str, Any]] = []
    for key, sides in sorted(all_by_key.items(), key=lambda item: repr(item[0])):
        left = valid_by_key.get(key, {}).get(config.pipeline_a)
        right = valid_by_key.get(key, {}).get(config.pipeline_b)
        if left is not None and right is not None:
            pair_status = "paired"
            diff = float(right["metric_value"] - left["metric_value"])
            stratum = _stratum_key(left, config)
        else:
            pair_status = "missing_partner" if left is None or right is None else "invalid_metric"
            diff = None
            source = left or right or next(iter(sides[config.pipeline_a]+sides[config.pipeline_b]),None)
            stratum = _stratum_key(source, config)
            missing_side = config.pipeline_a if left is None else config.pipeline_b
            exclusions.append({"task_id": _task_id(source, config), "pipeline": missing_side, "reason": "missing_valid_partner", "stratum": _stratum_label(stratum, config.stratum_fields)})
        source = left or right or next(iter(sides[config.pipeline_a]+sides[config.pipeline_b]),None)
        task_pair_rows.append({
            "task_id": _task_id(source, config),
            "dataset": source.get("dataset"),
            "stratum": _stratum_label(stratum, config.stratum_fields),
            "stratum_values": list(stratum),
            "pipeline_a": config.pipeline_a,
            "pipeline_b": config.pipeline_b,
            "metric_a": None if left is None else left["metric_value"],
            "metric_b": None if right is None else right["metric_value"],
            "difference_b_minus_a": diff,
            "pair_status": pair_status,
            "source_line_a": None if left is None else left.get("source_line"),
            "source_line_b": None if right is None else right.get("source_line"),
        })

    pair_frame = pd.DataFrame(task_pair_rows)
    dataset_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    strata = pair_frame.groupby("stratum", dropna=False) if not pair_frame.empty else []
    for stratum, stratum_frame in strata:
        paired = stratum_frame[stratum_frame["pair_status"] == "paired"].copy()
        for dataset, dataset_frame in paired.groupby("dataset", dropna=False):
            values = dataset_frame["difference_b_minus_a"].to_numpy(dtype=float)
            dataset_rows.append({
                "stratum": stratum,
                "dataset": dataset,
                "pipeline_a": config.pipeline_a,
                "pipeline_b": config.pipeline_b,
                "metric": config.metric,
                "dataset_difference": float(np.mean(values)),
                "n_intended_tasks": int(len(stratum_frame[stratum_frame["dataset"] == dataset])),
                "n_completed_a": int(dataset_frame["metric_a"].notna().sum()),
                "n_completed_b": int(dataset_frame["metric_b"].notna().sum()),
                "n_paired_tasks": int(len(dataset_frame)),
                "task_weighting": config.task_weighting,
                "task_ids": json.dumps(dataset_frame["task_id"].tolist()),
            })
        values = paired.groupby("dataset")["difference_b_minus_a"].mean().to_numpy(dtype=float)
        bootstrap = _bootstrap_dataset_mean(values, config, stratum_label=stratum, fingerprint=resampling_fingerprint)
        test = _sign_flip_test(values, config, stratum_label=stratum, fingerprint=resampling_fingerprint)
        summary_rows.append({
            "stratum": stratum,
            "metric": config.metric,
            "pipeline_a": config.pipeline_a,
            "pipeline_b": config.pipeline_b,
            "estimate_b_minus_a": float(np.mean(values)) if len(values) else None,
            "n_intended_tasks": int(len(stratum_frame)),
            "n_valid_tasks_a": int(stratum_frame["metric_a"].notna().sum()),
            "n_valid_tasks_b": int(stratum_frame["metric_b"].notna().sum()),
            "n_completed_tasks": int((stratum_frame["pair_status"] == "paired").sum()),
            "n_paired_tasks": int(len(paired)),
            "n_unpaired_tasks": int((stratum_frame["pair_status"] != "paired").sum()),
            "n_datasets": int(len(values)),
            "contributing_datasets": json.dumps(sorted(str(value) for value in paired["dataset"].dropna().unique())),
            "confidence_level": config.confidence_level,
            **bootstrap,
            **test,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        })

    summaries = pd.DataFrame(summary_rows)
    if not summaries.empty:
        valid_p = summaries["p_value"].notna()
        p_values = summaries.loc[valid_p, "p_value"].astype(float).tolist()
        adjusted, rejected = _holm(p_values, config.alpha)
        summaries["family_id"] = config.family_id
        summaries["family_size"] = int(len(p_values))
        summaries["p_value_adjusted_holm"] = np.nan
        summaries["rejected_holm"] = False
        if valid_p.any():
            summaries.loc[valid_p, "p_value_adjusted_holm"] = adjusted
            summaries.loc[valid_p, "rejected_holm"] = rejected
    multiplicity = summaries[[column for column in ("stratum", "family_id", "family_size", "p_value", "p_value_adjusted_holm", "rejected_holm", "alpha") if column in summaries.columns]].copy() if not summaries.empty else pd.DataFrame()
    if not multiplicity.empty:
        multiplicity["family_id"] = config.family_id
        multiplicity["alpha"] = config.alpha

    bundle = AnalysisBundle(
        config={**config.to_dict(), "schema_version": ANALYSIS_SCHEMA_VERSION, "input_ledger": str(path), "input_fingerprint_sha256": fingerprint,
            "resampling_fingerprint_sha256":resampling_fingerprint,
            "resampling_input_scope":"canonical scientific records; excludes runtime, export order, run labels and timestamps",
            "analysis_status":"supported_observed_pairs" if not pair_frame.empty else "empty_or_unsupported_observed_universe"},
        input_fingerprint=fingerprint,
        task_pairs=pair_frame,
        dataset_contrasts=pd.DataFrame(dataset_rows),
        summaries=summaries,
        exclusions=pd.DataFrame(exclusions),
        multiplicity=multiplicity,
    )
    if output_dir is not None:
        write_analysis_bundle(bundle, output_dir)
    return bundle


def write_analysis_bundle(bundle: AnalysisBundle, output_dir: str | Path) -> Path:
    """Persist traceable analysis tables and configuration."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "analysis_config.json").write_text(_canonical_json(bundle.config), encoding="utf-8")
    bundle.task_pairs.to_json(directory / "task_pairs.jsonl", orient="records", lines=True)
    bundle.dataset_contrasts.to_csv(directory / "dataset_contrasts.csv", index=False)
    bundle.summaries.to_csv(directory / "stratum_summaries.csv", index=False)
    bundle.exclusions.to_json(directory / "exclusions.jsonl", orient="records", lines=True)
    bundle.multiplicity.to_csv(directory / "multiplicity.csv", index=False)
    return directory


def run_dataset_level_analysis(
    ledger_path: str | Path,
    *,
    output_dir: str | Path = "reports/analysis/dataset_level",
    config: AnalysisConfig | None = None,
) -> AnalysisBundle:
    """Public reader entry point used by scripts and verification fixtures."""
    return analyze_ledger(ledger_path, config=config, output_dir=output_dir)
