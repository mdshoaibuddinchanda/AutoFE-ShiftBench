"""Dataset-level clean Jacobian/selection association with corrected AUC gain."""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.provenance import atomic_write_json, file_sha256, stable_digest


ROOT = Path(__file__).resolve().parents[1]
PERMUTATIONS = 10_000
BOOTSTRAPS = 10_000
ANALYSIS_SEED = 20261001


def _paired_clean_auc(results_path: Path, expected_pairs_per_dataset: int) -> tuple[dict[str, dict], dict[tuple[str, int, int], tuple[str, str]]]:
    """Stream only terminal clean Raw/Baseline successes; never load the full ledger."""
    ledger_path = results_path.parent / "manifest_outcomes.sqlite"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Primary corrected run lacks compact outcome ledger: {ledger_path}")
    connection = sqlite3.connect(ledger_path)
    scores: dict[tuple[str, int, int, str], dict[str, float]] = defaultdict(dict)
    baseline_hashes: dict[tuple[str, int, int], tuple[str, str]] = {}
    try:
        with results_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (row.get("status") != "success" or row.get("condition") != "clean"
                        or row.get("pipeline") not in {"Raw", "AutoFE_Baseline"}):
                    continue
                try:
                    auc = float(row.get("roc_auc"))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(auc):
                    continue
                terminal = connection.execute(
                    "SELECT status FROM outcomes WHERE task_key=?", (row["task_key"],)
                ).fetchone()
                if not terminal or terminal[0] != "success":
                    continue
                key = (str(row["dataset"]), int(row["seed"]), int(row["fold"]), str(row["model"]))
                scores[key][row["pipeline"]] = auc
                if row["pipeline"] == "AutoFE_Baseline":
                    fold_key = key[:3]
                    matrices = (row.get("train_matrix_sha256"), row.get("test_matrix_sha256"))
                    if fold_key in baseline_hashes and baseline_hashes[fold_key] != matrices:
                        raise ValueError(f"Baseline feature matrices differ across models: {fold_key}")
                    baseline_hashes[fold_key] = matrices
    finally:
        connection.close()
    differences: dict[str, list[float]] = defaultdict(list)
    for (dataset, _seed, _fold, _model), pair in scores.items():
        if "Raw" in pair and "AutoFE_Baseline" in pair:
            differences[dataset].append(pair["AutoFE_Baseline"] - pair["Raw"])
    dataset_scores = {
        dataset: {
            "n_paired_clean_cells": len(values),
            "expected_paired_clean_cells": expected_pairs_per_dataset,
            "complete": len(values) == expected_pairs_per_dataset,
            "mean_auc_delta": float(np.mean(values)) if values else None,
        }
        for dataset, values in differences.items()
    }
    return dataset_scores, baseline_hashes


def _validate_mechanism_manifest(manifest: dict, scope: dict, split_policy: str) -> None:
    """Require the exact frozen source and clean mechanism sampling grid."""
    configuration = manifest.get("configuration", {})
    expected_script = scope.get("analysis_sha256", {}).get("provenance/run_mechanism_history.py")
    expected_datasets = [item["name"] for item in scope["datasets"]]
    if (manifest.get("status") != "complete" or not expected_script
            or configuration.get("code_fingerprint") != scope["code_fingerprint"]
            or configuration.get("mechanism_script_sha256") != expected_script
            or configuration.get("split_policy") != split_policy
            or configuration.get("condition") != "clean"
            or configuration.get("pipeline") != "AutoFE_Baseline"
            or configuration.get("datasets") != expected_datasets
            or list(configuration.get("seeds", [])) != list(scope["seeds"])
            or list(configuration.get("folds", [])) != list(scope["folds"])
            or configuration.get("numerical_thread_environment") != scope["required_numerical_thread_environment"]
            or manifest.get("configuration_fingerprint") != stable_digest(configuration)):
        raise ValueError("Mechanism manifest differs from the selected frozen source, policy or grid")


def _mechanism_exposures(run_dir: Path, expected_folds_per_dataset: int, *,
                        scope: dict, split_policy: str) -> tuple[dict[str, dict], dict[tuple[str, int, int], tuple[str, str]]]:
    """Aggregate selected-candidate fold medians, then average within dataset."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    _validate_mechanism_manifest(manifest, scope, split_policy)
    dataset_identity = {item["name"]: item for item in scope["datasets"]}
    expected_keys = {(name, int(seed), int(fold)) for name in dataset_identity
                     for seed in scope["seeds"] for fold in scope["folds"]}
    ineligible = set(scope.get("group_auc_ineligible_datasets", [])) if split_policy == "group_aware" else set()
    seen_keys = set()
    fold_values: dict[str, list[float]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    matrix_hashes: dict[tuple[str, int, int], tuple[str, str]] = {}
    for path in (run_dir / "tasks").rglob("status.json"):
        task = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(task["dataset"])
        key = (dataset, int(task["seed"]), int(task["fold"]))
        identity = dataset_identity.get(dataset, {})
        if (key not in expected_keys or key in seen_keys
                or task.get("configuration_fingerprint") != manifest["configuration_fingerprint"]
                or task.get("split_policy") != split_policy
                or task.get("condition") != "clean" or task.get("pipeline") != "AutoFE_Baseline"
                or task.get("csv_sha256") != identity.get("csv_sha256")
                or task.get("sidecar_sha256") != identity.get("sidecar_sha256")
                or task.get("status") != ("skipped" if dataset in ineligible else "complete")):
            raise ValueError(f"Mechanism task differs from frozen identity or is duplicated: {path}")
        seen_keys.add(key)
        if task["status"] == "skipped":
            skipped[dataset] += 1
            continue
        matrix_hashes[key] = (
            task["train_matrix_sha256"], task["test_matrix_sha256"],
        )
        for path_field, hash_field in (("history_path", "history_sha256"),
                                       ("jacobian_path", "jacobian_sha256")):
            artifact_path = (path.parent / task[path_field]).resolve()
            if (not artifact_path.is_relative_to(run_dir.resolve()) or not artifact_path.is_file()
                    or file_sha256(artifact_path) != task[hash_field]):
                raise ValueError(f"Mechanism artifact missing or changed: {artifact_path}")
        jacobian_path = path.parent / task["jacobian_path"]
        selected_norms = []
        with gzip.open(jacobian_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["selection_decision"] != "selected":
                    continue
                median = (row.get("jacobian") or {}).get("median")
                if median is not None and np.isfinite(float(median)):
                    selected_norms.append(float(median))
        if selected_norms:
            fold_values[dataset].append(float(np.median(selected_norms)))
    if (seen_keys != expected_keys or manifest.get("expected_tasks") != len(expected_keys)
            or manifest.get("counts") != {"complete": len(expected_keys) - sum(skipped.values()),
                                          "skipped": sum(skipped.values())}):
        raise ValueError("Mechanism task coverage/counts do not match the complete frozen grid")
    exposures = {
        dataset: {
            "n_finite_folds": len(fold_values.get(dataset, [])),
            "expected_folds": expected_folds_per_dataset,
            "n_skipped_folds": skipped.get(dataset, 0),
            "complete": len(fold_values.get(dataset, [])) == expected_folds_per_dataset,
            "mean_fold_median_selected_scaled_jacobian_norm": (
                float(np.mean(fold_values[dataset])) if fold_values.get(dataset) else None
            ),
        }
        for dataset in manifest["configuration"]["datasets"]
    }
    return exposures, matrix_hashes


def _association(x: list[float], y: list[float], *, seed: int = ANALYSIS_SEED) -> dict:
    """Two-sided permutation p and dataset bootstrap interval for Spearman rho."""
    if len(x) != len(y) or len(x) < 3:
        return {"rho": None, "permutation_p": None, "ci_low": None, "ci_high": None,
                "status": "insufficient_datasets"}
    x_values, y_values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    observed = float(spearmanr(x_values, y_values).statistic)
    if not np.isfinite(observed):
        return {"rho": None, "permutation_p": None, "ci_low": None, "ci_high": None,
                "status": "undefined_constant_input"}
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(PERMUTATIONS):
        shuffled = float(spearmanr(x_values, rng.permutation(y_values)).statistic)
        if np.isfinite(shuffled) and abs(shuffled) >= abs(observed):
            extreme += 1
    boot = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, len(x_values), size=len(x_values))
        value = float(spearmanr(x_values[indices], y_values[indices]).statistic)
        if np.isfinite(value):
            boot.append(value)
    return {
        "rho": observed,
        "permutation_p": (extreme + 1) / (PERMUTATIONS + 1),
        "ci_low": float(np.quantile(boot, 0.025)) if boot else None,
        "ci_high": float(np.quantile(boot, 0.975)) if boot else None,
        "status": "exploratory_association",
    }


def analyze(row_results: Path, group_results: Path, row_mechanism: Path,
            group_mechanism: Path, output: Path, scope_path: Path | None = None) -> dict:
    scope = json.loads((scope_path or ROOT / 'provenance' / 'reviewer1_launch_scope_v3.json').read_text(encoding="utf-8"))
    expected_pairs = len(scope["seeds"]) * len(scope["folds"]) * len(scope["models"])
    expected_folds = len(scope["seeds"]) * len(scope["folds"])
    records = []
    associations = []
    for policy, results, mechanism in (
        ("row_level", row_results, row_mechanism),
        ("group_aware", group_results, group_mechanism),
    ):
        performance_manifest = json.loads((results.parent / "manifest.json").read_text(encoding="utf-8"))
        if (performance_manifest.get("status") != "complete"
                or performance_manifest.get("code_fingerprint") != scope["code_fingerprint"]
                or performance_manifest.get("split_policy") != policy):
            raise ValueError(f"Primary run is incomplete or has different identity: {results}")
        performance, baseline_hashes = _paired_clean_auc(results, expected_pairs)
        exposures, mechanism_hashes = _mechanism_exposures(
            mechanism, expected_folds, scope=scope, split_policy=policy)
        if set(mechanism_hashes) != set(baseline_hashes):
            raise ValueError("Mechanism and primary baseline fold coverage differ")
        for key, matrices in mechanism_hashes.items():
            if baseline_hashes[key] != matrices:
                raise ValueError(f"Mechanism feature matrices differ from primary run: {key}")
        x, y = [], []
        for item in scope["datasets"]:
            name = item["name"]
            auc = performance.get(name, {"complete": False, "mean_auc_delta": None,
                                         "n_paired_clean_cells": 0})
            exposure = exposures.get(name, {"complete": False,
                                            "mean_fold_median_selected_scaled_jacobian_norm": None,
                                            "n_finite_folds": 0, "n_skipped_folds": 0})
            complete = bool(auc["complete"] and exposure["complete"])
            if complete:
                x.append(exposure["mean_fold_median_selected_scaled_jacobian_norm"])
                y.append(auc["mean_auc_delta"])
            records.append({"split_policy": policy, "dataset": name,
                            "mechanism": exposure, "performance": auc,
                            "included_in_association": complete})
        associations.append({
            "split_policy": policy, "n_complete_datasets": len(x),
            "n_configured_datasets": len(scope["datasets"]),
            "exposure": "dataset mean of fold median selected-candidate scaled L2 Jacobian norms",
            "outcome": "dataset mean paired clean AutoFE_Baseline minus Raw ROC-AUC across all ten models and 25 seed-folds",
            **_association(x, y),
        })
    indexed = sorted(
        ((index, row["permutation_p"]) for index, row in enumerate(associations)
         if row["permutation_p"] is not None),
        key=lambda item: item[1],
    )
    running_adjusted = 0.0
    for rank, (index, value) in enumerate(indexed):
        running_adjusted = max(running_adjusted, min(1.0, (len(indexed) - rank) * value))
        associations[index]["holm_adjusted_permutation_p"] = running_adjusted
    for row in associations:
        row.setdefault("holm_adjusted_permutation_p", None)
    note = {
        "artifact_type": "reviewer1_exploratory_mechanism_association",
        "analysis_seed": ANALYSIS_SEED, "permutations": PERMUTATIONS,
        "bootstrap_replicates": BOOTSTRAPS,
        "inferential_unit": "dataset", "causal_claim": False,
        "multiplicity_family": "two split-policy mechanism associations, Holm adjusted",
        "dataset_records": records, "associations": associations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, note)
    return note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-results", required=True, type=Path)
    parser.add_argument("--group-results", required=True, type=Path)
    parser.add_argument("--row-mechanism", required=True, type=Path)
    parser.add_argument("--group-mechanism", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument('--scope', type=Path, default=ROOT / 'provenance' / 'reviewer1_launch_scope_v3.json')
    args = parser.parse_args()
    result = analyze(args.row_results, args.group_results, args.row_mechanism,
                     args.group_mechanism, args.output, args.scope)
    print(json.dumps({"output": str(args.output), "associations": result["associations"]}, indent=2))


if __name__ == "__main__":
    main()
