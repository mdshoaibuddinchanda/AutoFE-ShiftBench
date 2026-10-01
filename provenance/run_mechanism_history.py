"""Training-fold candidate histories and arithmetic Jacobians for the clean baseline.

This is a separate mechanism study. It does not fit a classifier or write
corrected ROC-AUC results. Use a unique run ID for each split policy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_csv_dataset, load_dataset_names
from src.feature_engineering import DFSConfig, expand_features_with_dfs
from src.group_splits import (
    assert_group_fold_integrity, canonical_feature_group_ids, get_group_stratified_splits,
)
from src.mechanism_audit import (
    ArithmeticCandidate, CandidateHistoryWriter, analytic_jacobian,
    scaled_jacobian_norm, summarize_jacobian_norm, training_input_scale,
)
from src.pipeline_runner import (
    PIPELINE_CONFIGS, _dataset_identity, _fence_coordinator, _stable_perturbation_seed,
)
from src.preprocessing import _build_preprocessor, _to_dense_array
from src.provenance import (
    atomic_write_json, code_fingerprint, file_sha256, frame_sha256,
    index_sha256, stable_digest, stable_seed,
)
from src.splitters import get_stratified_splits


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456, 789, 2025)
FOLDS = (1, 2, 3, 4, 5)
PIPELINE = "AutoFE_Baseline"
MAX_JACOBIAN_ROWS = 256
THREAD_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _histories_for_fold(
    *, dataset: str, split_policy: str, seed: int, fold: int,
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
    dataset_identity: dict, task_dir: Path, sample_cap: int = MAX_JACOBIAN_ROWS,
) -> dict:
    """Build one fold with the runner's clean training-only preprocessing."""
    condition = "clean"
    perturb_seed = _stable_perturbation_seed(
        dataset_identity, x_train, y_train, seed, fold, condition,
    )
    encoder = LabelEncoder().fit(y_train.astype(str))
    y_encoded = encoder.transform(y_train.astype(str))
    preprocessor = _build_preprocessor(x_train, encoding="onehot", scale_numeric=True)
    train_values = _to_dense_array(preprocessor.fit_transform(x_train))
    test_values = _to_dense_array(preprocessor.transform(x_test))
    columns = preprocessor.get_feature_names_out().tolist()
    prepared_train = pd.DataFrame(train_values, columns=columns).reset_index(drop=True)
    prepared_test = pd.DataFrame(test_values, columns=columns).reset_index(drop=True)
    y_series = pd.Series(y_encoded, index=prepared_train.index)
    cfg = DFSConfig(**asdict(PIPELINE_CONFIGS[PIPELINE]))
    cfg.random_seed = perturb_seed
    task_dir.mkdir(parents=True, exist_ok=True)
    history_path = task_dir / "candidate_history.jsonl.gz"
    temporary_history = task_dir / f"candidate_history.{os.getpid()}.partial.gz"
    if temporary_history.exists():
        temporary_history.unlink()
    with CandidateHistoryWriter(temporary_history) as writer:
        train_fe, test_fe, fe_meta = expand_features_with_dfs(
            prepared_train, prepared_test, y_series, config=cfg,
            audit_context={
                "dataset": dataset, "split_policy": split_policy,
                "seed": seed, "fold": fold, "condition": condition,
                "candidate_seed": perturb_seed,
            },
            candidate_history_writer=writer,
        )
        records_written = writer.records_written
    _publish_deterministic_gzip(temporary_history, history_path)

    sample_seed = stable_seed(
        {"dataset": dataset, "split_policy": split_policy}, seed, fold, "jacobian_rows",
    )
    rng = np.random.default_rng(sample_seed)
    sample_indices = np.sort(rng.choice(
        len(prepared_train), size=min(sample_cap, len(prepared_train)), replace=False,
    ))
    jacobian_path = task_dir / "candidate_jacobians.jsonl.gz"
    temporary_jacobians = task_dir / f"candidate_jacobians.{os.getpid()}.partial.gz"
    if temporary_jacobians.exists():
        temporary_jacobians.unlink()
    candidate_count = 0
    undefined_count = 0
    with gzip.open(history_path, "rt", encoding="utf-8") as source, gzip.open(
        temporary_jacobians, "wt", encoding="utf-8", newline=""
    ) as destination:
        for line in source:
            record = json.loads(line)
            parents = tuple(record["parent_features"])
            reason = None
            summary = None
            if len(parents) != 2 or any(parent not in prepared_train for parent in parents):
                reason = "parent_not_in_preprocessed_training_matrix"
            else:
                parent_matrix = prepared_train.loc[:, list(parents)].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=float)
                scale = training_input_scale(parent_matrix)
                candidate = ArithmeticCandidate(
                    candidate_id=record["candidate_id"],
                    operator=record["operator"], parent_features=parents,
                )
                derivative = analytic_jacobian(candidate, parent_matrix[sample_indices])
                norms = scaled_jacobian_norm(derivative, input_scale=scale)
                summary = asdict(summarize_jacobian_norm(
                    norms, undefined_reasons=derivative.undefined_reasons,
                ))
                if summary["n_valid"] == 0:
                    reason = "no_finite_scaled_jacobian_norm"
            if reason:
                undefined_count += 1
            destination.write(json.dumps({
                "candidate_id": record["candidate_id"],
                "operator": record["operator"],
                "selection_decision": record["selection_decision"],
                "jacobian": summary,
                "undefined_reason": reason,
            }, sort_keys=True, allow_nan=False) + "\n")
            candidate_count += 1
    _publish_deterministic_gzip(temporary_jacobians, jacobian_path)
    if candidate_count != records_written:
        raise RuntimeError("Candidate history and Jacobian row counts differ")
    return {
        "status": "complete", "candidate_count": candidate_count,
        "candidates_without_finite_jacobian": undefined_count,
        "history_path": history_path.name, "history_sha256": file_sha256(history_path),
        "jacobian_path": jacobian_path.name, "jacobian_sha256": file_sha256(jacobian_path),
        "sample_seed": sample_seed, "sample_rows": int(len(sample_indices)),
        "sample_indices_sha256": index_sha256(sample_indices),
        "train_matrix_sha256": frame_sha256(train_fe),
        "test_matrix_sha256": frame_sha256(test_fe),
        "operator_candidate_counts": fe_meta["operator_candidate_counts"],
        "candidate_history_metadata": fe_meta["candidate_history"],
        "stable_perturbation_seed": perturb_seed,
    }


def _publish_deterministic_gzip(temporary: Path, destination: Path) -> None:
    """Remove gzip timestamps/filenames, then atomically publish exact bytes."""
    uncompressed = gzip.decompress(temporary.read_bytes())
    normalized = temporary.with_suffix(".normalized")
    normalized.write_bytes(gzip.compress(uncompressed, mtime=0))
    os.replace(normalized, destination)
    temporary.unlink()


@_fence_coordinator
def run_mechanism(
    *, run_id: str, split_policy: str, max_datasets: int | None = None,
    max_seeds: int | None = None, max_folds: int | None = None,
    output_root: Path = ROOT / "corrected_runs",
    scope_path: Path = ROOT / 'provenance' / 'reviewer1_launch_scope_v3.json',
) -> dict:
    if split_policy not in {"row_level", "group_aware"}:
        raise ValueError("split_policy must be row_level or group_aware")
    if Path(sys.executable).resolve() != Path(r"D:\Conda\p12\python.exe").resolve():
        raise RuntimeError("Mechanism runs require the existing p12 interpreter")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise ValueError("Mechanism runs require four numerical thread variables set to 1")
    if any(value is not None and value < 1 for value in (max_datasets, max_seeds, max_folds)):
        raise ValueError("All bounds must be positive")
    names = load_dataset_names(ROOT / "config" / "dataset_list.yaml")[:max_datasets]
    seeds, folds = SEEDS[:max_seeds], FOLDS[:max_folds]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_fingerprint = code_fingerprint(ROOT)
    configuration = {
        "datasets": names, "seeds": seeds, "folds": folds,
        "split_policy": split_policy, "condition": "clean", "pipeline": PIPELINE,
        "max_jacobian_rows": MAX_JACOBIAN_ROWS,
        "pipeline_config": asdict(PIPELINE_CONFIGS[PIPELINE]),
        "code_fingerprint": source_fingerprint,
        "mechanism_script_sha256": file_sha256(Path(__file__)),
        "numerical_thread_environment": {name: os.environ[name] for name in THREAD_ENV},
    }
    config_fingerprint = stable_digest(configuration)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("configuration_fingerprint") != config_fingerprint:
            raise ValueError("Existing mechanism run ID has a different configuration")
    group_audit = json.loads((ROOT / "provenance" / "group_seed_grid_audit_v1.json").read_text(encoding="utf-8"))
    group_ineligible = {
        row["dataset"] for row in group_audit["records"]
        if any(row["seeds"][str(seed)]["auc_status"] != "supported" for seed in SEEDS)
    }
    frozen_scope = json.loads(scope_path.read_text(encoding="utf-8"))
    frozen_datasets = {item["name"]: item for item in frozen_scope["datasets"]}
    if source_fingerprint != frozen_scope["code_fingerprint"]:
        raise ValueError("Mechanism run source fingerprint differs from the frozen benchmark")
    counts = {"complete": 0, "skipped": 0}
    for name in names:
        csv_path = ROOT / "data" / "raw" / f"{name}.csv"
        sidecar_path = csv_path.with_name(f"{name}_meta.json")
        csv_hash, sidecar_hash = file_sha256(csv_path), file_sha256(sidecar_path)
        if (csv_hash, sidecar_hash) != (
            frozen_datasets[name]["csv_sha256"], frozen_datasets[name]["sidecar_sha256"]
        ):
            raise ValueError(f"Mechanism input differs from frozen benchmark: {name}")
        identity, sidecar = _dataset_identity(name, csv_path, sidecar_path)
        frame = load_csv_dataset(csv_path)
        target = sidecar.get("target_column", "target")
        if target not in frame:
            raise ValueError(f"{name}: target column is missing")
        X, y = frame.drop(columns=[target]), frame[target]
        group_ids = canonical_feature_group_ids(X) if split_policy == "group_aware" else None
        for seed in seeds:
            if split_policy == "group_aware":
                splits = get_group_stratified_splits(
                    X, y, 5, seed, target_column=target,
                    require_class_support=False, require_auc=False,
                )
                assert_group_fold_integrity(splits, group_ids)
            else:
                splits = get_stratified_splits(X, y, 5, seed, target_column=target)
            for fold in folds:
                train_idx, test_idx = splits[fold - 1]
                task = {"dataset": name, "split_policy": split_policy,
                        "seed": seed, "fold": fold, "condition": "clean", "pipeline": PIPELINE}
                task_key = stable_digest(task)
                task_dir = run_dir / "tasks" / task_key[:2] / task_key
                task_dir.mkdir(parents=True, exist_ok=True)
                status_path = task_dir / "status.json"
                fixed = {
                    **task, "task_key": task_key,
                    "csv_sha256": csv_hash,
                    "sidecar_sha256": sidecar_hash,
                    "train_indices_sha256": index_sha256(train_idx),
                    "test_indices_sha256": index_sha256(test_idx),
                    "configuration_fingerprint": config_fingerprint,
                }
                if status_path.exists():
                    previous = json.loads(status_path.read_text(encoding="utf-8"))
                    if any(previous.get(key) != value for key, value in fixed.items()):
                        raise ValueError(f"Mechanism task identity changed: {task_key}")
                    if previous["status"] == "complete" and any(
                        not (task_dir / previous[path_key]).exists()
                        or file_sha256(task_dir / previous[path_key]) != previous[hash_key]
                        for path_key, hash_key in (
                            ("history_path", "history_sha256"),
                            ("jacobian_path", "jacobian_sha256"),
                        )
                    ):
                        raise ValueError(f"Mechanism artifact missing or changed: {task_key}")
                    if previous["status"] in counts:
                        counts[previous["status"]] += 1
                        continue
                if split_policy == "group_aware" and name in group_ineligible:
                    result = {"status": "skipped", "skip_reason": "dataset_ineligible_for_all_five_seed_group_auc"}
                else:
                    result = _histories_for_fold(
                        dataset=name, split_policy=split_policy, seed=seed, fold=fold,
                        x_train=X.iloc[train_idx].copy(), y_train=y.iloc[train_idx].copy(),
                        x_test=X.iloc[test_idx].copy(), dataset_identity=identity,
                        task_dir=task_dir,
                    )
                atomic_write_json(status_path, {**fixed, **result})
                counts[result["status"]] += 1
                print(f"{name} {split_policy} seed={seed} fold={fold}: {result['status']}", flush=True)
    manifest = {
        "artifact_type": "reviewer1_clean_arithmetic_mechanism_history",
        "run_id": run_id, "configuration": configuration,
        "configuration_fingerprint": config_fingerprint,
        "expected_tasks": len(names) * len(seeds) * len(folds),
        "counts": counts,
        "status": "complete" if sum(counts.values()) == len(names) * len(seeds) * len(folds) else "partial",
        "scientific_status": "candidate histories and Jacobian diagnostics only; performance associations PENDING CORRECTED RUN",
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-policy", choices=["row_level", "group_aware"], required=True)
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument('--scope', type=Path, default=ROOT / 'provenance' / 'reviewer1_launch_scope_v3.json')
    args = parser.parse_args()
    print(json.dumps(run_mechanism(
        run_id=args.run_id, split_policy=args.split_policy,
        max_datasets=args.max_datasets, max_seeds=args.max_seeds,
        max_folds=args.max_folds, scope_path=args.scope,
    ), indent=2))


if __name__ == "__main__":
    main()
