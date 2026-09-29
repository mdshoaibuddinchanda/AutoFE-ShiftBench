from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.checkpoint import get_task_state
from src.evaluation import compute_classification_metrics
from src.feature_engineering import DFSConfig, _limit_features
from src.generate_tables import generate_table_2_robustness
from src.pipeline_runner import (
    PRIMARY_CONDITIONS,
    _load_or_create_feature_cache,
    _stable_perturbation_seed,
    prepare_task,
    run_experiment,
)
from src.preprocessing import basic_preprocess
from src.provenance import (
    PROTOCOL_VERSION,
    cache_fingerprint,
    frame_sha256,
    stable_digest,
    stable_seed,
    vector_sha256,
)
from src.splitters import (
    assert_fold_integrity,
    get_covariate_splits,
    get_population_splits,
    get_stratified_splits,
)
from src.stats_analysis import run_wilcoxon_analysis
from src.shift_generator import apply_perturbation


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_frame(n: int = 90, seed: int = 71) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    target = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
    frame = pd.DataFrame({
        "numeric_signal": signal,
        "numeric_noise": rng.normal(size=n),
        "category": np.where(np.arange(n) % 3 == 0, "a", np.where(np.arange(n) % 3 == 1, "b", "c")),
        "target_copy": target,
        "numeric_target": target,
    })
    for index in range(8):
        frame[f"numeric_extra_{index}"] = rng.normal(size=n)
    return frame


def _write_synthetic_dataset(path: Path) -> None:
    _synthetic_frame().to_csv(path, index=False)
    path.with_name(f"{path.stem}_meta.json").write_text(json.dumps({
        "target_column": "numeric_target",
        "dataset_identity": {"provider": "synthetic", "id": "synthetic-v1"},
    }), encoding="utf-8")


def test_target_permutation_does_not_change_target_free_domain_partitions(monkeypatch):
    frame = _synthetic_frame()
    y = frame["numeric_target"]
    X = frame.drop(columns=["numeric_target", "target_copy"])
    observed_widths = []
    original_fit_transform = StandardScaler.fit_transform

    def recording_fit_transform(self, values, *args, **kwargs):
        observed_widths.append(np.asarray(values).shape[1])
        return original_fit_transform(self, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit_transform", recording_fit_transform)
    y_permuted = y.sample(frac=1, random_state=9).reset_index(drop=True)
    permuted_frame = frame.copy()
    permuted_frame["numeric_target"] = y_permuted.to_numpy()
    X_after_target_permutation = permuted_frame.drop(columns=["numeric_target", "target_copy"])
    pd.testing.assert_frame_equal(X, X_after_target_permutation)
    for splitter in (get_covariate_splits, get_population_splits):
        first = splitter(X, 5, 12, target_column="numeric_target")
        second = splitter(X_after_target_permutation, 5, 12, target_column="numeric_target")
        assert all(np.array_equal(a, c) and np.array_equal(b, d)
                   for (a, b), (c, d) in zip(first, second))
        # The only change between the conceptual cases is y; y is not a splitter input.
        assert not y.equals(y_permuted) or len(y) == 1
        with pytest.raises(AssertionError, match="must be removed"):
            splitter(frame.drop(columns=["target_copy"]), 5, 12, target_column="numeric_target")
    assert observed_widths and set(observed_widths) == {X.select_dtypes(include=["number", "bool"]).shape[1]}


def test_stratified_folds_are_disjoint_and_hold_each_row_out_once():
    frame = _synthetic_frame()
    X = frame.drop(columns=["numeric_target", "target_copy"])
    y = frame["numeric_target"]
    splits = get_stratified_splits(X, y, 5, 42, target_column="numeric_target")
    assert_fold_integrity(splits, len(frame), held_out_once=True)
    assert all(not np.intersect1d(train, test).size for train, test in splits)


def test_perturbation_seed_uses_training_data_but_not_held_out_rows():
    frame = _synthetic_frame()
    X = frame.drop(columns=["numeric_target", "target_copy"])
    y = frame["numeric_target"]
    train_idx, test_idx = get_stratified_splits(
        X, y, 3, 31, target_column="numeric_target",
    )[0]
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    identity = {"provider": "OpenML", "openml_id": "42", "openml_version": "3",
                "saved_csv_sha256": "full-file-including-held-out-labels"}
    original = _stable_perturbation_seed(identity, x_train, y_train, 9, 1, "label_noise_0.05")

    poisoned_test = frame.copy()
    poisoned_test.loc[test_idx, "numeric_signal"] += 900
    poisoned_test.loc[test_idx, "numeric_target"] = 1 - poisoned_test.loc[test_idx, "numeric_target"]
    poison_x = poisoned_test.drop(columns=["numeric_target", "target_copy"])
    assert _stable_perturbation_seed(
        identity, poison_x.iloc[train_idx], poisoned_test["numeric_target"].iloc[train_idx],
        9, 1, "label_noise_0.05",
    ) == original

    poisoned_train = x_train.copy()
    poisoned_train.iloc[0, 0] += 1
    assert _stable_perturbation_seed(
        identity, poisoned_train, y_train, 9, 1, "label_noise_0.05",
    ) != original


def test_legacy_full_frame_preprocessing_fails_closed():
    with pytest.raises(RuntimeError, match="complete frame"):
        basic_preprocess(_synthetic_frame())


@pytest.mark.parametrize("pipeline_name", ["Raw", "AutoFE_MI"])
def test_held_out_label_poison_cannot_change_training_or_predictions(pipeline_name):
    frame = _synthetic_frame()
    X = frame.drop(columns=["numeric_target", "target_copy"])
    split = get_stratified_splits(
        X, frame["numeric_target"], 3, 5, target_column="numeric_target",
    )[0]
    train_idx, test_idx = split
    poisoned = frame.copy()
    poisoned.loc[test_idx, "numeric_target"] = 1 - poisoned.loc[test_idx, "numeric_target"]
    clean_out = prepare_task(
        frame, target_column="numeric_target", train_indices=train_idx, test_indices=test_idx,
        family="clean", severity=0.0, perturbation_seed=1402, pipeline_name=pipeline_name,
    )
    poisoned_out = prepare_task(
        poisoned, target_column="numeric_target", train_indices=train_idx, test_indices=test_idx,
        family="clean", severity=0.0, perturbation_seed=1402, pipeline_name=pipeline_name,
    )
    xtr, xte, ytr = clean_out[0], clean_out[1], clean_out[2]
    pxtr, pxte, pytr = poisoned_out[0], poisoned_out[1], poisoned_out[2]
    pd.testing.assert_frame_equal(xtr, pxtr)
    pd.testing.assert_frame_equal(xte, pxte)
    np.testing.assert_array_equal(ytr, pytr)
    assert [item["name"] for item in clean_out[5]["feature_metadata"]] == [
        item["name"] for item in poisoned_out[5]["feature_metadata"]
    ]
    model_a = LogisticRegression(random_state=4).fit(
        np.nan_to_num(xtr.to_numpy(dtype=np.float32), nan=0.0, posinf=1e10, neginf=-1e10), ytr,
    )
    model_b = LogisticRegression(random_state=4).fit(
        np.nan_to_num(pxtr.to_numpy(dtype=np.float32), nan=0.0, posinf=1e10, neginf=-1e10), pytr,
    )
    xte_safe = np.nan_to_num(xte.to_numpy(dtype=np.float32), nan=0.0, posinf=1e10, neginf=-1e10)
    pxte_safe = np.nan_to_num(pxte.to_numpy(dtype=np.float32), nan=0.0, posinf=1e10, neginf=-1e10)
    np.testing.assert_array_equal(model_a.predict(xte_safe), model_b.predict(pxte_safe))


@pytest.mark.parametrize("pipeline_name", ["Raw", "AutoFE_Baseline"])
def test_held_out_feature_poison_does_not_change_fitted_training_matrix(pipeline_name):
    frame = _synthetic_frame()
    splits = get_stratified_splits(
        frame.drop(columns=["numeric_target", "target_copy"]), frame["numeric_target"], 3, 4,
        target_column="numeric_target",
    )
    train_idx, test_idx = splits[0]
    poisoned = frame.copy()
    poisoned.loc[test_idx, "numeric_signal"] = poisoned.loc[test_idx, "numeric_signal"] + 500.0
    poisoned.loc[test_idx, "category"] = "unseen-held-out-category"
    clean_out = prepare_task(
        frame, target_column="numeric_target", train_indices=train_idx, test_indices=test_idx,
        family="clean", severity=0.0, perturbation_seed=72, pipeline_name=pipeline_name,
    )
    poisoned_out = prepare_task(
        poisoned, target_column="numeric_target", train_indices=train_idx, test_indices=test_idx,
        family="clean", severity=0.0, perturbation_seed=72, pipeline_name=pipeline_name,
    )
    pd.testing.assert_frame_equal(clean_out[0], poisoned_out[0])
    np.testing.assert_array_equal(clean_out[2], poisoned_out[2])
    assert not clean_out[1].equals(poisoned_out[1])


def test_mi_requires_equal_length_and_identical_training_row_order():
    train = pd.DataFrame(np.arange(80, dtype=float).reshape(16, 5), index=np.arange(100, 116))
    labels = pd.Series([0, 1] * 8, index=train.index)
    config = DFSConfig(enable_dfs=False, selection_method="mi", max_features=2, max_base_features=None)
    with pytest.raises(ValueError, match="length"):
        _limit_features(train, train.copy(), labels.iloc[:-1], config)
    with pytest.raises(ValueError, match="index/order"):
        _limit_features(train, train.copy(), labels.sample(frac=1, random_state=3), config)
    baseline = _limit_features(train, train.copy(), labels, config)[2]
    # Test labels are not part of this selector's API or its fitting inputs.
    poisoned_test_labels = pd.Series([1, 0] * 8)
    assert not poisoned_test_labels.equals(pd.Series([0, 1] * 8))
    assert baseline == _limit_features(train, train.copy(), labels, config)[2]


def test_numeric_target_proxy_candidates_are_reported_not_removed():
    from src.data_loader import inspect_target_proxy_candidates

    frame = _synthetic_frame()
    audit = inspect_target_proxy_candidates(
        frame.drop(columns=["numeric_target"]), frame["numeric_target"],
        target_column="numeric_target",
    )
    assert "target_copy" in audit["exact_target_copies_for_manual_review"]
    assert "target_copy" in frame.columns


def test_primary_and_separate_condition_semantics_are_distinct():
    assert {family for family, _ in PRIMARY_CONDITIONS} == {
        "clean", "gaussian_noise", "missing_values", "label_noise",
    }
    X = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0) ** 2})
    y = pd.Series([0] * 20 + [1] * 10)
    x_ablation, y_ablation = apply_perturbation(
        X, y, "feature_availability_ablation", 0.5, random_state=4,
    )
    assert len(x_ablation.columns) < len(X.columns)
    assert y_ablation.equals(y)
    x_relabel, y_relabel = apply_perturbation(
        X, y, "majority_label_relabeling", 0.0, random_state=4,
    )
    assert x_relabel.equals(X)
    assert len(y_relabel) == len(y) and y_relabel.index.equals(y.index)
    assert not y_relabel.equals(y)
    with pytest.raises(ValueError, match="Unknown shift family"):
        apply_perturbation(X, y, "class_prior_shift", 0.0, random_state=4)


def test_probability_metrics_align_encoded_targets_with_string_class_names():
    binary = compute_classification_metrics(
        np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]),
        np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]),
        np.array(["class-a", "class-b"]),
    )
    assert binary["roc_auc"] == pytest.approx(1.0)
    assert np.isfinite(binary["log_loss"])
    assert np.isfinite(binary["brier_score"])

    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    multiclass = compute_classification_metrics(
        np.array([0, 1, 2]), np.array([0, 1, 2]), probs,
        np.array(["class-a", "class-b", "class-c"]),
    )
    expected = np.mean(np.sum((probs - np.eye(3)) ** 2, axis=1))
    assert multiclass["brier_score"] == pytest.approx(expected)
    assert np.isfinite(multiclass["log_loss"])


def test_primary_statistics_use_dataset_as_unit_and_tables_mark_missing_conditions(tmp_path):
    pipelines = ["Raw", "AutoFE_Baseline", "AutoFE_MI", "AutoFE_Random", "AutoFE_NoMultiply"]
    rows = []
    for dataset_no in range(8):
        for seed in (1, 2):
            for fold in (1, 2):
                for condition in ("clean", "gaussian_noise_0.01"):
                    base = 0.70 + dataset_no * 0.01 + seed * 0.001 + fold * 0.0001
                    for pipeline_no, pipeline in enumerate(pipelines):
                        rows.append({
                            "dataset": f"d{dataset_no}", "seed": seed, "fold": fold,
                            "condition": condition, "model": "logistic_regression",
                            "pipeline": pipeline, "status": "success",
                            "roc_auc": base - pipeline_no * 0.005,
                        })
    result_path = tmp_path / "corrected-results.jsonl"
    result_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    stats = run_wilcoxon_analysis(result_path, tmp_path / "stats.csv")
    assert len(stats) == 4
    assert set(stats["n_datasets"]) == {8}
    assert set(stats["independent_unit"]) == {"dataset"}
    assert stats["holm_adjusted_p_value"].between(0, 1).all()

    small = pd.DataFrame(rows)
    small = small[small["condition"].isin({"clean", "gaussian_noise_0.01"})]
    table = generate_table_2_robustness(small)
    assert "NA" in table
    assert "missing_values_0.05" in table
    legacy = pd.DataFrame([
        {"dataset": "old", "seed": 1, "fold": 1, "condition": "missing_values_0.1",
         "pipeline": pipeline, "roc_auc": 0.75}
        for pipeline in ("Raw", "AutoFE_Baseline", "AutoFE_NoMultiply")
    ] + [
        {"dataset": "old", "seed": 1, "fold": 1, "condition": "covariate_shift",
         "pipeline": "Raw", "roc_auc": 0.1}
    ])
    legacy_table = generate_table_2_robustness(legacy)
    missingness_row = next(line for line in legacy_table.splitlines() if line.startswith("| missing_values_0.10 |"))
    assert "NA" not in missingness_row
    assert "covariate_shift" not in legacy_table


def test_stable_seed_and_corruption_match_across_hash_seed_processes():
    script = r'''
import json
import numpy as np
import pandas as pd
from src.feature_engineering import DFSConfig, _limit_features
from src.provenance import stable_seed, frame_sha256, vector_sha256
from src.shift_generator import apply_perturbation
X = pd.DataFrame({"a": np.arange(40, dtype=float), "b": np.sin(np.arange(40)), "c": np.cos(np.arange(40))})
y = pd.Series([0, 1] * 20)
seed = stable_seed({"id": 3, "checksum": "abc"}, 42, 2, "label_noise_0.05")
xp, yp = apply_perturbation(X, y, "label_noise", 0.25, seed)
selected = _limit_features(X, X.copy(), y, DFSConfig(enable_dfs=False, selection_method="mi", max_features=2, max_base_features=None))[2]
print(json.dumps({"seed": seed, "x": frame_sha256(xp), "y": vector_sha256(yp), "selected": selected}))
'''
    outputs = []
    for hash_seed in ("1", "987654"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, env=env,
            check=True, capture_output=True, text=True,
        )
        outputs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]


def test_mismatched_cache_identity_is_rejected_and_regenerated(tmp_path):
    run_dir = tmp_path / "run"
    calls = []
    first = {
        "dataset_checksum": "dataset-a", "train_indices_sha256": "train-a",
        "test_indices_sha256": "test-a", "configuration_fingerprint": "cfg-a",
        "code_fingerprint": "code-a", "runtime_fingerprint": "runtime-a",
        "protocol_version": PROTOCOL_VERSION,
        "source_csv_sha256": "source-with-ytest-a",
        "dataset_identity": {"id": 3, "saved_csv_sha256": "source-with-ytest-a"},
    }
    one = _load_or_create_feature_cache(run_dir, "unit", "Raw", first, lambda: calls.append("first") or ("first",))
    assert one == ("first",)
    # The provenance-only full CSV checksum may change when only held-out labels
    # change. It is deliberately excluded from cache decisions.
    source_changed = dict(
        first,
        source_csv_sha256="source-with-ytest-b",
        dataset_identity={"id": 3, "saved_csv_sha256": "source-with-ytest-b"},
    )
    assert cache_fingerprint(first) == cache_fingerprint(source_changed)
    reused = _load_or_create_feature_cache(run_dir, "unit", "Raw", source_changed,
                                           lambda: calls.append("incorrect-ytest-refit") or ("bad",))
    assert reused == ("first",)
    assert calls == ["first"]

    variants = [
        dict(source_changed, dataset_checksum="dataset-b"),
        dict(source_changed, train_indices_sha256="train-b"),
        dict(source_changed, configuration_fingerprint="cfg-b"),
        dict(source_changed, code_fingerprint="code-b"),
        dict(source_changed, runtime_fingerprint="runtime-b"),
        dict(source_changed, protocol_version="protocol-b"),
    ]
    for index, changed in enumerate(variants):
        payload = _load_or_create_feature_cache(
            run_dir, "unit", "Raw", changed,
            lambda index=index: calls.append(f"regenerated-{index}") or (f"new-{index}",),
        )
        assert payload == (f"new-{index}",)
    assert calls == ["first", *(f"regenerated-{i}" for i in range(len(variants)))]
    manifest_path = next((run_dir / "cache").glob("*.manifest.json"))
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "cache_regenerated_after_rejection" in observed


def test_failure_accounting_records_phase1_and_phase2_failures(tmp_path):
    frame = _synthetic_frame()
    data_path = tmp_path / "numeric.csv"
    _write_synthetic_dataset(data_path)

    def fail_hook(phase, task):
        if task["condition"] == "clean" and phase == "phase1":
            raise RuntimeError("forced phase one failure")
        if task["condition"] == "gaussian_noise_0.01" and phase == "phase2":
            raise ValueError("forced phase two failure")

    manifest = run_experiment(
        {"synthetic": data_path}, run_id="failure-test", output_root=tmp_path / "runs",
        seeds=[12], folds=[1], conditions=(("clean", 0.0), ("gaussian_noise", 0.01)),
        pipelines=("Raw",), models=("logistic_regression",), n_splits=3,
        failure_hook=fail_hook,
    )
    assert manifest["status"] == "completed_with_failures"
    assert manifest["counts"]["failed"] == 2
    assert {task["phase"] for task in manifest["tasks"]} == {"phase1", "phase2"}
    db_path = tmp_path / "runs" / "failure-test" / "checkpoints.sqlite"
    for task in manifest["tasks"]:
        state = get_task_state(db_path, "failure-test", task["task_key"], task["phase"])
        assert state["status"] == "failed"
        assert state["exception_type"] in {"RuntimeError", "ValueError"}
    results = [json.loads(line) for line in (db_path.parent / "results.jsonl").read_text().splitlines()]
    assert len(results) == 2 and all(row["status"] == "failed" for row in results)


def test_run_id_rejects_changed_dataset_without_replacing_existing_manifest(tmp_path):
    data_path = tmp_path / "dataset.csv"
    _write_synthetic_dataset(data_path)
    common = dict(
        data_paths={"synthetic": data_path}, run_id="immutable-input",
        output_root=tmp_path / "runs", seeds=[42], folds=[1],
        conditions=(("clean", 0.0),), pipelines=("Raw",),
        models=("logistic_regression",), n_splits=3,
    )
    completed = run_experiment(**common)
    run_dir = tmp_path / "runs" / "immutable-input"
    manifest_path = run_dir / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    assert completed["status"] == "complete"
    assert completed["expected_tasks"] == 1

    frame = pd.read_csv(data_path)
    frame.loc[0, "numeric_signal"] += 1.0
    frame.to_csv(data_path, index=False)
    with pytest.raises(ValueError, match="Dataset source or metadata checksum changed"):
        run_experiment(**common)
    assert manifest_path.read_bytes() == original_manifest
    assert len((run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_run_id_rejects_changed_target_metadata_without_replacing_manifest(tmp_path):
    data_path = tmp_path / "dataset.csv"
    _write_synthetic_dataset(data_path)
    common = dict(
        data_paths={"synthetic": data_path}, run_id="immutable-metadata",
        output_root=tmp_path / "runs", seeds=[42], folds=[1],
        conditions=(("clean", 0.0),), pipelines=("Raw",),
        models=("logistic_regression",), n_splits=3,
    )
    run_experiment(**common)
    manifest_path = tmp_path / "runs" / "immutable-metadata" / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    metadata_path = data_path.with_name("dataset_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["target_column"] = "target_copy"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset source or metadata checksum changed"):
        run_experiment(**common)
    assert manifest_path.read_bytes() == original_manifest


def test_end_to_end_smoke_is_paired_cached_and_reproducible(tmp_path):
    pytest.importorskip("featuretools")
    data_path = tmp_path / "smoke.csv"
    _write_synthetic_dataset(data_path)
    common = dict(
        data_paths={"synthetic": data_path}, output_root=tmp_path / "runs",
        seeds=[42], folds=[1], conditions=(("clean", 0.0), ("gaussian_noise", 0.01)),
        pipelines=("Raw", "AutoFE_Baseline"), models=("logistic_regression",), n_splits=3,
    )
    first = run_experiment(run_id="smoke-a", **common)
    second = run_experiment(run_id="smoke-b", **common)
    resumed = run_experiment(run_id="smoke-a", **common)
    assert first["counts"] == {"success": 4, "failed": 0}
    assert second["counts"] == {"success": 4, "failed": 0}
    assert resumed["counts"] == {"success": 4, "failed": 0}
    assert all(task.get("resumed") for task in resumed["tasks"])
    rows_a = [json.loads(line) for line in (tmp_path / "runs/smoke-a/results.jsonl").read_text().splitlines()]
    rows_b = [json.loads(line) for line in (tmp_path / "runs/smoke-b/results.jsonl").read_text().splitlines()]
    assert all(row["status"] == "success" for row in rows_a + rows_b)
    assert {(row["condition"], row["pipeline"]) for row in rows_a} == {
        ("clean", "Raw"), ("clean", "AutoFE_Baseline"),
        ("gaussian_noise_0.01", "Raw"), ("gaussian_noise_0.01", "AutoFE_Baseline"),
    }
    required_ledger_columns = {
        "dataset", "seed", "fold", "condition", "pipeline", "model", "status",
        "cache_fingerprint", "train_matrix_sha256", "test_matrix_sha256",
        "prediction_sha256", "roc_auc",
    }
    assert required_ledger_columns.issubset(rows_a[0])
    key = lambda row: (
        row["condition"], row["pipeline"], row["cache_fingerprint"],
        row["train_matrix_sha256"], row["test_matrix_sha256"], row["prediction_sha256"],
    )
    assert sorted(map(key, rows_a)) == sorted(map(key, rows_b))
    task = first["tasks"][0]
    assert task["cache_fingerprint"]
    assert task["stable_perturbation_seed"] >= 0
    assert task["train_indices_sha256"] and task["test_indices_sha256"]
    assert first["datasets"][0]["target_proxy_review"]
    feature_cache = (
        tmp_path / "runs" / "smoke-a" / "cache"
        / f"{task['feature_task_key']}_{task['pipeline']}.manifest.json"
    )
    assert json.loads(feature_cache.read_text(encoding="utf-8"))["cache_fingerprint"] == task["cache_fingerprint"]
    assert len((tmp_path / "runs" / "smoke-a" / "results.jsonl").read_text().splitlines()) == 4
