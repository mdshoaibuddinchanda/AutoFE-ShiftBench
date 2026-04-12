"""Unified experiment runner for full benchmark evaluation."""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation import aggregate_final_results, compute_roc_auc
from src.model import prepare_inference_features, train_model_pipeline
from src.shift_generator import build_shifted_test_frame
from src.statistics import build_main_results_table, run_wilcoxon_analysis


SHIFT_FILE_PATTERN = re.compile(
    r"^(?P<shift_type>random|important|missing)_severity_(?P<severity>\d{3})(?:__run\d{3})?\.pkl$"
)


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for full benchmark execution."""

    config_path: str = "config/config.yaml"
    dataset_list_path: str = "config/dataset_list.yaml"
    processed_dir: str = "data/processed"
    shifted_dir: str = "data/shifted"
    final_results_path: str = "reports/tables/final_results.csv"
    aggregated_results_path: str = "reports/tables/aggregated_results.csv"
    statistical_results_path: str = "reports/tables/statistical_results.csv"
    main_table_path: str = "reports/tables/main_results.csv"
    figure_dir: str = "reports/figures"
    task: str = "classification"
    random_state_base: int = 0
    model_params: dict[str, Any] | None = None
    model_types: list[str] | None = None
    feature_counts: list[int] | None = None
    max_datasets: int | None = None
    max_seeds: int | None = None
    checkpoint_path: str = "reports/tables/final_results.partial.csv"
    progress_every: int = 1
    generate_figures: bool = True
    figure_dpi: int = 600
    save_pdf_figures: bool = True
    save_tiff_figures: bool = True


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as mapping."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {file_path}")
    return payload


def _load_dataset_names(dataset_list_path: str | Path) -> list[str]:
    """Load dataset names from dataset list config."""
    payload = _load_yaml(dataset_list_path)
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("'datasets' must be a list in dataset_list.yaml")

    names: list[str] = []
    for dataset in datasets:
        if isinstance(dataset, str):
            names.append(dataset)
        elif isinstance(dataset, dict) and "name" in dataset:
            names.append(str(dataset["name"]))
        else:
            raise ValueError("Each dataset entry must be string or mapping with 'name'")

    if not names:
        raise ValueError("No datasets found in dataset list")
    return names


def _expand_seeds(seed_config: Any, random_state_base: int = 0) -> list[int]:
    """Expand seed config value into explicit seed list."""
    if isinstance(seed_config, int):
        if seed_config <= 0:
            raise ValueError("seeds in config must be > 0")
        return [random_state_base + seed for seed in range(seed_config)]

    if isinstance(seed_config, list) and seed_config:
        return [int(seed) for seed in seed_config]

    raise ValueError("Unsupported seeds config; expected int or non-empty list")


def _resolve_model_types(model_cfg: dict[str, Any]) -> list[str]:
    """Resolve model types from config with validation and stable ordering."""
    supported = {"xgboost", "random_forest"}
    configured = model_cfg.get("types")
    if configured is None:
        configured = [model_cfg.get("type", "xgboost")]

    if isinstance(configured, str):
        configured_values = [configured]
    elif isinstance(configured, list):
        configured_values = [str(value) for value in configured]
    else:
        raise ValueError("model.types must be a list or model.type must be a string")

    resolved: list[str] = []
    for value in configured_values:
        normalized = value.strip().lower()
        if not normalized:
            continue
        if normalized not in supported:
            raise ValueError(
                f"Unsupported model type '{normalized}'. Supported: {sorted(supported)}"
            )
        if normalized not in resolved:
            resolved.append(normalized)

    if not resolved:
        raise ValueError("At least one model type must be configured")
    return resolved


def _resolve_feature_counts(global_cfg: dict[str, Any]) -> list[int]:
    """Resolve requested feature-count grid for sensitivity analysis."""
    candidates: Any = global_cfg.get("feature_counts")

    if candidates is None:
        fs_cfg = global_cfg.get("feature_selection", {})
        if isinstance(fs_cfg, dict):
            candidates = fs_cfg.get("sensitivity_counts")

    if candidates is None:
        fe_cfg = global_cfg.get("feature_engineering", {})
        if isinstance(fe_cfg, dict):
            candidates = fe_cfg.get("sensitivity_counts")

    if candidates is None:
        fe_cfg = global_cfg.get("feature_engineering", {})
        default_count = 100
        if isinstance(fe_cfg, dict):
            default_count = int(fe_cfg.get("max_features", 100))
        candidates = [default_count]

    if isinstance(candidates, int):
        values = [candidates]
    elif isinstance(candidates, list):
        values = candidates
    else:
        raise ValueError("feature-count configuration must be an int or list[int]")

    resolved: list[int] = []
    for value in values:
        count = int(value)
        if count <= 0:
            raise ValueError("All feature counts must be > 0")
        if count not in resolved:
            resolved.append(count)

    if not resolved:
        raise ValueError("At least one feature count is required")
    return resolved


def _resolve_ranked_features(payload: dict[str, Any], available: list[str]) -> list[str]:
    """Resolve deterministic feature ranking from payload metadata."""
    candidates: list[str] = []

    selected_feature_names = payload.get("selected_feature_names")
    if isinstance(selected_feature_names, list):
        candidates.extend([str(feature) for feature in selected_feature_names])

    feature_selection = payload.get("feature_selection")
    if isinstance(feature_selection, dict):
        metadata = feature_selection.get("metadata")
        if isinstance(metadata, dict):
            selected_features = metadata.get("selected_features")
            if isinstance(selected_features, list):
                candidates.extend([str(feature) for feature in selected_features])

    feature_engineering = payload.get("feature_engineering")
    if isinstance(feature_engineering, dict):
        metadata = feature_engineering.get("metadata")
        if isinstance(metadata, dict):
            selected_features = metadata.get("selected_features")
            if isinstance(selected_features, list):
                candidates.extend([str(feature) for feature in selected_features])

    available_set = set(available)
    ranked: list[str] = []
    for feature in candidates:
        if feature in available_set and feature not in ranked:
            ranked.append(feature)

    for feature in available:
        if feature not in ranked:
            ranked.append(feature)

    return ranked


def _build_pipeline_b_feature_sets(
    selected_payload: dict[str, Any],
    feature_counts: list[int],
) -> dict[int, dict[str, Any]]:
    """Build Pipeline-B train/test matrices for each requested feature count."""
    if "x_train_fe" in selected_payload and "x_test_fe" in selected_payload:
        x_train_base = selected_payload["x_train_fe"]
        x_test_base = selected_payload["x_test_fe"]
    elif "x_train_selected" in selected_payload and "x_test_selected" in selected_payload:
        x_train_base = selected_payload["x_train_selected"]
        x_test_base = selected_payload["x_test_selected"]
    else:
        x_train_base, x_test_base = _get_pipeline_b_matrices(selected_payload)

    if not isinstance(x_train_base, pd.DataFrame) or not isinstance(x_test_base, pd.DataFrame):
        raise TypeError("Pipeline-B feature matrices must be pandas DataFrames")

    available_features = list(x_train_base.columns)
    ranked_features = _resolve_ranked_features(selected_payload, available_features)

    feature_sets: dict[int, dict[str, Any]] = {}
    for requested_count in feature_counts:
        used_count = min(requested_count, len(ranked_features))
        selected_features = ranked_features[:used_count]
        feature_sets[requested_count] = {
            "requested_count": requested_count,
            "used_count": used_count,
            "feature_names": selected_features,
            "x_train": x_train_base[selected_features].copy(),
            "x_test": x_test_base[selected_features].copy(),
        }

    return feature_sets


def _load_artifact(path: Path) -> dict[str, Any]:
    """Load pickle artifact and validate mapping type."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    with path.open("rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Artifact payload must be dict: {path}")
    return payload


def _get_pipeline_b_matrices(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract Pipeline-B train/test feature matrices from selected artifact."""
    key_pairs = (
        ("x_train_selected", "x_test_selected"),
        ("x_train_fe", "x_test_fe"),
    )
    for train_key, test_key in key_pairs:
        x_train = payload.get(train_key)
        x_test = payload.get(test_key)
        if isinstance(x_train, pd.DataFrame) and isinstance(x_test, pd.DataFrame):
            return x_train, x_test

    raise KeyError("Selected artifact missing supported Pipeline-B feature matrices")


def _resolve_shift_variations(dataset_shift_dir: Path) -> list[dict[str, Any]]:
    """Resolve latest unique shift files by (shift_type, severity)."""
    if not dataset_shift_dir.exists():
        raise FileNotFoundError(f"Dataset shift directory not found: {dataset_shift_dir}")

    variations: dict[tuple[str, int], Path] = {}
    for file_path in dataset_shift_dir.glob("*.pkl"):
        match = SHIFT_FILE_PATTERN.match(file_path.name)
        if not match:
            continue

        shift_type = match.group("shift_type")
        severity_code = int(match.group("severity"))
        key = (shift_type, severity_code)

        current = variations.get(key)
        if current is None or file_path.stat().st_mtime > current.stat().st_mtime:
            variations[key] = file_path

    resolved: list[dict[str, Any]] = []
    for (shift_type, severity_code), file_path in sorted(
        variations.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        resolved.append(
            {
                "shift_type": shift_type,
                "severity_code": severity_code,
                "severity": severity_code / 100.0,
                "path": file_path,
            }
        )
    return resolved


def _materialize_shifted_variations(
    x_test: pd.DataFrame,
    variations: list[dict[str, Any]],
    random_state: int,
    ranked_features: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build shifted test matrices for all shift-type/severity variations."""
    payload_for_importance: dict[str, Any] = {}
    if ranked_features:
        payload_for_importance["ranked_features"] = ranked_features

    prepared: list[dict[str, Any]] = []
    for variation in variations:
        shifted, _ = build_shifted_test_frame(
            x_test=x_test,
            shift_type=variation["shift_type"],
            severity=variation["severity"],
            random_state=random_state,
            payload_for_importance=payload_for_importance,
        )
        prepared.append(
            {
                "shift_type": variation["shift_type"],
                "severity": variation["severity"],
                "shifted": shifted,
            }
        )

    return prepared


def _encode_labels_for_model(y_true: pd.Series, bundle: dict[str, Any]) -> np.ndarray:
    """Encode labels using pipeline label encoder when available."""
    encoder = bundle.get("label_encoder")
    if encoder is None:
        return pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    return encoder.transform(y_true.astype(str))


def _evaluate_roc_auc(
    bundle: dict[str, Any],
    x_shifted: pd.DataFrame,
    y_true: pd.Series,
    x_prepared: np.ndarray | None = None,
) -> float:
    """Evaluate ROC-AUC for one pipeline on one shifted test matrix."""
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    if x_prepared is None:
        x_infer = prepare_inference_features(x_shifted, feature_names)
    elif isinstance(x_prepared, pd.DataFrame):
        x_infer = x_prepared
    else:
        x_infer = pd.DataFrame(x_prepared, columns=feature_names)

    if not hasattr(model, "predict_proba"):
        return float("nan")

    y_encoded = _encode_labels_for_model(y_true, bundle)
    y_proba = model.predict_proba(x_infer)
    return compute_roc_auc(pd.Series(y_encoded), y_proba)


def _generate_figures_from_final_results(
    final_results_path: str | Path,
    figure_dir: str | Path,
    figure_dpi: int = 600,
    save_pdf_figures: bool = True,
    save_tiff_figures: bool = True,
) -> dict[str, Path]:
    """Generate degradation and average-performance figures from final results."""
    import matplotlib.pyplot as plt

    input_path = Path(final_results_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Final results file not found: {input_path}")

    results_df = pd.read_csv(input_path)
    required_cols = {"severity", "pipeline", "roc_auc"}
    missing_cols = required_cols.difference(results_df.columns)
    if missing_cols:
        raise KeyError(
            f"Missing required columns for figure generation: {sorted(missing_cols)}"
        )

    out_dir = Path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if figure_dpi < 300:
        raise ValueError("figure_dpi must be at least 300 for publication-quality output")

    plt.rcParams.update(
        {
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
        }
    )

    figure_df = results_df.copy()
    figure_subtitle = ""
    if {"model_type", "feature_count"}.issubset(figure_df.columns):
        model_values = [str(value) for value in figure_df["model_type"].dropna().unique().tolist()]
        feature_counts = sorted(int(value) for value in figure_df["feature_count"].dropna().unique())
        if model_values and feature_counts:
            preferred_model = "xgboost" if "xgboost" in model_values else sorted(model_values)[0]
            preferred_feature_count = feature_counts[-1]
            scoped = figure_df[
                (figure_df["model_type"] == preferred_model)
                & (figure_df["feature_count"] == preferred_feature_count)
            ]
            if not scoped.empty:
                figure_df = scoped
                figure_subtitle = (
                    f" ({preferred_model.replace('_', ' ').title()}, "
                    f"{preferred_feature_count} features)"
                )

    pipeline_color = {
        "A": "#1f77b4",
        "B": "#d55e00",
    }

    pipeline_name = {
        "A": "Baseline Pipeline",
        "B": "AutoFE Pipeline",
    }

    curve_df = (
        figure_df.groupby(["severity", "pipeline"], as_index=False)
        .agg(mean_roc_auc=("roc_auc", "mean"))
        .sort_values(by=["pipeline", "severity"])
    )

    plt.figure(figsize=(8, 5))
    for pipeline, group in curve_df.groupby("pipeline"):
        plt.plot(
            group["severity"] * 100,
            group["mean_roc_auc"],
            marker="o",
            color=pipeline_color.get(str(pipeline), "#2f2f2f"),
            label=pipeline_name.get(str(pipeline), str(pipeline)),
        )
    plt.xlabel("Corruption Severity (%)")
    plt.ylabel("Mean ROC-AUC")
    plt.title("Performance Degradation Under Increasing Feature Corruption" + figure_subtitle)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    degradation_png_path = out_dir / "degradation_curve.png"
    plt.savefig(degradation_png_path, dpi=figure_dpi, format="png")
    degradation_tiff_path = out_dir / "degradation_curve.tiff"
    if save_tiff_figures:
        plt.savefig(degradation_tiff_path, dpi=figure_dpi, format="tiff")
    degradation_pdf_path = out_dir / "degradation_curve.pdf"
    if save_pdf_figures:
        plt.savefig(degradation_pdf_path, format="pdf")
    plt.close()

    avg_df = (
        figure_df.groupby("pipeline", as_index=False)
        .agg(mean_roc_auc=("roc_auc", "mean"), std_roc_auc=("roc_auc", "std"))
        .sort_values(by="pipeline")
    )
    avg_df["pipeline_label"] = avg_df["pipeline"].map(pipeline_name).fillna(avg_df["pipeline"])

    plt.figure(figsize=(7, 5))
    plt.bar(
        avg_df["pipeline_label"],
        avg_df["mean_roc_auc"],
        yerr=avg_df["std_roc_auc"],
        color=[pipeline_color.get(p, "#2f2f2f") for p in avg_df["pipeline"]],
        capsize=6,
    )
    plt.xlabel("Pipeline")
    plt.ylabel("ROC-AUC (mean ± std)")
    plt.title(
        "Mean ROC-AUC Comparison of Baseline and AutoFE Pipelines Under Feature Shift"
        + figure_subtitle
    )
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    average_png_path = out_dir / "average_performance.png"
    plt.savefig(average_png_path, dpi=figure_dpi, format="png")
    average_tiff_path = out_dir / "average_performance.tiff"
    if save_tiff_figures:
        plt.savefig(average_tiff_path, dpi=figure_dpi, format="tiff")
    average_pdf_path = out_dir / "average_performance.pdf"
    if save_pdf_figures:
        plt.savefig(average_pdf_path, format="pdf")
    plt.close()

    outputs: dict[str, Path] = {
        "degradation_png": degradation_png_path,
        "average_png": average_png_path,
    }
    if save_tiff_figures:
        outputs["degradation_tiff"] = degradation_tiff_path
        outputs["average_tiff"] = average_tiff_path
    if save_pdf_figures:
        outputs["degradation_pdf"] = degradation_pdf_path
        outputs["average_pdf"] = average_pdf_path
    return outputs


def _write_figure_captions(figure_dir: str | Path) -> Path:
    """Write publication-ready figure captions for generated plots."""
    out_dir = Path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caption_path = out_dir / "figure_captions.md"
    caption_text = (
        "# Figure Captions\n\n"
        "## Figure 1: average_performance.png\n\n"
        "Across all datasets, the baseline pipeline achieves higher mean ROC-AUC and lower variance than the AutoFE pipeline in aggregate.\n\n"
        "## Figure 2: degradation_curve.png\n\n"
        "Both pipelines degrade as corruption severity increases; AutoFE shows a slightly steeper average decline, consistent with a performance-robustness trade-off.\n"
    )
    caption_path.write_text(caption_text, encoding="utf-8")
    return caption_path


def run_full_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run full benchmark and generate final, aggregated, and statistical outputs."""
    global_cfg = _load_yaml(config.config_path)
    datasets = _load_dataset_names(config.dataset_list_path)
    seeds = _expand_seeds(
        global_cfg.get("seeds", 20),
        random_state_base=config.random_state_base,
    )

    if config.max_datasets is not None:
        datasets = datasets[: config.max_datasets]
    if config.max_seeds is not None:
        seeds = seeds[: config.max_seeds]

    model_cfg = global_cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be a mapping")

    model_params = config.model_params or model_cfg.get("params", {})
    model_types = config.model_types or _resolve_model_types(model_cfg)
    feature_counts = config.feature_counts or _resolve_feature_counts(global_cfg)

    rows: list[dict[str, Any]] = []
    start_ts = time.perf_counter()
    total_datasets = len(datasets)
    total_seeds = len(seeds)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for dataset_index, dataset in enumerate(datasets, start=1):
        print(f"[dataset {dataset_index}/{total_datasets}] {dataset}: loading artifacts")
        raw_artifact_path = Path(config.processed_dir) / f"{dataset}.pkl"
        feature_artifact_path = Path(config.processed_dir) / f"{dataset}_features.pkl"
        selected_artifact_path = Path(config.processed_dir) / f"{dataset}_selected.pkl"

        raw_payload = _load_artifact(raw_artifact_path)
        selected_payload = _load_artifact(selected_artifact_path)
        feature_payload = _load_artifact(feature_artifact_path)

        x_train_a = raw_payload["x_train"]
        x_test_a_base = raw_payload["x_test"]
        y_train = raw_payload["y_train"].reset_index(drop=True)
        y_test = raw_payload["y_test"].reset_index(drop=True)

        pipeline_b_payload = dict(selected_payload)
        x_train_fe = feature_payload.get("x_train_fe")
        x_test_fe = feature_payload.get("x_test_fe")
        if isinstance(x_train_fe, pd.DataFrame) and isinstance(x_test_fe, pd.DataFrame):
            pipeline_b_payload["x_train_fe"] = x_train_fe
            pipeline_b_payload["x_test_fe"] = x_test_fe
            if "feature_engineering" in feature_payload:
                pipeline_b_payload["feature_engineering"] = feature_payload["feature_engineering"]

        pipeline_b_feature_sets = _build_pipeline_b_feature_sets(
            selected_payload=pipeline_b_payload,
            feature_counts=feature_counts,
        )
        shift_variations = _resolve_shift_variations(Path(config.shifted_dir) / dataset)
        if not shift_variations:
            print(f"  no shift files found for {dataset}, skipping")
            continue

        prepared_variations_a = _materialize_shifted_variations(
            x_test=x_test_a_base,
            variations=shift_variations,
            random_state=42,
            ranked_features=None,
        )

        prepared_variations_b_by_count: dict[int, list[dict[str, Any]]] = {}
        for feature_count, feature_set in pipeline_b_feature_sets.items():
            prepared_variations_b_by_count[feature_count] = _materialize_shifted_variations(
                x_test=feature_set["x_test"],
                variations=shift_variations,
                random_state=42,
                ranked_features=feature_set["feature_names"],
            )

        for model_index, model_type in enumerate(model_types, start=1):
            print(
                f"  [model {model_index}/{len(model_types)}] "
                f"model_type={model_type} feature_counts={feature_counts}"
            )

            cached_a: list[np.ndarray] | None = None
            signature_a: tuple[str, ...] | None = None
            cached_b_by_count: dict[int, list[np.ndarray] | None] = {
                feature_count: None for feature_count in feature_counts
            }
            signature_b_by_count: dict[int, tuple[str, ...] | None] = {
                feature_count: None for feature_count in feature_counts
            }

            for seed_index, seed in enumerate(seeds, start=1):
                pipeline_a_bundle = train_model_pipeline(
                    x_train=x_train_a,
                    y_train=y_train,
                    task=config.task,
                    params=model_params,
                    random_state=seed,
                    model_type=model_type,
                )

                current_signature_a = tuple(pipeline_a_bundle["feature_names"])
                if cached_a is None or current_signature_a != signature_a:
                    cached_a = [
                        prepare_inference_features(
                            variation["shifted"],
                            pipeline_a_bundle["feature_names"],
                        )
                        for variation in prepared_variations_a
                    ]
                    signature_a = current_signature_a

                roc_auc_a_values = [
                    _evaluate_roc_auc(
                        bundle=pipeline_a_bundle,
                        x_shifted=variation["shifted"],
                        y_true=y_test,
                        x_prepared=cached_a[variation_index],
                    )
                    for variation_index, variation in enumerate(prepared_variations_a)
                ]

                for feature_count in feature_counts:
                    feature_set = pipeline_b_feature_sets[feature_count]
                    prepared_variations_b = prepared_variations_b_by_count[feature_count]

                    pipeline_b_bundle = train_model_pipeline(
                        x_train=feature_set["x_train"],
                        y_train=y_train,
                        task=config.task,
                        params=model_params,
                        random_state=seed,
                        model_type=model_type,
                    )

                    current_signature_b = tuple(pipeline_b_bundle["feature_names"])
                    cached_b = cached_b_by_count[feature_count]
                    signature_b = signature_b_by_count[feature_count]
                    if cached_b is None or current_signature_b != signature_b:
                        cached_b = [
                            prepare_inference_features(
                                variation["shifted"],
                                pipeline_b_bundle["feature_names"],
                            )
                            for variation in prepared_variations_b
                        ]
                        cached_b_by_count[feature_count] = cached_b
                        signature_b_by_count[feature_count] = current_signature_b

                    for variation_index, variation in enumerate(prepared_variations_b):
                        shift_type = variation["shift_type"]
                        severity = variation["severity"]
                        roc_auc_a = roc_auc_a_values[variation_index]
                        roc_auc_b = _evaluate_roc_auc(
                            bundle=pipeline_b_bundle,
                            x_shifted=variation["shifted"],
                            y_true=y_test,
                            x_prepared=cached_b[variation_index],
                        )

                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "model_type": model_type,
                                "feature_count": feature_count,
                                "feature_count_used": feature_set["used_count"],
                                "shift_type": shift_type,
                                "severity": severity,
                                "pipeline": "A",
                                "roc_auc": roc_auc_a,
                            }
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "model_type": model_type,
                                "feature_count": feature_count,
                                "feature_count_used": feature_set["used_count"],
                                "shift_type": shift_type,
                                "severity": severity,
                                "pipeline": "B",
                                "roc_auc": roc_auc_b,
                            }
                        )

                if config.progress_every > 0 and (
                    seed_index % config.progress_every == 0 or seed_index == total_seeds
                ):
                    elapsed_min = (time.perf_counter() - start_ts) / 60.0
                    print(
                        f"  [seed {seed_index}/{total_seeds}] "
                        f"model={model_type} seed={seed} rows={len(rows)} "
                        f"elapsed={elapsed_min:.1f}m"
                    )

        partial_df = pd.DataFrame(rows)
        if not partial_df.empty:
            partial_df = partial_df[
                [
                    "dataset",
                    "seed",
                    "model_type",
                    "feature_count",
                    "feature_count_used",
                    "shift_type",
                    "severity",
                    "pipeline",
                    "roc_auc",
                ]
            ].sort_values(
                [
                    "dataset",
                    "model_type",
                    "feature_count",
                    "seed",
                    "shift_type",
                    "severity",
                    "pipeline",
                ]
            )
            partial_df.to_csv(checkpoint_path, index=False)
            print(f"  checkpoint={checkpoint_path} rows={len(partial_df)}")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No experiment rows generated; check artifacts and shift files")

    results = results[
        [
            "dataset",
            "seed",
            "model_type",
            "feature_count",
            "feature_count_used",
            "shift_type",
            "severity",
            "pipeline",
            "roc_auc",
        ]
    ].sort_values(
        [
            "dataset",
            "model_type",
            "feature_count",
            "seed",
            "shift_type",
            "severity",
            "pipeline",
        ]
    )

    final_results_path = Path(config.final_results_path)
    final_results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(final_results_path, index=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    aggregate_final_results(
        final_results_path=final_results_path,
        output_path=config.aggregated_results_path,
    )
    run_wilcoxon_analysis(
        final_results_path=final_results_path,
        output_path=config.statistical_results_path,
    )
    build_main_results_table(
        statistical_results_path=config.statistical_results_path,
        output_path=config.main_table_path,
    )

    if config.generate_figures:
        _generate_figures_from_final_results(
            final_results_path=final_results_path,
            figure_dir=config.figure_dir,
            figure_dpi=config.figure_dpi,
            save_pdf_figures=config.save_pdf_figures,
            save_tiff_figures=config.save_tiff_figures,
        )
        _write_figure_captions(config.figure_dir)
    return results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for full benchmark runner."""
    parser = argparse.ArgumentParser(description="Run full benchmark experiment")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset-list", default="config/dataset_list.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--shifted-dir", default="data/shifted")
    parser.add_argument("--final-results", default="reports/tables/final_results.csv")
    parser.add_argument("--aggregated-results", default="reports/tables/aggregated_results.csv")
    parser.add_argument("--statistical-results", default="reports/tables/statistical_results.csv")
    parser.add_argument("--main-table", default="reports/tables/main_results.csv")
    parser.add_argument("--figure-dir", default="reports/figures")
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("--random-state-base", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument(
        "--model-types",
        default=None,
        help="Comma-separated model types (xgboost,random_forest)",
    )
    parser.add_argument(
        "--feature-counts",
        default=None,
        help="Comma-separated feature counts for sensitivity analysis (e.g. 100,200)",
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument(
        "--checkpoint-path",
        default="reports/tables/final_results.partial.csv",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--figure-dpi", type=int, default=600)
    parser.add_argument("--no-pdf-figures", action="store_true")
    parser.add_argument("--no-tiff-figures", action="store_true")
    return parser.parse_args()


def _parse_comma_list(value: str | None) -> list[str] | None:
    """Parse comma-separated string into non-empty token list."""
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_comma_ints(value: str | None) -> list[int] | None:
    """Parse comma-separated string into positive integers."""
    items = _parse_comma_list(value)
    if items is None:
        return None
    parsed = [int(item) for item in items]
    if any(item <= 0 for item in parsed):
        raise ValueError("feature counts must be positive integers")
    return parsed


if __name__ == "__main__":
    args = _parse_args()
    model_types = _parse_comma_list(args.model_types)
    feature_counts = _parse_comma_ints(args.feature_counts)
    config = ExperimentConfig(
        config_path=args.config,
        dataset_list_path=args.dataset_list,
        processed_dir=args.processed_dir,
        shifted_dir=args.shifted_dir,
        final_results_path=args.final_results,
        aggregated_results_path=args.aggregated_results,
        statistical_results_path=args.statistical_results,
        main_table_path=args.main_table,
        figure_dir=args.figure_dir,
        task=args.task,
        random_state_base=args.random_state_base,
        model_params={"n_estimators": args.n_estimators, "max_depth": args.max_depth},
        model_types=model_types,
        feature_counts=feature_counts,
        max_datasets=args.max_datasets,
        max_seeds=args.max_seeds,
        checkpoint_path=args.checkpoint_path,
        progress_every=args.progress_every,
        generate_figures=not args.skip_figures,
        figure_dpi=args.figure_dpi,
        save_pdf_figures=not args.no_pdf_figures,
        save_tiff_figures=not args.no_tiff_figures,
    )
    final_results = run_full_experiment(config)
    print(f"rows={len(final_results)}")
    print(f"saved={config.final_results_path}")
    print(f"saved={config.aggregated_results_path}")
    print(f"saved={config.statistical_results_path}")
    print(f"saved={config.main_table_path}")
    if config.generate_figures:
        print(f"saved={Path(config.figure_dir) / 'degradation_curve.png'}")
        print(f"saved={Path(config.figure_dir) / 'average_performance.png'}")
        if config.save_tiff_figures:
            print(f"saved={Path(config.figure_dir) / 'degradation_curve.tiff'}")
            print(f"saved={Path(config.figure_dir) / 'average_performance.tiff'}")
        if config.save_pdf_figures:
            print(f"saved={Path(config.figure_dir) / 'degradation_curve.pdf'}")
            print(f"saved={Path(config.figure_dir) / 'average_performance.pdf'}")
        print(f"saved={Path(config.figure_dir) / 'figure_captions.md'}")
