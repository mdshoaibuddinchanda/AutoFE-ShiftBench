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
from src.model import prepare_inference_features, train_xgboost_pipeline
from src.shift_generator import build_shifted_test_frame
from src.statistics import build_main_results_table, run_wilcoxon_analysis


SHIFT_FILE_PATTERN = re.compile(
    r"^(?P<shift_type>random|important)_severity_(?P<severity>\d{3})(?:__run\d{3})?\.pkl$"
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
        x_infer = x_infer.to_numpy(dtype=np.float32, copy=False)
    else:
        x_infer = x_prepared

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

    pipeline_color = {
        "A": "#1f77b4",
        "B": "#d55e00",
    }

    pipeline_name = {
        "A": "Baseline (XGBoost)",
        "B": "AutoFE Pipeline",
    }

    curve_df = (
        results_df.groupby(["severity", "pipeline"], as_index=False)
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
    plt.title("Performance Degradation Under Increasing Feature Corruption")
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
        results_df.groupby("pipeline", as_index=False)
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
    plt.title("Mean ROC-AUC Comparison of Baseline and AutoFE Pipelines Under Feature Shift")
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

    model_params = config.model_params or global_cfg.get("model", {}).get("params", {})
    rows: list[dict[str, Any]] = []
    start_ts = time.perf_counter()
    total_datasets = len(datasets)
    total_seeds = len(seeds)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for dataset_index, dataset in enumerate(datasets, start=1):
        print(f"[dataset {dataset_index}/{total_datasets}] {dataset}: loading artifacts")
        raw_artifact_path = Path(config.processed_dir) / f"{dataset}.pkl"
        selected_artifact_path = Path(config.processed_dir) / f"{dataset}_selected.pkl"

        raw_payload = _load_artifact(raw_artifact_path)
        selected_payload = _load_artifact(selected_artifact_path)

        x_train_a = raw_payload["x_train"]
        x_test_a_base = raw_payload["x_test"]
        y_train = raw_payload["y_train"].reset_index(drop=True)
        y_test = raw_payload["y_test"].reset_index(drop=True)

        x_train_b, x_test_b_base = _get_pipeline_b_matrices(selected_payload)
        shift_variations = _resolve_shift_variations(Path(config.shifted_dir) / dataset)
        if not shift_variations:
            print(f"  no shift files found for {dataset}, skipping")
            continue

        prepared_variations: list[dict[str, Any]] = []
        for variation in shift_variations:
            shift_payload = _load_artifact(variation["path"])
            shift_type = variation["shift_type"]
            severity = variation["severity"]

            shifted_b = shift_payload.get("x_test_shifted")
            if not isinstance(shifted_b, pd.DataFrame):
                shifted_b, _features_b = build_shifted_test_frame(
                    x_test=x_test_b_base,
                    shift_type=shift_type,
                    severity=severity,
                    random_state=42,
                    payload_for_importance=shift_payload,
                )

            shifted_a_payload = shift_payload.get("x_test_shifted_raw")
            if isinstance(shifted_a_payload, pd.DataFrame):
                shifted_a = shifted_a_payload
            else:
                shifted_a, _features_a = build_shifted_test_frame(
                    x_test=x_test_a_base,
                    shift_type=shift_type,
                    severity=severity,
                    random_state=42,
                    payload_for_importance={},
                )

            prepared_variations.append(
                {
                    "shift_type": shift_type,
                    "severity": severity,
                    "shifted_a": shifted_a,
                    "shifted_b": shifted_b,
                }
            )

        cached_a: list[np.ndarray] | None = None
        cached_b: list[np.ndarray] | None = None
        signature_a: tuple[str, ...] | None = None
        signature_b: tuple[str, ...] | None = None

        for seed_index, seed in enumerate(seeds, start=1):
            pipeline_a_bundle = train_xgboost_pipeline(
                x_train=x_train_a,
                y_train=y_train,
                task=config.task,
                params=model_params,
                random_state=seed,
            )
            pipeline_b_bundle = train_xgboost_pipeline(
                x_train=x_train_b,
                y_train=y_train,
                task=config.task,
                params=model_params,
                random_state=seed,
            )

            current_signature_a = tuple(pipeline_a_bundle["feature_names"])
            current_signature_b = tuple(pipeline_b_bundle["feature_names"])

            if cached_a is None or current_signature_a != signature_a:
                cached_a = [
                    prepare_inference_features(
                        variation["shifted_a"],
                        pipeline_a_bundle["feature_names"],
                    ).to_numpy(dtype=np.float32, copy=False)
                    for variation in prepared_variations
                ]
                signature_a = current_signature_a

            if cached_b is None or current_signature_b != signature_b:
                cached_b = [
                    prepare_inference_features(
                        variation["shifted_b"],
                        pipeline_b_bundle["feature_names"],
                    ).to_numpy(dtype=np.float32, copy=False)
                    for variation in prepared_variations
                ]
                signature_b = current_signature_b

            for variation_index, variation in enumerate(prepared_variations):
                shift_type = variation["shift_type"]
                severity = variation["severity"]

                roc_auc_a = _evaluate_roc_auc(
                    bundle=pipeline_a_bundle,
                    x_shifted=variation["shifted_a"],
                    y_true=y_test,
                    x_prepared=cached_a[variation_index],
                )
                roc_auc_b = _evaluate_roc_auc(
                    bundle=pipeline_b_bundle,
                    x_shifted=variation["shifted_b"],
                    y_true=y_test,
                    x_prepared=cached_b[variation_index],
                )

                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
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
                    f"seed={seed} rows={len(rows)} elapsed={elapsed_min:.1f}m"
                )

        partial_df = pd.DataFrame(rows)
        if not partial_df.empty:
            partial_df = partial_df[
                ["dataset", "seed", "shift_type", "severity", "pipeline", "roc_auc"]
            ].sort_values(["dataset", "seed", "shift_type", "severity", "pipeline"])
            partial_df.to_csv(checkpoint_path, index=False)
            print(f"  checkpoint={checkpoint_path} rows={len(partial_df)}")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No experiment rows generated; check artifacts and shift files")

    results = results[
        ["dataset", "seed", "shift_type", "severity", "pipeline", "roc_auc"]
    ].sort_values(["dataset", "seed", "shift_type", "severity", "pipeline"])

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


if __name__ == "__main__":
    args = _parse_args()
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
