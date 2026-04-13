"""Shift generation utilities that operate on test data only."""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHIFT_TYPES = {"random", "important", "missing"}


@dataclass(slots=True)
class ShiftConfig:
    """Configuration for dataset shift generation."""

    severities: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    shift_types: tuple[str, ...] = ("random", "important", "missing")
    random_state: int = 42


def _get_non_overwriting_path(output_path: Path) -> Path:
    """Return a unique file path without overwriting existing files."""
    if not output_path.exists():
        return output_path

    index = 1
    while True:
        candidate = output_path.with_name(
            f"{output_path.stem}__run{index:03d}{output_path.suffix}"
        )
        if not candidate.exists():
            return candidate
        index += 1


def _get_test_frames(payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Extract all available test feature matrices from artifact payload.

    We persist shifts for each available test-view so evaluation can consume
    fixed artifacts for both Pipeline A (raw) and Pipeline B (AutoFE views).
    """
    frames: dict[str, pd.DataFrame] = {}
    for key in ("x_test", "x_test_fe", "x_test_selected"):
        value = payload.get(key)
        if isinstance(value, pd.DataFrame):
            frames[key] = value.copy()

    if not frames:
        raise KeyError("Payload must contain one of x_test/x_test_fe/x_test_selected")
    return frames


def _importance_ranking_from_payload(x_test: pd.DataFrame, payload: dict[str, Any]) -> list[str]:
    """Build feature ranking from explicit/train-derived payload information."""
    available_features = list(x_test.columns)
    available_set = set(available_features)

    explicit_ranked = payload.get("ranked_features")
    if isinstance(explicit_ranked, list):
        ranked = [
            str(feature)
            for feature in explicit_ranked
            if str(feature) in available_set
        ]
        if ranked:
            for feature in available_features:
                if feature not in ranked:
                    ranked.append(feature)
            return ranked

    selection_meta = payload.get("feature_selection", {})
    metadata = selection_meta.get("metadata", {}) if isinstance(selection_meta, dict) else {}
    combined_scores = metadata.get("combined_scores") if isinstance(metadata, dict) else None

    if isinstance(combined_scores, dict) and combined_scores:
        ranked = [
            feature
            for feature, _score in sorted(
                combined_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if feature in x_test.columns
        ]
        if ranked:
            for feature in available_features:
                if feature not in ranked:
                    ranked.append(feature)
            return ranked

    for train_key in ("x_train_reference", "x_train_selected", "x_train_fe", "x_train"):
        train_source = payload.get(train_key)
        if not isinstance(train_source, pd.DataFrame):
            continue

        train_aligned = train_source.reindex(columns=available_features, fill_value=np.nan)
        numeric = train_aligned.select_dtypes(include=["number", "bool"])
        if numeric.empty:
            continue

        cleaned = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ranked = cleaned.var(axis=0, numeric_only=True).fillna(0.0).sort_values(
            ascending=False
        )
        ranked_features = ranked.index.tolist()
        for feature in available_features:
            if feature not in ranked_features:
                ranked_features.append(feature)
        return ranked_features

    return available_features


def _resolve_feature_scales(
    payload: dict[str, Any],
    features: list[str],
) -> dict[str, float]:
    """Resolve per-feature numeric scales from explicit/train-derived payload fields."""
    explicit_scales = payload.get("feature_scales")
    if isinstance(explicit_scales, dict):
        resolved: dict[str, float] = {}
        for feature in features:
            raw_value = explicit_scales.get(feature)
            if raw_value is None:
                continue
            value = float(raw_value)
            if np.isfinite(value) and value > 0.0:
                resolved[feature] = value
        if resolved:
            return resolved

    train_source: pd.DataFrame | None = None
    for train_key in ("x_train_reference", "x_train_selected", "x_train_fe", "x_train"):
        value = payload.get(train_key)
        if isinstance(value, pd.DataFrame):
            train_source = value
            break

    if train_source is None:
        return {}

    aligned_train = train_source.reindex(columns=features, fill_value=np.nan)
    numeric = aligned_train.select_dtypes(include=["number", "bool"])
    if numeric.empty:
        return {}

    cleaned = numeric.replace([np.inf, -np.inf], np.nan)
    scales: dict[str, float] = {}
    for feature in numeric.columns:
        std_value = float(cleaned[feature].std(ddof=0))
        if np.isfinite(std_value) and std_value > 0.0:
            scales[feature] = std_value

    return scales


def _apply_numeric_noise(
    series: pd.Series,
    severity: float,
    rng: np.random.Generator,
    base_scale: float | None = None,
) -> pd.Series:
    """Apply Gaussian noise corruption to a numeric column."""
    numeric = pd.to_numeric(series, errors="coerce")
    effective_scale = float(base_scale) if base_scale is not None else float("nan")
    if not np.isfinite(effective_scale) or effective_scale <= 0.0:
        std = float(numeric.std(ddof=0))
        effective_scale = std if std > 0 else 1.0

    noise = rng.normal(loc=0.0, scale=effective_scale * severity, size=len(series))
    fill_value = float(numeric.mean()) if numeric.notna().any() else 0.0
    return numeric.fillna(fill_value).to_numpy() + noise


def _apply_categorical_corruption(
    series: pd.Series,
    severity: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Apply random/missing corruption to a categorical column."""
    corrupted = series.copy()
    mask = rng.random(len(series)) < severity
    if not mask.any():
        return corrupted

    candidate_values = series.dropna()
    indices = np.flatnonzero(mask.to_numpy())
    random_mask = rng.random(len(indices)) < 0.5

    if not candidate_values.empty:
        random_indices = indices[random_mask]
        if len(random_indices) > 0:
            sampled_values = rng.choice(candidate_values.to_numpy(), size=len(random_indices))
            corrupted.iloc[random_indices] = sampled_values

    missing_indices = indices[~random_mask]
    if len(missing_indices) > 0:
        corrupted.iloc[missing_indices] = np.nan

    return corrupted


def _apply_missing_corruption(
    series: pd.Series,
    severity: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Apply missing-value corruption to one feature."""
    corrupted = series.copy()
    mask = rng.random(len(series)) < severity
    if not mask.any():
        return corrupted
    corrupted.loc[mask] = np.nan
    return corrupted


def _apply_shift_to_test_only(
    x_test: pd.DataFrame,
    features_to_shift: list[str],
    shift_type: str,
    severity: float,
    rng: np.random.Generator,
    feature_scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Corrupt only selected test features and return shifted test matrix."""
    shifted = x_test.copy()
    for feature in features_to_shift:
        if feature not in shifted.columns:
            continue

        series = shifted[feature]
        if shift_type == "missing":
            shifted[feature] = _apply_missing_corruption(series, severity, rng)
            continue

        if pd.api.types.is_numeric_dtype(series):
            base_scale = None
            if feature_scales is not None:
                base_scale = feature_scales.get(feature)
            shifted[feature] = _apply_numeric_noise(
                series,
                severity,
                rng,
                base_scale=base_scale,
            )
        else:
            shifted[feature] = _apply_categorical_corruption(series, severity, rng)

    return shifted


def build_shifted_test_frame(
    x_test: pd.DataFrame,
    shift_type: str,
    severity: float,
    random_state: int = 42,
    payload_for_importance: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate one shifted test frame for a given shift configuration."""
    if shift_type not in SHIFT_TYPES:
        raise ValueError(f"Unsupported shift type: {shift_type}")

    if x_test.empty:
        raise ValueError("x_test is empty")

    severity_value = float(severity)
    all_features = list(x_test.columns)
    n_features = max(1, min(len(all_features), math.ceil(len(all_features) * severity_value)))

    ranked_features = _importance_ranking_from_payload(
        x_test,
        payload_for_importance or {},
    )
    feature_scales = _resolve_feature_scales(payload_for_importance or {}, all_features)
    shift_offset = {
        "random": 0,
        "important": 1000,
        "missing": 2000,
    }[shift_type]
    seed = random_state + int(severity_value * 100) + shift_offset
    rng = np.random.default_rng(seed)

    if shift_type in {"random", "missing"}:
        selected_features = rng.choice(all_features, size=n_features, replace=False).tolist()
    else:
        selected_features = ranked_features[:n_features]

    shifted = _apply_shift_to_test_only(
        x_test=x_test,
        features_to_shift=selected_features,
        shift_type=shift_type,
        severity=severity_value,
        rng=rng,
        feature_scales=feature_scales,
    )
    return shifted, selected_features


def generate_shifts_for_dataset(
    artifact_path: str | Path,
    output_root: str | Path = "data/shifted",
    config: ShiftConfig | None = None,
) -> dict[str, Path]:
    """Generate test-only shift variants for one dataset artifact."""
    cfg = config or ShiftConfig()
    for shift_type in cfg.shift_types:
        if shift_type not in SHIFT_TYPES:
            raise ValueError(f"Unsupported shift type: {shift_type}")

    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    with path.open("rb") as file:
        payload = pickle.load(file)

    x_test_frames = _get_test_frames(payload)

    dataset_name = str(payload.get("dataset_name", path.stem.replace("_selected", "")))
    output_dir = Path(output_root) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for source_test_key, x_test in x_test_frames.items():
        if x_test.empty:
            raise ValueError(
                f"Test matrix is empty for source '{source_test_key}' in artifact: {path}"
            )

        all_features = list(x_test.columns)
        ranked_features = _importance_ranking_from_payload(x_test, payload)
        feature_scales = _resolve_feature_scales(payload, all_features)

        for shift_type in cfg.shift_types:
            for severity in cfg.severities:
                severity_value = float(severity)
                n_features = max(
                    1,
                    min(len(all_features), math.ceil(len(all_features) * severity_value)),
                )

                shift_offset = {
                    "random": 0,
                    "important": 1000,
                    "missing": 2000,
                }[shift_type]
                seed = cfg.random_state + int(severity_value * 100) + shift_offset
                rng = np.random.default_rng(seed)

                if shift_type in {"random", "missing"}:
                    selected_features = rng.choice(
                        all_features,
                        size=n_features,
                        replace=False,
                    ).tolist()
                else:
                    selected_features = ranked_features[:n_features]

                x_test_shifted = _apply_shift_to_test_only(
                    x_test=x_test,
                    features_to_shift=selected_features,
                    shift_type=shift_type,
                    severity=severity_value,
                    rng=rng,
                    feature_scales=feature_scales,
                )

                variation = f"{shift_type}_severity_{int(severity_value * 100):03d}"
                variation_key = f"{variation}__{source_test_key}"
                output_path = _get_non_overwriting_path(
                    output_dir / f"{variation_key}.pkl"
                )

                shifted_payload = {
                    "dataset_name": dataset_name,
                    "source_artifact": str(path),
                    "source_test_key": source_test_key,
                    "shift_type": shift_type,
                    "severity": severity_value,
                    "selected_features": selected_features,
                    "x_test_original": x_test.reset_index(drop=True),
                    "x_test_shifted": x_test_shifted.reset_index(drop=True),
                    "y_test": payload.get("y_test"),
                }

                with output_path.open("wb") as file:
                    pickle.dump(shifted_payload, file)
                outputs[variation_key] = output_path

    return outputs


def generate_shifts_for_all(
    input_dir: str | Path = "data/processed",
    output_root: str | Path = "data/shifted",
    config: ShiftConfig | None = None,
) -> dict[str, dict[str, Path]]:
    """Generate shift variants for all available selected and engineered artifacts.

    The generation order is important per dataset:
    1) *_selected.pkl (provides x_test_selected and compatibility views)
    2) *_features.pkl (provides full-width x_test_fe for higher feature-count runs)

    Because files are written with a non-overwrite policy, later artifacts in this
    order become the latest persisted source that the benchmark runner resolves.
    """
    cfg = config or ShiftConfig()

    input_path = Path(input_dir)
    selected_candidates = sorted(input_path.glob("*_selected.pkl"))
    feature_candidates = sorted(input_path.glob("*_features.pkl"))

    if not selected_candidates and not feature_candidates:
        raise FileNotFoundError(
            f"No *_selected.pkl or *_features.pkl artifacts found in: {input_dir}"
        )

    generation_plan: dict[str, list[Path]] = {}

    for candidate in selected_candidates:
        dataset_name = candidate.stem.replace("_selected", "")
        generation_plan.setdefault(dataset_name, []).append(candidate)

        feature_path = input_path / f"{dataset_name}_features.pkl"
        if feature_path.exists():
            generation_plan[dataset_name].append(feature_path)

    for candidate in feature_candidates:
        dataset_name = candidate.stem.replace("_features", "")
        if dataset_name not in generation_plan:
            generation_plan[dataset_name] = [candidate]

    outputs: dict[str, dict[str, Path]] = {}
    for dataset_name in sorted(generation_plan):
        dataset_outputs: dict[str, Path] = {}
        for artifact_path in generation_plan[dataset_name]:
            generated = generate_shifts_for_dataset(
                artifact_path=artifact_path,
                output_root=output_root,
                config=cfg,
            )
            dataset_outputs.update(generated)
        outputs[dataset_name] = dataset_outputs

    return outputs


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for shift generation stage."""
    parser = argparse.ArgumentParser(description="Generate test-only shifted datasets")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-root", default="data/shifted")
    parser.add_argument("--shift-types", default="random,important,missing")
    parser.add_argument("--severities", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    shift_types = tuple(
        part.strip() for part in args.shift_types.split(",") if part.strip()
    )
    severities = tuple(
        float(part.strip()) for part in args.severities.split(",") if part.strip()
    )
    config = ShiftConfig(
        severities=severities,
        shift_types=shift_types,
        random_state=args.random_state,
    )
    results = generate_shifts_for_all(
        input_dir=args.input_dir,
        output_root=args.output_root,
        config=config,
    )
    for dataset_name, variations in results.items():
        for variation, path in variations.items():
            print(f"{dataset_name}/{variation}: {path}")
