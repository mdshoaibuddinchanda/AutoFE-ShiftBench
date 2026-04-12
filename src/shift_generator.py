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


def _get_test_frame(payload: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Extract test feature matrix from artifact payload."""
    for key in ("x_test_selected", "x_test_fe", "x_test"):
        value = payload.get(key)
        if isinstance(value, pd.DataFrame):
            return value.copy(), key
    raise KeyError("Payload must contain one of x_test_selected/x_test_fe/x_test")


def _importance_ranking_from_payload(x_test: pd.DataFrame, payload: dict[str, Any]) -> list[str]:
    """Build feature ranking from payload metadata or test variance fallback."""
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
            return ranked

    numeric = x_test.select_dtypes(include=["number", "bool"])
    if not numeric.empty:
        cleaned = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ranked = cleaned.var(axis=0, numeric_only=True).fillna(0.0).sort_values(
            ascending=False
        )
        return ranked.index.tolist()

    return list(x_test.columns)


def _apply_numeric_noise(
    series: pd.Series,
    severity: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Apply Gaussian noise corruption to a numeric column."""
    numeric = pd.to_numeric(series, errors="coerce")
    std = float(numeric.std(ddof=0))
    base_scale = std if std > 0 else 1.0
    noise = rng.normal(loc=0.0, scale=base_scale * severity, size=len(series))
    return numeric.fillna(numeric.mean()).to_numpy() + noise


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
            shifted[feature] = _apply_numeric_noise(series, severity, rng)
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

    x_test, source_test_key = _get_test_frame(payload)
    if x_test.empty:
        raise ValueError(f"Test matrix is empty in artifact: {path}")

    ranked_features = _importance_ranking_from_payload(x_test, payload)
    all_features = list(x_test.columns)

    dataset_name = str(payload.get("dataset_name", path.stem.replace("_selected", "")))
    output_dir = Path(output_root) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for shift_type in cfg.shift_types:
        for severity in cfg.severities:
            severity_value = float(severity)
            n_features = max(1, min(len(all_features), math.ceil(len(all_features) * severity_value)))

            shift_offset = {
                "random": 0,
                "important": 1000,
                "missing": 2000,
            }[shift_type]
            seed = cfg.random_state + int(severity_value * 100) + shift_offset
            rng = np.random.default_rng(seed)

            if shift_type in {"random", "missing"}:
                selected_features = rng.choice(all_features, size=n_features, replace=False).tolist()
            else:
                selected_features = ranked_features[:n_features]

            x_test_shifted = _apply_shift_to_test_only(
                x_test=x_test,
                features_to_shift=selected_features,
                shift_type=shift_type,
                severity=severity_value,
                rng=rng,
            )

            variation = f"{shift_type}_severity_{int(severity_value * 100):03d}"
            output_path = _get_non_overwriting_path(output_dir / f"{variation}.pkl")

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
            outputs[variation] = output_path

    return outputs


def generate_shifts_for_all(
    input_dir: str | Path = "data/processed",
    output_root: str | Path = "data/shifted",
    config: ShiftConfig | None = None,
) -> dict[str, dict[str, Path]]:
    """Generate shift variants for all available selected/engineered artifacts."""
    cfg = config or ShiftConfig()

    candidates = sorted(Path(input_dir).glob("*_selected.pkl"))
    if not candidates:
        candidates = sorted(Path(input_dir).glob("*_features.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_selected.pkl or *_features.pkl artifacts found in: {input_dir}"
        )

    outputs: dict[str, dict[str, Path]] = {}
    for candidate in candidates:
        dataset_name = candidate.stem.replace("_selected", "").replace("_features", "")
        outputs[dataset_name] = generate_shifts_for_dataset(
            artifact_path=candidate,
            output_root=output_root,
            config=cfg,
        )

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
