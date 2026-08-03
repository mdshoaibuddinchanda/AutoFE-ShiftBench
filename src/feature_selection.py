"""Feature selection using L1 regularization and mutual information."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import Lasso, LogisticRegression


@dataclass(slots=True)
class FeatureSelectionConfig:
    """Configuration for feature selection."""

    max_features: int = 100
    l1_strength: float = 0.1
    random_state: int = 42
    task: str | None = None


def _infer_task(y: pd.Series) -> str:
    """Infer task type from target properties."""
    if y.dtype.kind in {"O", "b", "U", "S"}:
        return "classification"

    unique_values = y.nunique(dropna=False)
    threshold = max(20, int(0.1 * len(y)))
    if unique_values <= threshold:
        return "classification"
    return "regression"


def _encode_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all columns numerically for score computation."""
    encoded = pd.DataFrame(index=df.index)
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
            fill_value = float(numeric_series.mean()) if numeric_series.notna().any() else 0.0
            encoded[column] = numeric_series.fillna(fill_value).clip(-1e12, 1e12)
        else:
            categories, _ = pd.factorize(series.fillna("missing"), sort=True)
            encoded[column] = categories.astype(float)
    return encoded


def _normalize_scores(scores: pd.Series) -> pd.Series:
    """Min-max normalize score series to [0, 1]."""
    if scores.empty:
        return scores
    min_value = float(scores.min())
    max_value = float(scores.max())
    if np.isclose(min_value, max_value):
        return pd.Series(np.zeros(len(scores)), index=scores.index)
    return (scores - min_value) / (max_value - min_value)


def _compute_l1_scores(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    task: str,
    l1_strength: float,
    random_state: int,
) -> pd.Series:
    """Compute feature importance scores from L1-regularized models."""
    x_encoded = _encode_feature_frame(x_train)

    if task == "classification":
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=l1_strength,
            max_iter=2000,
            random_state=random_state,
        )
        model.fit(x_encoded, y_train)
        coefficients = np.abs(np.asarray(model.coef_))
        score_values = coefficients.mean(axis=0)
    else:
        alpha = max(1e-4, 1.0 / max(l1_strength, 1e-4))
        model = Lasso(alpha=alpha, random_state=random_state, max_iter=5000)
        model.fit(x_encoded, y_train)
        score_values = np.abs(np.asarray(model.coef_))

    return pd.Series(score_values, index=x_train.columns, dtype=float)


def _compute_mi_scores(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    task: str,
    random_state: int,
) -> pd.Series:
    """Compute feature importance scores from mutual information."""
    x_encoded = _encode_feature_frame(x_train)
    if task == "classification":
        score_values = mutual_info_classif(
            x_encoded,
            y_train,
            random_state=random_state,
        )
    else:
        score_values = mutual_info_regression(
            x_encoded,
            y_train,
            random_state=random_state,
        )

    return pd.Series(score_values, index=x_train.columns, dtype=float)


def select_top_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    config: FeatureSelectionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select top features using combined L1 and MI importance."""
    cfg = config or FeatureSelectionConfig()
    if cfg.max_features <= 0:
        raise ValueError("max_features must be > 0")

    task = cfg.task or _infer_task(y_train)
    l1_scores = _compute_l1_scores(
        x_train=x_train,
        y_train=y_train,
        task=task,
        l1_strength=cfg.l1_strength,
        random_state=cfg.random_state,
    )
    mi_scores = _compute_mi_scores(
        x_train=x_train,
        y_train=y_train,
        task=task,
        random_state=cfg.random_state,
    )

    combined_scores = (
        0.5 * _normalize_scores(l1_scores) + 0.5 * _normalize_scores(mi_scores)
    )
    selected_features = combined_scores.sort_values(ascending=False).head(
        cfg.max_features
    ).index.tolist()

    x_train_selected = (
        x_train[selected_features]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )
    x_test_selected = (
        x_test[selected_features]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )

    train_fill_values = x_train_selected.mean(axis=0, numeric_only=True)
    x_train_selected = x_train_selected.fillna(train_fill_values).fillna(0.0)
    x_test_selected = x_test_selected.fillna(train_fill_values).fillna(0.0)

    metadata: dict[str, Any] = {
        "task": task,
        "max_features": cfg.max_features,
        "l1_strength": cfg.l1_strength,
        "methods": ["l1_regularization", "mutual_information"],
        "num_features_in": int(x_train.shape[1]),
        "num_features_out": int(len(selected_features)),
        "selected_features": selected_features,
        "l1_scores": l1_scores[selected_features].to_dict(),
        "mi_scores": mi_scores[selected_features].to_dict(),
        "combined_scores": combined_scores[selected_features].to_dict(),
    }

    return x_train_selected, x_test_selected, metadata


def _extract_feature_matrices(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract feature matrices with engineered-feature preference."""
    if "x_train_fe" in payload and "x_test_fe" in payload:
        return payload["x_train_fe"], payload["x_test_fe"]
    if "x_train" in payload and "x_test" in payload:
        return payload["x_train"], payload["x_test"]
    raise KeyError("Payload must contain x_train/x_test or x_train_fe/x_test_fe")


def select_features_for_dataset(
    engineered_path: str | Path,
    output_dir: str | Path = "data/processed",
    config: FeatureSelectionConfig | None = None,
) -> Path:
    """Run feature selection for one engineered dataset artifact."""
    cfg = config or FeatureSelectionConfig()
    path = Path(engineered_path)
    if not path.exists():
        raise FileNotFoundError(f"Engineered artifact not found: {path}")

    with path.open("rb") as file:
        payload = pickle.load(file)

    if "y_train" not in payload:
        raise KeyError(f"Artifact missing y_train: {path}")

    x_train, x_test = _extract_feature_matrices(payload)
    y_train = payload["y_train"]

    x_train_selected, x_test_selected, metadata = select_top_features(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        config=cfg,
    )

    payload["x_train_selected"] = x_train_selected.reset_index(drop=True)
    payload["x_test_selected"] = x_test_selected.reset_index(drop=True)
    payload["selected_feature_names"] = metadata["selected_features"]
    payload["feature_selection"] = {
        "method": "l1_plus_mutual_information",
        "metadata": metadata,
    }

    dataset_stem = path.stem.replace("_features", "")
    output_path = Path(output_dir) / f"{dataset_stem}_selected.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(payload, file)

    return output_path


def select_features_for_all(
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/processed",
    config: FeatureSelectionConfig | None = None,
) -> dict[str, Path]:
    """Run feature selection for all engineered dataset artifacts."""
    cfg = config or FeatureSelectionConfig()
    feature_paths = [
        path
        for path in sorted(Path(processed_dir).glob("*_features.pkl"))
        if "_selected" not in path.stem
    ]
    if not feature_paths:
        raise FileNotFoundError(
            f"No engineered artifacts (*_features.pkl) found in: {processed_dir}"
        )

    selected_paths: dict[str, Path] = {}
    for feature_path in feature_paths:
        dataset_name = feature_path.stem.replace("_features", "")
        selected_paths[dataset_name] = select_features_for_dataset(
            engineered_path=feature_path,
            output_dir=output_dir,
            config=cfg,
        )

    return selected_paths


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for feature selection stage."""
    parser = argparse.ArgumentParser(description="Run feature selection")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--l1-strength", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = FeatureSelectionConfig(
        max_features=args.max_features,
        l1_strength=args.l1_strength,
        random_state=args.random_state,
        task=args.task,
    )
    results = select_features_for_all(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        config=config,
    )
    for dataset_name, output_path in results.items():
        print(f"{dataset_name}: {output_path}")
