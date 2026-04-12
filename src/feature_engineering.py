"""Feature engineering helpers, including Featuretools DFS expansion."""

from __future__ import annotations

import argparse
import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DFSConfig:
    """Configuration for DFS feature generation."""

    depth: int = 2
    max_features: int = 100
    max_base_features: int = 40
    monitor_ram: bool = True


def _get_process_ram_mb() -> float | None:
    """Get current process RAM usage in MB when psutil is available."""
    try:
        import psutil
    except ModuleNotFoundError:
        return None

    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def _build_entityset(df: pd.DataFrame, entityset_id: str):
    """Build a Featuretools EntitySet for a flat dataframe."""
    try:
        import featuretools as ft
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "featuretools is required for DFS. Add it to requirements and install it."
        ) from exc

    working_df = df.reset_index(drop=True).copy()
    working_df.insert(0, "__row_id", range(len(working_df)))

    entityset = ft.EntitySet(id=entityset_id)
    return entityset.add_dataframe(
        dataframe_name="samples",
        dataframe=working_df,
        index="__row_id",
    )


def _top_variance_columns(df: pd.DataFrame, max_columns: int) -> list[str]:
    """Select top-variance columns to control DFS expansion size."""
    if max_columns <= 0:
        raise ValueError("max_columns must be > 0")

    if df.shape[1] <= max_columns:
        return list(df.columns)

    cleaned = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    variance = cleaned.var(axis=0, numeric_only=True).fillna(0.0)
    if variance.empty:
        return list(df.columns[:max_columns])

    ranked = variance.sort_values(ascending=False).index.tolist()
    return ranked[:max_columns]


def _limit_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Limit output matrix width to avoid feature explosion."""
    if max_features <= 0:
        raise ValueError("max_features must be > 0")

    if train_df.shape[1] <= max_features:
        selected_cols = list(train_df.columns)
        return train_df, test_df, selected_cols

    cleaned = train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    variance = cleaned.var(axis=0, numeric_only=True).fillna(0.0)
    ranked = variance.sort_values(ascending=False).index.tolist()
    selected_cols = ranked[:max_features]

    if len(selected_cols) < max_features:
        for col in train_df.columns:
            if col not in selected_cols:
                selected_cols.append(col)
            if len(selected_cols) == max_features:
                break

    return train_df[selected_cols], test_df[selected_cols], selected_cols


def expand_features_with_dfs(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    config: DFSConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Expand train/test features with DFS depth=2 and controlled output width."""
    cfg = config or DFSConfig()

    try:
        import featuretools as ft
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "featuretools is required for DFS. Install dependency and retry."
        ) from exc

    numeric_train = x_train.select_dtypes(include=["number", "bool"]).copy()
    numeric_test = x_test.select_dtypes(include=["number", "bool"]).copy()

    if numeric_train.empty:
        raise ValueError("No numeric columns available for DFS feature generation")

    base_columns = _top_variance_columns(numeric_train, cfg.max_base_features)
    train_base = numeric_train[base_columns].copy()
    test_base = numeric_test[base_columns].copy()

    ram_before = _get_process_ram_mb() if cfg.monitor_ram else None

    train_entityset = _build_entityset(train_base, entityset_id="train_es")
    trans_primitives = [
        "add_numeric",
        "subtract_numeric",
        "multiply_numeric",
        "divide_numeric",
    ]

    # DFS depth > 1 requires multi-table relationships. For this benchmark's
    # flat tabular inputs, Featuretools effectively supports depth 1.
    effective_depth = 1 if cfg.depth > 1 else cfg.depth

    train_feature_matrix, feature_defs = ft.dfs(
        entityset=train_entityset,
        target_dataframe_name="samples",
        trans_primitives=trans_primitives,
        max_depth=effective_depth,
        verbose=False,
    )
    train_feature_matrix = (
        train_feature_matrix.drop(columns=["__row_id"], errors="ignore")
        .reset_index(drop=True)
        .fillna(0.0)
    )

    test_entityset = _build_entityset(test_base, entityset_id="test_es")
    test_feature_matrix = ft.calculate_feature_matrix(
        features=feature_defs,
        entityset=test_entityset,
        verbose=False,
    )
    test_feature_matrix = (
        test_feature_matrix.drop(columns=["__row_id"], errors="ignore")
        .reset_index(drop=True)
        .fillna(0.0)
    )

    features_before_limit = int(train_feature_matrix.shape[1])
    train_limited, test_limited, selected_features = _limit_features(
        train_df=train_feature_matrix,
        test_df=test_feature_matrix,
        max_features=cfg.max_features,
    )

    ram_after = _get_process_ram_mb() if cfg.monitor_ram else None

    metadata: dict[str, Any] = {
        "requested_depth": cfg.depth,
        "effective_depth": effective_depth,
        "max_base_features": cfg.max_base_features,
        "max_features": cfg.max_features,
        "base_features_used": len(base_columns),
        "features_before_limit": features_before_limit,
        "features_after_limit": int(train_limited.shape[1]),
        "selected_features": selected_features,
        "ram_before_mb": ram_before,
        "ram_after_mb": ram_after,
    }

    return train_limited, test_limited, metadata


def engineer_processed_dataset(
    processed_path: str | Path,
    output_dir: str | Path = "data/processed",
    config: DFSConfig | None = None,
) -> Path:
    """Run DFS feature engineering on one processed dataset artifact."""
    cfg = config or DFSConfig()
    path = Path(processed_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    with path.open("rb") as file:
        payload = pickle.load(file)

    required_keys = {"x_train", "x_test", "y_train", "y_test"}
    missing_keys = required_keys.difference(payload.keys())
    if missing_keys:
        raise KeyError(
            f"Processed payload missing keys: {sorted(missing_keys)} in {path.name}"
        )

    x_train_fe, x_test_fe, metadata = expand_features_with_dfs(
        x_train=payload["x_train"],
        x_test=payload["x_test"],
        config=cfg,
    )

    payload["x_train_fe"] = x_train_fe
    payload["x_test_fe"] = x_test_fe
    payload["feature_engineering"] = {
        "method": "featuretools_dfs",
        "metadata": metadata,
    }

    output_path = Path(output_dir) / f"{path.stem}_features.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(payload, file)

    return output_path


def engineer_all_processed_datasets(
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/processed",
    config: DFSConfig | None = None,
) -> dict[str, Path]:
    """Run DFS feature engineering on all processed dataset artifacts."""
    cfg = config or DFSConfig()
    processed_paths = [
        path
        for path in sorted(Path(processed_dir).glob("*.pkl"))
        if not path.stem.endswith("_features") and not path.stem.endswith("_selected")
    ]
    if not processed_paths:
        raise FileNotFoundError(f"No processed .pkl files found in: {processed_dir}")

    feature_paths: dict[str, Path] = {}
    for processed_path in processed_paths:
        feature_paths[processed_path.stem] = engineer_processed_dataset(
            processed_path=processed_path,
            output_dir=output_dir,
            config=cfg,
        )

    return feature_paths


def add_interaction_terms(
    df: pd.DataFrame,
    max_terms: int = 10,
) -> pd.DataFrame:
    """Create pairwise interaction features for numeric columns."""
    engineered = df.copy()
    numeric_cols = list(engineered.select_dtypes(include=["number"]).columns)

    pairs = itertools.islice(itertools.combinations(numeric_cols, 2), max_terms)
    for left, right in pairs:
        engineered[f"int_{left}_x_{right}"] = engineered[left] * engineered[right]

    return engineered


def add_ratio_features(
    df: pd.DataFrame,
    epsilon: float = 1e-6,
    max_terms: int = 10,
) -> pd.DataFrame:
    """Create pairwise ratio features for numeric columns."""
    engineered = df.copy()
    numeric_cols = list(engineered.select_dtypes(include=["number"]).columns)

    pairs = itertools.islice(itertools.combinations(numeric_cols, 2), max_terms)
    for num_col, den_col in pairs:
        engineered[f"ratio_{num_col}_over_{den_col}"] = (
            engineered[num_col] / (engineered[den_col].abs() + epsilon)
        )

    return engineered


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for DFS feature engineering."""
    parser = argparse.ArgumentParser(description="Run DFS feature engineering")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-base-features", type=int, default=40)
    parser.add_argument("--disable-ram-monitor", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = DFSConfig(
        depth=args.depth,
        max_features=args.max_features,
        max_base_features=args.max_base_features,
        monitor_ram=not args.disable_ram_monitor,
    )
    results = engineer_all_processed_datasets(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        config=config,
    )
    for dataset_name, path in results.items():
        print(f"{dataset_name}: {path}")
