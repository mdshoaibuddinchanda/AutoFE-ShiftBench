"""Feature engineering helpers, including Featuretools DFS expansion and ablations."""

from __future__ import annotations

import argparse
import itertools
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


@dataclass(slots=True)
class DFSConfig:
    """Configuration for DFS feature generation and ablations."""
    enable_dfs: bool = True
    depth: int = 1
    max_features: int | None = 100
    max_base_features: int | None = 20
    selection_method: str = "variance"  # "variance", "mi", "random", "none"
    trans_primitives: list[str] = field(
        default_factory=lambda: [
            "add_numeric",
            "subtract_numeric",
            "multiply_numeric",
            "divide_numeric",
        ]
    )
    monitor_ram: bool = True
    random_seed: int = 42

def _get_process_ram_mb() -> float | None:
    try:
        import psutil
    except ModuleNotFoundError:
        return None
    return float(psutil.Process().memory_info().rss / (1024 * 1024))

def _build_entityset(df: pd.DataFrame, entityset_id: str):
    try:
        import featuretools as ft
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("featuretools is required.") from exc

    working_df = df.reset_index(drop=True).copy()
    working_df.insert(0, "__row_id", range(len(working_df)))
    entityset = ft.EntitySet(id=entityset_id)
    return entityset.add_dataframe(
        dataframe_name="samples",
        dataframe=working_df,
        index="__row_id",
    )

def _limit_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray | None,
    cfg: DFSConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if cfg.selection_method == "none" or cfg.max_features is None or train_df.shape[1] <= cfg.max_features:
        return train_df, test_df, list(train_df.columns)

    cleaned = train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    if cfg.selection_method == "variance":
        variance = cleaned.var(axis=0, numeric_only=True).fillna(0.0)
        ranked = variance.sort_values(ascending=False).index.tolist()
        
    elif cfg.selection_method == "mi":
        if y_train is None:
            raise ValueError("y_train is required for MI selection.")
        # Ensure y_train matches cleaned length
        if len(y_train) != len(cleaned):
            y_train = y_train[:len(cleaned)]
        mi_scores = mutual_info_classif(cleaned, y_train, random_state=cfg.random_seed)
        mi_series = pd.Series(mi_scores, index=cleaned.columns)
        ranked = mi_series.sort_values(ascending=False).index.tolist()
        
    elif cfg.selection_method == "random":
        # Average random ranking over 10 draws
        rng = np.random.default_rng(cfg.random_seed)
        cols = list(cleaned.columns)
        draws = []
        for _ in range(10):
            d = cols.copy()
            rng.shuffle(d)
            draws.append(d)
        
        rank_scores = {c: 0 for c in cols}
        for d in draws:
            for i, c in enumerate(d):
                rank_scores[c] += i
        ranked = sorted(cols, key=lambda c: rank_scores[c])
    else:
        raise ValueError(f"Unknown selection method {cfg.selection_method}")

    selected_cols = ranked[:cfg.max_features]
    return train_df[selected_cols], test_df[selected_cols], selected_cols

def expand_features_with_dfs(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray | None = None,
    config: DFSConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = config or DFSConfig()
    
    ram_before = _get_process_ram_mb() if cfg.monitor_ram else None

    numeric_train = x_train.select_dtypes(include=["number", "bool"]).copy()
    numeric_test = x_test.select_dtypes(include=["number", "bool"]).copy()

    if numeric_train.empty:
        raise ValueError("No numeric columns available.")

    # Base feature selection if needed
    if cfg.max_base_features and numeric_train.shape[1] > cfg.max_base_features:
        # We always use variance for base selection to prevent DFS explosion unless otherwise specified
        variance = numeric_train.var(axis=0, numeric_only=True).fillna(0.0)
        base_columns = variance.sort_values(ascending=False).index.tolist()[:cfg.max_base_features]
    else:
        base_columns = list(numeric_train.columns)

    train_base = numeric_train[base_columns].copy()
    test_base = numeric_test[base_columns].copy()

    if not cfg.enable_dfs:
        # Just selection control
        train_out, test_out, selected = _limit_features(train_base, test_base, y_train, cfg)
        metadata = {
            "n_generated": 0,
            "n_retained": len(selected),
            "ram_used_mb": (_get_process_ram_mb() - ram_before) if ram_before else 0,
            "feature_metadata": [{"name": c, "primitive": "raw", "parents": [], "depth": 0} for c in selected]
        }
        return train_out, test_out, metadata

    import featuretools as ft

    train_entityset = _build_entityset(train_base, entityset_id="train_es")
    
    train_feature_matrix, feature_defs = ft.dfs(
        entityset=train_entityset,
        target_dataframe_name="samples",
        trans_primitives=cfg.trans_primitives,
        max_depth=cfg.depth,
        verbose=False,
    )
    
    train_feature_matrix = train_feature_matrix.drop(columns=["__row_id"], errors="ignore").reset_index(drop=True).fillna(0.0)

    test_entityset = _build_entityset(test_base, entityset_id="test_es")
    test_feature_matrix = ft.calculate_feature_matrix(
        features=feature_defs,
        entityset=test_entityset,
        verbose=False,
    )
    test_feature_matrix = test_feature_matrix.drop(columns=["__row_id"], errors="ignore").reset_index(drop=True).fillna(0.0)

    # Feature Metadata
    generated_features = []
    for f in feature_defs:
        try:
            prim = f.primitive.name if hasattr(f, 'primitive') and f.primitive else "raw"
            parents = [p.get_name() for p in f.base_features] if hasattr(f, 'base_features') else []
            depth = f.get_depth()
        except:
            prim = "unknown"
            parents = []
            depth = 1
        generated_features.append({"name": f.get_name(), "primitive": prim, "parents": parents, "depth": depth})

    train_out, test_out, selected_cols = _limit_features(train_feature_matrix, test_feature_matrix, y_train, cfg)
    
    retained_meta = [g for g in generated_features if g["name"] in selected_cols]

    metadata = {
        "n_generated": len(feature_defs),
        "n_retained": len(selected_cols),
        "ram_used_mb": (_get_process_ram_mb() - ram_before) if ram_before else 0,
        "feature_metadata": retained_meta
    }

    return train_out, test_out, metadata
