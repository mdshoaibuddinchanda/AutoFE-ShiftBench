"""Feature engineering helpers, including Featuretools DFS expansion and ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import itertools
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


ARITHMETIC_PRIMITIVES: tuple[str, ...] = (
    "add_numeric",
    "subtract_numeric",
    "multiply_numeric",
    "divide_numeric",
)
_PRIMITIVE_ALIASES = {
    "add": "add_numeric", "add_numeric": "add_numeric",
    "subtract": "subtract_numeric", "subtract_numeric": "subtract_numeric",
    "multiply": "multiply_numeric", "multiply_numeric": "multiply_numeric",
    "divide": "divide_numeric", "divide_numeric": "divide_numeric",
}


def _arithmetic_candidate_counts(
    generated_features: list[dict[str, Any]],
    raw_train_feature_matrix: pd.DataFrame,
    selected_cols: list[str],
) -> dict[str, dict[str, int]]:
    """Count generated, duplicate, finite-eligible, rejected, and selected candidates."""
    counts = {
        operator: {"generated": 0, "rejected": 0, "eligible": 0, "selected": 0, "duplicates": 0}
        for operator in ARITHMETIC_PRIMITIVES
    }
    selected_set = set(selected_cols)
    seen_candidates: set[tuple[str, tuple[str, ...]]] = set()
    for feature in generated_features:
        operator = _PRIMITIVE_ALIASES.get(str(feature.get("primitive", "")).lower())
        if operator is None:
            continue
        parents = tuple(str(item) for item in feature.get("parents", ()))
        identity_parents = tuple(sorted(parents)) if operator in {"add_numeric", "multiply_numeric"} else parents
        identity = (operator, identity_parents)
        counts[operator]["generated"] += 1
        if identity in seen_candidates:
            counts[operator]["duplicates"] += 1
        seen_candidates.add(identity)
        values = pd.to_numeric(raw_train_feature_matrix[str(feature["name"])], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).all():
            counts[operator]["eligible"] += 1
            if str(feature["name"]) in selected_set:
                counts[operator]["selected"] += 1
        else:
            counts[operator]["rejected"] += 1
    return counts


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
        if not isinstance(y_train, pd.Series):
            raise TypeError("MI selection requires y_train as a pandas Series so row alignment can be verified")
        if len(y_train) != len(cleaned):
            raise ValueError(f"MI alignment error (length mismatch): {len(y_train)} labels for {len(cleaned)} feature rows")
        if not cleaned.index.equals(y_train.index):
            raise ValueError("MI alignment error: y_train index/order does not match the training feature rows")
        mi_scores = mutual_info_classif(cleaned, y_train.to_numpy(), random_state=cfg.random_seed)
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
    *,
    audit_context: dict[str, Any] | None = None,
    candidate_history_writer: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = config or DFSConfig()
    enabled_operators = tuple(str(item) for item in cfg.trans_primitives)
    unknown_operators = sorted(set(enabled_operators).difference(ARITHMETIC_PRIMITIVES))
    if unknown_operators:
        raise ValueError(f"Unsupported arithmetic primitives: {unknown_operators}")
    if len(set(enabled_operators)) != len(enabled_operators):
        raise ValueError("trans_primitives must not contain duplicate operators")
    
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
            "feature_metadata": [{"name": c, "primitive": "raw", "parents": [], "depth": 0} for c in selected],
            "operator_configuration": {
                "enabled_operators": list(enabled_operators),
                "excluded_operators": [item for item in ARITHMETIC_PRIMITIVES if item not in enabled_operators],
                "candidate_counts": {item: {"generated": 0, "rejected": 0, "eligible": 0, "selected": 0, "duplicates": 0} for item in ARITHMETIC_PRIMITIVES},
            },
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
    
    train_feature_matrix = train_feature_matrix.drop(columns=["__row_id"], errors="ignore").reset_index(drop=True)
    raw_train_feature_matrix = train_feature_matrix.copy()
    train_feature_matrix = train_feature_matrix.fillna(0.0)

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

    # Reject arithmetic candidates with non-finite training values before
    # selection.  Filling NaNs with zero is retained for valid downstream
    # matrices, but an invalid candidate must not become selectable merely
    # because its missing values were imputed to zero.
    invalid_candidate_names: set[str] = set()
    for feature in generated_features:
        operator = _PRIMITIVE_ALIASES.get(str(feature.get("primitive", "")).lower())
        if operator is None:
            continue
        name = str(feature["name"])
        raw_values = pd.to_numeric(raw_train_feature_matrix[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(raw_values).all():
            invalid_candidate_names.add(name)
    train_for_selection = train_feature_matrix.drop(columns=sorted(invalid_candidate_names), errors="ignore")
    test_for_selection = test_feature_matrix.drop(columns=sorted(invalid_candidate_names), errors="ignore")
    train_out, test_out, selected_cols = _limit_features(train_for_selection, test_for_selection, y_train, cfg)
    
    retained_meta = [g for g in generated_features if g["name"] in selected_cols]

    # Optional candidate ledger.  Every score is computed from the training
    # matrix only; the test feature matrix is never consulted for selection.
    # The writer is deliberately opt-in because a full benchmark can generate
    # many gigabytes of JSONL history.
    history_records = 0
    history_operator_counts: dict[str, int] = {}
    history_score_status: dict[str, int] = {"finite": 0, "undefined": 0, "not_scored": 0}
    if candidate_history_writer is not None:
        from src.mechanism_audit import CandidateHistoryRecord

        context = dict(audit_context or {})
        required = {"dataset", "split_policy", "seed", "fold", "condition"}
        missing = sorted(required.difference(context))
        if missing:
            raise ValueError(f"candidate history context is missing: {missing}")
        selected_set = set(selected_cols)
        for iteration, feature in enumerate(generated_features):
            primitive = str(feature.get("primitive", "")).lower()
            operator = _PRIMITIVE_ALIASES.get(primitive)
            if operator is None:
                continue
            parents = tuple(str(item) for item in feature.get("parents", ()))
            candidate_id = hashlib.sha256(json.dumps({
                "name": feature["name"], "parents": parents, "operator": operator,
            }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
            raw_values = pd.to_numeric(raw_train_feature_matrix[feature["name"]], errors="coerce").to_numpy(dtype=float)
            values = pd.to_numeric(train_feature_matrix[feature["name"]], errors="coerce").to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                score = float(np.nanvar(values)) if np.isfinite(values).any() else None
            score_status = "finite" if score is not None and np.isfinite(score) else "undefined"
            admissible = bool(np.isfinite(raw_values).all())
            rejection_reason = None if admissible else "nonfinite_candidate_value"
            if not admissible:
                score = None
                score_status = "undefined"
            if not admissible:
                decision = "rejected"
            elif feature["name"] in selected_set:
                decision = "selected"
            else:
                decision = "not_selected"
            if score is None:
                decision = "rejected" if not admissible else "not_selected"
                score_status = "undefined"
            record = CandidateHistoryRecord(
                dataset=str(context["dataset"]), split_policy=str(context["split_policy"]),
                seed=int(context["seed"]), fold=int(context["fold"]),
                condition=str(context["condition"]), iteration=int(iteration),
                candidate_id=candidate_id, parent_features=parents, operator=operator,
                admissible=admissible, rejection_reason=rejection_reason,
                selection_score=score, selection_score_status=score_status,
                selection_decision=decision,
                candidate_seed=int(context.get("candidate_seed", cfg.random_seed)),
            )
            candidate_history_writer.append(record)
            history_records += 1
            history_operator_counts[operator] = history_operator_counts.get(operator, 0) + 1
            history_score_status[score_status] += 1

    candidate_counts = _arithmetic_candidate_counts(
        generated_features, raw_train_feature_matrix, selected_cols,
    )

    metadata = {
        "n_generated": len(feature_defs),
        "n_retained": len(selected_cols),
        "ram_used_mb": (_get_process_ram_mb() - ram_before) if ram_before else 0,
        "feature_metadata": retained_meta,
        "generated_feature_metadata_count": len(generated_features),
        "operator_counts": {operator: counts["generated"] for operator, counts in candidate_counts.items()},
        "operator_candidate_counts": candidate_counts,
        "operator_configuration": {
            "enabled_operators": list(enabled_operators),
            "excluded_operators": [item for item in ARITHMETIC_PRIMITIVES if item not in enabled_operators],
            "candidate_counts": candidate_counts,
        },
        "candidate_history": {
            "enabled": candidate_history_writer is not None,
            "records_written": history_records,
            "operator_counts": history_operator_counts,
            "selection_score_status": history_score_status,
            "score_scope": "train",
        },
    }

    return train_out, test_out, metadata
