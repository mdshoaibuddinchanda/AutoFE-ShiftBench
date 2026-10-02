"""Deterministic arithmetic candidate generation and baseline controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.operator_registry import (
    OPERATOR_REGISTRY,
    OPERATOR_REGISTRY_VERSION,
    OPERATOR_SEMANTICS_VERSION,
    OPERATOR_SET_REGISTRY,
    Expression,
    candidate_id,
    evaluate_operator,
    expression_columns,
    expression_depth,
    expression_operators,
    expression_to_dict,
    expression_to_string,
    op_expression,
    raw_expression,
    validate_operator_set,
    operator_set_manifest,
)


CAP_POLICY_VERSION = "post_candidate_topk_v1"
BASE_FEATURE_POLICY_VERSION = "training_variance_topk_v1"


@dataclass(slots=True)
class DFSConfig:
    """Configuration for raw controls and arithmetic feature generation."""

    enable_dfs: bool = True
    depth: int = 1
    max_features: int | None = 100
    max_base_features: int | None = 20
    selection_method: str = "variance"
    trans_primitives: list[str] | None = None
    monitor_ram: bool = True
    random_seed: int = 42
    operator_set_id: str = "full_arithmetic_v1"
    baseline_kind: str | None = None
    display_identity: str | None = None

    def __post_init__(self) -> None:
        if self.trans_primitives is None:
            self.trans_primitives = list(OPERATOR_SET_REGISTRY[self.operator_set_id])
        else:
            primitive_tuple = tuple(self.trans_primitives)
            if primitive_tuple != tuple(OPERATOR_SET_REGISTRY[self.operator_set_id]):
                matches = [key for key, value in OPERATOR_SET_REGISTRY.items() if value == primitive_tuple]
                if not matches:
                    raise ValueError(f"Primitive list is not a registered operator set: {primitive_tuple}")
                self.operator_set_id = matches[0]


def _get_process_ram_mb() -> float | None:
    try:
        import psutil
    except ModuleNotFoundError:
        return None
    return float(psutil.Process().memory_info().rss / (1024 * 1024))


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=["number", "bool"]).copy()
    if numeric.empty:
        raise ValueError("No numeric columns available.")
    numeric = numeric.astype(float).replace([np.inf, -np.inf], np.nan)
    return numeric.fillna(0.0)


def _base_columns(x_train: pd.DataFrame, cfg: DFSConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric = _numeric_frame(x_train)
    eligible = list(numeric.columns)
    if cfg.max_base_features is None or len(eligible) <= cfg.max_base_features:
        return numeric, eligible, []
    variances = numeric.var(axis=0).fillna(0.0)
    ranked = sorted(eligible, key=lambda col: (-float(variances[col]), str(col)))
    retained = ranked[: cfg.max_base_features]
    excluded = [col for col in eligible if col not in set(retained)]
    return numeric[retained], retained, excluded


def _generate_expressions(columns: list[str], operator_set_id: str, depth: int) -> list[Expression]:
    operators = validate_operator_set(operator_set_id)
    if depth < 0:
        raise ValueError("depth must be non-negative")
    expressions: list[Expression] = [raw_expression(col) for col in columns]
    if not operators or depth == 0:
        return expressions

    by_depth: dict[int, list[Expression]] = {0: list(expressions)}
    for current_depth in range(1, depth + 1):
        left_pool = [expr for d in range(current_depth) for expr in by_depth.get(d, [])]
        right_pool = by_depth[0]
        generated: list[Expression] = []
        for operator in operators:
            spec = OPERATOR_REGISTRY[operator]
            if spec.commutative:
                for left in left_pool:
                    for right in right_pool:
                        if expression_to_string(left) <= expression_to_string(right):
                            generated.append(op_expression(operator, left, right))
            else:
                for left in left_pool:
                    for right in right_pool:
                        generated.append(op_expression(operator, left, right))
        generated.sort(key=candidate_id)
        by_depth[current_depth] = generated
        expressions.extend(generated)
    return expressions


def _evaluate_expression(expression: Expression, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    if expression[0] == "raw":
        return frame[str(expression[1])].to_numpy(dtype=float), {"nonfinite_count": 0, "clipped_count": 0, "protected_count": 0}
    left, left_status = _evaluate_expression(expression[2], frame)
    right, right_status = _evaluate_expression(expression[3], frame)
    values, status = evaluate_operator(str(expression[1]), left, right)
    merged = {key: left_status.get(key, 0) + right_status.get(key, 0) + status.get(key, 0) for key in ("nonfinite_count", "clipped_count", "protected_count")}
    return values, merged


def _score_candidates(matrix: pd.DataFrame, y_train: np.ndarray | None, cfg: DFSConfig) -> dict[str, float | None]:
    method = cfg.selection_method
    if method == "none":
        return {column: None for column in matrix.columns}
    cleaned = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if method == "variance":
        return {column: float(cleaned[column].var(ddof=1)) for column in matrix.columns}
    if method == "mi":
        if y_train is None:
            raise ValueError("y_train is required for MI selection.")
        scores = mutual_info_classif(cleaned, y_train, random_state=cfg.random_seed)
        return {column: float(score) for column, score in zip(matrix.columns, scores)}
    if method == "random":
        rng = np.random.default_rng(cfg.random_seed)
        return {column: float(rng.random()) for column in matrix.columns}
    raise ValueError(f"Unknown selection method {method!r}")


def _metadata_for_expression(expression: Expression, feature_id: str, status: dict[str, int]) -> dict[str, Any]:
    return {
        "feature_id": feature_id,
        "name": expression_to_string(expression),
        "expression": expression_to_dict(expression),
        "parents": list(expression_columns(expression)),
        "operators": list(expression_operators(expression)),
        "depth": expression_depth(expression),
        "validity": status,
    }


def expand_features_with_dfs(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray | None = None,
    config: DFSConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit candidate construction and selection on training rows only."""
    cfg = config or DFSConfig()
    if cfg.operator_set_id not in OPERATOR_SET_REGISTRY:
        raise ValueError(f"Unknown operator set: {cfg.operator_set_id}")
    if tuple(cfg.trans_primitives) != tuple(validate_operator_set(cfg.operator_set_id)):
        raise ValueError("trans_primitives must exactly match the declared operator set")

    ram_before = _get_process_ram_mb() if cfg.monitor_ram else None
    train_base, base_columns, excluded_base = _base_columns(x_train, cfg)
    test_numeric = _numeric_frame(x_test)
    missing = [col for col in base_columns if col not in test_numeric.columns]
    if missing:
        raise ValueError(f"Held-out predictors are missing training columns: {missing}")
    test_base = test_numeric[base_columns]

    expressions = _generate_expressions(base_columns, cfg.operator_set_id if cfg.enable_dfs else "none_v1", cfg.depth if cfg.enable_dfs else 0)
    train_values: dict[str, np.ndarray] = {}
    test_values: dict[str, np.ndarray] = {}
    validity: dict[str, dict[str, int]] = {}
    expression_by_id: dict[str, Expression] = {}
    for expression in expressions:
        fid = candidate_id(expression)
        train_value, train_status = _evaluate_expression(expression, train_base)
        test_value, _test_status = _evaluate_expression(expression, test_base)
        train_values[fid] = train_value
        test_values[fid] = test_value
        validity[fid] = train_status
        expression_by_id[fid] = expression

    train_matrix = pd.DataFrame(train_values, index=train_base.index)
    test_matrix = pd.DataFrame(test_values, index=test_base.index)
    scores = _score_candidates(train_matrix, y_train, cfg)
    ranked = sorted(train_matrix.columns, key=lambda fid: (-(scores[fid] if scores[fid] is not None and np.isfinite(scores[fid]) else -np.inf), fid))
    selected = ranked if cfg.max_features is None else ranked[: cfg.max_features]
    selected = list(selected)
    selected_set = set(selected)

    rank_by_id = {fid: rank for rank, fid in enumerate(ranked, start=1)}
    history: list[dict[str, Any]] = []
    for fid in train_matrix.columns:
        expr = expression_by_id[fid]
        selected_flag = fid in selected_set
        score = scores[fid]
        history.append({
            "candidate_id": fid,
            "expression": expression_to_dict(expr),
            "expression_text": expression_to_string(expr),
            "parents": list(expression_columns(expr)),
            "operators": list(expression_operators(expr)),
            "depth": expression_depth(expr),
            "selection_stage": cfg.selection_method,
            "score": score,
            "eligible": True,
            "evaluated": True,
            "selected": selected_flag,
            "final_retained": selected_flag,
            "candidate_rank": rank_by_id[fid],
            "decision": "selected" if selected_flag else "not_selected_by_cap",
            "rejection_reason": None if selected_flag else "post_candidate_cap",
            "pruned": not selected_flag,
            "validity": validity[fid],
        })
    for column in excluded_base:
        expression = raw_expression(column)
        history.append({
            "candidate_id": candidate_id(expression),
            "expression": expression_to_dict(expression),
            "expression_text": expression_to_string(expression),
            "parents": [column],
            "operators": [],
            "depth": 0,
            "selection_stage": "base_feature_cap",
            "score": None,
            "eligible": False,
            "evaluated": False,
            "selected": False,
            "final_retained": False,
            "candidate_rank": None,
            "decision": "not_generated_by_base_cap",
            "rejection_reason": "base_feature_cap",
            "pruned": True,
            "validity": {},
        })
    history.sort(key=lambda row: row["candidate_id"])

    selected_train = train_matrix[selected].copy()
    selected_test = test_matrix[selected].copy()
    selected_meta = [_metadata_for_expression(expression_by_id[fid], fid, validity[fid]) for fid in selected]
    generated_selected = sum(bool(expression_operators(expression_by_id[fid])) for fid in selected)
    raw_selected = len(selected) - generated_selected
    metadata: dict[str, Any] = {
        "operator_registry_version": OPERATOR_REGISTRY_VERSION,
        "operator_semantics_version": OPERATOR_SEMANTICS_VERSION,
        "operator_set_id": cfg.operator_set_id,
        "operator_set": list(validate_operator_set(cfg.operator_set_id)),
        "operator_set_manifest": operator_set_manifest()[cfg.operator_set_id],
        "baseline_kind": cfg.baseline_kind or ("autofe" if cfg.enable_dfs else "raw"),
        "display_identity": cfg.display_identity,
        "cap_policy_version": CAP_POLICY_VERSION if cfg.max_features is not None else "none_v1",
        "base_feature_policy_version": BASE_FEATURE_POLICY_VERSION if cfg.max_base_features is not None else "none_v1",
        "requested_cap": cfg.max_features,
        "requested_base_cap": cfg.max_base_features,
        "eligible_base_feature_count": len(base_columns) + len(excluded_base),
        "base_feature_count": len(base_columns),
        "candidate_count": len(train_matrix.columns),
        "num_original": len(base_columns),
        "num_generated": sum(bool(expression_operators(expression_by_id[fid])) for fid in train_matrix.columns),
        "num_selected": len(selected),
        "retained_raw_count": raw_selected,
        "retained_generated_count": generated_selected,
        "actual_estimator_input_dimension": len(selected),
        "selector_identity": cfg.selection_method,
        "selected_feature_identities": selected,
        "selected_feature_expressions": selected_meta,
        "selection_history": history,
        "generation_time_s": 0.0,
        "ram_used_mb": (_get_process_ram_mb() - ram_before) if ram_before else 0,
        "feature_metadata": selected_meta,
    }
    return selected_train, selected_test, metadata
