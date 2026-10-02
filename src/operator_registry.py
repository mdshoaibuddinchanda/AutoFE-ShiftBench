"""Versioned arithmetic operators and stable expression identities.

The benchmark only uses the operators declared here.  Expression trees are
represented as nested tuples so candidate generation, evaluation, derivative
calculation, and persisted histories share one executable definition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


OPERATOR_REGISTRY_VERSION = "arithmetic_operator_registry_v1"
OPERATOR_SEMANTICS_VERSION = "safe_division_1e-12_clip_1e12_v1"
SAFE_DIVISION_EPSILON = 1e-12
FINITE_CLIP = 1e12


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    symbol: str
    commutative: bool


OPERATOR_REGISTRY: dict[str, OperatorSpec] = {
    "add_numeric": OperatorSpec("add_numeric", "+", True),
    "subtract_numeric": OperatorSpec("subtract_numeric", "-", False),
    "multiply_numeric": OperatorSpec("multiply_numeric", "*", True),
    "divide_numeric": OperatorSpec("divide_numeric", "/", False),
}

OPERATOR_SET_REGISTRY: dict[str, tuple[str, ...]] = {
    "full_arithmetic_v1": tuple(OPERATOR_REGISTRY),
    "add_sub_v1": ("add_numeric", "subtract_numeric"),
    "add_sub_div_v1": ("add_numeric", "subtract_numeric", "divide_numeric"),
    "add_sub_mul_v1": ("add_numeric", "subtract_numeric", "multiply_numeric"),
    "multiply_only_v1": ("multiply_numeric",),
    "divide_only_v1": ("divide_numeric",),
    "add_only_v1": ("add_numeric",),
    "subtract_only_v1": ("subtract_numeric",),
    "none_v1": (),
}


def operator_set_manifest() -> dict[str, dict[str, Any]]:
    """Return a serializable registry manifest for result and artifact metadata."""
    return {
        set_id: {
            "operator_set_id": set_id,
            "registry_version": OPERATOR_REGISTRY_VERSION,
            "semantics_version": OPERATOR_SEMANTICS_VERSION,
            "operators": list(operators),
        }
        for set_id, operators in OPERATOR_SET_REGISTRY.items()
    }

Expression = tuple[Any, ...]


def validate_operator_set(operator_set_id: str) -> tuple[str, ...]:
    try:
        return OPERATOR_SET_REGISTRY[operator_set_id]
    except KeyError as exc:
        raise ValueError(f"Unknown operator set: {operator_set_id}") from exc


def raw_expression(column: str) -> Expression:
    return ("raw", str(column))


def op_expression(operator: str, left: Expression, right: Expression) -> Expression:
    if operator not in OPERATOR_REGISTRY:
        raise ValueError(f"Unsupported arithmetic operator: {operator}")
    return ("op", operator, left, right)


def expression_depth(expression: Expression) -> int:
    if expression[0] == "raw":
        return 0
    return 1 + max(expression_depth(expression[2]), expression_depth(expression[3]))


def expression_operators(expression: Expression) -> tuple[str, ...]:
    if expression[0] == "raw":
        return ()
    return (str(expression[1]),) + expression_operators(expression[2]) + expression_operators(expression[3])


def expression_columns(expression: Expression) -> tuple[str, ...]:
    if expression[0] == "raw":
        return (str(expression[1]),)
    return expression_columns(expression[2]) + expression_columns(expression[3])


def expression_to_string(expression: Expression) -> str:
    if expression[0] == "raw":
        return f"[{expression[1]}]"
    spec = OPERATOR_REGISTRY[expression[1]]
    return f"({expression_to_string(expression[2])}{spec.symbol}{expression_to_string(expression[3])})"


def expression_to_dict(expression: Expression) -> dict[str, Any]:
    if expression[0] == "raw":
        return {"kind": "raw", "column": str(expression[1])}
    return {
        "kind": "operation",
        "operator": str(expression[1]),
        "left": expression_to_dict(expression[2]),
        "right": expression_to_dict(expression[3]),
    }


def expression_from_dict(value: dict[str, Any]) -> Expression:
    if value.get("kind") == "raw":
        return raw_expression(str(value["column"]))
    if value.get("kind") == "operation":
        return op_expression(
            str(value["operator"]),
            expression_from_dict(value["left"]),
            expression_from_dict(value["right"]),
        )
    raise ValueError(f"Invalid expression mapping: {value!r}")


def candidate_id(expression: Expression) -> str:
    payload = {
        "registry_version": OPERATOR_REGISTRY_VERSION,
        "semantics_version": OPERATOR_SEMANTICS_VERSION,
        "expression": expression_to_dict(expression),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "cand_" + hashlib.sha256(encoded).hexdigest()[:20]


def _finite_output(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    clipped = np.clip(np.where(finite, values, 0.0), -FINITE_CLIP, FINITE_CLIP)
    clipped_count = int(np.count_nonzero(finite & (values != clipped)))
    nonfinite_count = int(np.count_nonzero(~finite))
    return clipped, {"nonfinite_count": nonfinite_count, "clipped_count": clipped_count}


def evaluate_operator(operator: str, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Evaluate one registered operation with the production safety policy."""
    if operator == "add_numeric":
        raw = left + right
    elif operator == "subtract_numeric":
        raw = left - right
    elif operator == "multiply_numeric":
        raw = left * right
    elif operator == "divide_numeric":
        safe = np.abs(right) >= SAFE_DIVISION_EPSILON
        raw = np.zeros_like(left, dtype=float)
        np.divide(left, right, out=raw, where=safe)
        raw[~safe] = 0.0
    else:
        raise ValueError(f"Unsupported arithmetic operator: {operator}")
    output, counts = _finite_output(raw)
    counts["protected_count"] = int(np.count_nonzero(np.abs(right) < SAFE_DIVISION_EPSILON)) if operator == "divide_numeric" else 0
    return output, counts
