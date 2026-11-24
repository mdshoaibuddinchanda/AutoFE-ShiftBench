"""Direct, bounded diagnostics for Feature-Synthesis Variance Amplification."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from src.resource_limits import require_bytes

from src.operator_registry import (
    FINITE_CLIP,
    SAFE_DIVISION_EPSILON,
    Expression,
    OPERATOR_REGISTRY,
    candidate_id,
    expression_from_dict,
    expression_operators,
    expression_to_string,
    expression_columns,
    raw_expression,
)


FSVA_SCHEMA_VERSION = "fsva_diagnostics_v1"
DEFAULT_PERTURBATION_MAGNITUDES = (1e-3, 1e-2, 5e-2)


def _sample_rows(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows < 1:
        raise ValueError('Diagnostic max_rows must be positive')
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=random_state).copy()


def _derivative(expression: Expression, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    n_rows = len(frame)
    columns = list(frame.columns)
    positions = {column: index for index, column in enumerate(columns)}
    if expression[0] == "raw":
        values = frame[str(expression[1])].to_numpy(dtype=float)
        jac = np.zeros((n_rows, len(columns)), dtype=float)
        jac[:, positions[str(expression[1])]] = 1.0
        return values, jac, {"protected_count": 0, "nonsmooth_count": 0, "nonfinite_count": int(np.count_nonzero(~np.isfinite(values)))}

    left_value, left_jac, left_status = _derivative(expression[2], frame)
    right_value, right_jac, right_status = _derivative(expression[3], frame)
    operator = str(expression[1])
    status = {
        "protected_count": left_status.get("protected_count", 0) + right_status.get("protected_count", 0),
        "nonsmooth_count": left_status.get("nonsmooth_count", 0) + right_status.get("nonsmooth_count", 0),
        "nonfinite_count": left_status.get("nonfinite_count", 0) + right_status.get("nonfinite_count", 0),
    }
    if operator == "add_numeric":
        raw_values = left_value + right_value
        jac = left_jac + right_jac
    elif operator == "subtract_numeric":
        raw_values = left_value - right_value
        jac = left_jac - right_jac
    elif operator == "multiply_numeric":
        raw_values = left_value * right_value
        jac = left_jac * right_value[:, None] + right_jac * left_value[:, None]
    elif operator == "divide_numeric":
        safe = np.abs(right_value) >= SAFE_DIVISION_EPSILON
        raw_values = np.zeros_like(left_value, dtype=float)
        np.divide(left_value, right_value, out=raw_values, where=safe)
        jac = np.zeros_like(left_jac)
        valid = safe & np.isfinite(raw_values) & (np.abs(raw_values) <= FINITE_CLIP)
        jac[valid] = (left_jac[valid] * right_value[valid, None] - right_jac[valid] * left_value[valid, None]) / (right_value[valid, None] ** 2)
        status["protected_count"] += int(np.count_nonzero(~safe))
        status["nonsmooth_count"] += int(np.count_nonzero(~valid))
    else:
        raise ValueError(f"Unsupported operator in expression: {operator}")
    finite = np.isfinite(raw_values) & np.all(np.isfinite(jac), axis=1)
    clipped = finite & (np.abs(raw_values) > FINITE_CLIP)
    valid = finite & ~clipped
    status["nonfinite_count"] += int(np.count_nonzero(~finite))
    status["nonsmooth_count"] += int(np.count_nonzero(clipped))
    values = np.clip(np.where(np.isfinite(raw_values), raw_values, 0.0), -FINITE_CLIP, FINITE_CLIP)
    jac[~valid] = 0.0
    jac = np.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
    return values, jac, status


def _summary(jacobian: np.ndarray, *, status: dict[str, Any], method: str = "exact") -> dict[str, Any]:
    # jacobian shape: rows x outputs x inputs
    fro = float(np.linalg.norm(jacobian.reshape(-1), ord=2))
    n_outputs = int(jacobian.shape[1])
    n_inputs = int(jacobian.shape[2])
    per_output = np.linalg.norm(jacobian, axis=2)
    return {
        "matrix_norm": "frobenius",
        "norm_method": method,
        "measurement_scope": "exact_on_sampled_rows",
        "frobenius_norm": fro,
        "dimension_adjusted_frobenius": float(fro / np.sqrt(max(n_outputs, 1))),
        "mean_output_l2": float(np.mean(per_output)) if per_output.size else 0.0,
        "max_output_l2": float(np.max(per_output)) if per_output.size else 0.0,
        "n_rows": int(jacobian.shape[0]),
        "n_outputs": n_outputs,
        "n_inputs": n_inputs,
        "status": status,
    }


def _stream_summary(frame,expressions):
    """Retain all rows/outputs/coordinates without an outputs x inputs tensor."""
    rows,inputs=len(frame),len(frame.columns)
    require_bytes(rows*(inputs*8*5+len(expressions)*8*3),purpose='streamed exact Jacobian workspace')
    norms=np.empty((rows,len(expressions)),dtype=float)
    status=Counter()
    for index,expression in enumerate(expressions):
        if expression[0]=='raw':
            values=frame[str(expression[1])].to_numpy(dtype=float)
            norms[:,index]=1.
            status.update({'protected_count':0,'nonsmooth_count':0,'nonfinite_count':int(np.count_nonzero(~np.isfinite(values)))})
        else:
            _,jac,counts=_derivative(expression,frame)
            norms[:,index]=np.linalg.norm(jac,axis=1)
            status.update(counts)
    fro=float(np.sqrt(np.sum(norms*norms)))
    summary={'matrix_norm':'frobenius','norm_method':'exact','measurement_scope':'exact_on_sampled_rows',
        'frobenius_norm':fro,'dimension_adjusted_frobenius':float(fro/np.sqrt(max(len(expressions),1))),
        'mean_output_l2':float(np.mean(norms)) if norms.size else 0.,'max_output_l2':float(np.max(norms)) if norms.size else 0.,
        'n_rows':rows,'n_outputs':len(expressions),'n_inputs':inputs,'status':dict(status)}
    per_output=[{'candidate_id':candidate_id(expr),'expression':expression_to_string(expr),
        'mean_l2':float(np.mean(norms[:,index])) if rows else 0.} for index,expr in enumerate(expressions)]
    return summary,per_output


def compute_jacobian_diagnostic(
    base_inputs: pd.DataFrame,
    expressions: Iterable[Expression],
    *,
    raw_control_expressions: Iterable[Expression] | None = None,
    max_rows: int = 128,
    random_state: int = 0,
) -> dict[str, Any]:
    """Measure exact derivatives of the frozen executed arithmetic mapping."""
    sampled = _sample_rows(base_inputs, max_rows, random_state)
    expressions = list(expressions)
    if raw_control_expressions is not None:
        raw_control_expressions=list(raw_control_expressions)
    selected,per_output=_stream_summary(sampled,expressions)
    result = {
        "schema_version": FSVA_SCHEMA_VERSION,
        "diagnostic_status": "diagnostic_complete" if expressions else "unsupported",
        "input_coordinates": list(base_inputs.columns),
        "input_definition": "preprocessed_corrupted_training_rows; frozen during diagnostic",
        "output_definition": "selected_executed_arithmetic_features",
        "unsupported_coordinates": [
            "categorical_level_changes",
            "estimator_predictions",
        ],
        "sample_identity": {"random_state": int(random_state), "max_rows": int(max_rows)},
        "selected": selected,
        "per_output": per_output,
    }
    if raw_control_expressions is not None:
        raw_expressions = list(raw_control_expressions)
        result["raw_control"] = _stream_summary(sampled,raw_expressions)[0]
    return result


def _evaluate_mapping(
    expressions: list[Expression], frame: pd.DataFrame, *, return_status: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    outputs: list[np.ndarray] = []
    status = Counter()
    for expression in expressions:
        if expression[0]=='raw':
            values=frame[str(expression[1])].to_numpy(dtype=float)
            expression_status={'protected_count':0,'nonsmooth_count':0,'nonfinite_count':int(np.count_nonzero(~np.isfinite(values)))}
        else:
            # Derivative validity remains part of the executed mapping contract.
            # Drop only provably irrelevant coordinates on finite wide inputs.
            dependencies=set(expression_columns(expression))
            relevant=[column for column in frame if column in dependencies]
            working=frame
            if len(frame.columns)>=64 and len(relevant)<len(frame.columns) and np.isfinite(frame.to_numpy(dtype=float)).all():
                working=frame[relevant]
            values, _jac, expression_status = _derivative(expression, working)
        outputs.append(values)
        status.update(expression_status)
    values = np.stack(outputs, axis=1) if outputs else np.zeros((len(frame), 0))
    return (values, dict(status)) if return_status else values


def compute_empirical_amplification(
    base_inputs: pd.DataFrame,
    expressions: Iterable[Expression],
    *,
    raw_control_expressions: Iterable[Expression] | None = None,
    magnitudes: Iterable[float] = DEFAULT_PERTURBATION_MAGNITUDES,
    max_rows: int = 128,
    random_state: int = 0,
) -> dict[str, Any]:
    """Measure frozen-map output/input ratios for bounded deterministic perturbations."""
    sampled = _sample_rows(base_inputs, max_rows, random_state)
    expressions = list(expressions)
    raw_expressions = list(raw_control_expressions or [])
    rng = np.random.default_rng(random_state)
    directions = rng.normal(size=sampled.shape)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.where(norms == 0.0, 1.0, norms)
    baseline, baseline_status = _evaluate_mapping(expressions, sampled, return_status=True)
    if raw_expressions:
        raw_baseline, raw_status = _evaluate_mapping(raw_expressions, sampled, return_status=True)
    else:
        raw_baseline, raw_status = None, {}
    rows: list[dict[str, Any]] = []
    for magnitude in magnitudes:
        magnitude = float(magnitude)
        perturbed = sampled + magnitude * directions
        perturbed_values, perturbed_status = _evaluate_mapping(expressions, perturbed, return_status=True)
        changed = perturbed_values - baseline
        input_norm = np.linalg.norm(magnitude * directions, axis=1)
        output_norm = np.linalg.norm(changed, axis=1)
        valid = input_norm > 0.0
        ratios = output_norm[valid] / input_norm[valid] if np.any(valid) else np.array([], dtype=float)
        row: dict[str, Any] = {
            "magnitude": magnitude,
            "input_norm_mean": float(np.mean(input_norm)) if len(input_norm) else 0.0,
            "output_norm_mean": float(np.mean(output_norm)) if len(output_norm) else 0.0,
            "amplification_ratio_mean": float(np.mean(ratios)) if len(ratios) else None,
            "amplification_ratio_max": float(np.max(ratios)) if len(ratios) else None,
            "valid_rows": int(np.count_nonzero(valid)),
            "baseline_validity": baseline_status,
            "perturbed_validity": perturbed_status,
            "status": "diagnostic_complete" if len(ratios) else "zero_input_norm",
        }
        if raw_baseline is not None:
            raw_perturbed, raw_perturbed_status = _evaluate_mapping(raw_expressions, perturbed, return_status=True)
            raw_changed = raw_perturbed - raw_baseline
            raw_output_norm = np.linalg.norm(raw_changed, axis=1)
            raw_ratios = raw_output_norm[valid] / input_norm[valid] if np.any(valid) else np.array([], dtype=float)
            row["raw_control_amplification_ratio_mean"] = float(np.mean(raw_ratios)) if len(raw_ratios) else None
            row["raw_control_baseline_validity"] = raw_status
            row["raw_control_perturbed_validity"] = raw_perturbed_status
            row["dimension_adjusted_ratio"] = (
                float(np.mean(ratios) / max(np.sqrt(len(expressions)), 1.0)) if len(ratios) else None
            )
        rows.append(row)
    return {
        "schema_version": FSVA_SCHEMA_VERSION,
        "diagnostic_status": "diagnostic_complete" if rows else "unsupported",
        "perturbation_definition": "normalized deterministic Gaussian directions on frozen preprocessed inputs",
        "sample_identity": {"random_state": int(random_state), "max_rows": int(max_rows)},
        "magnitudes": rows,
    }


def _tensor_finite_difference(
    base_inputs: pd.DataFrame,
    expressions: Iterable[Expression],
    *,
    epsilon: float = 1e-6,
    max_rows: int = 32,
    random_state: int = 0,
) -> dict[str, Any]:
    """Compare analytic derivatives with bounded central finite differences."""
    sampled = _sample_rows(base_inputs, max_rows, random_state)
    expressions = list(expressions)
    require_bytes(len(sampled)*len(expressions)*len(sampled.columns)*8*4,purpose='nonfinite-input finite-difference tensor fallback')
    analytic: list[np.ndarray] = []
    nonsmooth = 0
    for expression in expressions:
        _values, jac, status = _derivative(expression, sampled)
        analytic.append(jac)
        nonsmooth += int(status.get("nonsmooth_count", 0))
    analytic_tensor = np.stack(analytic, axis=1) if analytic else np.zeros((len(sampled), 0, len(sampled.columns)))
    finite = np.zeros_like(analytic_tensor)
    for column_index, column in enumerate(sampled.columns):
        plus = sampled.copy()
        minus = sampled.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        finite[:, :, column_index] = (
            _evaluate_mapping(expressions, plus) - _evaluate_mapping(expressions, minus)
        ) / (2.0 * epsilon)
    error = np.abs(analytic_tensor - finite)
    max_error = float(np.max(error)) if error.size else 0.0
    return {
        "method": "central_finite_difference",
        "epsilon": float(epsilon),
        "max_absolute_error": max_error,
        "mean_absolute_error": float(np.mean(error)) if error.size else 0.0,
        "nonsmooth_count": nonsmooth,
        "status": "validated" if max_error <= max(1e-4, epsilon * 1000) else "tolerance_exceeded",
    }


def validate_jacobian_finite_difference(base_inputs,expressions,*,epsilon=1e-6,max_rows=32,random_state=0):
    sampled=_sample_rows(base_inputs,max_rows,random_state)
    expressions=list(expressions)
    if not np.isfinite(sampled.to_numpy(dtype=float)).all():
        return _tensor_finite_difference(base_inputs,expressions,epsilon=epsilon,max_rows=max_rows,random_state=random_state)
    require_bytes(len(sampled)*(len(expressions)+len(sampled.columns))*8*5,purpose='exact dependency-aware finite differences')
    errors=np.zeros((len(sampled),len(expressions)),dtype=float)
    max_error=0.
    nonsmooth=0
    for index,expression in enumerate(expressions):
        dependencies=set(expression_columns(expression))
        columns=[column for column in sampled if column in dependencies]
        working=sampled[columns]
        _,analytic,status=_derivative(expression,working)
        nonsmooth+=int(status.get('nonsmooth_count',0))
        for position,column in enumerate(columns):
            plus,minus=working.copy(),working.copy()
            plus[column]+=epsilon
            minus[column]-=epsilon
            # Same central directions, magnitudes and arithmetic as the reference.
            finite=(_evaluate_mapping([expression],plus)[:,0]-_evaluate_mapping([expression],minus)[:,0])/(2.*epsilon)
            error=np.abs(analytic[:,position]-finite)
            errors[:,index]+=error
            if error.size: max_error=max(max_error,float(np.max(error)))
    count=len(sampled)*len(expressions)*len(sampled.columns)
    return {'method':'central_finite_difference','epsilon':float(epsilon),'max_absolute_error':max_error,
        'mean_absolute_error':float(np.sum(errors)/count) if count else 0.,'nonsmooth_count':nonsmooth,
        'status':'validated' if max_error<=max(1e-4,epsilon*1000) else 'tolerance_exceeded'}


def selection_stability(histories: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize selected candidate overlap without pooled inference."""
    histories = list(histories)
    selected_sets = [set(row.get("selected_feature_identities", [])) for row in histories]
    pairs = []
    for left, right in combinations(selected_sets, 2):
        union = left | right
        pairs.append(float(len(left & right) / len(union)) if union else 1.0)
    availability = Counter()
    frequencies = Counter()
    for history in histories:
        if 'selection_history' in history and 'candidate_history' in history:
            raise ValueError('Ambiguous selection and legacy candidate history keys')
        events=history.get('selection_history',history.get('candidate_history'))
        if events is None:
            raise ValueError('Candidate availability requires real selection history')
        if len({event['candidate_id'] for event in events}) != len(events):
            raise ValueError('Duplicate candidates in selection history')
        for candidate in events:
            availability[candidate["candidate_id"]] += int(candidate.get("eligible", False))
            frequencies[candidate["candidate_id"]] += int(candidate.get("selected", False))
    return {
        "schema_version": FSVA_SCHEMA_VERSION,
        "n_histories": len(histories),
        "pairwise_jaccard_mean": float(np.mean(pairs)) if pairs else None,
        "pairwise_jaccard_values": pairs,
        "candidate_availability": dict(availability),
        "candidate_selection_frequency": dict(frequencies),
        "empty_set_jaccard": 1.0,
    }
