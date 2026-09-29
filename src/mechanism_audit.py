"""Auditable FSVA and operator-isolation primitives.

This module is deliberately independent of the benchmark runner.  It defines
the measurement contract that an experiment can use on one training fold:

* an arithmetic candidate is a map from its parent feature values to one
  candidate value;
* its Jacobian is the row-wise derivative with respect to those parents;
* a scaled row-wise norm is computed only for finite, well-defined rows; and
* candidate selection history is written as flushed JSON Lines records.

The current Featuretools pipeline is not instrumented by this module.  Keeping
the contract separate makes the definition and synthetic checks reviewable
before adding logging to an expensive campaign.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np


ARITHMETIC_OPERATORS: tuple[str, ...] = (
    "add_numeric",
    "subtract_numeric",
    "multiply_numeric",
    "divide_numeric",
)
_OPERATOR_PARENT_COUNTS = {operator: 2 for operator in ARITHMETIC_OPERATORS}


class UndefinedDerivative(ValueError):
    """Raised when a derivative map cannot be defined for a candidate."""


@dataclass(frozen=True, slots=True)
class ArithmeticCandidate:
    """A binary arithmetic candidate and the names of its parent features."""

    candidate_id: str
    operator: str
    parent_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if self.operator not in ARITHMETIC_OPERATORS:
            raise ValueError(
                f"Unsupported arithmetic operator {self.operator!r}; "
                f"expected one of {ARITHMETIC_OPERATORS}"
            )
        expected = _OPERATOR_PARENT_COUNTS[self.operator]
        if len(self.parent_features) != expected:
            raise ValueError(
                f"{self.operator} requires {expected} parents, "
                f"received {len(self.parent_features)}"
            )
        if any(not parent for parent in self.parent_features):
            raise ValueError("parent feature names must be non-empty")


@dataclass(slots=True)
class DerivativeResult:
    """A Jacobian plus an explicit validity/undefined reason per row."""

    values: np.ndarray
    valid_rows: np.ndarray
    undefined_reasons: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Derivative values must have shape (n_rows, n_parents)")
        if self.values.shape[0] != len(self.valid_rows):
            raise ValueError("valid_rows length does not match derivative rows")
        if self.values.shape[0] != len(self.undefined_reasons):
            raise ValueError("undefined_reasons length does not match derivative rows")


@dataclass(slots=True)
class EvaluationResult:
    """Candidate values and explicit finite/undefined row status."""

    values: np.ndarray
    valid_rows: np.ndarray
    undefined_reasons: tuple[str | None, ...]


def _input_matrix(values: Any, expected_columns: int = 2) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
        raise ValueError(
            f"Expected a two-dimensional matrix with {expected_columns} columns; "
            f"received shape {matrix.shape}"
        )
    return matrix


def _row_reasons(valid: np.ndarray, reason: str) -> tuple[str | None, ...]:
    return tuple(None if bool(ok) else reason for ok in valid)


def evaluate_arithmetic(candidate: ArithmeticCandidate, values: Any) -> EvaluationResult:
    """Evaluate a candidate without converting undefined values to zero.

    Non-finite parents and non-finite outputs are marked undefined.  The
    corresponding output is retained as ``nan`` so callers cannot mistake an
    undefined value for a valid measurement.
    """

    matrix = _input_matrix(values)
    finite_parents = np.isfinite(matrix).all(axis=1)
    left, right = matrix[:, 0], matrix[:, 1]
    with np.errstate(all="ignore"):
        if candidate.operator == "add_numeric":
            output = left + right
            base_reason = "nonfinite_parent_or_output"
        elif candidate.operator == "subtract_numeric":
            output = left - right
            base_reason = "nonfinite_parent_or_output"
        elif candidate.operator == "multiply_numeric":
            output = left * right
            base_reason = "nonfinite_parent_or_output"
        elif candidate.operator == "divide_numeric":
            output = left / right
            base_reason = "zero_denominator_or_nonfinite_parent_or_output"
        else:  # guarded by ArithmeticCandidate, retained for defensive use
            raise UndefinedDerivative(f"Unsupported operator: {candidate.operator}")
    finite_output = np.isfinite(output)
    valid = finite_parents & finite_output
    result = np.asarray(output, dtype=float)
    result[~valid] = np.nan
    reasons: list[str | None] = []
    for row, ok in enumerate(valid):
        if ok:
            reasons.append(None)
        elif not finite_parents[row]:
            reasons.append("nonfinite_parent")
        elif candidate.operator == "divide_numeric" and right[row] == 0.0:
            reasons.append("zero_denominator")
        else:
            reasons.append(base_reason)
    return EvaluationResult(result, valid, tuple(reasons))


def analytic_jacobian(candidate: ArithmeticCandidate, values: Any) -> DerivativeResult:
    """Return the analytic row-wise Jacobian for supported arithmetic maps.

    For ``u=(a,b)`` the maps and derivatives are ``a+b -> (1,1)``,
    ``a-b -> (1,-1)``, ``a*b -> (b,a)``, and ``a/b -> (1/b,-a/b**2)``.
    Division by zero, non-finite inputs, overflow, and non-finite derivative
    values are undefined and remain ``nan`` in the returned matrix.
    """

    matrix = _input_matrix(values)
    finite_parents = np.isfinite(matrix).all(axis=1)
    left, right = matrix[:, 0], matrix[:, 1]
    jacobian = np.full((len(matrix), 2), np.nan, dtype=float)
    valid = finite_parents.copy()
    reasons: list[str | None] = [None if ok else "nonfinite_parent" for ok in valid]
    with np.errstate(all="ignore"):
        if candidate.operator == "add_numeric":
            jacobian[:, :] = (1.0, 1.0)
        elif candidate.operator == "subtract_numeric":
            jacobian[:, :] = (1.0, -1.0)
        elif candidate.operator == "multiply_numeric":
            jacobian[:, 0] = right
            jacobian[:, 1] = left
        elif candidate.operator == "divide_numeric":
            jacobian[:, 0] = 1.0 / right
            jacobian[:, 1] = -left / np.square(right)
        else:  # guarded by ArithmeticCandidate
            raise UndefinedDerivative(f"Unsupported operator: {candidate.operator}")
    if candidate.operator == "divide_numeric":
        zero_denominator = finite_parents & (right == 0.0)
        for row in np.flatnonzero(zero_denominator):
            valid[row] = False
            reasons[row] = "zero_denominator"
    finite_jacobian = np.isfinite(jacobian).all(axis=1)
    for row in np.flatnonzero(valid & ~finite_jacobian):
        valid[row] = False
        reasons[row] = "nonfinite_derivative"
    jacobian[~valid, :] = np.nan
    return DerivativeResult(jacobian, valid, tuple(reasons))


def finite_difference_jacobian(
    candidate: ArithmeticCandidate,
    values: Any,
    *,
    relative_step: float = 1e-6,
) -> DerivativeResult:
    """Check a candidate derivative with central finite differences.

    The step is ``relative_step * max(1, abs(parent))`` for each parent.  A
    row is valid only when both perturbed evaluations and the resulting
    difference are finite.  No one-sided or clipped derivative is substituted
    at an undefined point.
    """

    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    matrix = _input_matrix(values)
    n_rows = len(matrix)
    finite_parents = np.isfinite(matrix).all(axis=1)
    derivative = np.full((n_rows, 2), np.nan, dtype=float)
    valid = finite_parents.copy()
    reasons: list[str | None] = [None if ok else "nonfinite_parent" for ok in valid]
    for column in range(2):
        step = relative_step * np.maximum(1.0, np.abs(matrix[:, column]))
        plus = matrix.copy()
        minus = matrix.copy()
        plus[:, column] += step
        minus[:, column] -= step
        plus_eval = evaluate_arithmetic(candidate, plus)
        minus_eval = evaluate_arithmetic(candidate, minus)
        with np.errstate(all="ignore"):
            values_col = (plus_eval.values - minus_eval.values) / (2.0 * step)
        column_valid = valid & plus_eval.valid_rows & minus_eval.valid_rows & np.isfinite(values_col)
        for row in np.flatnonzero(valid & ~column_valid):
            reasons[row] = (
                plus_eval.undefined_reasons[row]
                or minus_eval.undefined_reasons[row]
                or "nonfinite_finite_difference"
            )
        valid &= column_valid
        derivative[:, column] = values_col
    derivative[~valid, :] = np.nan
    return DerivativeResult(derivative, valid, tuple(reasons))


@dataclass(slots=True)
class DerivativeCheck:
    """Comparison between analytic and finite-difference Jacobians."""

    status: str
    valid_rows: int
    valid_entries: int
    max_absolute_error: float | None
    max_relative_error: float | None
    undefined_reason_counts: dict[str, int]


def check_derivative(
    candidate: ArithmeticCandidate,
    values: Any,
    *,
    relative_step: float = 1e-6,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-4,
) -> DerivativeCheck:
    """Compare analytic and numerical derivatives without hiding undefined rows."""

    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Derivative tolerances must be non-negative")
    analytic = analytic_jacobian(candidate, values)
    numerical = finite_difference_jacobian(candidate, values, relative_step=relative_step)
    valid_rows = analytic.valid_rows & numerical.valid_rows
    reasons = Counter()
    for row, ok in enumerate(valid_rows):
        if not ok:
            reasons[analytic.undefined_reasons[row] or numerical.undefined_reasons[row] or "undefined"] += 1
    if not valid_rows.any():
        return DerivativeCheck("undefined", 0, 0, None, None, dict(reasons))
    analytic_values = analytic.values[valid_rows]
    numerical_values = numerical.values[valid_rows]
    difference = np.abs(analytic_values - numerical_values)
    scale = np.maximum(np.abs(analytic_values), np.abs(numerical_values))
    relative = difference / np.maximum(scale, np.finfo(float).eps)
    finite = np.isfinite(difference) & np.isfinite(relative)
    if not finite.all():
        return DerivativeCheck(
            "undefined",
            int(valid_rows.sum()),
            int(finite.sum()),
            None,
            None,
            {**dict(reasons), "nonfinite_comparison": int((~finite).sum())},
        )
    passed = np.all(
        (difference <= absolute_tolerance)
        | (difference <= relative_tolerance * scale)
    )
    return DerivativeCheck(
        "pass" if passed else "fail",
        int(valid_rows.sum()),
        int(difference.size),
        float(np.max(difference)),
        float(np.max(relative)),
        dict(reasons),
    )


def training_input_scale(values: Any) -> np.ndarray:
    """Compute population standard deviations from training parents only.

    A non-finite or zero-variance scale is returned as ``nan``.  The caller
    must record the resulting undefined norm rather than silently using one.
    """

    matrix = _input_matrix(values)
    with np.errstate(all="ignore"):
        scale = np.std(matrix, axis=0, ddof=0)
    scale = np.asarray(scale, dtype=float)
    scale[(~np.isfinite(scale)) | (scale <= 0.0)] = np.nan
    return scale


def scaled_jacobian_norm(
    derivative: DerivativeResult,
    *,
    input_scale: Iterable[float] | None = None,
    norm: str = "l2",
) -> np.ndarray:
    """Return one scaled local sensitivity norm per sample.

    ``input_scale`` is estimated on the training fold and the dimensionless
    local gain is ``||J(x) diag(scale)||``.  ``nan`` is retained for undefined
    rows, invalid scales, or non-finite results.  Supported norms are ``l1``,
    ``l2`` (the default), and ``linf``.
    """

    if norm not in {"l1", "l2", "linf"}:
        raise ValueError("norm must be one of 'l1', 'l2', or 'linf'")
    jacobian = np.asarray(derivative.values, dtype=float)
    if input_scale is None:
        scale = np.ones(jacobian.shape[1], dtype=float)
    else:
        scale = np.asarray(tuple(input_scale), dtype=float)
        if scale.shape != (jacobian.shape[1],):
            raise ValueError("input_scale length must equal the number of parents")
    scale_ok = np.isfinite(scale) & (scale > 0.0)
    scaled = jacobian * scale
    with np.errstate(all="ignore"):
        if norm == "l1":
            output = np.sum(np.abs(scaled), axis=1)
        elif norm == "linf":
            output = np.max(np.abs(scaled), axis=1)
        else:
            output = np.sqrt(np.sum(np.square(scaled), axis=1))
    valid = derivative.valid_rows & bool(scale_ok.all()) & np.isfinite(output)
    output = np.asarray(output, dtype=float)
    output[~valid] = np.nan
    return output


@dataclass(slots=True)
class NormSummary:
    """Finite-only summary with undefined rows counted separately."""

    norm: str
    n_rows: int
    n_valid: int
    n_undefined: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    undefined_reason_counts: dict[str, int] = field(default_factory=dict)


def summarize_jacobian_norm(
    values: Any,
    *,
    norm: str = "l2",
    undefined_reasons: Iterable[str | None] | None = None,
) -> NormSummary:
    """Aggregate only finite norms and report undefined cases explicitly."""

    array = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(array)
    if undefined_reasons is None:
        reason_values = ["undefined" for ok in finite if not ok]
    else:
        supplied_reasons = tuple(undefined_reasons)
        if len(supplied_reasons) != len(array):
            raise ValueError("undefined_reasons length must match values length")
        reason_values = [
            reason or "undefined"
            for ok, reason in zip(finite, supplied_reasons)
            if not ok
        ]
    finite_values = array[finite]
    if len(finite_values):
        stats: dict[str, float | None] = {
            "mean": float(np.mean(finite_values)),
            "median": float(np.median(finite_values)),
            "minimum": float(np.min(finite_values)),
            "maximum": float(np.max(finite_values)),
        }
    else:
        stats = {"mean": None, "median": None, "minimum": None, "maximum": None}
    return NormSummary(
        norm=norm,
        n_rows=int(array.size),
        n_valid=int(finite.sum()),
        n_undefined=int((~finite).sum()),
        undefined_reason_counts=dict(Counter(reason_values)),
        **stats,
    )


@dataclass(frozen=True, slots=True)
class CandidateHistoryRecord:
    """One candidate decision, suitable for a JSONL audit stream.

    ``score_scope`` is intentionally fixed to ``"train"``.  A held-out score
    must never be used to choose a candidate; held-out evaluation belongs in a
    separate result ledger.
    """

    dataset: str
    split_policy: str
    seed: int
    fold: int
    condition: str
    iteration: int
    candidate_id: str
    parent_features: tuple[str, ...]
    operator: str
    admissible: bool
    rejection_reason: str | None
    selection_score: float | None
    selection_score_status: str
    selection_decision: str
    candidate_seed: int
    score_scope: str = "train"

    def __post_init__(self) -> None:
        if not self.dataset or not self.split_policy or not self.condition:
            raise ValueError("dataset, split_policy, and condition must be non-empty")
        if self.seed < 0 or self.fold < 1 or self.iteration < 0 or self.candidate_seed < 0:
            raise ValueError("seed/candidate_seed must be non-negative; fold must be positive")
        if not self.candidate_id or not self.parent_features or not self.operator:
            raise ValueError("candidate identity, parents, and operator are required")
        if self.score_scope != "train":
            raise ValueError("candidate selection scores must have score_scope='train'")
        if self.selection_decision not in {"selected", "not_selected", "rejected"}:
            raise ValueError("selection_decision must be selected, not_selected, or rejected")
        if self.selection_score_status not in {"finite", "undefined", "not_scored"}:
            raise ValueError("invalid selection_score_status")
        if self.selection_score is not None and not np.isfinite(self.selection_score):
            raise ValueError("undefined selection scores must be represented as None")
        if not self.admissible and not self.rejection_reason:
            raise ValueError("an inadmissible candidate requires rejection_reason")
        if self.admissible and self.rejection_reason is not None:
            raise ValueError("admissible candidates cannot carry a rejection_reason")
        if self.selection_decision == "selected":
            if not self.admissible or self.selection_score is None:
                raise ValueError("selected candidates require admissibility and a finite score")
            if self.selection_score_status != "finite":
                raise ValueError("selected candidates require selection_score_status='finite'")
        if self.selection_decision == "rejected" and self.admissible:
            raise ValueError("rejected candidates must be inadmissible")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parent_features"] = list(self.parent_features)
        return payload


class CandidateHistoryWriter:
    """Flush-each-record JSONL writer, optionally gzip-compressed.

    A ``.gz`` suffix selects gzip.  The writer never drops a record; malformed
    records fail before writing.  ``uncompressed_bytes_written`` is useful for
    preflight storage accounting and ``path.stat().st_size`` gives compressed
    on-disk size after a flush/close.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO | gzip.GzipFile
        if self.path.suffix == ".gz":
            self._stream = gzip.open(self.path, "at", encoding="utf-8", newline="")
        else:
            self._stream = self.path.open("a", encoding="utf-8", newline="")
        self.records_written = 0
        self.uncompressed_bytes_written = 0
        self._closed = False

    def append(self, record: CandidateHistoryRecord) -> None:
        if self._closed:
            raise RuntimeError("CandidateHistoryWriter is closed")
        if not isinstance(record, CandidateHistoryRecord):
            raise TypeError("append expects CandidateHistoryRecord")
        line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        self._stream.write(line)
        self._stream.flush()
        self.records_written += 1
        self.uncompressed_bytes_written += len(line.encode("utf-8"))

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True

    def __enter__(self) -> "CandidateHistoryWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class CandidateHistoryStorageEstimate:
    records: int
    bytes_per_record: int
    uncompressed_bytes: int
    compressed_bytes: int
    uncompressed_mib: float
    compressed_mib: float
    compression_ratio: float


def estimate_candidate_history_storage(
    *,
    tasks: int,
    candidates_per_task: int,
    bytes_per_record: int = 420,
    compression_ratio: float = 0.35,
) -> CandidateHistoryStorageEstimate:
    """Estimate candidate-history storage before enabling detailed logging."""

    for name, value in {
        "tasks": tasks,
        "candidates_per_task": candidates_per_task,
        "bytes_per_record": bytes_per_record,
    }.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not np.isfinite(compression_ratio) or not (0.0 < compression_ratio <= 1.0):
        raise ValueError("compression_ratio must be in (0, 1]")
    records = tasks * candidates_per_task
    uncompressed = records * bytes_per_record
    compressed = math.ceil(uncompressed * compression_ratio)
    mib = 1024 * 1024
    return CandidateHistoryStorageEstimate(
        records=records,
        bytes_per_record=bytes_per_record,
        uncompressed_bytes=uncompressed,
        compressed_bytes=compressed,
        uncompressed_mib=uncompressed / mib,
        compressed_mib=compressed / mib,
        compression_ratio=float(compression_ratio),
    )


@dataclass(frozen=True, slots=True)
class OperatorIsolationSpec:
    """One explicit operator scope with shared budget requirements."""

    name: str
    enabled_operators: tuple[str, ...]
    excluded_operators: tuple[str, ...]
    purpose: str
    historical_alias: str | None = None
    depth: int = 1
    max_base_features: int = 20
    max_features: int = 100
    selection_method: str = "variance"

    def __post_init__(self) -> None:
        enabled = set(self.enabled_operators)
        excluded = set(self.excluded_operators)
        unknown = (enabled | excluded) - set(ARITHMETIC_OPERATORS)
        if unknown:
            raise ValueError(f"Unknown operators: {sorted(unknown)}")
        if enabled & excluded:
            raise ValueError("An operator cannot be both enabled and excluded")
        if self.depth < 1 or self.max_base_features < 1 or self.max_features < 1:
            raise ValueError("operator budgets must be positive")


def recommended_operator_isolation_configs() -> tuple[OperatorIsolationSpec, ...]:
    """Return prespecified isolate-one and leave-one-out operator scopes.

    Every generated scope retains the same depth, base-feature cap, output
    feature cap, and variance selection rule.  The historical
    ``AutoFE_NoMultiply`` label is included explicitly because the current
    implementation enables addition/subtraction only and therefore excludes
    division as well as multiplication.
    """

    all_ops = ARITHMETIC_OPERATORS
    specs: list[OperatorIsolationSpec] = [
        OperatorIsolationSpec(
            "AutoFE_Baseline_AllOperators", all_ops, (),
            "Reference scope with all supported arithmetic operators",
        ),
        OperatorIsolationSpec(
            "AutoFE_NoMultiply_Historical", ("add_numeric", "subtract_numeric"),
            ("multiply_numeric", "divide_numeric"),
            "Historical label; excludes multiplication and division",
            historical_alias="AutoFE_NoMultiply",
        ),
    ]
    for operator in all_ops:
        short = operator.removesuffix("_numeric").title().replace(" ", "")
        specs.append(OperatorIsolationSpec(
            f"AutoFE_Isolate_{short}", (operator,),
            tuple(item for item in all_ops if item != operator),
            f"Isolate the contribution of {operator}",
        ))
    for operator in all_ops:
        short = operator.removesuffix("_numeric").title().replace(" ", "")
        specs.append(OperatorIsolationSpec(
            f"AutoFE_LeaveOut_{short}",
            tuple(item for item in all_ops if item != operator),
            (operator,),
            f"Leave out {operator} from the all-operator reference",
        ))
    return tuple(specs)


def validate_operator_isolation_budget(specs: Iterable[OperatorIsolationSpec]) -> None:
    """Assert that an operator comparison uses identical search budgets."""

    values = list(specs)
    if not values:
        raise ValueError("At least one operator scope is required")
    budgets = {
        (spec.depth, spec.max_base_features, spec.max_features, spec.selection_method)
        for spec in values
    }
    if len(budgets) != 1:
        raise ValueError("Operator scopes must use identical generation/selection budgets")
