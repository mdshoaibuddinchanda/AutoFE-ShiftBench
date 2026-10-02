"""Stable, purpose-separated random seeds for benchmark tasks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


SEED_SCHEME_VERSION = "sha256_canonical_json_u32_v1"
SEED_MAX = (1 << 32) - 1


def _canonical_value(value: Any) -> Any:
    """Convert supported task identity values to deterministic JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Seed identities cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Seed identity mapping keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"Unsupported seed identity value: {type(value).__name__}")


def stable_seed(purpose: str, identity: Mapping[str, Any]) -> int:
    """Derive a uint32 seed from a versioned canonical JSON identity.

    Encoding: UTF-8 JSON with sorted object keys, compact separators, and
    non-ASCII text preserved. The seed is the unsigned big-endian integer from
    the first four bytes of SHA-256 over that encoding. It is in
    ``[0, 2**32 - 1]``, the range accepted by NumPy and scikit-learn.
    """
    if not purpose:
        raise ValueError("A randomness purpose is required")
    payload = {
        "seed_scheme_version": SEED_SCHEME_VERSION,
        "purpose": purpose,
        "identity": _canonical_value(identity),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], byteorder="big", signed=False)


def _task_identity(
    dataset: str,
    split_policy: str,
    repetition_seed: int,
    *,
    fold: int | None = None,
    condition: str | None = None,
    pipeline: str | None = None,
    model: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "dataset": dataset,
        "split_policy": split_policy,
        "repetition_seed": int(repetition_seed),
        "fold": fold,
        "condition": condition,
        "pipeline": pipeline,
        "model": model,
    }
    if extra:
        identity.update(extra)
    return identity


def split_seed(dataset: str, split_policy: str, repetition_seed: int, n_splits: int) -> int:
    """Seed a complete fold assignment; shared by every condition/pipeline."""
    return stable_seed(
        "split_assignment",
        _task_identity(
            dataset, split_policy, repetition_seed,
            extra={"n_splits": int(n_splits)},
        ),
    )


def corruption_seed(
    dataset: str,
    split_policy: str,
    repetition_seed: int,
    fold: int,
    condition: str,
) -> int:
    """Seed the shared corruption realization for a fold/condition."""
    return stable_seed(
        "training_corruption",
        _task_identity(
            dataset, split_policy, repetition_seed,
            fold=fold, condition=condition,
        ),
    )


def feature_selection_seed(
    dataset: str,
    split_policy: str,
    repetition_seed: int,
    fold: int,
    condition: str,
    pipeline: str,
) -> int:
    """Seed selector randomness in one pipeline without changing shared corruption."""
    return stable_seed(
        "feature_selection",
        _task_identity(
            dataset, split_policy, repetition_seed,
            fold=fold, condition=condition, pipeline=pipeline,
        ),
    )


def estimator_seed(
    dataset: str,
    split_policy: str,
    repetition_seed: int,
    fold: int,
    condition: str,
    model: str,
) -> int:
    """Seed a model fit, shared across pipelines for paired comparisons."""
    return stable_seed(
        "estimator_fit",
        _task_identity(
            dataset, split_policy, repetition_seed,
            fold=fold, condition=condition, model=model,
        ),
    )


def distance_sample_seed(
    dataset: str,
    split_policy: str,
    repetition_seed: int,
    fold: int,
    condition: str,
) -> int:
    """Seed paired row subsampling for held-out distribution diagnostics."""
    return stable_seed(
        "distribution_distance_sample",
        _task_identity(
            dataset, split_policy, repetition_seed,
            fold=fold, condition=condition,
        ),
    )
