"""Version identifiers for incompatible benchmark protocols and outputs."""

from pathlib import Path

from src.seeding import SEED_SCHEME_VERSION


# Geometry-based folds now use predictors only. Keep their artifacts in a new
# namespace so that historical splits, representations, and task checkpoints
# cannot be mistaken for outputs from this corrected evaluation protocol.
EVALUATION_PROTOCOL_VERSION = "predictor_only_geometry_v2_baselines_operators_fsva_v1"


def cache_root() -> Path:
    """Return the cache namespace for the active evaluation protocol."""
    return Path("data/cache") / EVALUATION_PROTOCOL_VERSION / SEED_SCHEME_VERSION


def results_ledger_path() -> Path:
    """Return a new ledger path without appending to historical results."""
    return Path("reports/tables") / (
        f"results_stream_{EVALUATION_PROTOCOL_VERSION}_{SEED_SCHEME_VERSION}.jsonl"
    )
