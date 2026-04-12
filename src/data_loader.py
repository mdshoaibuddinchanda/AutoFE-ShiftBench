"""Utilities for downloading and loading benchmark tabular datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


DEFAULT_TARGET_COLUMN = "target"
DEFAULT_MAX_ROWS = 15_000

# Candidate names are tried in order to tolerate naming variants in OpenML.
OPENML_NAME_CANDIDATES: dict[str, list[str]] = {
    "credit-g": ["credit-g", "credit_g"],
    "phoneme": ["phoneme"],
    "diabetes": ["diabetes"],
    "bank-marketing": ["bank-marketing", "bank_marketing"],
    "blood-transfusion": [
        "blood-transfusion-service-center",
        "blood-transfusion",
        "blood_transfusion_service_center",
    ],
    "ilpd": ["ilpd"],
    "kc1": ["kc1"],
    "pc1": ["pc1"],
    "adult": ["adult"],
    "jungle_chess": [
        "jungle_chess_2pcs_raw_endgame_complete",
        "jungle_chess",
    ],
    "churn": ["churn", "churn_modelling"],
    "credit-approval": ["credit-approval", "credit_approval"],
}

# Use explicit OpenML versions for reproducibility and to avoid ambiguous
# multi-active-version warnings from name-only lookups.
OPENML_PREFERRED_VERSION: dict[str, int] = {
    "credit-g": 1,
    "phoneme": 1,
    "diabetes": 1,
    "bank-marketing": 1,
    "blood-transfusion": 1,
    "ilpd": 1,
    "kc1": 1,
    "pc1": 1,
    "adult": 1,
    "jungle_chess": 1,
    "churn": 1,
    "credit-approval": 1,
}


def load_dataset_names(dataset_list_path: str | Path) -> list[str]:
    """Load dataset names from config/dataset_list.yaml."""
    path = Path(dataset_list_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset list not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("dataset_list.yaml must contain a YAML mapping")

    raw_datasets = config.get("datasets")
    if not isinstance(raw_datasets, list):
        raise ValueError("'datasets' must be a list")

    dataset_names: list[str] = []
    for dataset in raw_datasets:
        if isinstance(dataset, str):
            dataset_names.append(dataset)
            continue

        if isinstance(dataset, dict) and "name" in dataset:
            dataset_names.append(str(dataset["name"]))
            continue

        raise ValueError("Each dataset entry must be a string or {name: ...}")

    if not dataset_names:
        raise ValueError("No datasets provided in dataset_list.yaml")

    return dataset_names


def _fetch_openml_with_fallbacks(dataset_name: str):
    """Fetch an OpenML dataset by trying supported candidate names."""
    # Imported lazily so lightweight utilities in this module do not require
    # importing the full scikit-learn stack unless OpenML download is invoked.
    from sklearn.datasets import fetch_openml

    candidates = OPENML_NAME_CANDIDATES.get(dataset_name, [dataset_name])
    preferred_version = OPENML_PREFERRED_VERSION.get(dataset_name, 1)
    versions_to_try: tuple[int | str, ...] = (preferred_version, "active")
    last_error: Exception | None = None

    for candidate in candidates:
        for version in versions_to_try:
            try:
                dataset = fetch_openml(
                    name=candidate,
                    version=version,
                    as_frame=True,
                    parser="auto",
                )
                return dataset, candidate
            except Exception as exc:  # pragma: no cover - network/API dependent
                last_error = exc

    raise RuntimeError(
        f"Failed to fetch OpenML dataset '{dataset_name}'. "
        f"Tried names: {candidates} with versions {list(versions_to_try)}. "
        f"Last error: {last_error}"
    )


def download_openml_dataset(
    dataset_name: str,
    output_dir: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = 42,
) -> Path:
    """Download one dataset from OpenML and save it as CSV in data/raw."""
    dataset, _resolved_name = _fetch_openml_with_fallbacks(dataset_name)

    x = dataset.data
    y = dataset.target
    if x is None or y is None:
        raise ValueError(f"Dataset '{dataset_name}' does not provide X/y data")

    features = x.copy()
    target_column = (
        DEFAULT_TARGET_COLUMN
        if DEFAULT_TARGET_COLUMN not in features.columns
        else "target_label"
    )

    combined = features.copy()
    combined[target_column] = pd.Series(y).reset_index(drop=True)

    # Keep benchmark datasets bounded for stable runtimes and fair comparisons.
    if len(combined) > max_rows:
        combined = combined.sample(
            n=max_rows,
            random_state=random_state,
        ).reset_index(drop=True)

    output_path = Path(output_dir) / f"{dataset_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path


def download_datasets_from_list(
    dataset_list_path: str | Path = "config/dataset_list.yaml",
    output_dir: str | Path = "data/raw",
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = 42,
) -> dict[str, Path]:
    """Download all datasets from dataset_list.yaml into data/raw."""
    dataset_names = load_dataset_names(dataset_list_path)

    saved_paths: dict[str, Path] = {}
    for dataset_name in dataset_names:
        saved_paths[dataset_name] = download_openml_dataset(
            dataset_name=dataset_name,
            output_dir=output_dir,
            max_rows=max_rows,
            random_state=random_state,
        )

    return saved_paths


def load_csv_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix and target vector."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset")

    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target


if __name__ == "__main__":
    results = download_datasets_from_list()
    for name, path in results.items():
        print(f"{name}: {path}")
