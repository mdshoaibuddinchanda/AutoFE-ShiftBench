"""Utilities for downloading, loading, and analyzing benchmark tabular datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_TARGET_COLUMN = "target"
DEFAULT_MAX_ROWS = 100_000

# Candidate names to tolerate naming variants in OpenML.
OPENML_NAME_CANDIDATES: dict[str, list[str]] = {
    "adult": ["adult", "adult-income"],
    "bank-marketing": ["bank-marketing", "bank_marketing", "Bank_Marketing"],
    "aps_failure": ["aps_failure", "aps-failure", "aps_failure_training_set"],
    "electricity": ["electricity"],
    "covertype": ["covertype", "Covertype"],
    "dry-bean-dataset": ["dry-bean-dataset", "Dry_Bean_Dataset"],
    "crop-recommendation": ["crop-recommendation", "Crop_Recommendation", "crop_recommendation"],
    "breast-cancer-wisconsin": ["breast-cancer-wisconsin", "breast_cancer", "wdbc"],
    "heart-disease": ["heart-disease", "heart_disease", "heart-statlog", "heart"],
    "diabetes": ["diabetes"],
    "haberman": ["haberman", "haberman-survival", "Haberman"],
    "ionosphere": ["ionosphere"],
    "sonar": ["sonar"],
    "credit-g": ["credit-g", "credit_g", "german_credit"],
    "default-of-credit-card-clients": ["default-of-credit-card-clients", "default_of_credit_card_clients"],
    "mushroom": ["mushroom"],
    "magic-telescope": ["magic-telescope", "MagicTelescope", "magic"],
    "spambase": ["spambase"],
    "wine-quality-red": ["wine-quality-red", "wine_quality", "wine-quality"],
    "rice-cammeo-and-osmancik": ["rice-cammeo-and-osmancik", "Rice_Cammeo_Osmancik"],
}


def load_dataset_names(dataset_list_path: str | Path) -> list[str]:
    """Load dataset names from config/dataset_list.yaml."""
    path = Path(dataset_list_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset list not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_datasets = config.get("datasets")
    if not isinstance(raw_datasets, list):
        raise ValueError("'datasets' must be a list")

    dataset_names: list[str] = []
    for dataset in raw_datasets:
        if isinstance(dataset, str):
            dataset_names.append(dataset)
        elif isinstance(dataset, dict) and "name" in dataset:
            dataset_names.append(str(dataset["name"]))

    if not dataset_names:
        raise ValueError("No datasets provided in dataset_list.yaml")

    return dataset_names


def _fetch_openml_with_fallbacks(dataset_name: str):
    """Fetch an OpenML dataset by trying supported candidate names."""
    from sklearn.datasets import fetch_openml

    candidates = OPENML_NAME_CANDIDATES.get(dataset_name, [dataset_name])
    versions_to_try: tuple[int | str, ...] = (1, "active")
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
            except Exception as exc:
                last_error = exc

    raise RuntimeError(
        f"Failed to fetch OpenML dataset '{dataset_name}'. "
        f"Tried names: {candidates} with versions {list(versions_to_try)}. "
        f"Last error: {last_error}"
    )


def compute_meta_features(features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Compute meta-features for dataset analysis."""
    meta = {}
    
    n_samples = len(features)
    n_features = features.shape[1]
    meta["number_of_samples"] = float(n_samples)
    meta["number_of_features"] = float(n_features)
    
    if n_samples == 0 or n_features == 0:
        return meta
        
    num_cols = features.select_dtypes(include=["number"]).columns
    cat_cols = features.select_dtypes(exclude=["number"]).columns
    
    meta["numerical_percent"] = float(len(num_cols) / n_features * 100.0)
    meta["categorical_percent"] = float(len(cat_cols) / n_features * 100.0)
    
    total_cells = n_samples * n_features
    missing_cells = features.isna().sum().sum()
    meta["missing_percent"] = float(missing_cells / total_cells * 100.0) if total_cells > 0 else 0.0
    
    # Class imbalance (majority class % / minority class %)
    value_counts = target.value_counts(dropna=False)
    meta["number_of_classes"] = float(len(value_counts))
    if len(value_counts) > 1:
        majority = value_counts.max()
        minority = value_counts.min()
        meta["class_imbalance_ratio"] = float(majority / minority) if minority > 0 else float("inf")
    else:
        meta["class_imbalance_ratio"] = 1.0

    # Average correlation (numeric features only)
    if len(num_cols) > 1:
        corr_matrix = features[num_cols].corr().abs().to_numpy()
        np.fill_diagonal(corr_matrix, np.nan)
        avg_corr = np.nanmean(corr_matrix)
        meta["average_correlation"] = float(avg_corr) if not np.isnan(avg_corr) else 0.0
    else:
        meta["average_correlation"] = 0.0
        
    # We leave entropy, redundancy and MI for a more advanced script if needed, 
    # but basic ones are fast to compute here.
    
    return meta


def download_openml_dataset(
    dataset_name: str,
    output_dir: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = 42,
) -> Path:
    """Download one dataset from OpenML, downsample if needed, save as CSV and JSON metadata."""
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

    # Dataset cap at 100K
    if len(combined) > max_rows:
        combined = combined.sample(
            n=max_rows,
            random_state=random_state,
        ).reset_index(drop=True)

    output_path = Path(output_dir) / f"{dataset_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    # Compute and save meta-features
    features_sampled = combined.drop(columns=[target_column])
    target_sampled = combined[target_column]
    meta_features = compute_meta_features(features_sampled, target_sampled)
    
    meta_path = Path(output_dir) / f"{dataset_name}_meta.json"
    meta_path.write_text(json.dumps(meta_features, indent=2))
    
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
        print(f"Downloading {dataset_name}...")
        try:
            saved_paths[dataset_name] = download_openml_dataset(
                dataset_name=dataset_name,
                output_dir=output_dir,
                max_rows=max_rows,
                random_state=random_state,
            )
            print(f"  -> Saved to {saved_paths[dataset_name]}")
        except Exception as e:
            print(f"  -> Failed to download {dataset_name}: {e}")

    return saved_paths


def load_csv_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

if __name__ == "__main__":
    results = download_datasets_from_list()
