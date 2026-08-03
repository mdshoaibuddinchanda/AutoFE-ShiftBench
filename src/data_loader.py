"""Utilities for downloading, loading, and analyzing benchmark tabular datasets."""

from __future__ import annotations

import warnings
from pathlib import Path


import json

import numpy as np
import pandas as pd
import yaml


DEFAULT_TARGET_COLUMN = "target"
DEFAULT_MAX_ROWS = 100_000

# Candidate names or IDs to tolerate naming variants in OpenML.
OPENML_NAME_CANDIDATES: dict[str, list[str | int]] = {
    "adult": ["adult", "adult-income"],
    "bank-marketing": ["bank-marketing", "bank_marketing", "Bank_Marketing"],
    "aps_failure": [41138, "aps_failure", "aps-failure"],
    "electricity": ["electricity"],
    "covertype": ["covertype", "Covertype"],
    "dry-bean-dataset": [42585, "dry-bean-dataset", "Dry_Bean_Dataset"],
    "crop-recommendation": [43491, "crop-recommendation", "Crop_Recommendation"],
    "breast-cancer-wisconsin": ["breast-cancer-wisconsin", "breast_cancer", "wdbc"],
    "heart-disease": [43398, "heart-disease", "heart-statlog", 53],
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
    "rice-cammeo-and-osmancik": [43586, "rice-cammeo-and-osmancik", "Rice_Cammeo_Osmancik"],
    "titanic": [40945, "titanic", "Titanic"],
    "airlines": [1169, "airlines", "Airlines"],
    "kddcup99": [1113, "kddcup99", "KDDCup99"],
    "kr-vs-kp": [3, "kr-vs-kp", "kr-vs-kp"],
    "blood-transfusion-service-center": [1464, "blood-transfusion-service-center", "blood-transfusion-service-center"],
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
                if isinstance(candidate, int):
                    dataset = fetch_openml(
                        data_id=candidate,
                        as_frame=True,
                        parser="auto",
                    )
                else:
                    dataset = fetch_openml(
                        name=candidate,
                        version=version,
                        as_frame=True,
                        parser="auto",
                    )
                return dataset, str(candidate)
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
        print(f"Computing correlation matrix for {len(num_cols)} numerical features...")
        corr_matrix = features[num_cols].corr().abs().to_numpy()
        np.fill_diagonal(corr_matrix, np.nan)
        meta["average_correlation"] = float(np.nanmean(corr_matrix))
        print("Correlation matrix done.")
    else:
        meta["average_correlation"] = 0.0
        
    # Skewness and Kurtosis
    if len(num_cols) > 0:
        import scipy.stats
        num_data = features[num_cols].dropna()
        if len(num_data) > 0:
            skew_vals = scipy.stats.skew(num_data, axis=0, nan_policy='omit')
            kurt_vals = scipy.stats.kurtosis(num_data, axis=0, nan_policy='omit')
            meta["average_skewness"] = float(np.nanmean(skew_vals))
            meta["average_kurtosis"] = float(np.nanmean(kurt_vals))
        else:
            meta["average_skewness"] = 0.0
            meta["average_kurtosis"] = 0.0
    else:
        meta["average_skewness"] = 0.0
        meta["average_kurtosis"] = 0.0

    # Sparsity (percentage of exactly 0 values)
    meta["sparsity"] = float((features == 0).sum().sum() / total_cells * 100.0) if total_cells > 0 else 0.0

    # Entropy (Shannon entropy)
    import scipy.stats
    entropies = []
    for col in features.columns:
        counts = features[col].value_counts(normalize=True)
        if len(counts) > 0:
            entropies.append(scipy.stats.entropy(counts))
    meta["average_entropy"] = float(np.mean(entropies)) if entropies else 0.0

    # Feature Redundancy (Percentage of highly correlated feature pairs |corr| > 0.8)
    if len(num_cols) > 1 and "corr_matrix" in locals():
        high_corr_pairs = np.sum(corr_matrix > 0.8) / 2.0  # symmetric
        total_pairs = (len(num_cols) * (len(num_cols) - 1)) / 2.0
        meta["feature_redundancy"] = float(high_corr_pairs / total_pairs)
    else:
        meta["feature_redundancy"] = 0.0

    # Average Mutual Information (AMI) with Target
    try:
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import LabelEncoder
        # Sample to speed up
        sample_size = min(n_samples, 2000)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        f_sample = features.iloc[sample_idx]
        t_sample = target.iloc[sample_idx]
        
        # Prepare numeric/encoded for AMI
        f_numeric = pd.DataFrame()
        for col in f_sample.columns:
            if pd.api.types.is_numeric_dtype(f_sample[col]):
                f_numeric[col] = f_sample[col].fillna(f_sample[col].median())
            else:
                f_numeric[col] = LabelEncoder().fit_transform(f_sample[col].astype(str))
                
        t_encoded = LabelEncoder().fit_transform(t_sample.astype(str))
        ami_scores = mutual_info_classif(f_numeric, t_encoded, random_state=42)
        meta["average_mutual_information"] = float(np.mean(ami_scores))
    except Exception as e:
        print(f"Skipping AMI: {e}")
        meta["average_mutual_information"] = 0.0

    # Intrinsic Dimension (PCA 95% variance)
    try:
        if len(num_cols) > 1:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            num_data = features[num_cols].fillna(features[num_cols].median())
            scaled_data = StandardScaler().fit_transform(num_data)
            pca = PCA(n_components=0.95, random_state=42)
            pca.fit(scaled_data)
            meta["intrinsic_dimension"] = float(pca.n_components_)
        else:
            meta["intrinsic_dimension"] = 1.0
    except Exception as e:
        print(f"Skipping Intrinsic Dimension: {e}")
        meta["intrinsic_dimension"] = 1.0
    
    return meta


def download_openml_dataset(
    dataset_name: str,
    output_dir: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = 42,
) -> Path:
    """Download one dataset from OpenML, downsample if needed, save as CSV and JSON metadata."""
    print(f"Fetching {dataset_name}...")
    dataset, _resolved_name = _fetch_openml_with_fallbacks(dataset_name)
    print(f"Fetched {dataset_name}. Processing X/y...")

    x = dataset.data
    y = dataset.target

    if y is None and dataset.frame is not None:
        fallback_targets = {
            "heart-disease": "target",
        }
        target_name = fallback_targets.get(dataset_name)
        if target_name and target_name in dataset.frame.columns:
            y = dataset.frame[target_name]
            x = dataset.frame.drop(columns=[target_name])
        else:
            target_name = dataset.frame.columns[-1]
            y = dataset.frame[target_name]
            x = dataset.frame.drop(columns=[target_name])

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
    print(f"Computing meta features for {dataset_name}...")
    meta_features = compute_meta_features(features_sampled, target_sampled)
    print(f"Meta features done for {dataset_name}.")
    
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
