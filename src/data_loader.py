"""Utilities for downloading, loading, and analyzing benchmark tabular datasets."""

from __future__ import annotations

import re
import warnings
from pathlib import Path


import json

import numpy as np
import pandas as pd
import yaml

from src.provenance import file_sha256


DEFAULT_TARGET_COLUMN = "target"
DEFAULT_MAX_ROWS = 100_000

# Candidate names or IDs to tolerate naming variants in OpenML.
OPENML_NAME_CANDIDATES: dict[str, list[str | int]] = {
    "adult": ["adult", "adult-income"],
    "bank-marketing": ["bank-marketing", "bank_marketing", "Bank_Marketing"],
    "aps_failure": [41138, "aps_failure", "aps-failure"],
    "electricity": ["electricity"],
    "covertype": ["covertype", "Covertype"],
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


def compute_meta_features(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> dict[str, float]:
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
        sample_idx = np.random.default_rng(random_state).choice(n_samples, sample_size, replace=False)
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
    """Download one configured benchmark dataset, downsample if needed, and save CSV plus metadata."""
    print(f"Fetching {dataset_name}...")
    uci_metadata = None
    if dataset_name == "dry-bean-dataset":
        # Dry Bean is a UCI dataset, not OpenML data_id 42585 (which is a different
        # dataset). Keep the benchmark's intended dataset name and record its source.
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=602)
        x = dataset.data.features
        target_frame = dataset.data.targets
        if target_frame is None or target_frame.shape[1] != 1:
            raise ValueError("UCI Dry Bean must expose exactly one target column")
        y = target_frame.iloc[:, 0]
        uci_metadata = dataset.metadata
        resolved_name = str(uci_metadata.get("name", "Dry Bean"))
    else:
        dataset, resolved_name = _fetch_openml_with_fallbacks(dataset_name)
        x = dataset.data
        y = dataset.target
    print(f"Fetched {dataset_name}. Processing X/y...")

    if y is None and dataset.frame is not None:
        fallback_targets = {
            "heart-disease": "target",
        }
        target_name = fallback_targets.get(dataset_name)
        if target_name and target_name in dataset.frame.columns:
            y = dataset.frame[target_name]
            x = dataset.frame.drop(columns=[target_name])
        else:
            raise ValueError(
                f"OpenML dataset '{dataset_name}' has no target metadata and no explicit fallback target is configured"
            )

    if x is None or y is None:
        raise ValueError(f"Dataset '{dataset_name}' does not provide X/y data")

    features = x.copy()
    target_column = DEFAULT_TARGET_COLUMN
    while target_column in features.columns:
        target_column = "target_label" if target_column == DEFAULT_TARGET_COLUMN else f"_{target_column}_"

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
    meta_features = compute_meta_features(features_sampled, target_sampled, random_state=random_state)
    print(f"Meta features done for {dataset_name}.")
    
    if uci_metadata is not None:
        target_columns = uci_metadata.get("target_col", [])
        source_target_name = str(target_columns[0]) if target_columns else "Class"
        dataset_identity = {
            "provider": "UCI",
            "requested_name": dataset_name,
            "resolved_name": resolved_name,
            "uci_id": str(uci_metadata.get("uci_id", 602)),
            "doi": uci_metadata.get("dataset_doi"),
            "repository_url": uci_metadata.get("repository_url"),
            "last_updated": uci_metadata.get("last_updated"),
        }
    else:
        details = getattr(dataset, "details", {}) or {}
        dataset_id = details.get("id") if isinstance(details, dict) else None
        dataset_version = details.get("version") if isinstance(details, dict) else None
        target_names = getattr(dataset, "target_names", None)
        if isinstance(target_names, (list, tuple, np.ndarray)):
            source_target_name = str(target_names[0]) if len(target_names) else "target"
        else:
            source_target_name = str(target_names or "target")
        dataset_identity = {
            "provider": "OpenML",
            "requested_name": dataset_name,
            "resolved_name": str(resolved_name),
            "openml_id": str(dataset_id) if dataset_id is not None else None,
            "openml_version": str(dataset_version) if dataset_version is not None else None,
        }
    proxy_audit = inspect_target_proxy_candidates(
        features_sampled,
        target_sampled,
        target_column=target_column,
        source_target_name=source_target_name,
    )
    meta_features["dataset_identity"] = dataset_identity
    meta_features["target_column"] = target_column
    meta_features["source_target_name"] = source_target_name
    meta_features["saved_csv_sha256"] = file_sha256(output_path)
    meta_features["schema_columns"] = [str(column) for column in combined.columns]
    meta_features["target_proxy_review"] = proxy_audit

    meta_path = Path(output_dir) / f"{dataset_name}_meta.json"
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(child) for key, child in value.items()}
        if isinstance(value, list):
            return [json_safe(child) for child in value]
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    meta_path.write_text(json.dumps(json_safe(meta_features), indent=2, allow_nan=False))
    
    return output_path


def download_datasets_from_list(
    dataset_list_path: str | Path = "config/dataset_list.yaml",
    output_dir: str | Path = "data/raw",
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = 42,
    dataset_names: list[str] | None = None,
) -> dict[str, Path]:
    """Download selected configured datasets into the raw-data directory."""
    configured_names = load_dataset_names(dataset_list_path)
    selected_names = configured_names if dataset_names is None else list(dataset_names)
    unknown = sorted(set(selected_names).difference(configured_names))
    if unknown:
        raise ValueError(f"Requested datasets are not configured: {unknown}")

    saved_paths: dict[str, Path] = {}
    failures: dict[str, str] = {}
    for dataset_name in selected_names:
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
            failures[dataset_name] = f"{type(e).__name__}: {e}"

    if failures:
        names = ", ".join(sorted(failures))
        raise RuntimeError(f"Dataset download incomplete for: {names}. Details: {failures}")

    return saved_paths


def load_csv_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def inspect_target_proxy_candidates(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    target_column: str,
    source_target_name: str | None = None,
) -> dict[str, list[str]]:
    """Return exact-copy and name-based target proxy candidates for manual review.

    Candidates are reported, never deleted automatically. Similarity can indicate
    a legitimate feature or a leakage proxy and requires dataset-level review.
    """
    exact_copies: list[str] = []
    suspicious_names: list[str] = []
    target_values = target.reset_index(drop=True).astype("string")
    tokens = {"target", "label", "class", "outcome", "response"}
    if source_target_name:
        tokens.add(str(source_target_name).casefold())
    for column in features.columns:
        candidate = features[column].reset_index(drop=True)
        if len(candidate) == len(target_values) and candidate.astype("string").equals(target_values):
            exact_copies.append(str(column))
        name_tokens = set(re.findall(r"[a-z0-9]+", str(column).casefold()))
        if name_tokens.intersection(tokens):
            suspicious_names.append(str(column))
    return {
        "target_column": [target_column],
        "exact_target_copies_for_manual_review": exact_copies,
        "target_like_names_for_manual_review": suspicious_names,
    }

if __name__ == "__main__":
    results = download_datasets_from_list()
