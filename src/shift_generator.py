"""Shift generation utilities for the 10 perturbation conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _apply_gaussian_noise(
    x: pd.DataFrame,
    sigma: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply Gaussian noise to numeric columns."""
    shifted = x.copy()
    numeric_cols = shifted.select_dtypes(include=["number", "bool"]).columns
    
    for col in numeric_cols:
        series = pd.to_numeric(shifted[col], errors="coerce")
        # Standardize noise relative to the feature's std dev
        std = float(series.std(ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0
            
        noise = rng.normal(loc=0.0, scale=std * sigma, size=len(series))
        
        # We don't impute here; the pipeline handles imputation. We just add noise to non-NaNs.
        # Actually, adding noise to NaNs makes them non-NaN, so let's only add noise to non-NaNs.
        mask = series.notna()
        # Cast to float to avoid FutureWarning when adding float noise to int columns
        if not pd.api.types.is_float_dtype(shifted[col]):
            shifted[col] = shifted[col].astype(float)
        shifted.loc[mask, col] = series[mask] + noise[mask]
        
    return shifted


def _apply_missing_values(
    x: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Randomly inject NaNs across all features."""
    shifted = x.copy()
    n_samples = len(shifted)
    
    for col in shifted.columns:
        mask = rng.random(n_samples) < fraction
        if mask.any():
            shifted.loc[mask, col] = np.nan
            
    return shifted


def _apply_label_noise(
    y: pd.Series,
    fraction: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Randomly flip labels."""
    shifted_y = y.copy()
    n_samples = len(shifted_y)
    classes = shifted_y.dropna().unique()
    
    if len(classes) < 2:
        return shifted_y
        
    mask = rng.random(n_samples) < fraction
    indices = np.flatnonzero(mask)
    
    if len(indices) == 0:
        return shifted_y
        
    for idx in indices:
        current_class = shifted_y.iloc[idx]
        if pd.isna(current_class):
            continue
            
        possible_classes = [c for c in classes if c != current_class]
        if possible_classes:
            new_class = rng.choice(possible_classes)
            shifted_y.iloc[idx] = new_class
            
    return shifted_y

def _apply_class_prior_shift(
    y: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Class Prior Shift.
    Resample the labels to artificially change the class distribution.
    For simplicity, we randomly drop 50% of the majority class.
    Since we only perturb training data, this creates a mismatch with the test data prior.
    """
    shifted_y = y.copy()
    counts = shifted_y.value_counts()
    if len(counts) < 2:
        return shifted_y
        
    majority_class = counts.idxmax()
    majority_indices = shifted_y[shifted_y == majority_class].index
    
    # Drop half of the majority class by setting them to NaN (or we can just leave as is since we need equal length array?)
    # Wait, apply_perturbation returns x and y of the same length.
    # To truly do class prior shift without dropping rows (which messes up x alignment), we can flip some majority class instances to minority class.
    # This simulates a different prior without changing dataset size.
    minority_classes = [c for c in counts.index if c != majority_class]
    
    n_to_flip = int(len(majority_indices) * 0.5)
    flip_indices = rng.choice(majority_indices, size=n_to_flip, replace=False)
    
    for idx in flip_indices:
        shifted_y.loc[idx] = rng.choice(minority_classes)
        
    return shifted_y

def _apply_structured_label_drift(
    x: pd.DataFrame,
    y: pd.Series,
    fraction: float,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Structured Label Drift (replaces Concept Drift).
    Flip labels based on the distance from the mean of the most variant feature.
    """
    shifted_y = y.copy()
    numeric_cols = x.select_dtypes(include=["number"]).columns
    classes = shifted_y.dropna().unique()
    
    if len(classes) < 2 or len(numeric_cols) == 0:
        return _apply_label_noise(y, fraction, rng)
        
    # Find most variant feature (normalized)
    variances = x[numeric_cols].apply(lambda col: np.var((col - np.mean(col)) / (np.std(col) + 1e-9)))
    top_col = variances.idxmax()
    
    feature_vals = x[top_col]
    mean_val = np.mean(feature_vals)
    std_val = np.std(feature_vals) + 1e-9
    
    # Probability of flip increases with distance from mean
    distances = np.abs(feature_vals - mean_val) / std_val
    probs = 1 / (1 + np.exp(-distances))  # Sigmoid scaling
    
    # Normalize probabilities so average is `fraction`
    probs = probs * (fraction / (np.mean(probs) + 1e-9))
    probs = np.clip(probs, 0.0, 1.0)
    
    mask = rng.random(len(shifted_y)) < probs
    indices = np.flatnonzero(mask)
    
    for idx in indices:
        current_class = shifted_y.iloc[idx]
        if pd.isna(current_class):
            continue
        possible_classes = [c for c in classes if c != current_class]
        if possible_classes:
            shifted_y.iloc[idx] = rng.choice(possible_classes)
            
    return shifted_y

def _apply_feature_removal(
    x: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Feature Removal Shift.
    Drops the top `fraction` of features (simulated by dropping highest variance features).
    """
    shifted = x.copy()
    n_drop = max(1, int(len(shifted.columns) * fraction))
    
    if n_drop >= len(shifted.columns):
        n_drop = len(shifted.columns) - 1 # Keep at least one feature
        
    # We drop randomly to simulate sensor failure, but you could drop based on importance.
    # To keep it simple and robust, we randomly select cols to drop.
    cols_to_drop = rng.choice(shifted.columns, size=n_drop, replace=False)
    shifted = shifted.drop(columns=cols_to_drop)
    
    return shifted



def apply_perturbation(
    x: pd.DataFrame,
    y: pd.Series,
    shift_family: str,
    severity: float | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply one of the 8 distinct shift families.
    """
    rng = np.random.default_rng(random_state)
    
    if shift_family in ["clean", "covariate_shift", "population_shift"]:
        # covariate and population are split-level shifts, so no in-place perturbation needed here
        return x.copy(), y.copy()
        
    elif shift_family == "gaussian_noise":
        if severity is None:
            severity = 0.05
        x_shifted = _apply_gaussian_noise(x, severity, rng)
        return x_shifted, y.copy()
        
    elif shift_family == "missing_values":
        if severity is None:
            severity = 0.10
        x_shifted = _apply_missing_values(x, severity, rng)
        return x_shifted, y.copy()
        
    elif shift_family == "label_noise":
        if severity is None:
            severity = 0.10
        y_shifted = _apply_label_noise(y, severity, rng)
        return x.copy(), y_shifted

    elif shift_family == "class_prior_shift":
        y_shifted = _apply_class_prior_shift(y, rng)
        return x.copy(), y_shifted
        
    elif shift_family == "feature_removal":
        if severity is None:
            severity = 0.20
        x_shifted = _apply_feature_removal(x, severity, rng)
        return x_shifted, y.copy()
        
    else:
        raise ValueError(f"Unknown shift family: {shift_family}")
