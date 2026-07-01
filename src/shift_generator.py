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


def apply_perturbation(
    x: pd.DataFrame,
    y: pd.Series,
    condition: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply one of the 10 defined perturbation conditions.
    
    Conditions:
    - clean
    - gaussian_0.01, gaussian_0.05, gaussian_0.10
    - missing_0.05, missing_0.10, missing_0.20
    - label_0.05, label_0.10, label_0.20
    """
    rng = np.random.default_rng(random_state)
    
    if condition == "clean":
        return x.copy(), y.copy()
        
    elif condition.startswith("gaussian_"):
        sigma = float(condition.split("_")[1])
        x_shifted = _apply_gaussian_noise(x, sigma, rng)
        return x_shifted, y.copy()
        
    elif condition.startswith("missing_"):
        fraction = float(condition.split("_")[1])
        x_shifted = _apply_missing_values(x, fraction, rng)
        return x_shifted, y.copy()
        
    elif condition.startswith("label_"):
        fraction = float(condition.split("_")[1])
        y_shifted = _apply_label_noise(y, fraction, rng)
        return x.copy(), y_shifted
        
    else:
        raise ValueError(f"Unknown perturbation condition: {condition}")
