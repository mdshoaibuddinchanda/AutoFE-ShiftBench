"""SHAP value computation with intelligent subsampling for fast evaluation."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans


def compute_shap_values(
    model: BaseEstimator,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    model_type: str,
    max_background: int = 100,
    max_eval: int = 500,
) -> dict[str, Any]:
    """
    Compute SHAP values efficiently using K-means subsampling for slow explainers
    and TreeExplainer for tree-based models.
    """
    # Downsample evaluation set if too large
    if len(x_test) > max_eval:
        x_eval = x_test.sample(n=max_eval, random_state=42)
    else:
        x_eval = x_test

    # Determine Explainer Type
    normalized_type = model_type.strip().lower()
    tree_models = {"random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"}
    
    try:
        if normalized_type in tree_models:
            # Tree explainer is fast, but we still limit background if required by the model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_eval)
        else:
            # For linear, SVM, KNN, MLP, we use KernelExplainer which is very slow
            # So we use KMeans to summarize the background dataset
            if len(x_train) > max_background:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=max_background, random_state=42, n_init="auto")
                    kmeans.fit(x_train)
                    background = pd.DataFrame(kmeans.cluster_centers_, columns=x_train.columns)
            else:
                background = x_train.to_numpy()
                
            predict_fn = getattr(model, "predict_proba", model.predict)
            explainer = shap.KernelExplainer(predict_fn, background)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(x_eval, silent=True)

        # Handle different return formats (list of arrays for multiclass, single array for binary)
        if isinstance(shap_values, list):
            # Take the mean absolute SHAP value across all classes and samples
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
        # Format as a dictionary {feature_name: importance}
        feature_names = x_eval.columns.tolist()
        importance_dict = {
            feat: float(val) for feat, val in zip(feature_names, mean_abs_shap)
        }
        
        # Sort by importance descending
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    except Exception as e:
        return {"error": str(e)}
