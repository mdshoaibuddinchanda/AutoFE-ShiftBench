"""XGBoost model training for raw and AutoFE pipelines."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from src.evaluation import evaluate_predictions


@dataclass(slots=True)
class ModelConfig:
    """Configuration for XGBoost training and evaluation."""

    task: str | None = None
    random_state: int = 42
    params: dict[str, Any] | None = None


def _infer_task(y: pd.Series) -> str:
    """Infer task from target characteristics."""
    if y.dtype.kind in {"O", "b", "U", "S"}:
        return "classification"

    unique_values = y.nunique(dropna=False)
    threshold = max(20, int(0.1 * len(y)))
    if unique_values <= threshold:
        return "classification"
    return "regression"


def build_xgboost_model(
    task: str = "classification",
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> BaseEstimator:
    """Build an XGBoost estimator for classification or regression."""
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required. Install requirements and retry."
        ) from exc

    default_params: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    normalized_task = task.strip().lower()
    if normalized_task == "classification":
        return XGBClassifier(
            **default_params,
            eval_metric="mlogloss",
        )
    if normalized_task == "regression":
        return XGBRegressor(
            **default_params,
            objective="reg:squarederror",
            eval_metric="rmse",
        )
    raise ValueError("Task must be 'classification' or 'regression'")


def _extract_feature_matrices(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract train/test feature matrices from payload."""
    key_pairs = (
        ("x_train_selected", "x_test_selected"),
        ("x_train_fe", "x_test_fe"),
        ("x_train", "x_test"),
    )
    for train_key, test_key in key_pairs:
        x_train = payload.get(train_key)
        x_test = payload.get(test_key)
        if isinstance(x_train, pd.DataFrame) and isinstance(x_test, pd.DataFrame):
            return x_train, x_test

    raise KeyError("Payload does not contain supported feature matrix keys")


def _fit_pipeline(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task: str,
    params: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    """Fit one XGBoost pipeline and return model and metrics."""
    x_train_model, x_test_model, feature_names = _sanitize_features(x_train, x_test)
    model = build_xgboost_model(task=task, params=params, random_state=random_state)

    label_encoder: LabelEncoder | None = None
    y_train_model = y_train
    y_train_eval = y_train
    y_test_eval = y_test

    if task == "classification":
        label_encoder = LabelEncoder()
        y_train_model = pd.Series(label_encoder.fit_transform(y_train.astype(str)))
        y_train_eval = pd.Series(label_encoder.transform(y_train.astype(str)))
        y_test_eval = pd.Series(label_encoder.transform(y_test.astype(str)))

    model.fit(x_train_model, y_train_model)
    y_pred_train = pd.Series(model.predict(x_train_model))
    y_pred_test = pd.Series(model.predict(x_test_model))

    train_metrics = evaluate_predictions(y_train_eval, y_pred_train, task=task)
    test_metrics = evaluate_predictions(y_test_eval, y_pred_test, task=task)

    if task == "classification":
        overfit_gap = float(train_metrics["accuracy"] - test_metrics["accuracy"])
    else:
        overfit_gap = float(train_metrics["r2"] - test_metrics["r2"])

    return {
        "model": model,
        "label_encoder": label_encoder,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfit_gap": overfit_gap,
        "feature_names": feature_names,
    }


def _sanitize_column_names(columns: pd.Index) -> list[str]:
    """Sanitize and de-duplicate feature names for XGBoost."""
    renamed_columns: list[str] = []
    seen: dict[str, int] = {}
    for index, original_name in enumerate(columns):
        sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", str(original_name))
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            sanitized = f"f_{index}"

        count = seen.get(sanitized, 0)
        seen[sanitized] = count + 1
        final_name = sanitized if count == 0 else f"{sanitized}_{count}"
        renamed_columns.append(final_name)

    return renamed_columns


def _sanitize_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Sanitize feature names and values for XGBoost compatibility."""
    x_train_clean = x_train.replace([np.inf, -np.inf], np.nan).copy()
    x_test_clean = x_test.replace([np.inf, -np.inf], np.nan).copy()

    train_means = x_train_clean.mean(axis=0, numeric_only=True)
    x_train_clean = x_train_clean.fillna(train_means).fillna(0.0)
    x_test_clean = x_test_clean.fillna(train_means).fillna(0.0)

    renamed_columns = _sanitize_column_names(x_train_clean.columns)

    x_train_clean.columns = renamed_columns
    x_test_clean.columns = renamed_columns
    return x_train_clean, x_test_clean, renamed_columns


def prepare_inference_features(
    x_input: pd.DataFrame,
    trained_feature_names: list[str],
) -> pd.DataFrame:
    """Sanitize and align inference features to trained model columns."""
    clean = x_input.replace([np.inf, -np.inf], np.nan).copy()
    means = clean.mean(axis=0, numeric_only=True)
    clean = clean.fillna(means).fillna(0.0)

    clean.columns = _sanitize_column_names(clean.columns)
    aligned = clean.reindex(columns=trained_feature_names, fill_value=0.0)
    return aligned


def train_xgboost_pipeline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    task: str,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train one XGBoost pipeline and return model bundle for inference."""
    model = build_xgboost_model(task=task, params=params, random_state=random_state)
    x_train_clean, _x_dummy, feature_names = _sanitize_features(x_train, x_train)

    label_encoder: LabelEncoder | None = None
    y_model = y_train
    if task == "classification":
        label_encoder = LabelEncoder()
        y_model = pd.Series(label_encoder.fit_transform(y_train.astype(str)))

    model.fit(x_train_clean, y_model)
    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "task": task,
    }


def _primary_score(metrics: dict[str, float], task: str) -> float:
    """Return primary metric used for pipeline ranking."""
    if task == "classification":
        return float(metrics["accuracy"])
    return float(metrics["r2"])


def _load_payload(path: Path) -> dict[str, Any]:
    """Load a pickle artifact payload."""
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact payload must be dict: {path}")
    return payload


def train_two_pipelines_for_dataset(
    raw_artifact_path: str | Path,
    autofe_artifact_path: str | Path,
    output_root: str | Path = "experiments/results/models",
    config: ModelConfig | None = None,
) -> Path:
    """Train and compare Pipeline A (raw) vs Pipeline B (AutoFE + selection)."""
    cfg = config or ModelConfig()
    raw_path = Path(raw_artifact_path)
    autofe_path = Path(autofe_artifact_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw artifact not found: {raw_path}")
    if not autofe_path.exists():
        raise FileNotFoundError(f"AutoFE artifact not found: {autofe_path}")

    raw_payload = _load_payload(raw_path)
    autofe_payload = _load_payload(autofe_path)

    for key in ("x_train", "x_test", "y_train", "y_test"):
        if key not in raw_payload:
            raise KeyError(f"Raw artifact missing key '{key}': {raw_path}")

    x_train_raw = raw_payload["x_train"]
    x_test_raw = raw_payload["x_test"]
    y_train = raw_payload["y_train"]
    y_test = raw_payload["y_test"]

    x_train_autofe, x_test_autofe = _extract_feature_matrices(autofe_payload)

    task = (cfg.task or _infer_task(y_train)).strip().lower()
    params = cfg.params or {"n_estimators": 100, "max_depth": 6}

    pipeline_a = _fit_pipeline(
        x_train=x_train_raw,
        x_test=x_test_raw,
        y_train=y_train,
        y_test=y_test,
        task=task,
        params=params,
        random_state=cfg.random_state,
    )
    pipeline_b = _fit_pipeline(
        x_train=x_train_autofe,
        x_test=x_test_autofe,
        y_train=y_train,
        y_test=y_test,
        task=task,
        params=params,
        random_state=cfg.random_state,
    )

    score_a = _primary_score(pipeline_a["test_metrics"], task)
    score_b = _primary_score(pipeline_b["test_metrics"], task)
    if score_a > score_b:
        preferred_pipeline = "pipeline_a_raw"
    elif score_b > score_a:
        preferred_pipeline = "pipeline_b_autofe_selection"
    else:
        preferred_pipeline = (
            "pipeline_a_raw"
            if abs(pipeline_a["overfit_gap"]) <= abs(pipeline_b["overfit_gap"])
            else "pipeline_b_autofe_selection"
        )

    dataset_name = str(raw_payload.get("dataset_name", raw_path.stem))
    dataset_dir = Path(output_root) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    model_a_path = dataset_dir / "pipeline_a_raw_model.pkl"
    model_b_path = dataset_dir / "pipeline_b_autofe_selection_model.pkl"
    with model_a_path.open("wb") as file:
        pickle.dump(
            {
                "pipeline": "pipeline_a_raw",
                "task": task,
                "model": pipeline_a["model"],
                "label_encoder": pipeline_a["label_encoder"],
                "feature_names": pipeline_a["feature_names"],
            },
            file,
        )
    with model_b_path.open("wb") as file:
        pickle.dump(
            {
                "pipeline": "pipeline_b_autofe_selection",
                "task": task,
                "model": pipeline_b["model"],
                "label_encoder": pipeline_b["label_encoder"],
                "feature_names": pipeline_b["feature_names"],
            },
            file,
        )

    summary = {
        "dataset_name": dataset_name,
        "task": task,
        "model_type": "xgboost",
        "params": params,
        "pipeline_a_raw": {
            "train_metrics": pipeline_a["train_metrics"],
            "test_metrics": pipeline_a["test_metrics"],
            "overfit_gap": pipeline_a["overfit_gap"],
            "num_features": len(pipeline_a["feature_names"]),
            "model_path": str(model_a_path),
        },
        "pipeline_b_autofe_selection": {
            "train_metrics": pipeline_b["train_metrics"],
            "test_metrics": pipeline_b["test_metrics"],
            "overfit_gap": pipeline_b["overfit_gap"],
            "num_features": len(pipeline_b["feature_names"]),
            "model_path": str(model_b_path),
        },
        "preferred_pipeline": preferred_pipeline,
    }

    summary_path = dataset_dir / "pipeline_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def train_two_pipelines_for_all(
    processed_dir: str | Path = "data/processed",
    output_root: str | Path = "experiments/results/models",
    config: ModelConfig | None = None,
) -> dict[str, Path]:
    """Train two pipelines for every dataset with available artifacts."""
    cfg = config or ModelConfig()
    raw_paths = [
        path
        for path in sorted(Path(processed_dir).glob("*.pkl"))
        if not path.stem.endswith("_features") and not path.stem.endswith("_selected")
    ]
    if not raw_paths:
        raise FileNotFoundError(f"No raw processed artifacts found in: {processed_dir}")

    outputs: dict[str, Path] = {}
    for raw_path in raw_paths:
        dataset_name = raw_path.stem
        selected_path = Path(processed_dir) / f"{dataset_name}_selected.pkl"
        engineered_path = Path(processed_dir) / f"{dataset_name}_features.pkl"

        if selected_path.exists():
            autofe_path = selected_path
        elif engineered_path.exists():
            autofe_path = engineered_path
        else:
            continue

        outputs[dataset_name] = train_two_pipelines_for_dataset(
            raw_artifact_path=raw_path,
            autofe_artifact_path=autofe_path,
            output_root=output_root,
            config=cfg,
        )

    return outputs


def build_default_model(
    task: str = "classification",
    random_state: int = 42,
) -> BaseEstimator:
    """Backward-compatible wrapper for building default XGBoost model."""
    return build_xgboost_model(task=task, params=None, random_state=random_state)


def train_model(model: BaseEstimator, x_train, y_train) -> BaseEstimator:
    """Backward-compatible estimator fit wrapper."""
    model.fit(x_train, y_train)
    return model


def predict_model(model: BaseEstimator, x_test):
    """Backward-compatible estimator predict wrapper."""
    return model.predict(x_test)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model training."""
    parser = argparse.ArgumentParser(description="Train two XGBoost pipelines")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-root", default="experiments/results/models")
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default=None,
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = ModelConfig(
        task=args.task,
        random_state=args.random_state,
        params={"n_estimators": args.n_estimators, "max_depth": args.max_depth},
    )
    results = train_two_pipelines_for_all(
        processed_dir=args.processed_dir,
        output_root=args.output_root,
        config=config,
    )
    for dataset_name, summary_path in results.items():
        print(f"{dataset_name}: {summary_path}")
