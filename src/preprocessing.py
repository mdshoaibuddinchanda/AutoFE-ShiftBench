"""Train/test-safe preprocessing for benchmark tabular datasets."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


SUPPORTED_ENCODINGS = {"onehot", "label"}


@dataclass(slots=True)
class PreprocessingConfig:
    """Configuration for preprocessing raw benchmark datasets."""

    target_column: str = "target"
    test_size: float = 0.2
    random_state: int = 42
    encoding: str = "onehot"
    scale_numeric: bool = False


def _resolve_target_column(df: pd.DataFrame, target_column: str) -> str:
    """Resolve target column name with fallback support."""
    if target_column in df.columns:
        return target_column
    if "target_label" in df.columns:
        return "target_label"
    raise KeyError(
        f"Target column '{target_column}' not found and fallback 'target_label' is absent"
    )


def _build_preprocessor(
    x_train: pd.DataFrame,
    encoding: str,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build a split-safe sklearn preprocessor fitted only on training features."""
    if encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(
            f"Unsupported encoding '{encoding}'. Choose from {sorted(SUPPORTED_ENCODINGS)}"
        )

    numeric_cols = list(x_train.select_dtypes(include=["number", "bool"]).columns)
    categorical_cols = list(x_train.select_dtypes(exclude=["number", "bool"]).columns)

    transformers = []
    if numeric_cols:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="mean")),
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=numeric_steps), numeric_cols))

    if categorical_cols:
        if encoding == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No feature columns available for preprocessing")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _get_stratify_target(target: pd.Series) -> pd.Series | None:
    """Return stratification target when safe, otherwise None."""
    if target.nunique(dropna=False) < 2:
        return None

    class_counts = target.value_counts(dropna=False)
    if class_counts.min() < 2:
        return None

    return target


def _to_dense_array(values: Any) -> np.ndarray:
    """Convert scipy sparse outputs to dense arrays when needed."""
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values)


def preprocess_dataset(
    raw_csv_path: str | Path,
    output_dir: str | Path = "data/processed",
    config: PreprocessingConfig | None = None,
) -> Path:
    """Preprocess one raw dataset and store split-safe artifacts as .pkl."""
    cfg = config or PreprocessingConfig()
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    df = pd.read_csv(raw_path)
    target_column = _resolve_target_column(df, cfg.target_column)

    x = df.drop(columns=[target_column])
    y = df[target_column]

    # Critical rule: split before fitting imputation/encoding/scaling.
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=_get_stratify_target(y),
    )

    preprocessor = _build_preprocessor(
        x_train=x_train_raw,
        encoding=cfg.encoding,
        scale_numeric=cfg.scale_numeric,
    )

    x_train_array = _to_dense_array(preprocessor.fit_transform(x_train_raw))
    x_test_array = _to_dense_array(preprocessor.transform(x_test_raw))
    feature_names = preprocessor.get_feature_names_out().tolist()

    x_train_processed = pd.DataFrame(x_train_array, columns=feature_names)
    x_test_processed = pd.DataFrame(x_test_array, columns=feature_names)

    payload = {
        "dataset_name": raw_path.stem,
        "target_column": target_column,
        "encoding": cfg.encoding,
        "scale_numeric": cfg.scale_numeric,
        "x_train": x_train_processed.reset_index(drop=True),
        "x_test": x_test_processed.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "feature_names": feature_names,
        "preprocessor": preprocessor,
    }

    output_path = Path(output_dir) / f"{raw_path.stem}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(payload, file)

    return output_path


def preprocess_raw_datasets(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    config: PreprocessingConfig | None = None,
) -> dict[str, Path]:
    """Preprocess all CSV files in data/raw and save to data/processed."""
    csv_paths = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {Path(raw_dir)}")

    processed_paths: dict[str, Path] = {}
    for csv_path in csv_paths:
        processed_paths[csv_path.stem] = preprocess_dataset(
            raw_csv_path=csv_path,
            output_dir=output_dir,
            config=config,
        )

    return processed_paths


def load_processed_dataset(file_path: str | Path) -> dict[str, Any]:
    """Load a saved preprocessing artifact (.pkl)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")

    with path.open("rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Processed payload must be a dictionary")

    return payload


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper for quick in-memory preprocessing."""
    processed = df.copy().drop_duplicates()

    numeric_cols = processed.select_dtypes(include=["number"]).columns
    categorical_cols = processed.select_dtypes(exclude=["number"]).columns

    for col in numeric_cols:
        processed[col] = processed[col].fillna(processed[col].mean())

    for col in categorical_cols:
        mode_value = processed[col].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else "missing"
        processed[col] = processed[col].fillna(fill_value)

    return processed


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess raw datasets safely")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--target-column", default="target")
    parser.add_argument(
        "--encoding",
        default="onehot",
        choices=sorted(SUPPORTED_ENCODINGS),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--scale-numeric", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = PreprocessingConfig(
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        encoding=args.encoding,
        scale_numeric=args.scale_numeric,
    )
    paths = preprocess_raw_datasets(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        config=config,
    )
    for dataset_name, path in paths.items():
        print(f"{dataset_name}: {path}")
