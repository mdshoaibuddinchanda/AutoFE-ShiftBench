"""Statistical summaries and significance testing for benchmark experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _safe_std(series: pd.Series) -> float:
    """Return sample std when possible, else 0.0 for singleton groups."""
    values = series.to_numpy(dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _benjamini_hochberg_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted p-values."""
    if p_values.size == 0:
        return p_values

    order = np.argsort(p_values)
    ordered = p_values[order]
    m = float(len(ordered))

    adjusted_ordered = np.empty_like(ordered, dtype=float)
    prev = 1.0
    for idx in range(len(ordered) - 1, -1, -1):
        rank = float(idx + 1)
        value = min(prev, (ordered[idx] * m) / rank)
        adjusted_ordered[idx] = value
        prev = value

    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return np.clip(adjusted, 0.0, 1.0)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two equally indexed series, returning NaN on zero denominator."""
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator.notna() & (~np.isclose(denominator.to_numpy(dtype=float), 0.0))
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def compare_mean_std_shift(
    baseline_df: pd.DataFrame,
    shifted_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare mean and standard deviation between two datasets."""
    baseline_numeric = baseline_df.select_dtypes(include=["number"])
    shifted_numeric = shifted_df.select_dtypes(include=["number"])

    common_cols = baseline_numeric.columns.intersection(shifted_numeric.columns)
    if common_cols.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "baseline_mean",
                "shifted_mean",
                "mean_shift",
                "baseline_std",
                "shifted_std",
                "std_shift",
            ]
        )

    baseline_stats = baseline_numeric[common_cols].agg(["mean", "std"]).T
    shifted_stats = shifted_numeric[common_cols].agg(["mean", "std"]).T

    summary = pd.DataFrame(
        {
            "feature": common_cols,
            "baseline_mean": baseline_stats["mean"].values,
            "shifted_mean": shifted_stats["mean"].values,
            "mean_shift": (shifted_stats["mean"] - baseline_stats["mean"]).values,
            "baseline_std": baseline_stats["std"].values,
            "shifted_std": shifted_stats["std"].values,
            "std_shift": (shifted_stats["std"] - baseline_stats["std"]).values,
        }
    )

    return summary


def run_wilcoxon_analysis(
    final_results_path: str | Path = "reports/tables/final_results.csv",
    output_path: str | Path = "reports/tables/statistical_results.csv",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run per-dataset Wilcoxon tests on per-seed aggregated A vs B performance.

    Inferential unit is one seed. For each dataset (and optional model/feature
    slice), we first aggregate ROC-AUC across all shift types and severities
    within each seed and pipeline, then apply paired Wilcoxon A vs B.
    """
    input_path = Path(final_results_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Final results file not found: {input_path}")

    results = pd.read_csv(input_path)
    required_cols = {"dataset", "seed", "pipeline", "roc_auc"}
    missing_cols = required_cols.difference(results.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns in final results: {sorted(missing_cols)}")

    results = results.dropna(subset=["roc_auc"]).copy()
    if results.empty:
        raise ValueError("No valid roc_auc rows available for statistical analysis")

    optional_group_cols = [
        column
        for column in ["model_type", "feature_count", "feature_count_used"]
        if column in results.columns
    ]
    group_cols = ["dataset", *optional_group_cols]

    dataset_rows: list[dict[str, object]] = []
    for group_key, group in results.groupby(group_cols):
        if isinstance(group_key, tuple):
            group_values = list(group_key)
        else:
            group_values = [group_key]
        group_record = dict(zip(group_cols, group_values, strict=True))

        paired_scores = (
            group.groupby(["seed", "pipeline"], as_index=False)
            .agg(mean_roc_auc=("roc_auc", "mean"))
            .pivot(index="seed", columns="pipeline", values="mean_roc_auc")
        )

        if "A" not in paired_scores.columns or "B" not in paired_scores.columns:
            continue

        paired = paired_scores[["A", "B"]].dropna()
        scores_a = paired["A"].to_numpy(dtype=float)
        scores_b = paired["B"].to_numpy(dtype=float)

        if len(scores_a) >= 2 and not np.allclose(scores_a, scores_b):
            _stat, p_value = wilcoxon(scores_a, scores_b, alternative="two-sided")
            p_value_float = float(p_value)
        else:
            p_value_float = float("nan")

        mean_a = float(np.mean(scores_a)) if len(scores_a) else float("nan")
        mean_b = float(np.mean(scores_b)) if len(scores_b) else float("nan")
        std_a = float(np.std(scores_a, ddof=1)) if len(scores_a) > 1 else 0.0
        std_b = float(np.std(scores_b, ddof=1)) if len(scores_b) > 1 else 0.0

        if np.isnan(mean_a) or np.isnan(mean_b):
            winner = "undetermined"
        elif mean_a > mean_b:
            winner = "A"
        elif mean_b > mean_a:
            winner = "B"
        else:
            winner = "tie"

        dataset_rows.append(
            {
                **group_record,
                "n_seeds": int(paired.index.nunique()),
                "n_pairs": int(len(scores_a)),
                "pipeline_a_mean": mean_a,
                "pipeline_b_mean": mean_b,
                "pipeline_a_std": std_a,
                "pipeline_b_std": std_b,
                "p_value": p_value_float,
                "significant": bool(p_value_float < alpha) if not np.isnan(p_value_float) else False,
                "winner": winner,
            }
        )

    stats_df = pd.DataFrame(dataset_rows)
    if stats_df.empty:
        stats_df = pd.DataFrame(
            columns=[
                *group_cols,
                "n_seeds",
                "pipeline_a_mean",
                "pipeline_b_mean",
                "pipeline_a_std",
                "pipeline_b_std",
                "p_value",
                "significant",
                "winner",
            ]
        )
    else:
        valid_mask = stats_df["p_value"].notna().to_numpy()
        valid_p_values = stats_df.loc[valid_mask, "p_value"].to_numpy(dtype=float)

        stats_df["p_value_bonferroni"] = np.nan
        stats_df["p_value_fdr_bh"] = np.nan
        stats_df["significant_bonferroni"] = False
        stats_df["significant_fdr_bh"] = False

        if valid_p_values.size > 0:
            n_tests = float(valid_p_values.size)
            bonferroni_values = np.minimum(valid_p_values * n_tests, 1.0)
            fdr_values = _benjamini_hochberg_adjust(valid_p_values)

            stats_df.loc[valid_mask, "p_value_bonferroni"] = bonferroni_values
            stats_df.loc[valid_mask, "p_value_fdr_bh"] = fdr_values
            stats_df.loc[valid_mask, "significant_bonferroni"] = (
                bonferroni_values < alpha
            )
            stats_df.loc[valid_mask, "significant_fdr_bh"] = (
                fdr_values < alpha
            )

        stats_df = stats_df.sort_values(group_cols)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_path, index=False)
    return stats_df


def build_runtime_comparison_table(
    final_results_path: str | Path = "reports/tables/final_results.csv",
    output_path: str | Path = "reports/tables/runtime_results.csv",
) -> pd.DataFrame:
    """Build runtime comparison table between Pipeline A and Pipeline B.

    This function expects runtime columns to exist in final_results and computes
    per-slice means/stds for both pipelines, along with B-over-A ratios.
    """
    input_path = Path(final_results_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Final results file not found: {input_path}")

    results = pd.read_csv(input_path)
    required_cols = {
        "dataset",
        "seed",
        "pipeline",
        "train_time_s",
        "inference_time_s",
        "total_time_s",
    }
    missing_cols = required_cols.difference(results.columns)
    if missing_cols:
        raise KeyError(
            "Missing required columns in final results for runtime analysis: "
            f"{sorted(missing_cols)}"
        )

    optional_group_cols = [
        column
        for column in ["model_type", "feature_count", "feature_count_used"]
        if column in results.columns
    ]

    seed_runtime_cols = ["dataset", "seed", *optional_group_cols, "pipeline"]
    seed_runtime = (
        results.sort_values(seed_runtime_cols)
        .drop_duplicates(subset=seed_runtime_cols, keep="first")
    )

    grouped = (
        seed_runtime.groupby(["dataset", *optional_group_cols, "pipeline"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            train_time_mean_s=("train_time_s", "mean"),
            train_time_std_s=("train_time_s", _safe_std),
            inference_time_mean_s=("inference_time_s", "mean"),
            inference_time_std_s=("inference_time_s", _safe_std),
            total_time_mean_s=("total_time_s", "mean"),
            total_time_std_s=("total_time_s", _safe_std),
        )
        .sort_values(["dataset", *optional_group_cols, "pipeline"])
    )

    base_cols = ["dataset", *optional_group_cols]

    pipeline_a = grouped[grouped["pipeline"] == "A"].drop(columns=["pipeline"]).rename(
        columns={
            "n_seeds": "pipeline_a_n_seeds",
            "train_time_mean_s": "pipeline_a_train_time_mean_s",
            "train_time_std_s": "pipeline_a_train_time_std_s",
            "inference_time_mean_s": "pipeline_a_inference_time_mean_s",
            "inference_time_std_s": "pipeline_a_inference_time_std_s",
            "total_time_mean_s": "pipeline_a_total_time_mean_s",
            "total_time_std_s": "pipeline_a_total_time_std_s",
        }
    )

    pipeline_b = grouped[grouped["pipeline"] == "B"].drop(columns=["pipeline"]).rename(
        columns={
            "n_seeds": "pipeline_b_n_seeds",
            "train_time_mean_s": "pipeline_b_train_time_mean_s",
            "train_time_std_s": "pipeline_b_train_time_std_s",
            "inference_time_mean_s": "pipeline_b_inference_time_mean_s",
            "inference_time_std_s": "pipeline_b_inference_time_std_s",
            "total_time_mean_s": "pipeline_b_total_time_mean_s",
            "total_time_std_s": "pipeline_b_total_time_std_s",
        }
    )

    comparison = pipeline_a.merge(pipeline_b, on=base_cols, how="outer")
    comparison["train_time_diff_b_minus_a_s"] = (
        comparison["pipeline_b_train_time_mean_s"]
        - comparison["pipeline_a_train_time_mean_s"]
    )
    comparison["inference_time_diff_b_minus_a_s"] = (
        comparison["pipeline_b_inference_time_mean_s"]
        - comparison["pipeline_a_inference_time_mean_s"]
    )
    comparison["total_time_diff_b_minus_a_s"] = (
        comparison["pipeline_b_total_time_mean_s"]
        - comparison["pipeline_a_total_time_mean_s"]
    )

    comparison["train_time_ratio_b_over_a"] = _safe_divide(
        comparison["pipeline_b_train_time_mean_s"],
        comparison["pipeline_a_train_time_mean_s"],
    )
    comparison["inference_time_ratio_b_over_a"] = _safe_divide(
        comparison["pipeline_b_inference_time_mean_s"],
        comparison["pipeline_a_inference_time_mean_s"],
    )
    comparison["total_time_ratio_b_over_a"] = _safe_divide(
        comparison["pipeline_b_total_time_mean_s"],
        comparison["pipeline_a_total_time_mean_s"],
    )

    comparison = comparison.sort_values(base_cols)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_path, index=False)
    return comparison


def build_main_results_table(
    statistical_results_path: str | Path = "reports/tables/statistical_results.csv",
    output_path: str | Path = "reports/tables/main_results.csv",
) -> pd.DataFrame:
    """Build paper-ready main results table from statistical results."""
    input_path = Path(statistical_results_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Statistical results file not found: {input_path}")

    stats_df = pd.read_csv(input_path)
    required_cols = {
        "dataset",
        "pipeline_a_mean",
        "pipeline_b_mean",
        "pipeline_a_std",
        "pipeline_b_std",
        "p_value",
        "winner",
    }
    missing_cols = required_cols.difference(stats_df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns in statistical results: {sorted(missing_cols)}")

    table = stats_df.copy()
    table["std_deviation"] = (table["pipeline_a_std"] + table["pipeline_b_std"]) / 2.0
    optional_cols = [
        column
        for column in [
            "model_type",
            "feature_count",
            "feature_count_used",
            "n_seeds",
            "n_pairs",
            "p_value_bonferroni",
            "p_value_fdr_bh",
            "significant",
            "significant_bonferroni",
            "significant_fdr_bh",
        ]
        if column in table.columns
    ]
    table = table[
        [
            "dataset",
            *optional_cols,
            "pipeline_a_mean",
            "pipeline_b_mean",
            "std_deviation",
            "p_value",
            "winner",
        ]
    ].sort_values(["dataset", *optional_cols])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return table
