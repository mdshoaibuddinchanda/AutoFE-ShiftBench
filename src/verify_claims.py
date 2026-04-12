"""Verify benchmark claims against generated result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FINAL_COLS = {
    "dataset",
    "seed",
    "shift_type",
    "severity",
    "pipeline",
    "roc_auc",
}

REQUIRED_STATS_COLS = {
    "dataset",
    "pipeline_a_mean",
    "pipeline_b_mean",
    "p_value",
    "significant",
    "winner",
}


def _check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {name}: {sorted(missing)}")


def verify_claims(final_results_path: Path, statistical_results_path: Path) -> dict[str, float | int | bool]:
    """Verify core empirical claims from generated CSV outputs."""
    final_df = pd.read_csv(final_results_path)
    stats_df = pd.read_csv(statistical_results_path)

    _check_columns(final_df, REQUIRED_FINAL_COLS, "final_results")
    _check_columns(stats_df, REQUIRED_STATS_COLS, "statistical_results")

    # Dataset-level wins and significance.
    winner_a = int((stats_df["winner"] == "A").sum())
    winner_b = int((stats_df["winner"] == "B").sum())
    winner_tie = int((stats_df["winner"] == "tie").sum())
    all_significant = bool(stats_df["significant"].astype(bool).all())

    # Aggregate performance and spread.
    agg = final_df.groupby("pipeline")["roc_auc"].agg(["mean", "std"])
    mean_a = float(agg.loc["A", "mean"])
    mean_b = float(agg.loc["B", "mean"])
    std_a = float(agg.loc["A", "std"])
    std_b = float(agg.loc["B", "std"])

    # Shift degradation trend.
    sev_mean = (
        final_df.groupby(["severity", "pipeline"], as_index=False)
        .agg(mean_roc_auc=("roc_auc", "mean"))
        .pivot(index="severity", columns="pipeline", values="mean_roc_auc")
        .sort_index()
    )
    slope_a = float(np.polyfit(sev_mean.index.values, sev_mean["A"].values, 1)[0])
    slope_b = float(np.polyfit(sev_mean.index.values, sev_mean["B"].values, 1)[0])

    gap = sev_mean["A"] - sev_mean["B"]
    gap_first = float(gap.iloc[0])
    gap_last = float(gap.iloc[-1])

    # Output integrity.
    expected_rows = (
        final_df["dataset"].nunique()
        * final_df["seed"].nunique()
        * final_df["shift_type"].nunique()
        * final_df["severity"].nunique()
        * final_df["pipeline"].nunique()
    )

    return {
        "rows": int(len(final_df)),
        "expected_rows": int(expected_rows),
        "rows_match_expected": bool(len(final_df) == expected_rows),
        "winner_a": winner_a,
        "winner_b": winner_b,
        "winner_tie": winner_tie,
        "all_significant": all_significant,
        "p_value_min": float(stats_df["p_value"].min()),
        "p_value_max": float(stats_df["p_value"].max()),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": std_a,
        "std_b": std_b,
        "slope_a": slope_a,
        "slope_b": slope_b,
        "gap_first": gap_first,
        "gap_last": gap_last,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify benchmark claims from CSV outputs")
    parser.add_argument("--final-results", default="reports/tables/final_results.csv")
    parser.add_argument("--stats", default="reports/tables/statistical_results.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = verify_claims(Path(args.final_results), Path(args.stats))

    print("CLAIM CHECK SUMMARY")
    print(f"rows={result['rows']} expected_rows={result['expected_rows']} rows_match={result['rows_match_expected']}")
    print(f"wins: A={result['winner_a']} B={result['winner_b']} tie={result['winner_tie']}")
    print(
        "significance: all_significant="
        f"{result['all_significant']} p_range=[{result['p_value_min']:.6g}, {result['p_value_max']:.6g}]"
    )
    print(
        "overall: "
        f"mean_A={result['mean_a']:.6f} mean_B={result['mean_b']:.6f} "
        f"std_A={result['std_a']:.6f} std_B={result['std_b']:.6f}"
    )
    print(
        "degradation slopes: "
        f"slope_A={result['slope_a']:.6f} slope_B={result['slope_b']:.6f} "
        f"gap_first={result['gap_first']:.6f} gap_last={result['gap_last']:.6f}"
    )


if __name__ == "__main__":
    main()
