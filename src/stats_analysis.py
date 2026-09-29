"""Paired, dataset-level analysis for the prespecified primary contrasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PRESPECIFIED_PIPELINES = (
    "AutoFE_Baseline", "AutoFE_MI", "AutoFE_Random", "AutoFE_NoMultiply",
)
PRIMARY_CONDITIONS = {
    "clean",
    "gaussian_noise_0.01", "gaussian_noise_0.05", "gaussian_noise_0.10",
    "missing_values_0.05", "missing_values_0.10", "missing_values_0.20",
    "label_noise_0.05", "label_noise_0.10", "label_noise_0.20",
}
CONDITION_ALIASES = {
    "gaussian_noise_0.1": "gaussian_noise_0.10",
    "missing_values_0.1": "missing_values_0.10",
    "missing_values_0.2": "missing_values_0.20",
    "label_noise_0.1": "label_noise_0.10",
    "label_noise_0.2": "label_noise_0.20",
}


def _holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down family-wise adjusted p-values, retaining NaN positions."""
    adjusted = [float("nan")] * len(pvalues)
    valid = [(index, value) for index, value in enumerate(pvalues) if np.isfinite(value)]
    ordered = sorted(valid, key=lambda item: item[1])
    running_max = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - rank) * value))
        adjusted[index] = running_max
    return adjusted


def _read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    if path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        frame = pd.DataFrame(records)
    else:
        frame = pd.read_csv(path)
    required = {"dataset", "seed", "fold", "condition", "pipeline", "model", "roc_auc"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required results columns: {sorted(missing)}")
    if "status" in frame:
        frame = frame[frame["status"] == "success"].copy()
    frame["condition"] = frame["condition"].replace(CONDITION_ALIASES)
    frame = frame[frame["condition"].isin(PRIMARY_CONDITIONS)].copy()
    frame["roc_auc"] = pd.to_numeric(frame["roc_auc"], errors="coerce")
    frame = frame.dropna(subset=["roc_auc"])
    if frame.empty:
        raise ValueError("No successful primary-condition ROC-AUC rows are available")
    return frame


def run_wilcoxon_analysis(
    final_results_path: str | Path | None = None,
    output_path: str | Path | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare Raw with four fixed AutoFE variants using datasets as n.

    Scores are first paired on dataset/seed/fold/condition/model, then averaged
    within dataset and pipeline. Each dataset contributes one paired value to
    each Wilcoxon test; Holm correction covers the four prespecified contrasts.
    """
    if final_results_path is None:
        raise ValueError("Pass a corrected-run results path explicitly; historical estimates are not corrected results")
    input_path = Path(final_results_path)
    results = _read_results(input_path)
    pipelines_present = set(results["pipeline"].astype(str))
    missing_pipelines = {"Raw", *PRESPECIFIED_PIPELINES}.difference(pipelines_present)
    if missing_pipelines:
        raise ValueError(f"Primary comparison is missing pipelines: {sorted(missing_pipelines)}")

    pairing = ["dataset", "seed", "fold", "condition", "model", "pipeline"]
    # Collapse retried duplicate successes deterministically before pairing.
    results = results.groupby(pairing, as_index=False, dropna=False)["roc_auc"].mean()
    pair_cols = ["dataset", "seed", "fold", "condition", "model"]
    wide = results.pivot(index=pair_cols, columns="pipeline", values="roc_auc").reset_index()

    rows: list[dict[str, object]] = []
    pvalues: list[float] = []
    for pipeline in PRESPECIFIED_PIPELINES:
        paired = wide[[*pair_cols, "Raw", pipeline]].dropna(subset=["Raw", pipeline]).copy()
        dataset_scores = paired.groupby("dataset")[["Raw", pipeline]].mean()
        differences = (dataset_scores[pipeline] - dataset_scores["Raw"]).to_numpy(dtype=float)
        if len(differences) >= 2 and not np.allclose(differences, 0.0):
            p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
        elif len(differences) >= 2:
            p_value = 1.0
        else:
            p_value = float("nan")
        pvalues.append(p_value)
        rows.append({
            "comparison": f"Raw vs {pipeline}",
            "autofe_pipeline": pipeline,
            "n_datasets": int(len(differences)),
            "mean_dataset_auc_difference": float(np.mean(differences)) if len(differences) else np.nan,
            "median_dataset_auc_difference": float(np.median(differences)) if len(differences) else np.nan,
            "dataset_wins": int(np.sum(differences > 0)),
            "dataset_ties": int(np.sum(np.isclose(differences, 0.0))),
            "dataset_losses": int(np.sum(differences < 0)),
            "p_value": p_value,
            "conditions_used": ",".join(sorted(set(results["condition"].astype(str)))),
            "independent_unit": "dataset",
        })

    adjusted = _holm_adjust(pvalues)
    for row, p_adjusted in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = p_adjusted
        row["significant"] = bool(np.isfinite(p_adjusted) and p_adjusted < alpha)
    output = pd.DataFrame(rows)
    out_path = Path(output_path) if output_path else input_path.with_name("statistical_results_dataset_level.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    run_wilcoxon_analysis(args.results, args.output)
