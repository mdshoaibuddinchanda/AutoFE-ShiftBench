"""Statistical summaries, non-parametric tests, and effect sizes."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's Delta effect size for two non-parametric samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if len(x) == 0 or len(y) == 0:
        return np.nan
        
    m, n = len(x), len(y)
    
    # Efficient broadcasting for pairwise comparisons
    x_matrix = np.tile(x, (n, 1)).T
    y_matrix = np.tile(y, (m, 1))
    
    diff = np.sign(x_matrix - y_matrix)
    return float(diff.mean())


def run_friedman_nemenyi(data: pd.DataFrame, value_col: str, group_col: str, block_col: str):
    """
    Run Friedman test and Nemenyi post-hoc on a DataFrame.
    Returns the p-value of the Friedman test and the Nemenyi p-value matrix.
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        warnings.warn("scikit-posthocs not installed. Nemenyi test skipped.")
        return np.nan, pd.DataFrame()
        
    # Pivot to get blocks as rows, groups as columns
    pivot = data.pivot(index=block_col, columns=group_col, values=value_col).dropna()
    
    if pivot.empty or pivot.shape[1] < 3:
        return np.nan, pd.DataFrame()
        
    # Friedman
    stat, p_val = friedmanchisquare(*[pivot[c] for c in pivot.columns])
    
    # Nemenyi
    # scikit-posthocs requires a melted format or block format depending on the function
    # posthoc_nemenyi_friedman takes a matrix
    nemenyi_res = sp.posthoc_nemenyi_friedman(pivot.values)
    nemenyi_res.columns = pivot.columns
    nemenyi_res.index = pivot.columns
    
    return float(p_val), nemenyi_res


def run_wilcoxon_analysis(
    final_results_path: str | Path = "reports/tables/results_stream.jsonl",
    output_path: str | Path = "reports/tables/statistical_results.csv",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run per-dataset Wilcoxon tests on paired cross-validation outcomes with Effect Sizes."""
    
    input_path = Path(final_results_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
        
    if input_path.suffix == ".jsonl":
        # Aggregate the stream first
        records = []
        with open(input_path, "r") as f:
            import json
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        results = pd.DataFrame(records)
    else:
        results = pd.read_csv(input_path)
        
    # We want to compare Pipeline A (Raw) vs B (AutoFE)
    # The experimental unit is a single Fold outcome for a specific Condition and Seed and Model
    # So we pair them on [dataset, seed, fold, condition, model]
    
    required_cols = {"dataset", "seed", "fold", "condition", "pipeline", "model", "roc_auc"}
    missing_cols = required_cols.difference(results.columns)
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")
        
    results = results.dropna(subset=["roc_auc"]).copy()
    if results.empty:
        raise ValueError("No valid rows for stats.")
        
    # Pair by all variables except pipeline
    pair_cols = ["dataset", "seed", "fold", "condition", "model"]
    
    pivoted = results.pivot(index=pair_cols, columns="pipeline", values="roc_auc").reset_index()
    if "Raw" not in pivoted.columns or "AutoFE" not in pivoted.columns:
        raise ValueError("Both 'Raw' and 'AutoFE' pipelines must exist to run paired tests.")
        
    pivoted = pivoted.dropna(subset=["Raw", "AutoFE"])
    
    dataset_rows = []
    
    for dataset, group in pivoted.groupby("dataset"):
        scores_raw = group["Raw"].to_numpy(dtype=float)
        scores_autofe = group["AutoFE"].to_numpy(dtype=float)
        
        if len(scores_raw) >= 2 and not np.allclose(scores_raw, scores_autofe):
            _stat, p_value = wilcoxon(scores_autofe, scores_raw, alternative="two-sided")
            p_value_float = float(p_value)
        else:
            p_value_float = float("nan")
            
        effect_size = cliffs_delta(scores_autofe, scores_raw)
        
        mean_raw = float(np.mean(scores_raw))
        mean_autofe = float(np.mean(scores_autofe))
        
        if np.isnan(mean_raw) or np.isnan(mean_autofe):
            winner = "undetermined"
        elif mean_autofe > mean_raw:
            winner = "AutoFE"
        elif mean_raw > mean_autofe:
            winner = "Raw"
        else:
            winner = "tie"
            
        dataset_rows.append({
            "dataset": dataset,
            "n_pairs": len(scores_raw),
            "raw_mean_auc": mean_raw,
            "autofe_mean_auc": mean_autofe,
            "mean_diff": mean_autofe - mean_raw,
            "cliffs_delta": effect_size,
            "p_value": p_value_float,
            "significant": bool(p_value_float < alpha) if not np.isnan(p_value_float) else False,
            "winner": winner
        })
        
    stats_df = pd.DataFrame(dataset_rows)
    stats_df = stats_df.sort_values("dataset")
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_path, index=False)
    
    return stats_df

if __name__ == "__main__":
    run_wilcoxon_analysis()
