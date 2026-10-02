"""Statistical summaries, non-parametric tests, and effect sizes."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

from src.dataset_statistics import AnalysisConfig, run_dataset_level_analysis
from src.protocol import results_ledger_path


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
    Legacy pooled-row Friedman/Nemenyi helper.

    Corrected confirmatory analysis uses :func:`run_dataset_level_analysis`;
    this function remains only for historical exploratory notebooks.
    Returns the p-value of the Friedman test and the Nemenyi p-value matrix.
    """
    warnings.warn("Pooled-row Friedman/Nemenyi is legacy exploratory output; use dataset-level analysis.", RuntimeWarning)
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
    final_results_path: str | Path = results_ledger_path(),
    output_path: str | Path = "reports/tables/statistical_results.csv",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run the active dataset-cluster analysis and persist its summaries.

    The historical row-level Wilcoxon implementation is intentionally no
    longer used for corrected outputs.  This compatibility entry point keeps
    the old command name while applying the versioned dataset-level contract.
    """
    input_path = Path(final_results_path)
    out_path = Path(output_path)
    bundle = run_dataset_level_analysis(
        input_path,
        output_dir=out_path.parent / "dataset_level",
        config=AnalysisConfig(alpha=alpha),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.summaries.to_csv(out_path, index=False)
    return bundle.summaries

if __name__ == "__main__":
    run_wilcoxon_analysis()
