"""Statistical summaries, non-parametric tests, and effect sizes."""

from __future__ import annotations

import warnings
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

from src.dataset_statistics import AnalysisConfig, run_dataset_level_analysis
from src.protocol import results_ledger_path
from src.sensitivity_analysis import SensitivityConfig, run_sensitivity_analysis


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
    pipeline_a: str = "Raw",
    pipeline_b: str = "AutoFE_Baseline",
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
        config=AnalysisConfig(alpha=alpha, pipeline_a=pipeline_a, pipeline_b=pipeline_b),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.summaries.to_csv(out_path, index=False)
    return bundle.summaries


def run_incomplete_run_sensitivity(
    manifest_db: str | Path,
    ledger_path: str | Path,
    run_id: str,
    *,
    output_dir: str | Path = "reports/analysis/sensitivity",
    config: SensitivityConfig | None = None,
):
    """Run the active manifest-backed incomplete-run sensitivity reader."""
    return run_sensitivity_analysis(manifest_db, ledger_path, run_id, output_dir=output_dir, config=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run corrected dataset-level or incomplete-run analyses")
    parser.add_argument("--sensitivity", action="store_true", help="Use the manifest-backed incomplete-run sensitivity analysis")
    parser.add_argument("--ledger", type=Path, default=results_ledger_path())
    parser.add_argument("--output", type=Path, default=Path("reports/tables/statistical_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/analysis/sensitivity"))
    parser.add_argument("--manifest-db", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--pipeline-a", default="Raw")
    parser.add_argument("--pipeline-b", default="AutoFE_Baseline")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutation-resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.sensitivity:
        if args.manifest_db is None or args.run_id is None:
            parser.error("--sensitivity requires --manifest-db and --run-id")
        result = run_incomplete_run_sensitivity(
            args.manifest_db,
            args.ledger,
            args.run_id,
            output_dir=args.output_dir,
            config=SensitivityConfig(bootstrap_resamples=args.bootstrap_resamples, permutation_resamples=args.permutation_resamples),
        )
        print(json.dumps({"snapshot_id": result.snapshot["snapshot_id"], "output_dir": str(args.output_dir), "summary_rows": len(result.summaries), "coverage_rows": len(result.coverage_tasks)}, sort_keys=True))
    else:
        run_wilcoxon_analysis(args.ledger, args.output, pipeline_a=args.pipeline_a, pipeline_b=args.pipeline_b)
