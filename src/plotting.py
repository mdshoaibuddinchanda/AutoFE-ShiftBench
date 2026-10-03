"""Publication-quality plotting for the AutoFE robustness benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.protocol import results_ledger_path


def set_publication_style():
    """Set seaborn styles for IEEE/ACM publication quality."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="colorblind",
        font="serif",
        rc={
            "font.family": "serif",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        },
    )


def plot_overall_robustness_cd(
    results_path: str | Path,
    output_dir: str | Path = "reports/figures"
):
    """
    Plot Critical Difference (CD) diagram for overall robustness across 10 models 
    under different shift conditions. (Figure 1-5 style)
    """
    raise NotImplementedError("Empty CD placeholder retired; use corrected src.plotting_q1 dataset sign-flip/Holm panels.")


def plot_performance_degradation(results_path=results_ledger_path(),output_dir='reports/figures'):
    from src.reporting import report_inputs
    from src.plotting_q1 import plot_fig5_robustness_shift
    df,_=report_inputs(results_path)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    return plot_fig5_robustness_shift(df,out)

def plot_runtime_efficiency(results_path=results_ledger_path(),output_dir='reports/figures'):
    from src.reporting import report_inputs
    from src.plotting_q1 import plot_fig6_runtime_memory
    df,_=report_inputs(results_path)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    return plot_fig6_runtime_memory(df,out)

def generate_all_plots(results_path=results_ledger_path()):
    from src.plotting_q1 import generate_all
    return generate_all(results_path)


if __name__ == '__main__':
    generate_all_plots()
