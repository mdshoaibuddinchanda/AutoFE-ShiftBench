"""Publication-quality figures for AutoFE-ShiftBench.

The original figures were written as one-off PNGs and used a few crowded
layouts.  This module now supports an explicit analysis scope and writes both
300-DPI PNG and vector PDF outputs without touching the historical figures.
"""

import json
import textwrap
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scikit_posthocs as sp
import scipy.stats as ss


# The primary paper scope is the 22 datasets for which the run produced a
# complete or near-complete block of results.  The three remaining configured
# datasets are deliberately excluded from regenerated publication assets:
# aps_failure was not completed and covertype/kddcup99 were stopped while
# their long-running final blocks were still in progress.
PRIMARY_DATASETS = {
    "haberman",
    "sonar",
    "ionosphere",
    "heart-disease",
    "breast-cancer-wisconsin",
    "blood-transfusion-service-center",
    "diabetes",
    "titanic",
    "credit-g",
    "wine-quality-red",
    "kr-vs-kp",
    "mushroom",
    "spambase",
    "jm1",
    "PhishingWebsites",
    "default-of-credit-card-clients",
    "magic-telescope",
    "dry-bean-dataset",
    "adult",
    "bank-marketing",
    "electricity",
    "airlines",
}

DOMAIN_BY_DATASET = {
    "haberman": "Healthcare",
    "heart-disease": "Healthcare",
    "breast-cancer-wisconsin": "Healthcare",
    "blood-transfusion-service-center": "Healthcare",
    "diabetes": "Healthcare",
    "adult": "Finance",
    "bank-marketing": "Finance",
    "credit-g": "Finance",
    "default-of-credit-card-clients": "Finance",
    "spambase": "Cybersecurity",
    "PhishingWebsites": "Cybersecurity",
    "jm1": "Software",
    "mushroom": "Biology",
    "dry-bean-dataset": "Biology",
    "electricity": "Engineering",
    "airlines": "Engineering",
    "sonar": "Other",
    "ionosphere": "Other",
    "titanic": "Other",
    "wine-quality-red": "Other",
    "kr-vs-kp": "Other",
    "magic-telescope": "Other",
}

def set_q1_publication_style():
    """Set seaborn styles for Elsevier/IEEE publication quality."""
    custom_palette = ["#2C3E50", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#7F8C8D", "#BDC3C7"]
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=custom_palette,
        font="sans-serif",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # Figures are commonly reduced to roughly 0.5--0.7 of their
            # generated width in a two-column manuscript.  Use deliberately
            # large print fonts so labels remain legible after that scaling.
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "grid.color": "#E5E5E5",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "legend.frameon": True,
            "legend.edgecolor": "#E5E5E5",
        },
    )


def _set_print_text(ax, *, tick=14, label=16, title=20, legend=14):
    """Apply a print-safe text hierarchy after seaborn has created artists."""
    ax.title.set_fontsize(title)
    ax.xaxis.label.set_fontsize(label)
    ax.yaxis.label.set_fontsize(label)
    ax.tick_params(axis="both", which="major", labelsize=tick, length=5, width=1)
    ax.tick_params(axis="both", which="minor", labelsize=max(tick - 1, 10))
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(legend)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(legend)

def _save_figure(fig, out_dir: Path, stem: str, dpi: int = 300):
    """Save a figure as a high-resolution PNG and a vector PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_data(
    results_path="reports/tables/results_stream.jsonl",
    datasets: Iterable[str] | None = PRIMARY_DATASETS,
):
    path = Path(results_path)
    if not path.exists():
        return pd.DataFrame()
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()
    if datasets is not None and "dataset" in df.columns:
        df = df[df["dataset"].isin(set(datasets))].copy()
    return df

def plot_fig2_dataset_diversity(out_dir: Path, datasets: Iterable[str] = PRIMARY_DATASETS, dpi: int = 300):
    set_q1_publication_style()
    domain_counts = pd.Series({d: DOMAIN_BY_DATASET[d] for d in datasets}).value_counts()
    domains = domain_counts.to_dict()
    df_dom = pd.DataFrame(list(domains.items()), columns=["Domain", "Count"]).sort_values("Count", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    sns.barplot(data=df_dom, x="Count", y="Domain", color="#3498DB", ax=ax)
    ax.set_title(f"Figure 2: Dataset Domain Diversity (N={len(set(datasets))})", weight="bold", pad=15)
    ax.set_xlabel("Number of datasets")
    ax.set_ylabel("")
    ax.bar_label(ax.containers[0], fmt="%d", fontsize=15, padding=5, color="#1f2937")
    _set_print_text(ax, tick=15, label=17, title=21)
    _save_figure(fig, out_dir, "Figure_2_Dataset_Diversity", dpi)

def plot_fig3_pipeline_ranking(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    if df.empty or "roc_auc" not in df.columns: return
    set_q1_publication_style()
    
    pipeline_order = df.groupby("pipeline")["roc_auc"].mean().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    sns.barplot(
        data=df, x="pipeline", y="roc_auc", 
        order=pipeline_order,
        estimator=np.mean, errorbar=("ci", 95),
        color="#2C3E50", alpha=0.9, ax=ax
    )
    ax.set_title("Figure 3: Overall Pipeline Ranking (Mean ROC-AUC)", weight="bold", pad=15)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("")
    ax.set_xticks(range(len(pipeline_order)))
    ax.set_xticklabels([textwrap.fill(str(x), 20) for x in pipeline_order], rotation=25, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=14, padding=5, color="#1f2937")
    _set_print_text(ax, tick=14, label=17, title=21)
    _save_figure(fig, out_dir, "Figure_3_Pipeline_Ranking", dpi)

def plot_fig4_model_ranking(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    if df.empty or "roc_auc" not in df.columns: return
    set_q1_publication_style()
    
    model_means = df.groupby("model")["roc_auc"].mean().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    sns.barplot(
        data=df, y="model", x="roc_auc", 
        order=model_means,
        estimator=np.mean, errorbar=("ci", 95),
        color="#95A5A6", ax=ax
    )
    ax.set_title("Figure 4: Classifier Robustness Ranking", weight="bold", pad=15)
    ax.set_xlabel("Mean ROC-AUC across shifts and pipelines")
    ax.set_ylabel("")
    ax.set_xlim(0.5, 1.0)
    ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=14, padding=5, color="#1f2937")
    _set_print_text(ax, tick=15, label=17, title=21)
    _save_figure(fig, out_dir, "Figure_4_Model_Ranking", dpi)

def plot_fig5_robustness_shift(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    if df.empty or "roc_auc" not in df.columns: return
    set_q1_publication_style()
    
    shift_df = df[df["condition"].str.startswith("missing_") | (df["condition"] == "clean")].copy()
    if shift_df.empty: return
    
    shift_df["severity"] = shift_df["condition"].apply(
        lambda x: 0.0 if x == "clean" else float(x.split("_")[-1])
    )
    pipelines_to_plot = ["Raw", "AutoFE_Baseline", "AutoFE_NoMultiply"]
    shift_df = shift_df[shift_df["pipeline"].isin(pipelines_to_plot)]
    
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    sns.lineplot(
        data=shift_df, x="severity", y="roc_auc", hue="pipeline",
        style="pipeline", markers=True, dashes=False, linewidth=2, markersize=8, ax=ax
    )
    ax.set_xlabel("Missing value fraction")
    ax.set_ylabel("ROC-AUC")
    # A frameless legend keeps the line trajectories visible and avoids an
    # opaque patch obscuring the Raw/AutoFE curves at their crossings.
    ax.legend(
        title="", loc="lower center", bbox_to_anchor=(0.5, 1.01),
        ncol=3, frameon=False, handlelength=2.4, columnspacing=1.0,
    )
    ax.set_title("Figure 5: Performance Degradation under Missing Values", weight="bold", pad=38)
    _set_print_text(ax, tick=14, label=17, title=21, legend=15)
    _save_figure(fig, out_dir, "Figure_5_Robustness_Shift", dpi)

def plot_fig6_runtime_memory(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    if df.empty or "train_time_s" not in df.columns: return
    set_q1_publication_style()
    
    # Calculate median times per pipeline
    agg = df.groupby("pipeline")[["autofe_gen_time_s", "train_time_s", "infer_time_s"]].median().reset_index()
    
    # Stacked bar
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    pipelines = agg["pipeline"].values
    
    ax.bar(pipelines, agg["autofe_gen_time_s"], label="Feature generation", color="#95A5A6")
    ax.bar(pipelines, agg["train_time_s"], bottom=agg["autofe_gen_time_s"], label="Training", color="#3498DB")
    ax.bar(pipelines, agg["infer_time_s"], bottom=agg["autofe_gen_time_s"] + agg["train_time_s"], label="Inference", color="#E74C3C")
    ax.set_title("Figure 6: Computational Overhead (Median Seconds)", weight="bold", pad=15)
    ax.set_ylabel("Time (seconds; log scale)")
    ax.set_xticks(range(len(pipelines)))
    ax.set_xticklabels([textwrap.fill(str(x), 20) for x in pipelines], rotation=30, ha="right")
    ax.set_yscale("log")
    ax.legend()
    _set_print_text(ax, tick=14, label=17, title=21, legend=15)
    _save_figure(fig, out_dir, "Figure_6_Runtime_Comparison", dpi)

def plot_fig7_feature_complexity(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    if df.empty or "n_retained" not in df.columns: return
    set_q1_publication_style()
    
    pipeline_order = df.groupby("pipeline")["n_retained"].median().sort_values().index
    
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    sns.boxplot(
        data=df, x="pipeline", y="n_retained", order=pipeline_order,
        color="#BDC3C7", showfliers=False, ax=ax
    )
    ax.set_title("Figure 7: Feature Dimensionality per Pipeline", weight="bold", pad=15)
    ax.set_ylabel("Number of retained features")
    ax.set_xlabel("")
    ax.set_xticks(range(len(pipeline_order)))
    ax.set_xticklabels([textwrap.fill(str(x), 20) for x in pipeline_order], rotation=30, ha="right")
    ax.set_yscale("log")
    _set_print_text(ax, tick=14, label=17, title=21)
    _save_figure(fig, out_dir, "Figure_7_Feature_Complexity", dpi)

def plot_fig8_cd_diagram(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    if df.empty or "roc_auc" not in df.columns: return
    set_q1_publication_style()
    
    # Aggregate to dataset-fold level so models are averaged out for the test
    agg = df.groupby(["dataset", "seed", "fold", "condition", "pipeline"])["roc_auc"].mean().reset_index()
    
    # Pivot so each row is a task, each col is a pipeline
    pivot = agg.pivot(index=["dataset", "seed", "fold", "condition"], columns="pipeline", values="roc_auc").dropna()
    
    if len(pivot) < 10: return # not enough data
    
    # Nemenyi critical difference plot requires scikit-posthocs
    avg_ranks = pivot.rank(axis=1, ascending=False).mean()
    
    # Compute the Nemenyi matrix directly.  This avoids a blank panel with
    # older scikit-posthocs releases whose deprecated NumPy aliases and
    # studentized-range wrapper are incompatible with modern environments.
    from statsmodels.stats.libqsturng import psturng

    rank_matrix = pivot.rank(axis=1, method="average", ascending=False)
    mean_ranks = rank_matrix.mean(axis=0).to_numpy(dtype=float)
    k = len(mean_ranks)
    n = len(rank_matrix)
    p_values_array = np.ones((k, k), dtype=float)
    standard_error = np.sqrt(k * (k + 1.0) / (6.0 * n))
    for i in range(k):
        for j in range(i + 1, k):
            q_value = np.sqrt(2.0) * abs(mean_ranks[i] - mean_ranks[j]) / standard_error
            p_value = float(np.asarray(psturng(q_value, k, np.inf)).reshape(-1)[0])
            p_values_array[i, j] = p_value
            p_values_array[j, i] = p_value
    p_values = pd.DataFrame(p_values_array, index=pivot.columns, columns=pivot.columns)
    
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.heatmap(
        p_values,
        annot=True,
        fmt=".3g",
        vmin=0,
        vmax=1,
        cmap="RdYlGn_r",
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 20, "weight": "bold"},
        cbar_kws={"label": "Nemenyi p-value"},
        ax=ax,
    )
    ax.set_title("Figure 8: Nemenyi Post-hoc P-value Heatmap", weight="bold", pad=15)
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Pipeline")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    _set_print_text(ax, tick=17, label=19, title=22)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label("Nemenyi p-value", fontsize=16)
    _save_figure(fig, out_dir, "Figure_8_CD_Diagram", dpi)

def plot_fig9_heatmap(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    if df.empty or "roc_auc" not in df.columns: return
    set_q1_publication_style()
    
    pivot = df.pivot_table(index="dataset", columns="pipeline", values="roc_auc", aggfunc="mean")
    
    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 20, "weight": "bold"},
        cmap="YlGnBu",
        cbar_kws={"label": "Mean ROC-AUC"},
        ax=ax,
    )
    ax.set_title("Figure 9: Dataset vs Pipeline Performance Heatmap", weight="bold", pad=15)
    ax.set_ylabel("Dataset")
    ax.set_xlabel("Pipeline")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    _set_print_text(ax, tick=17, label=19, title=22)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Mean ROC-AUC", fontsize=18)
    _save_figure(fig, out_dir, "Figure_9_Heatmap", dpi)

def plot_fig10_ablation(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    if df.empty or "roc_auc" not in df.columns: return
    set_q1_publication_style()
    
    ablation_pl = ["Raw", "Raw_MI", "AutoFE_Baseline", "AutoFE_MI", "AutoFE_Random", "AutoFE_NoMultiply"]
    adf = df[df["pipeline"].isin(ablation_pl)].copy()
    if adf.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.barplot(
        data=adf, x="pipeline", y="roc_auc",
        order=ablation_pl, hue="pipeline", palette="viridis", legend=False,
        errorbar=("ci", 95), ax=ax
    )
    ax.set_title("Figure 10: Ablation Study of Feature Generation/Selection", weight="bold", pad=15)
    ax.set_ylabel("Mean ROC-AUC")
    ax.set_xlabel("")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(range(len(ablation_pl)))
    ax.set_xticklabels([textwrap.fill(str(x), 20) for x in ablation_pl], rotation=25, ha="right")
    # Seaborn creates one BarContainer per hue level.  Label every container,
    # not only the first (Raw) bar, so the ablation comparison is numerically
    # self-contained in the printed figure.
    for container in ax.containers:
        if getattr(container, "patches", None):
            ax.bar_label(container, fmt="%.3f", fontsize=14, padding=5, color="#1f2937")
    _set_print_text(ax, tick=14, label=17, title=21)
    _save_figure(fig, out_dir, "Figure_10_Ablation_Study", dpi)


def generate_all(
    results_path="reports/tables/results_stream.jsonl",
    out_dir="reports/figures/q1_paper_regenerated",
    datasets: Iterable[str] | None = PRIMARY_DATASETS,
    dpi: int = 300,
):
    df = _load_data(results_path, datasets=datasets)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_fig2_dataset_diversity(out_dir, datasets=datasets or PRIMARY_DATASETS, dpi=dpi)
    plot_fig3_pipeline_ranking(df, out_dir, dpi=dpi)
    plot_fig4_model_ranking(df, out_dir, dpi=dpi)
    plot_fig5_robustness_shift(df, out_dir, dpi=dpi)
    plot_fig6_runtime_memory(df, out_dir, dpi=dpi)
    plot_fig7_feature_complexity(df, out_dir, dpi=dpi)
    plot_fig8_cd_diagram(df, out_dir, dpi=dpi)
    plot_fig9_heatmap(df, out_dir, dpi=dpi)
    plot_fig10_ablation(df, out_dir, dpi=dpi)
    print(f"Generated Figures 2-10 for {len(set(datasets or PRIMARY_DATASETS))} datasets in {out_dir.absolute()}")

if __name__ == "__main__":
    generate_all()
