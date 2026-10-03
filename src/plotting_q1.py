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

from src.dataset_statistics import AnalysisConfig, AnalysisInputError, run_dataset_level_analysis
from src.protocol import results_ledger_path
from src.reporting import report_inputs,summaries,dataset_descriptive,caption


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

# Regenerated readers keep the historical figures available while exposing all
# new fair-baseline and operator-ablation identities when those rows exist.
ABLATION_PIPELINES = [
    "Raw", "Raw_Full", "Raw_Capped", "Raw_Variance", "Raw_MI",
    "AutoFE_Baseline", "AutoFE_NoMultiply", "AutoFE_AddSub",
    "AutoFE_AddSubDiv", "AutoFE_NoDivision", "AutoFE_MultiplyOnly",
    "AutoFE_DivideOnly", "AutoFE_MI", "AutoFE_Random",
]

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
    results_path=results_ledger_path(),
    datasets: Iterable[str] | None = None,
):
    return report_inputs(results_path,datasets=datasets)[0] if Path(results_path).exists() else pd.DataFrame()


def _contrast_panel(df,out_dir,stem,title,dpi=300,condition=None):
    table=summaries(df)
    if condition and not table.empty:
        table=table[table['stratum'].str.contains(condition,regex=False)]
    if table.empty:
        (out_dir/(stem+'_unsupported.txt')).write_text('No supported corrected contrast rows in declared scope.',encoding='utf-8')
        return
    fig,ax=plt.subplots(figsize=(12,max(5,len(table)*.35)),constrained_layout=True)
    for i,row in enumerate(table.itertuples()):
        value=row.estimate_b_minus_a
        if pd.notna(value):
            ax.plot(value,i,'o',color='#3498DB')
        if pd.notna(row.ci_lower) and pd.notna(row.ci_upper):
            ax.hlines(i,row.ci_lower,row.ci_upper,color='#2C3E50',linewidth=2)
    ax.set_yticks(range(len(table)),[textwrap.fill(str(row.stratum),70) for row in table.itertuples()])
    ax.axvline(0,color='grey',linestyle='--')
    ax.set_xlabel('Dataset-equal paired ROC-AUC difference (B - A); dataset bootstrap CI')
    ax.set_title(title+'; declared strata kept separate')
    _save_figure(fig,out_dir,stem,dpi)


def plot_fig2_dataset_diversity(out_dir: Path, datasets: Iterable[str] = PRIMARY_DATASETS, dpi: int = 300):
    set_q1_publication_style()
    domain_counts = pd.Series({d: DOMAIN_BY_DATASET.get(d,"Other") for d in datasets}).value_counts()
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
    _contrast_panel(df,out_dir,'Figure_3_Pipeline_Ranking','Primary paired pipeline contrast',dpi)

def plot_fig4_model_ranking(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    _contrast_panel(df,out_dir,'Figure_4_Model_Ranking','Paired contrasts by estimator and condition',dpi)

def plot_fig5_robustness_shift(df: pd.DataFrame, out_dir: Path, dpi: int = 300):
    _contrast_panel(df,out_dir,'Figure_5_Robustness_Shift','Training missingness contrast',dpi,condition='missing_values')

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
    table=summaries(df)
    if table.empty: return
    matrix=table.set_index('stratum')[['p_value_adjusted_holm']]
    if matrix.notna().any().any():
        fig,ax=plt.subplots(figsize=(7,max(5,len(matrix)*.35)),constrained_layout=True)
        sns.heatmap(matrix,annot=True,fmt='.3g',vmin=0,vmax=1,ax=ax,cbar_kws={'label':'Holm adjusted p'})
        ax.set_title('Declared dataset sign-flip family; unsupported cells missing')
        _save_figure(fig,out_dir,'Figure_8_CD_Diagram',dpi)
    else:
        (out_dir/'Figure_8_CD_Diagram_unsupported.txt').write_text('No supported adjusted p values; no CD or Nemenyi inference.',encoding='utf-8')

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
    _contrast_panel(df,out_dir,'Figure_10_Ablation_Study','Configured paired contrast (ablation requires explicit pipeline_b)',dpi)

def generate_all(
    results_path=results_ledger_path(),
    out_dir="reports/figures/q1_paper_regenerated",
    datasets: Iterable[str] | None = None,
    dpi: int = 300,
    analysis_config: AnalysisConfig | None = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df,bundle=report_inputs(results_path,config=analysis_config,datasets=datasets,output_dir=out_dir/'dataset_level_analysis')
    (out_dir/'figure_captions.md').write_text(caption(bundle),encoding='utf-8')
    actual_datasets=sorted(df['dataset'].unique()) if not df.empty else []
    figure_config={**bundle.config,'report_schema':'dataset_unit_reports_v2','dpi':dpi,'formats':['png','pdf'],
        'analysis_input_task_ids':sorted(df['scientific_task_id'].dropna().unique()) if 'scientific_task_id' in df else []}
    (out_dir/'figure_config.json').write_text(json.dumps(figure_config,sort_keys=True),encoding='utf-8')
    if not actual_datasets:
        (out_dir/'unsupported.txt').write_text('No valid selected run records; no empirical figures generated.',encoding='utf-8')
        return bundle
    plot_fig2_dataset_diversity(out_dir, datasets=actual_datasets, dpi=dpi)
    plot_fig3_pipeline_ranking(df, out_dir, dpi=dpi)
    plot_fig4_model_ranking(df, out_dir, dpi=dpi)
    plot_fig5_robustness_shift(df, out_dir, dpi=dpi)
    plot_fig6_runtime_memory(df, out_dir, dpi=dpi)
    plot_fig7_feature_complexity(df, out_dir, dpi=dpi)
    plot_fig8_cd_diagram(df, out_dir, dpi=dpi)
    plot_fig9_heatmap(df, out_dir, dpi=dpi)
    plot_fig10_ablation(df, out_dir, dpi=dpi)
    print(f"Generated Figures 2-10 for {len(actual_datasets)} datasets in {out_dir.absolute()}")
    return bundle

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--ledger',type=Path,default=results_ledger_path())
    parser.add_argument('--run-id')
    parser.add_argument('--datasets')
    parser.add_argument('--output-dir',default='reports/figures/q1_paper_regenerated')
    parser.add_argument('--bootstrap-resamples',type=int,default=2000)
    parser.add_argument('--permutation-resamples',type=int,default=5000)
    parser.add_argument('--dpi',type=int,default=300)
    args=parser.parse_args()
    generate_all(args.ledger,args.output_dir,datasets=None if args.datasets is None else args.datasets.split(','),dpi=args.dpi,
        analysis_config=AnalysisConfig(run_id=args.run_id,bootstrap_resamples=args.bootstrap_resamples,permutation_resamples=args.permutation_resamples))
