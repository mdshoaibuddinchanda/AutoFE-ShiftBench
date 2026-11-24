"""Regenerate the complete empirical figure set from the corrected audit.

The historical seven-pipeline ledger is unavailable. Figures 2-13 therefore
visualize the distinct corrected 14-pipeline audit, with scope and aggregation
stated in each figure. The workflow figure is a protocol schematic.
"""

from __future__ import annotations

import argparse
import csv
import collections
import json
import math
import shutil
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
from matplotlib.patches import Patch
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "reports" / "benchmark_runs" / "combined_results_20261122.jsonl"
AUDIT_PATH = ROOT / "reports" / "reviewer_experimental_audit_20261122.json"
PREFLIGHT_PATH = ROOT / "reports" / "benchmark_runs" / "preflight_eligibility.json"
DEFAULT_OUTPUT = ROOT / "paper" / "figures"
SUBMISSION_OUTPUT = ROOT / "KNOSYS-D-26-17927"
FIGURE_DPI = 350

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "font.size": 9.5,
    "axes.titleweight": "bold",
    "axes.titlepad": 16,
    "axes.labelweight": "bold",
    "axes.labelpad": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.pad": 7,
    "ytick.major.pad": 8,
    "legend.fontsize": 8.5,
    "savefig.dpi": FIGURE_DPI,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
PIPELINES = [
    "Raw", "Raw_Full", "Raw_Capped", "Raw_Variance", "Raw_MI",
    "AutoFE_Baseline", "AutoFE_MI", "AutoFE_Random", "AutoFE_NoMultiply",
    "AutoFE_AddSub", "AutoFE_AddSubDiv", "AutoFE_NoDivision",
    "AutoFE_MultiplyOnly", "AutoFE_DivideOnly",
]
PIPELINE_LABELS = {
    "Raw": "Raw",
    "Raw_Full": "Raw, full",
    "Raw_Capped": "Raw, capped",
    "Raw_Variance": "Raw, variance",
    "Raw_MI": "Raw, MI",
    "AutoFE_Baseline": "AutoFE, baseline",
    "AutoFE_MI": "AutoFE, MI selection",
    "AutoFE_Random": "AutoFE, random selection",
    "AutoFE_NoMultiply": "AutoFE, no multiplication",
    "AutoFE_AddSub": "AutoFE, add/subtract",
    "AutoFE_AddSubDiv": "AutoFE, add/subtract/divide",
    "AutoFE_NoDivision": "AutoFE, no division",
    "AutoFE_MultiplyOnly": "AutoFE, multiply only",
    "AutoFE_DivideOnly": "AutoFE, divide only",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "extra_trees": "Extra Trees",
    "linear_svm": "Linear SVM",
    "knn": "k-nearest neighbors",
    "gaussian_nb": "Gaussian Naive Bayes",
    "mlp": "Multilayer perceptron",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}
DATASET_LABELS = {
    "haberman": "Haberman",
    "sonar": "Sonar",
    "ionosphere": "Ionosphere",
    "heart-disease": "Heart disease",
    "breast-cancer-wisconsin": "Breast cancer",
    "blood-transfusion-service-center": "Blood transfusion",
    "diabetes": "Diabetes",
    "titanic": "Titanic",
    "credit-g": "Credit-G",
    "wine-quality-red": "Wine quality (red)",
    "kr-vs-kp": "KR-vs-KP",
    "mushroom": "Mushroom",
    "spambase": "Spambase",
    "jm1": "JM1",
    "PhishingWebsites": "Phishing Websites",
    "default-of-credit-card-clients": "Credit card default",
    "magic-telescope": "Magic Telescope",
    "dry-bean-dataset": "Dry Bean",
    "adult": "Adult",
    "bank-marketing": "Bank Marketing",
    "electricity": "Electricity",
    "aps_failure": "APS Failure",
    "covertype": "Covertype",
    "airlines": "Airlines",
    "kddcup99": "KDD Cup 99",
}
DOMAINS = {
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
    "kddcup99": "Cybersecurity",
    "jm1": "Software",
    "mushroom": "Biology",
    "dry-bean-dataset": "Biology",
    "electricity": "Engineering",
    "airlines": "Engineering",
    "aps_failure": "Engineering",
    "sonar": "Other",
    "ionosphere": "Other",
    "titanic": "Other",
    "wine-quality-red": "Other",
    "kr-vs-kp": "Other",
    "magic-telescope": "Other",
}
CONDITION_LABELS = {
    ("gaussian_noise", 0.01): "Gaussian noise (0.01)",
    ("gaussian_noise", 0.05): "Gaussian noise (0.05)",
    ("gaussian_noise", 0.1): "Gaussian noise (0.10)",
    ("missing_values", 0.05): "Missing values (0.05)",
    ("missing_values", 0.1): "Missing values (0.10)",
    ("missing_values", 0.2): "Missing values (0.20)",
    ("label_noise", 0.05): "Label noise (0.05)",
    ("label_noise", 0.1): "Label noise (0.10)",
    ("label_noise", 0.2): "Label noise (0.20)",
    ("covariate_shift", 0.0): "Covariate partition",
    ("feature_removal", 0.2): "Feature removal (0.20)",
    ("population_shift", 0.0): "Population partition",
    ("class_prior_shift", 0.0): "Class-prior relabeling",
}
CONDITION_COLORS = {
    "gaussian_noise": "#2f6f9f",
    "missing_values": "#2a9d8f",
    "label_noise": "#d97736",
    "covariate_shift": "#7a5195",
    "feature_removal": "#5a9367",
    "population_shift": "#bc5090",
    "class_prior_shift": "#ef5675",
}
HEATMAP_PIPELINE_LABELS = {
    "Raw": "Raw",
    "Raw_Full": "Raw full",
    "Raw_Capped": "Raw capped",
    "Raw_Variance": "Raw variance",
    "Raw_MI": "Raw MI",
    "AutoFE_Baseline": "AutoFE base",
    "AutoFE_MI": "AutoFE MI",
    "AutoFE_Random": "AutoFE random",
    "AutoFE_NoMultiply": "No multiply",
    "AutoFE_AddSub": "Add/subtract",
    "AutoFE_AddSubDiv": "Add/sub/divide",
    "AutoFE_NoDivision": "No division",
    "AutoFE_MultiplyOnly": "Multiply only",
    "AutoFE_DivideOnly": "Divide only",
}


def _emphasize_figure_text(fig: plt.Figure) -> None:
    """Keep manuscript labels bold and comfortably legible at print size."""
    for artist in fig.findobj(match=Text):
        artist.set_fontweight("bold")
        if artist.get_fontsize() < 8.5:
            artist.set_fontsize(8.5)


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _add(store: dict, key, value) -> None:
    entry = store.setdefault(key, [0.0, 0])
    entry[0] += float(value)
    entry[1] += 1


def _mean(entry):
    return entry[0] / entry[1] if entry and entry[1] else float("nan")


def _bootstrap_mean_ci(values, *, seed=20261123, resamples=10_000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[sampled].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def _save(fig, output: Path, stem: str) -> Path:
    _emphasize_figure_text(fig)
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{stem}.pdf"
    png = output / f"{stem}.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": stem.replace("_", " "),
            "Author": "AutoFE-ShiftBench corrected-protocol audit",
            "Subject": "Corrected 14-pipeline evidence; see figure caption for scope.",
        },
    )
    fig.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not pdf.is_file() or pdf.stat().st_size < 5_000:
        raise RuntimeError(f"Missing or unexpectedly small PDF: {pdf}")
    if not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError(f"Invalid PDF signature: {pdf}")
    from PIL import Image
    with Image.open(png) as image:
        image.verify()
    return pdf


def _scan(ledger: Path, progress_every: int = 100_000):
    """Accumulate task means and profile-local cost summaries in one stream."""
    pipeline_values = {}
    model_values = {}
    condition_values = {}
    profile_rows = Counter()
    profile_pipelines = defaultdict(set)
    profile_datasets = defaultdict(set)
    profile_times = defaultdict(list)
    profile_dimensions = defaultdict(list)
    profile_rss = defaultdict(list)
    observed_datasets = set()
    success_rows = 0
    parsed_rows = 0
    byte_count = 0
    start_size = ledger.stat().st_size

    with ledger.open("rb", buffering=8 * 1024 * 1024) as handle:
        for line_num, raw in enumerate(handle, 1):
            byte_count += len(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            parsed_rows += 1
            if row.get("status") != "success":
                continue
            success_rows += 1

            dataset = row.get("dataset")
            observed_datasets.add(dataset)
            pipeline = row.get("pipeline")
            source = row.get("source_code_fingerprint")
            environment = row.get("environment_fingerprint")
            profile = (source, environment)
            if source and environment and pipeline:
                profile_rows[profile] += 1
                profile_pipelines[profile].add(pipeline)
                profile_datasets[profile].add(dataset)

                for field in ("autofe_gen_time_s", "train_time_s", "infer_time_s"):
                    value = row.get(field)
                    if _finite(value) and float(value) >= 0:
                        profile_times[(profile, pipeline, field)].append(float(value))
                for field, target in (
                    ("actual_estimator_input_dimension", profile_dimensions),
                    ("ram_used_mb", profile_rss),
                ):
                    value = row.get(field)
                    if _finite(value):
                        target[(profile, pipeline)].append(float(value))

            # Covertype remains a single-seed supporting result, not a pooled
            # multi-seed performance estimate.
            if dataset == "covertype":
                continue
            metric = row.get("roc_auc")
            if not _finite(metric) or not 0.0 <= float(metric) <= 1.0:
                continue
            # `condition` includes its severity suffix in result rows, while
            # `condition_identity` carries the declared family name.
            condition = (
                row.get("shift_family")
                or row.get("condition_identity")
                or row.get("condition")
            )
            severity = float(row.get("severity") or 0.0)
            _add(pipeline_values, (dataset, pipeline), metric)
            _add(model_values, (dataset, row.get("model")), metric)
            _add(condition_values, (dataset, condition, severity), metric)

            if parsed_rows % progress_every == 0:
                print(
                    f"Scanned {parsed_rows:,} records "
                    f"({byte_count / 1e9:.2f}/{start_size / 1e9:.2f} GB); "
                    f"{success_rows:,} successful rows.",
                    flush=True,
                )
    if byte_count != start_size:
        raise RuntimeError(
            f"Ledger size changed during scan: {start_size:,} -> {byte_count:,} bytes"
        )
    print(
        f"Scan complete: {parsed_rows:,} ledger rows, {success_rows:,} successful rows; "
        f"{len(profile_rows)} source/environment profiles.",
        flush=True,
    )
    return {
        "pipeline_values": pipeline_values,
        "model_values": model_values,
        "condition_values": condition_values,
        "profile_rows": profile_rows,
        "profile_pipelines": profile_pipelines,
        "profile_datasets": profile_datasets,
        "profile_times": profile_times,
        "profile_dimensions": profile_dimensions,
        "profile_rss": profile_rss,
        "observed_datasets": observed_datasets,
        "parsed_rows": parsed_rows,
        "success_rows": success_rows,
    }


def _dataset_means(values, group_values):
    """Return group -> {dataset: task-mean}, preserving dataset as unit."""
    out = defaultdict(dict)
    for (dataset, group), entry in values.items():
        out[group][dataset] = _mean(entry)
    return out


def _pipeline_summary(data):
    by_pipeline = _dataset_means(data["pipeline_values"], group_values=None)
    summary = {}
    for pipeline in PIPELINES:
        dataset_scores = by_pipeline.get(pipeline, {})
        if not dataset_scores:
            raise ValueError(f"No ROC-AUC data for pipeline {pipeline}")
        names = sorted(dataset_scores)
        values = np.asarray([dataset_scores[n] for n in names], dtype=float)
        lower, upper = _bootstrap_mean_ci(values, seed=20261123 + PIPELINES.index(pipeline))
        summary[pipeline] = {
            "datasets": names,
            "values": values,
            "mean": float(values.mean()),
            "ci": (lower, upper),
        }
    dataset_counts = {len(row["datasets"]) for row in summary.values()}
    if dataset_counts != {23}:
        raise ValueError(f"Expected 23 ROC-AUC-supported datasets per pipeline: {dataset_counts}")
    return summary


def _plot_dataset_diversity(output: Path, data):
    datasets = sorted(
        {dataset for dataset in data["observed_datasets"] if dataset != "covertype"},
        key=lambda name: DATASET_LABELS.get(name, name),
    )
    if len(datasets) != 24:
        raise ValueError(f"Expected 24 non-Covertype corrected-run datasets, found {len(datasets)}")
    counts = Counter(DOMAINS[name] for name in datasets)
    order = sorted(counts, key=lambda x: (-counts[x], x))
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    bars = ax.barh(order, [counts[d] for d in order], color="#2f6f9f", height=0.68)
    ax.invert_yaxis()
    ax.set_xlabel("Datasets", labelpad=12, fontweight="bold")
    ax.set_title("Corrected-run dataset domain coverage", loc="center", weight="bold", pad=14)
    ax.set_xlim(0, max(counts.values()) + 1.8)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.bar_label(bars, padding=4, fontsize=10)
    fig.tight_layout()
    return _save(fig, output, "Figure_2_Dataset_Diversity")


def _plot_pipeline_ranking(output: Path, summary):
    rows = sorted(summary.items(), key=lambda pair: pair[1]["mean"])
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    y = np.arange(len(rows))
    for i, (pipeline, item) in enumerate(rows):
        color = "#355c7d" if pipeline.startswith("Raw") else "#c65d21"
        ax.errorbar(
            item["mean"], i,
            xerr=[[item["mean"] - item["ci"][0]], [item["ci"][1] - item["mean"]]],
            fmt="o", color=color, ecolor=color, capsize=3.2, markersize=5.5,
            linewidth=1.5,
        )
    ax.set_yticks(y, [PIPELINE_LABELS[p] for p, _ in rows])
    ax.set_xlabel("ROC-AUC (dataset-equal mean; 95% bootstrap CI)")
    ax.set_title("Corrected-protocol pipeline ranking", loc="center", weight="bold", pad=14)
    ax.set_xlim(0.68, 0.93)
    ax.axvline(0.5, color="#7b8794", linestyle=":", linewidth=1)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(color="#355c7d", label="Raw baselines"),
        Patch(color="#c65d21", label="AutoFE variants"),
    ], frameon=True, facecolor="white", edgecolor="#555555", framealpha=0.96,
       fancybox=True, borderpad=0.7, labelspacing=0.5, loc="lower right")
    fig.tight_layout()
    return _save(fig, output, "Figure_3_Pipeline_Ranking")


def _plot_model_ranking(output: Path, data):
    by_model = _dataset_means(data["model_values"], group_values=None)
    rows = []
    for model, dataset_scores in by_model.items():
        values = np.asarray([dataset_scores[d] for d in sorted(dataset_scores)], dtype=float)
        lo, hi = _bootstrap_mean_ci(values, seed=20261200 + len(rows))
        rows.append((model, float(values.mean()), lo, hi, len(values)))
    if len(rows) != 10 or {row[4] for row in rows} != {23}:
        raise ValueError(f"Expected ten classifier summaries over 23 datasets: {[(r[0], r[4]) for r in rows]}")
    rows.sort(key=lambda row: row[1])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    y = np.arange(len(rows))
    for i, (_, mean, lo, hi, _) in enumerate(rows):
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color="#3f7f70", ecolor="#3f7f70", capsize=3, markersize=5)
    ax.set_yticks(y, [MODEL_LABELS.get(row[0], row[0]) for row in rows])
    ax.set_xlabel("ROC-AUC (dataset-equal mean; 95% bootstrap CI)")
    ax.set_title("Classifier ranking across the corrected pipeline menu", loc="center", weight="bold", pad=14)
    ax.set_xlim(0.65, 0.95)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, output, "Figure_4_Model_Ranking")


def _plot_training_corruption(output: Path, data, condition_config):
    per_condition = defaultdict(dict)
    for (dataset, condition, severity), entry in data["condition_values"].items():
        per_condition[(condition, severity)][dataset] = _mean(entry)
    clean = per_condition.get(("clean", 0.0), {})
    rows = []
    for condition, severity in condition_config:
        key = (condition, float(severity))
        if condition == "clean":
            continue
        condition_means = per_condition.get(key, {})
        paired = sorted(set(clean) & set(condition_means))
        differences = np.asarray(
            [condition_means[d] - clean[d] for d in paired], dtype=float
        )
        if len(differences) < 2:
            raise ValueError(
                f"Too few paired dataset means for {key}: {len(differences)}"
            )
        lo, hi = _bootstrap_mean_ci(
            differences, seed=20261300 + len(rows), resamples=10_000
        )
        rows.append((key, differences.mean(), lo, hi, len(differences)))
    if len(rows) != 13:
        raise ValueError(f"Expected 13 corruption conditions, found {len(rows)}")
    labels = [
        f"{CONDITION_LABELS.get(key, f'{key[0]} ({key[1]:g})')} (n={n})"
        for key, _, _, _, n in rows
    ]
    colors = [CONDITION_COLORS.get(key[0], "#52616b") for key, *_ in rows]
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    y = np.arange(len(rows))
    for i, (key, mean, lo, hi, n) in enumerate(rows):
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=colors[i], ecolor=colors[i], capsize=3, markersize=5.2)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC difference from clean (condition − clean)")
    ax.set_title("Sensitivity to training-data corruption", loc="center", weight="bold", pad=14)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, output, "Figure_5_Robustness_Shift")


def _select_profile(data):
    candidates = [
        profile for profile, pipelines in data["profile_pipelines"].items()
        if set(PIPELINES).issubset(pipelines)
    ]
    if not candidates:
        raise ValueError("No single source/environment fingerprint contains all 14 pipelines")
    return max(candidates, key=lambda p: data["profile_rows"][p])


def _plot_runtime(output: Path, data, profile):
    rows = []
    fields = [
        ("autofe_gen_time_s", "Feature generation", "#4e79a7"),
        ("train_time_s", "Training", "#f28e2b"),
        ("infer_time_s", "Inference", "#59a14f"),
    ]
    for pipeline in PIPELINES:
        medians = {}
        for field, label, color in fields:
            values = data["profile_times"].get((profile, pipeline, field), [])
            values = [v for v in values if _finite(v) and v >= 0]
            if values:
                medians[field] = float(np.median(values))
            else:
                medians[field] = None
        rows.append((pipeline, medians))
    positive_values = [
        value for _, medians in rows for value in medians.values()
        if value is not None and value > 0
    ]
    if not positive_values:
        raise ValueError("No positive runtime values in the selected provenance profile")
    floor = min(positive_values) / 2
    rows.sort(key=lambda row: sum(v or 0.0 for v in row[1].values()))
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    y = np.arange(len(rows))
    offsets = [-0.18, 0, 0.18]
    for offset, (field, label, color) in zip(offsets, fields):
        for i, (_, item) in enumerate(rows):
            value = item[field]
            if value is None:
                continue
            if value == 0:
                ax.scatter(floor, i + offset, facecolors="white", edgecolors=color,
                           marker="s", s=42, linewidths=1.4, zorder=4)
            else:
                ax.scatter(value, i + offset, label=label if i == 0 else None,
                           color=color, s=34, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, [textwrap.fill(PIPELINE_LABELS[p], width=22) for p, _ in rows])
    ax.tick_params(axis="y", pad=9, labelsize=9)
    ax.set_xlabel("Median seconds per successful task (log scale)")
    ax.set_title("Runtime by pipeline within one provenance profile", loc="center", weight="bold", pad=14)
    ax.set_xlim(floor / 1.5, max(positive_values) * 1.35)
    ax.grid(axis="x", which="both", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    marker_handles = [
        Line2D([], [], marker="o", linestyle="None", color=color, label=label)
        for _, label, color in fields
    ] + [
        Line2D([], [], marker="s", markersize=7, markerfacecolor="white",
               markeredgecolor="#333333", linestyle="None",
               label="Measured zero (open square at edge)")
    ]
    fig.legend(handles=marker_handles, title="Marker key",
               title_fontproperties={"weight": "bold"},
               frameon=True, facecolor="white", edgecolor="#555555", framealpha=1.0,
               fancybox=True, borderpad=0.8, labelspacing=0.55, handletextpad=0.8,
               loc="center left", bbox_to_anchor=(0.77, 0.5),
               bbox_transform=fig.transFigure)
    fig.subplots_adjust(left=0.28, right=0.72, top=0.90, bottom=0.17)
    return _save(fig, output, "Figure_6_Runtime_Comparison")


def _plot_feature_complexity(output: Path, data, profile):
    stats = []
    for pipeline in PIPELINES:
        dims = np.asarray(data["profile_dimensions"].get((profile, pipeline), []), dtype=float)
        rss = np.asarray(data["profile_rss"].get((profile, pipeline), []), dtype=float)
        stats.append((pipeline, dims, rss))
    stats.sort(key=lambda row: np.median(row[1]) if len(row[1]) else float("inf"))
    fig, (ax_dim, ax_rss) = plt.subplots(1, 2, figsize=(10.5, 6.4), sharey=True)
    y = np.arange(len(stats))
    for i, (pipeline, dims, rss) in enumerate(stats):
        if len(dims):
            d25, d50, d75 = np.quantile(dims, [0.25, 0.5, 0.75])
            ax_dim.errorbar(d50, i, xerr=[[d50 - d25], [d75 - d50]], fmt="o",
                            color="#355c7d", capsize=2.5, markersize=4)
        else:
            ax_dim.text(0.98, i, "not recorded", transform=ax_dim.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7.5, color="#777777")
        if len(rss):
            r25, r50, r75 = np.quantile(rss, [0.25, 0.5, 0.75])
            ax_rss.errorbar(r50, i, xerr=[[r50 - r25], [r75 - r50]], fmt="o",
                            color="#c65d21", capsize=2.5, markersize=4)
        else:
            ax_rss.text(0.98, i, "not recorded", transform=ax_rss.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7.5, color="#777777")
    labels = [PIPELINE_LABELS[p] for p, _, _ in stats]
    ax_dim.set_yticks(y, labels)
    ax_dim.invert_yaxis()
    ax_dim.set_xlabel("Estimator input dimension")
    ax_rss.set_xlabel("Recorded process RSS delta (MB)")
    ax_dim.set_title("Representation size")
    ax_rss.set_title("Memory delta")
    for ax in (ax_dim, ax_rss):
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Feature and memory measures by pipeline", x=0.5, ha="center", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, output, "Figure_7_Feature_Complexity")


def _plot_pairwise_contrasts(output: Path, audit):
    block = audit["pipeline_analysis"]["roc_auc"]
    rows = block["selected_contrasts"]
    if len(rows) != 16:
        raise ValueError(f"Expected 16 selected paired contrasts, found {len(rows)}")
    rows = sorted(rows, key=lambda r: (r["mean_difference"], r["left"], r["right"]))
    # Keep this forest plot compact when it is placed at manuscript width.
    # The earlier canvas reserved a full-width footer and left a large blank
    # region beside the axes after LaTeX scaled the PDF.
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    short_labels = {
        "Raw_Full": "Raw full",
        "Raw_Capped": "Raw capped",
        "Raw_MI": "Raw MI",
        "AutoFE_Baseline": "AutoFE baseline",
        "AutoFE_MI": "AutoFE MI",
        "AutoFE_Random": "AutoFE random",
        "AutoFE_NoMultiply": "AutoFE no multiply",
        "AutoFE_AddSub": "AutoFE add/sub",
        "AutoFE_AddSubDiv": "AutoFE add/sub/div",
        "AutoFE_NoDivision": "AutoFE no division",
        "AutoFE_MultiplyOnly": "AutoFE multiply only",
        "AutoFE_DivideOnly": "AutoFE divide only",
    }
    y = np.arange(len(rows))
    for i, row in enumerate(rows):
        mean = float(row["mean_difference"])
        lo, hi = map(float, row["ci95"])
        sig = float(row["p_holm"]) < 0.05
        color = "#b44732" if sig else "#355c7d"
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, ecolor=color, capsize=2.8, markersize=4.8)
        ax.text(1.025, i, f"{float(row['p_holm']):.3g}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.5,
                clip_on=False)
    labels = []
    for r in rows:
        left = short_labels.get(r["left"], r["left"])
        right = short_labels.get(r["right"], r["right"])
        labels.append(f"{left} − {right}")
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Dataset-equal paired ROC-AUC difference (left − right)")
    fig.suptitle(
        "Selected corrected-protocol paired contrasts",
        x=0.5,
        y=0.985,
        ha="center",
        va="top",
        weight="bold",
    )
    ax.text(1.025, 1.01, "Holm p", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, weight="bold", clip_on=False)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(min(float(r["ci95"][0]) for r in rows) - 0.012, max(float(r["ci95"][1]) for r in rows) + 0.012)
    # Scope and interpretation are stated in the manuscript caption. Keeping
    # them out of the artwork prevents the caption-like footer from forcing a
    # wide, mostly empty image box in the paper.
    fig.subplots_adjust(left=0.36, right=0.88, top=0.90, bottom=0.14)
    return _save(fig, output, "Figure_8_Paired_Contrasts")


def _plot_heatmap(output: Path, data, pipeline_summary):
    by_pipeline = _dataset_means(data["pipeline_values"], group_values=None)
    pipeline_order = [p for p, _ in sorted(pipeline_summary.items(), key=lambda item: item[1]["mean"])]
    datasets = sorted(
        set.intersection(*(set(by_pipeline[p]) for p in pipeline_order)),
        key=lambda ds: by_pipeline["Raw_Capped"].get(ds, float("inf")),
    )
    if len(datasets) != 23:
        raise ValueError(f"Expected 23 fully supported dataset rows, found {len(datasets)}")
    matrix = np.asarray([[by_pipeline[p][ds] for p in pipeline_order] for ds in datasets], dtype=float)
    fig, ax = plt.subplots(figsize=(10.8, 8.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.55, vmax=1.0)
    ax.set_xticks(np.arange(len(pipeline_order)), [HEATMAP_PIPELINE_LABELS[p] for p in pipeline_order])
    ax.set_yticks(np.arange(len(datasets)), [DATASET_LABELS[d] for d in datasets])
    ax.tick_params(axis="x", labelrotation=38, labelsize=9, pad=8)
    ax.tick_params(axis="y", labelsize=9, pad=7)
    for i in range(len(datasets)):
        for j in range(len(pipeline_order)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white" if value > 0.82 else "#1c2730")
    fig.suptitle(
        "Corrected-run dataset × pipeline ROC-AUC",
        x=0.5,
        y=0.985,
        ha="center",
        va="top",
        weight="bold",
    )
    ax.set_xlabel("Pipeline", labelpad=12, fontweight="bold")
    ax.set_ylabel("Dataset", labelpad=12, fontweight="bold")
    cb = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    cb.set_label("Mean ROC-AUC within dataset", fontweight="bold", labelpad=10)
    fig.subplots_adjust(left=0.18, right=0.91, top=0.93, bottom=0.20)
    return _save(fig, output, "Figure_9_Heatmap")


def _plot_operator_contrasts(output: Path, audit):
    contrasts = audit["pipeline_analysis"]["roc_auc"]["selected_contrasts"]
    wanted = {
        ("AutoFE_AddSub", "AutoFE_AddSubDiv"):
            "Add division to add/subtract",
        ("AutoFE_AddSub", "AutoFE_NoDivision"):
            "Add multiplication to add/subtract",
    }
    rows = []
    for contrast in contrasts:
        pair = (contrast["left"], contrast["right"])
        if pair in wanted:
            rows.append((wanted[pair], contrast))
    if len(rows) != 2:
        raise ValueError(f"Expected two direct nested operator contrasts, found {len(rows)}")
    rows.sort(key=lambda item: item[0])
    fig, ax = plt.subplots(figsize=(8.8, 2.9))
    y = np.arange(len(rows))
    for i, (label, contrast) in enumerate(rows):
        mean = float(contrast["mean_difference"])
        lo, hi = map(float, contrast["ci95"])
        color = "#b44732" if float(contrast["p_holm"]) < 0.05 else "#355c7d"
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, ecolor=color, capsize=4, markersize=6)
        ax.text(1.035, i, f"{float(contrast['p_holm']):.3g}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5,
                clip_on=False)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Dataset-equal paired ROC-AUC difference (add/subtract − comparison)")
    ax.set_title("Operator-set ablations in the corrected run", loc="center", weight="bold", pad=12)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.01, 0.095)
    ax.text(1.035, 1.02, "Holm p", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, weight="bold", clip_on=False)
    fig.subplots_adjust(left=0.36, right=0.84, top=0.78, bottom=0.28)
    return _save(fig, output, "Figure_10_Ablation_Study")



FIELDS = (
    "run_id",
    "evaluation_protocol_version",
    "seed_scheme_version",
    "dataset_fingerprint",
    "source_code_fingerprint",
    "environment_fingerprint",
    "pipeline",
    "pipeline_identity",
    "operator_registry_version",
    "operator_semantics_version",
    "operator_set_id",
    "cap_policy_version",
    "metric_semantics_version",
    "status",
    "dataset",
    "split_policy",
    "seed",
    "fold",
    "condition",
    "severity",
    "model",
    "scientific_task_id",
    "roc_auc",
    "balanced_accuracy",
    "accuracy",
    "f1_macro",
    "train_auc",
    "test_auc",
    "autofe_gen_time_s",
    "train_time_s",
    "infer_time_s",
    "ram_used_mb",
    "n_original",
    "n_generated",
    "n_retained",
)
NUMERIC_FIELDS = {
    "severity",
    "roc_auc",
    "balanced_accuracy",
    "accuracy",
    "f1_macro",
    "train_auc",
    "test_auc",
    "autofe_gen_time_s",
    "train_time_s",
    "infer_time_s",
    "ram_used_mb",
    "n_original",
    "n_generated",
    "n_retained",
}
REQUIRED_IDENTITY_FIELDS = (
    "evaluation_protocol_version",
    "seed_scheme_version",
    "dataset_fingerprint",
    "source_code_fingerprint",
    "environment_fingerprint",
    "pipeline_identity",
    "operator_registry_version",
    "operator_semantics_version",
    "operator_set_id",
    "cap_policy_version",
    "metric_semantics_version",
)



JACOBIAN_SOURCE = ROOT / "reports" / "fsva_jacobian_measured_summary_20261122.json"
JACOBIAN_ASSOCIATION_SOURCE = (
    ROOT / "reports" / "revision_final" / "run_20261123T175648Z"
    / "04_fsva_association" / "jacobian_performance_matched.csv"
)
COVERAGE_SOURCE = ROOT / "reports" / "revision_run_coverage_by_dataset_20261122.csv"
# Keep revision outputs beside the existing manuscript figure set.
DEFAULT_OUTPUT = ROOT / "paper" / "figures"

JACOBIAN_PIPELINE_LABELS = {
    "AutoFE_AddSub": "AutoFE: add/subtract",
    "AutoFE_AddSubDiv": "AutoFE: add/subtract/divide",
    "AutoFE_Baseline": "AutoFE: baseline",
    "AutoFE_DivideOnly": "AutoFE: divide only",
    "AutoFE_MI": "AutoFE: mutual information",
    "AutoFE_MultiplyOnly": "AutoFE: multiply only",
    "AutoFE_NoDivision": "AutoFE: no division",
    "AutoFE_NoMultiply": "AutoFE: no multiplication",
    "AutoFE_Random": "AutoFE: random selection",
    "Raw": "Raw",
    "Raw_Capped": "Raw: capped",
    "Raw_Full": "Raw: full dimensionality",
    "Raw_MI": "Raw: mutual information",
    "Raw_Variance": "Raw: variance selection",
}

JACOBIAN_DATASET_LABELS = {
    "haberman": "Haberman",
    "sonar": "Sonar",
    "ionosphere": "Ionosphere",
    "heart-disease": "Heart disease",
    "breast-cancer-wisconsin": "Breast cancer (Wisconsin)",
    "blood-transfusion-service-center": "Blood transfusion service center",
    "diabetes": "Diabetes",
    "titanic": "Titanic",
    "credit-g": "Credit-G",
    "wine-quality-red": "Wine quality (red)",
    "kr-vs-kp": "KR-vs-KP",
    "mushroom": "Mushroom",
    "spambase": "Spambase",
    "jm1": "JM1",
    "PhishingWebsites": "Phishing Websites",
    "default-of-credit-card-clients": "Default of credit card clients",
    "magic-telescope": "Magic Telescope",
    "dry-bean-dataset": "Dry Bean",
    "adult": "Adult",
    "bank-marketing": "Bank Marketing",
    "electricity": "Electricity",
    "aps_failure": "APS Failure",
    "covertype": "Covertype",
    "airlines": "Airlines",
    "kddcup99": "KDD Cup 99",
}

JACOBIAN_SEEDS = ("42", "123", "456", "789", "2025")



NAVY = "#19324A"
BLUE = "#2F6F9F"
TEAL = "#2A9D8F"
ORANGE = "#E58F3A"
GREEN = "#5A9367"
PALE_BLUE = "#EAF2F8"
PALE_TEAL = "#E8F5F2"
PALE_ORANGE = "#FDF1E4"
PALE_GREEN = "#EEF6EE"
TEXT = "#17212B"


def _box(ax, x: float, y: float, w: float, h: float, title: str, body: str, *,
         edge: str, face: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    # The workflow is placed at manuscript width.  Use print-safe typography
    # and a compact vertical rhythm so the labels remain legible after the
    # vector PDF is reduced from the source canvas to a single text column.
    if h <= 0.17:
        title_y, body_y = y + h - 0.020, y + 0.035
        title_fs, body_fs, line_spacing = 12.5, 10.4, 1.12
    elif h <= 0.19:
        title_y, body_y = y + h - 0.018, y + 0.044
        title_fs, body_fs, line_spacing = 12.7, 10.6, 1.12
    elif h <= 0.22:
        title_y, body_y = y + h - 0.028, y + h / 2 - 0.006
        title_fs, body_fs, line_spacing = 15.5, 13.5, 1.15
    else:
        title_y, body_y = y + h - 0.040, y + h / 2 - 0.005
        title_fs, body_fs, line_spacing = 18, 15, 1.25
    ax.text(
        x + w / 2, title_y, title,
        transform=ax.transAxes, ha="center", va="top", color=edge,
        fontsize=title_fs, fontweight="bold", family="DejaVu Sans", zorder=3,
    )
    ax.text(
        x + w / 2, body_y, body,
        transform=ax.transAxes, ha="center", va="center", color=TEXT,
        fontsize=body_fs, linespacing=line_spacing, family="DejaVu Sans", zorder=3,
    )


def _arrow(ax, x1: float, x2: float, y: float, color: str = NAVY) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y), (x2, y),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=24,
            linewidth=2.2,
            color=color,
            shrinkA=3,
            shrinkB=3,
            zorder=1,
        )
    )


def _connector(ax, start: tuple[float, float], end: tuple[float, float], *,
               color: str = NAVY, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            transform=ax.transAxes,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.0,
            color=color,
            shrinkA=4,
            shrinkB=4,
            zorder=1,
        )
    )


def create_workflow_diagram(out_dir: str | Path = "results", dpi: int = FIGURE_DPI) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    # This is an internal method diagram rather than a project-management
    # checklist: it exposes the data path, representation branches, fitting,
    # evaluation, and aggregation used by every completed task.
    fig, ax = plt.subplots(figsize=(10, 8.6), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.5, 0.955,
        "AutoFE-ShiftBench: internal evaluation protocol",
        ha="center", va="top", color=NAVY, fontsize=21, fontweight="bold",
    )
    fig.text(
        0.5, 0.915,
        "Fold-local fitting; predictor-wide stress partitions are transductive",
        ha="center", va="top", color="#52616B", fontsize=12.5,
    )

    # Stage 1: data and shift control.
    ax.text(0.03, 0.865, "DATA AND SHIFT CONTROL", transform=ax.transAxes,
            ha="left", va="bottom", color=NAVY, fontsize=12.5, fontweight="bold")
    top_y, top_h, top_w = 0.66, 0.18, 0.205
    top_x = [0.03, 0.275, 0.52, 0.765]
    top = [
        ("INPUT\nDATA", "OpenML tables\nfeatures + target\n25 datasets", BLUE, PALE_BLUE),
        ("REPEATED\nSPLITS", "5 seeds × 5 folds\nstratified splits\npaired stress tasks", TEAL, PALE_TEAL),
        ("TRAINING\nCORRUPTION", "noise · missing\nrelabel · removal\n14 conditions", ORANGE, PALE_ORANGE),
        ("FOLD-LOCAL\nPREP", "impute · encode\ntrain-only scaling\ncache by task key", GREEN, PALE_GREEN),
    ]
    for x, (title, body, edge, face) in zip(top_x, top):
        _box(ax, x, top_y, top_w, top_h, title, body, edge=edge, face=face)
    for i in range(3):
        _arrow(ax, top_x[i] + top_w + 0.008, top_x[i + 1] - 0.008, top_y + top_h / 2)

    # Stage 2: the prepared fold branches into the capped raw controls and
    # the explicit AutoFE candidate/selection space.
    ax.text(0.03, 0.605, "REPRESENTATION AND MODEL FITTING", transform=ax.transAxes,
            ha="left", va="bottom", color=NAVY, fontsize=12.5, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5}, zorder=5)
    branch_y, branch_h, branch_w = 0.42, 0.17, 0.31
    _box(ax, 0.08, branch_y, branch_w, branch_h, "RAW CONTROLS",
         "Raw / full / capped\nvariance / MI controls\nno feature synthesis",
         edge=BLUE, face=PALE_BLUE)
    _box(ax, 0.61, branch_y, branch_w, branch_h, "AUTOFE SEARCH",
         "DFS depth-one arithmetic\n+/−/×/÷; selection\n100-column final cap",
         edge=GREEN, face=PALE_GREEN)
    _connector(ax, (top_x[-1] + top_w / 2, top_y - 0.008),
               (0.08 + branch_w / 2, branch_y + branch_h + 0.008), rad=0.18)
    _connector(ax, (top_x[-1] + top_w / 2, top_y - 0.008),
               (0.61 + branch_w / 2, branch_y + branch_h + 0.008), rad=-0.08)

    model_x, model_y, model_w, model_h = 0.355, 0.245, 0.29, 0.16
    _box(ax, model_x, model_y, model_w, model_h, "MODEL FITTING",
         "10 classifiers\nCPU + GPU families\none fit per task",
         edge=NAVY, face="#F7F9FB")
    _connector(ax, (0.08 + branch_w / 2, branch_y - 0.008),
               (model_x + model_w / 2, model_y + model_h + 0.008), rad=-0.12)
    _connector(ax, (0.61 + branch_w / 2, branch_y - 0.008),
               (model_x + model_w / 2, model_y + model_h + 0.008), rad=0.12)

    # Stage 3: every fit produces held-out evidence, a task ledger, and then
    # data-set-level summaries used by the manuscript.
    bottom_y, bottom_h, bottom_w = 0.035, 0.16, 0.205
    bottom_x = [0.03, 0.275, 0.52, 0.765]
    bottom = [
        ("HELD-OUT\nEVALUATION", "ROC-AUC · F1\ntrain–test gap\nclean test"),
        ("TASK\nLEDGER", "runtime + RSS\nfeature counts\nJSONL + SQLite"),
        ("DATASET-LEVEL\nANALYSIS", "balanced means\nrobustness/ablation\npaired contrasts"),
        ("PAPER\nEVIDENCE", "tables · figures\nFSVA diagnostics\nLaTeX manuscript"),
    ]
    for x, (title, body) in zip(bottom_x, bottom):
        _box(ax, x, bottom_y, bottom_w, bottom_h, title, body, edge=NAVY, face="#F7F9FB")
    _connector(ax, (model_x + model_w / 2, model_y - 0.008),
               (bottom_x[0] + bottom_w / 2, bottom_y + bottom_h + 0.008), rad=0.20)
    for i in range(3):
        _arrow(ax, bottom_x[i] + bottom_w + 0.008, bottom_x[i + 1] - 0.008,
               bottom_y + bottom_h / 2, color="#52616B")

    png_path = out_dir / "AutoFE_ShiftBench_Workflow.png"
    pdf_path = out_dir / "AutoFE_ShiftBench_Workflow.pdf"
    _emphasize_figure_text(fig)
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path



def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    metadata = {
        "Title": stem.replace("_", " "),
        "Author": "AutoFE-ShiftBench revision analysis",
        "Subject": "Corrected-protocol audit figure; see figure caption for scope.",
    }
    _emphasize_figure_text(fig)
    fig.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", metadata={"Title": stem})
    plt.close(fig)
    for path in (pdf_path, png_path):
        if not path.is_file() or path.stat().st_size < 5_000:
            raise RuntimeError(f"Figure export is missing or unexpectedly small: {path}")
    if not pdf_path.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError(f"Invalid PDF signature: {pdf_path}")
    return pdf_path, png_path


def plot_jacobian_summary(output_dir: Path) -> tuple[Path, Path]:
    summary = json.loads(JACOBIAN_SOURCE.read_text(encoding="utf-8"))
    pipeline_data = summary["by_pipeline"]
    rows = []
    for pipeline, values in pipeline_data.items():
        if pipeline not in JACOBIAN_PIPELINE_LABELS:
            raise ValueError(f"Unmapped pipeline in Jacobian summary: {pipeline}")
        ratio = values["selected_to_raw_control_ratio"]
        median = float(ratio["p50"])
        p95 = float(ratio["p95"])
        if median <= 0 or p95 <= 0:
            raise ValueError(f"Nonpositive log-scale ratio for {pipeline}")
        rows.append((median, pipeline, p95, int(values["artifacts"])))
    rows.sort(key=lambda row: (row[0], row[1]))

    fig_height = max(7.1, 0.34 * len(rows) + 2.2)
    fig, ax = plt.subplots(figsize=(7.8, fig_height))
    y = np.arange(len(rows))
    for i, (median, pipeline, p95, artifacts) in enumerate(rows):
        color = "#355c7d" if pipeline.startswith("Raw") else "#c65d21"
        ax.plot([median, p95], [i, i], color=color, linewidth=1.6, alpha=0.75, zorder=1)
        ax.scatter(median, i, color=color, marker="o", s=48, zorder=2,
                   label="Median" if i == 0 else None)
        ax.scatter(p95, i, facecolor="white", edgecolor=color, marker="D", s=42,
                   linewidth=1.4, zorder=3, label="95th percentile" if i == 0 else None)
    ax.axvline(1.0, color="#333333", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 4.0e7)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks(y, [JACOBIAN_PIPELINE_LABELS[pipeline] for _, pipeline, _, _ in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("Selected-to-raw normalized Jacobian norm ratio (log scale)")
    ax.set_title("Measured local sensitivity of selected-feature mappings", loc="center", pad=14)
    ax.grid(axis="x", which="major", color="#d9d9d9", linewidth=0.7)
    ax.grid(axis="x", which="minor", color="#eeeeee", linewidth=0.45)
    legend_handles = [
        Line2D([], [], color="#355c7d", linewidth=2, label="Raw pipelines"),
        Line2D([], [], color="#c65d21", linewidth=2, label="AutoFE pipelines"),
        Line2D([], [], color="#333333", marker="o", linestyle="None", label="Median"),
        Line2D([], [], color="#333333", marker="D", markerfacecolor="white",
               linestyle="None", label="95th percentile"),
    ]
    ax.legend(handles=legend_handles, title="Marker key",
              title_fontproperties={"weight": "bold"}, frameon=True,
              facecolor="white", edgecolor="#555555", framealpha=0.96,
              fancybox=True, borderpad=0.7, labelspacing=0.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=4,
              fontsize=8.5)
    fig.subplots_adjust(left=0.35, right=0.98, top=0.88, bottom=0.24)
    return _save_figure(fig, output_dir, "Figure_11_Measured_Jacobian_Sensitivity")


def plot_jacobian_performance_association(output_dir: Path) -> tuple[Path, Path]:
    """Plot saved local Jacobian sensitivity against matched AUC degradation."""
    if not JACOBIAN_ASSOCIATION_SOURCE.is_file():
        raise FileNotFoundError(JACOBIAN_ASSOCIATION_SOURCE)

    points = []
    with JACOBIAN_ASSOCIATION_SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pipeline = row.get("pipeline")
            x = row.get("log10_jacobian_ratio")
            y = row.get("mean_delta_auc_raw_minus_autofe")
            if not pipeline or not _finite(x) or not _finite(y):
                continue
            points.append((pipeline, float(x), float(y)))

    if len(points) != 19_856:
        raise ValueError(
            "Jacobian-performance association source no longer reconciles to "
            f"19,856 matched mapping units: {len(points):,}"
        )

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    grouped = defaultdict(lambda: ([], []))
    for pipeline, x, y in points:
        grouped[pipeline][0].append(x)
        grouped[pipeline][1].append(y)

    palette = plt.get_cmap("tab20").colors
    for index, (pipeline, (x_values, y_values)) in enumerate(grouped.items()):
        ax.scatter(
            x_values,
            y_values,
            s=12,
            alpha=0.35,
            color=palette[index % len(palette)],
            label=PIPELINE_LABELS.get(pipeline, pipeline),
            linewidths=0,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=7.5,
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#555555",
        framealpha=0.96,
        title="Pipeline",
        title_fontproperties={"weight": "bold"},
        loc="upper right",
    )
    ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.set_xlabel("log10(selected/raw normalized Jacobian ratio)")
    ax.set_ylabel("Raw AUC - AutoFE AUC (matched mapping mean)")
    ax.set_title("Saved Jacobian sensitivity versus matched AUC degradation")
    ax.grid(axis="both", color="#e6e6e6", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.13)
    return _save_figure(fig, output_dir, "Figure_13_Jacobian_Performance_Association")


def plot_seed_coverage(output_dir: Path) -> tuple[Path, Path]:
    rows = []
    with COVERAGE_SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            if dataset not in JACOBIAN_DATASET_LABELS:
                raise ValueError(f"Unmapped dataset in coverage CSV: {dataset}")
            observed = np.array([int(row[f"seed_{seed}_observed"]) for seed in JACOBIAN_SEEDS], dtype=float)
            eligible = np.array([int(row[f"seed_{seed}_eligible"]) for seed in JACOBIAN_SEEDS], dtype=float)
            if np.any(eligible <= 0) or np.any(observed > eligible):
                raise ValueError(f"Invalid seed coverage counts for {dataset}")
            rows.append((dataset, observed, eligible))
    if len(rows) != 25:
        raise ValueError(f"Expected 25 datasets in coverage ledger; found {len(rows)}")

    total_observed = sum(int(observed.sum()) for _, observed, _ in rows)
    total_eligible = sum(int(eligible.sum()) for _, _, eligible in rows)
    if (total_observed, total_eligible) != (1_130_779, 1_166_900):
        raise ValueError(
            "Coverage source no longer reconciles to the checked audit totals: "
            f"{total_observed}/{total_eligible}"
        )
    values = np.vstack([observed / eligible * 100 for _, observed, eligible in rows])

    fig, ax = plt.subplots(figsize=(8.0, 8.3))
    cmap = plt.get_cmap("YlGnBu")
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=Normalize(vmin=0, vmax=100))
    ax.set_xticks(np.arange(len(JACOBIAN_SEEDS)), [f"Seed {seed}" for seed in JACOBIAN_SEEDS])
    ax.set_yticks(np.arange(len(rows)), [JACOBIAN_DATASET_LABELS[name] for name, _, _ in rows])
    fig.suptitle(
        "Eligible-configuration completion by dataset and seed\n"
        f"{total_observed:,} / {total_eligible:,} eligible configurations observed "
        f"({total_observed / total_eligible:.1%});\n"
        "all missing configurations are Covertype",
        x=0.5,
        y=0.985,
        ha="center",
        va="top",
        weight="bold",
    )
    ax.set_xticks(np.arange(-0.5, len(JACOBIAN_SEEDS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, row in enumerate(values):
        for j, pct in enumerate(row):
            label = f"{pct:.1f}%" if 0 < pct < 100 else f"{pct:.0f}%"
            color = "white" if pct >= 58 else "#1a1a1a"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.2, color=color)
    covertype_row = next(i for i, (name, _, _) in enumerate(rows) if name == "covertype")
    ax.add_patch(
        plt.Rectangle(
            (-0.5, covertype_row - 0.5),
            len(JACOBIAN_SEEDS),
            1,
            fill=False,
            edgecolor="#b22222",
            linewidth=2,
            clip_on=False,
        )
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Eligible configurations observed (%)")
    fig.subplots_adjust(left=0.32, right=0.95, top=0.86, bottom=0.18)
    return _save_figure(fig, output_dir, "Figure_12_Eligible_Configuration_Coverage")



def prepare_figure_input_ledger(ledger: Path, run_id: str, output: Path) -> None:
    ledger = ledger if ledger.is_absolute() else ROOT / ledger
    output = output if output.is_absolute() else ROOT / output
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite figure input: {output}")
    if not ledger.is_file():
        raise FileNotFoundError(ledger)

    run_id_bytes = run_id.encode("utf-8")
    bytes_read = 0
    line_count = 0
    selected_count = 0
    parse_errors = []
    datasets = Counter()
    pipelines = Counter()
    statuses = Counter()
    identities: dict[str, set] = {field: set() for field in REQUIRED_IDENTITY_FIELDS}

    output.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("rb") as source, output.open("x", encoding="utf-8", newline="\n") as target:
        for line_count, raw_line in enumerate(source, start=1):
            bytes_read += len(raw_line)
            # The merged ledger preserves source formatting; inspect candidate
            # run rows before JSON parsing to keep this scan memory-efficient.
            if b'"run_id"' not in raw_line or run_id_bytes not in raw_line:
                if bytes_read // (1024**3) > (bytes_read - len(raw_line)) // (1024**3):
                    print(f"Scanned {bytes_read / 1e9:.1f}/{ledger.stat().st_size / 1e9:.1f} GB; selected {selected_count:,} rows.", flush=True)
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                parse_errors.append({"line": line_count, "error": str(exc)})
                continue
            if row.get("run_id") != run_id:
                continue
            compact = {}
            for field in FIELDS:
                value = row.get(field)
                if field in NUMERIC_FIELDS and isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    value = None
                compact[field] = value
            missing = [field for field in REQUIRED_IDENTITY_FIELDS if compact[field] is None]
            if missing:
                parse_errors.append({"line": line_count, "missing_identity_fields": missing})
                continue
            target.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            selected_count += 1
            datasets[compact["dataset"]] += 1
            pipelines[compact["pipeline"]] += 1
            statuses[compact["status"]] += 1
            for field in REQUIRED_IDENTITY_FIELDS:
                identities[field].add(compact[field])
            if selected_count % 100_000 == 0:
                print(f"Selected {selected_count:,} rows at {bytes_read / 1e9:.1f} GB scanned.", flush=True)

    profile_counts = {field: len(values) for field, values in identities.items()}
    metadata = {
        "input_ledger": str(ledger),
        "input_ledger_bytes": ledger.stat().st_size,
        "run_id": run_id,
        "source_lines_scanned": line_count,
        "source_bytes_scanned": bytes_read,
        "selected_rows": selected_count,
        "dataset_rows": dict(sorted(datasets.items())),
        "pipeline_rows": dict(sorted(pipelines.items())),
        "statuses": dict(statuses),
        "identity_profile_cardinalities": profile_counts,
        "identity_values": {field: sorted(values) for field, values in identities.items()},
        "parse_or_identity_errors": parse_errors,
        "compact_fields": list(FIELDS),
        "scope": "single run ID; no data from other run IDs copied",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "metadata": str(metadata_path), "selected_rows": selected_count,
                      "datasets": len(datasets), "pipelines": len(pipelines), "identity_profile_cardinalities": profile_counts,
                      "statuses": dict(statuses), "parse_or_identity_errors": len(parse_errors)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-ledger", action="store_true",
                        help="prepare a compact input for one run instead of regenerating figures")
    parser.add_argument("--run-id", help="run ID required with --prepare-ledger")
    parser.add_argument("--ledger-output", type=Path,
                        help="output JSONL required with --prepare-ledger")
    args = parser.parse_args()
    if args.prepare_ledger:
        if not args.run_id or not args.ledger_output:
            parser.error("--prepare-ledger requires both --run-id and --ledger-output")
        prepare_figure_input_ledger(args.ledger, args.run_id, args.ledger_output)
        return
    if args.run_id or args.ledger_output:
        parser.error("--run-id and --ledger-output are only used with --prepare-ledger")
    ledger = args.ledger.resolve()
    output = args.output_dir.resolve()
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8-sig"))
    configured_pipelines = preflight["configuration"]["pipelines"]
    if set(configured_pipelines) != set(PIPELINES):
        raise ValueError("Pipeline identities differ from the declared corrected grid")
    data = _scan(ledger)
    if data["parsed_rows"] != int(audit["rows_scanned"]):
        raise ValueError(
            f"Ledger row count mismatch: scanned {data['parsed_rows']:,}, audit says {audit['rows_scanned']:,}"
        )
    pipeline_summary = _pipeline_summary(data)
    summary_pipeline_names = set(audit["pipeline_analysis"]["roc_auc"]["pipelines"])
    if summary_pipeline_names != set(PIPELINES):
        raise ValueError("The reviewer audit and corrected grid do not contain the same pipeline menu")
    for pipeline, row in pipeline_summary.items():
        audited = audit["pipeline_analysis"]["roc_auc"]["pipelines"][pipeline]["mean"]
        print(
            f"AUC {pipeline}: {row['mean']:.6f} (streamed); reviewer audit {audited:.6f}; "
            f"delta {row['mean'] - audited:+.6f}",
            flush=True,
        )
    conditions = [(name, float(severity)) for name, severity in preflight["configuration"]["conditions"]]
    outputs = []

    # Method schematic: update its audited run-scale footer, while retaining
    # its role as a protocol drawing rather than an empirical result chart.
    _, workflow_pdf = create_workflow_diagram(output)
    outputs.append(workflow_pdf)

    outputs.extend([
        _plot_dataset_diversity(output, data),
        _plot_pipeline_ranking(output, pipeline_summary),
        _plot_model_ranking(output, data),
        _plot_training_corruption(output, data, conditions),
    ])
    profile = _select_profile(data)
    print(
        f"Runtime/features profile: {profile[0]} / {profile[1]} "
        f"({data['profile_rows'][profile]:,} successful rows; "
        f"{len(data['profile_datasets'][profile])} datasets; "
        f"{len(data['profile_pipelines'][profile])} pipelines)",
        flush=True,
    )
    outputs.extend([
        _plot_runtime(output, data, profile),
        _plot_feature_complexity(output, data, profile),
        _plot_pairwise_contrasts(output, audit),
        _plot_heatmap(output, data, pipeline_summary),
        _plot_operator_contrasts(output, audit),
    ])

    outputs.append(plot_jacobian_summary(output)[0])
    outputs.append(plot_jacobian_performance_association(output)[0])
    outputs.append(plot_seed_coverage(output)[0])

    if len(outputs) != 13:
        raise RuntimeError(f"Expected 13 empirical/method figure PDFs, generated {len(outputs)}")
    if SUBMISSION_OUTPUT.is_dir() and output == DEFAULT_OUTPUT.resolve():
        for pdf in outputs:
            shutil.copy2(pdf, SUBMISSION_OUTPUT / pdf.name)
    print("Verified figure PDFs:")
    for pdf in outputs:
        print(f"  {pdf} ({pdf.stat().st_size:,} bytes)")
    if SUBMISSION_OUTPUT.is_dir() and output == DEFAULT_OUTPUT.resolve():
        for pdf in outputs:
            copied = SUBMISSION_OUTPUT / pdf.name
            if not copied.is_file() or copied.read_bytes() != pdf.read_bytes():
                raise RuntimeError(f"Submission copy mismatch: {copied}")
        print(f"Submission PDFs verified byte-for-byte in {SUBMISSION_OUTPUT}")


if __name__ == "__main__":
    main()
