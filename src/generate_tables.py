"""Generate advanced publication-quality tables for Q1 journal submission."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as ss

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
PRIMARY_PIPELINE_CONTRASTS = [
    ("Raw", "AutoFE_Baseline"), ("Raw", "AutoFE_MI"),
    ("Raw", "AutoFE_Random"), ("Raw", "AutoFE_NoMultiply"),
]

def _load_data(results_path="reports/tables/results_stream.jsonl"):
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
    if "condition" in df.columns:
        df["condition"] = df["condition"].replace(CONDITION_ALIASES)
        df = df[df["condition"].isin(PRIMARY_CONDITIONS)].copy()
    return df

def _dataset_pipeline_metric_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Task-pair, then equally average one primary metric per dataset/pipeline."""
    pair_cols = ["dataset", "seed", "fold", "condition", "model"]
    required = {*pair_cols, "pipeline", metric}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["dataset", "pipeline", metric])
    df = df.copy()
    df["condition"] = df["condition"].replace(CONDITION_ALIASES)
    task_scores = (
        df[df["condition"].isin(PRIMARY_CONDITIONS)]
        .groupby([*pair_cols, "pipeline"], as_index=False)[metric]
        .mean()
    )
    wide = task_scores.pivot(index=pair_cols, columns="pipeline", values=metric)
    pipeline_names = list(wide.columns)
    matched = wide.dropna(subset=pipeline_names)
    if matched.empty:
        return pd.DataFrame(columns=["dataset", "pipeline", metric])
    return (
        matched.reset_index()
        .melt(id_vars=pair_cols, var_name="pipeline", value_name=metric)
        .groupby(["dataset", "pipeline"], as_index=False)[metric]
        .mean()
    )


def _dataset_pipeline_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight, task-paired primary mean ROC-AUC per dataset and pipeline."""
    return _dataset_pipeline_metric_scores(df, "roc_auc")

def generate_table_1_predictive_performance(df: pd.DataFrame):
    if df.empty or "roc_auc" not in df.columns: return "*(No data)*\n"
    grouped = _dataset_pipeline_scores(df).rename(columns={"roc_auc": "dataset_mean_roc_auc"})
    for metric in ("accuracy", "f1"):
        scores = _dataset_pipeline_metric_scores(df, metric)
        if not scores.empty:
            scores = scores.groupby("pipeline", as_index=False)[metric].mean()
            grouped = grouped.merge(scores, on="pipeline", how="left")
        else:
            grouped[metric] = np.nan
    grouped = grouped.groupby("pipeline", as_index=False).mean(numeric_only=True)
    grouped = grouped.sort_values("dataset_mean_roc_auc", ascending=False)

    out = "## Table 1: Dataset-Balanced Primary Predictive Performance\n\n"
    out += "| Rank | Pipeline | Mean ROC-AUC | Mean Accuracy | Mean F1-Score |\n"
    out += "| :---: | :--- | :---: | :---: | :---: |\n"

    for idx, row in enumerate(grouped.itertuples(), 1):
        roc_str = f"**{row.dataset_mean_roc_auc:.4f}**" if idx == 1 else f"{row.dataset_mean_roc_auc:.4f}"
        acc_str = f"**{row.accuracy:.4f}**" if row.accuracy == grouped["accuracy"].max() else f"{row.accuracy:.4f}"
        f1_str = f"**{row.f1:.4f}**" if row.f1 == grouped["f1"].max() else f"{row.f1:.4f}"
        out += f"| {idx} | {row.pipeline} | {roc_str} | {acc_str} | {f1_str} |\n"
    return out

def generate_table_2_robustness(df: pd.DataFrame):
    if df.empty or "roc_auc" not in df.columns: return "*(No data)*\n"
    df = df.copy()
    df["condition"] = df["condition"].replace(CONDITION_ALIASES)
    out = "## Table 2: Robustness Under Training-Data Corruption\n\n"

    shifts = [
        "clean", "missing_values_0.05", "missing_values_0.10", "missing_values_0.20",
        "gaussian_noise_0.01", "gaussian_noise_0.05", "gaussian_noise_0.10",
        "label_noise_0.05", "label_noise_0.10", "label_noise_0.20",
    ]
    pipelines = ["Raw", "AutoFE_Baseline", "AutoFE_NoMultiply"]
    dataset_scores = (
        df[df["condition"].isin(PRIMARY_CONDITIONS)]
        .groupby(["dataset", "condition", "pipeline"], as_index=False)["roc_auc"]
        .mean()
    )
    
    out += "| Shift Condition | " + " | ".join(pipelines) + " |\n"
    out += "| :--- | " + " | ".join([":---:"] * len(pipelines)) + " |\n"
    
    for shift in shifts:
        row_str = f"| {shift} | "
        for p in pipelines:
            val = dataset_scores[
                (dataset_scores["condition"] == shift) & (dataset_scores["pipeline"] == p)
            ]["roc_auc"].mean()
            row_str += (f"{val:.4f} | " if pd.notna(val) else "NA | ")
        out += row_str + "\n"
    return out

def generate_table_3_computational_cost(df: pd.DataFrame):
    if df.empty or "train_time_s" not in df.columns: return "*(No data)*\n"
    out = "## Table 3: Computational Cost & Efficiency (Median Seconds)\n\n"
    grouped = df.groupby("pipeline").agg({
        "autofe_gen_time_s": "median",
        "train_time_s": "median",
        "infer_time_s": "median"
    }).reset_index()
    
    out += "| Pipeline | FE Gen Time (s) | Train Time (s) | Infer Time (s) | Total (s) |\n"
    out += "| :--- | :---: | :---: | :---: | :---: |\n"
    
    for row in grouped.itertuples():
        total = row.autofe_gen_time_s + row.train_time_s + row.infer_time_s
        out += f"| {row.pipeline} | {row.autofe_gen_time_s:.3f} | {row.train_time_s:.3f} | {row.infer_time_s:.3f} | {total:.3f} |\n"
    return out

def generate_table_4_memory_and_features(df: pd.DataFrame):
    if df.empty or "ram_used_mb" not in df.columns: return "*(No data)*\n"
    out = "## Table 4: Memory Usage and Feature Complexity\n\n"
    
    # Safely compute median values without KeyError
    cols = ["ram_used_mb", "n_generated", "n_retained", "n_original"]
    agg_dict = {}
    for c in cols:
        if c == "ram_used_mb":
            agg_dict[c] = "max"
        elif c in df.columns:
            agg_dict[c] = "median"
            
    grouped = df.groupby("pipeline").agg(agg_dict).reset_index()
    
    out += "| Pipeline | Peak RAM (MB) | Original Features | Generated Features | Retained Features |\n"
    out += "| :--- | :---: | :---: | :---: | :---: |\n"
    
    for row in grouped.itertuples():
        n_orig = getattr(row, "n_original", 0)
        n_gen = getattr(row, "n_generated", 0)
        n_ret = getattr(row, "n_retained", 0)
        out += f"| {row.pipeline} | {row.ram_used_mb:.1f} | {n_orig:.1f} | {n_gen:.1f} | {n_ret:.1f} |\n"
    return out

def generate_table_5_statistical_tests(df: pd.DataFrame):
    if df.empty or "roc_auc" not in df.columns: return "*(No data)*\n"
    scores = _dataset_pipeline_scores(df)
    out = "## Table 5: Paired Primary Contrasts (Dataset-Level Wilcoxon; Holm Corrected)\n\n"
    out += "Each dataset contributes one paired mean across the matched primary conditions, folds, seeds, and models.\n\n"
    out += "| Comparison | Datasets | Mean Δ AUC (AutoFE − Raw) | Median Δ AUC | p-value | Holm p |\n"
    out += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    raw = scores[scores["pipeline"] == "Raw"].set_index("dataset")["roc_auc"]
    raw_diffs = []
    raw_rows = []
    for baseline, autofe in PRIMARY_PIPELINE_CONTRASTS:
        candidate = scores[scores["pipeline"] == autofe].set_index("dataset")["roc_auc"]
        paired = pd.concat([raw.rename("Raw"), candidate.rename("AutoFE")], axis=1).dropna()
        diffs = (paired["AutoFE"] - paired["Raw"]).to_numpy(dtype=float)
        pvalue = float(ss.wilcoxon(diffs).pvalue) if len(diffs) > 1 and not np.allclose(diffs, 0) else (1.0 if len(diffs) > 1 else np.nan)
        raw_diffs.append(pvalue)
        raw_rows.append((autofe, diffs, pvalue))
    valid = [(i, p) for i, p in enumerate(raw_diffs) if np.isfinite(p)]
    adjusted = [np.nan] * len(raw_diffs)
    running = 0.0
    for rank, (index, pvalue) in enumerate(sorted(valid, key=lambda item: item[1])):
        running = max(running, min(1.0, (len(valid) - rank) * pvalue))
        adjusted[index] = running
    for (autofe, diffs, pvalue), holm in zip(raw_rows, adjusted):
        mean_diff = float(np.mean(diffs)) if len(diffs) else np.nan
        median_diff = float(np.median(diffs)) if len(diffs) else np.nan
        out += f"| Raw vs {autofe} | {len(diffs)} | {mean_diff:.4f} | {median_diff:.4f} | "
        out += f"{pvalue:.3g} | {holm:.3g} |\n" if np.isfinite(pvalue) else "NA | NA |\n"
    return out

def generate_table_6_friedman_test(df: pd.DataFrame):
    if df.empty or "roc_auc" not in df.columns: return ""
    out = "## Table 6: Friedman Omnibus Test\n\n"
    scores = _dataset_pipeline_scores(df)
    pivot = scores.pivot(index="dataset", columns="pipeline", values="roc_auc").dropna()
    if len(pivot) < 3 or pivot.shape[1] < 3: return "*(Not enough complete paired datasets for Friedman)*\n"
    stat, p_val = ss.friedmanchisquare(*[pivot[c] for c in pivot.columns])
    out += f"**Null Hypothesis:** All feature engineering pipelines perform equally.\n\n"
    out += f"- **Independent datasets:** {len(pivot)}\n"
    out += f"- **Friedman Chi-Square Statistic:** {stat:.3f}\n"
    out += f"- **p-value:** {p_val:.2e}\n\n"
    if p_val < 0.05:
        out += "*(Statistically significant: We reject the null hypothesis. Pipelines differ significantly.)*\n"
        
    return out

def generate_table_7_win_tie_loss(df: pd.DataFrame):
    if df.empty or "roc_auc" not in df.columns: return "*(No data)*\n"
    out = "## Table 7: Win / Tie / Loss Analysis\n\n"
    
    out += "| Comparison | Wins | Ties | Losses |\n"
    out += "| :--- | :---: | :---: | :---: |\n"
    
    scores = _dataset_pipeline_scores(df)
    comparisons = PRIMARY_PIPELINE_CONTRASTS
    threshold = 0.001
    
    for p1, p2 in comparisons:
        df1 = scores[scores["pipeline"] == p1].set_index("dataset")["roc_auc"]
        df2 = scores[scores["pipeline"] == p2].set_index("dataset")["roc_auc"]
        paired = pd.concat({p1: df1, p2: df2}, axis=1).dropna()
        if paired.empty: continue
        
        wins = (paired[p1] > paired[p2] + threshold).sum()
        losses = (paired[p1] < paired[p2] - threshold).sum()
        ties = len(paired) - wins - losses
        out += f"| {p1} vs {p2} | {wins} | {ties} | {losses} |\n"
        
    return out

def generate_table_8_feature_stability(df: pd.DataFrame):
    if df.empty or "wasserstein" not in df.columns: return "*(No data)*\n"
    if df["wasserstein"].isna().all() and ("ks_stat" not in df.columns or df["ks_stat"].isna().all()):
        return "## Table 8: Feature Distribution Distance\n\n*(Not reported: held-out features are unchanged in the primary training-corruption protocol.)*\n"
    out = "## Table 8: Feature Distribution Distance\n\n"
    
    shift_df = df[df["condition"] != "clean"].copy()
    if shift_df.empty: return "*(No shift data available)*\n"
    
    grouped = shift_df.groupby("pipeline").agg({
        "wasserstein": "mean",
        "ks_stat": "mean"
    }).reset_index()
    
    out += "| Pipeline | Mean Wasserstein Distance | Mean KS-Statistic |\n"
    out += "| :--- | :---: | :---: |\n"
    
    for row in grouped.itertuples():
        out += f"| {row.pipeline} | {row.wasserstein:.4f} | {row.ks_stat:.4f} |\n"
    
    return out

def generate_table_9_overfitting_gap(df: pd.DataFrame):
    """Calculates Train AUC - Test AUC to measure overfitting."""
    if df.empty or "train_auc" not in df.columns or "roc_auc" not in df.columns: return "*(No data)*\n"
    out = "## Table 9: Overfitting Analysis (Train AUC - Test AUC)\n\n"
    df = df.copy()
    df["overfitting_gap"] = pd.to_numeric(df["train_auc"], errors="coerce") - pd.to_numeric(df["roc_auc"], errors="coerce")
    
    grouped = df.groupby("pipeline")["overfitting_gap"].mean().reset_index().sort_values("overfitting_gap")
    
    out += "| Pipeline | Mean Overfitting Gap |\n"
    out += "| :--- | :---: |\n"
    
    for row in grouped.itertuples():
        out += f"| {row.pipeline} | {row.overfitting_gap:+.4f} |\n"
        
    return out

def generate_table_10_robustness_score(df: pd.DataFrame):
    """Calculates Robustness Score = Shifted AUC / Clean AUC"""
    if df.empty or "roc_auc" not in df.columns: return "*(No data)*\n"
    out = "## Table 10: Robustness Score (Shifted AUC / Clean AUC)\n\n"
    
    clean_df = df[df["condition"] == "clean"].groupby(["dataset", "seed", "fold", "pipeline", "model"])["roc_auc"].mean().reset_index()
    shift_df = df[df["condition"] != "clean"].groupby(["dataset", "seed", "fold", "pipeline", "model", "condition"])["roc_auc"].mean().reset_index()
    
    if shift_df.empty: return "*(No shift data available)*\n"
    
    merged = pd.merge(shift_df, clean_df, on=["dataset", "seed", "fold", "pipeline", "model"], suffixes=("_shift", "_clean"))
    merged["robustness_score"] = merged["roc_auc_shift"] / merged["roc_auc_clean"]
    
    grouped = merged.groupby("pipeline")["robustness_score"].mean().reset_index().sort_values("robustness_score", ascending=False)
    
    out += "| Pipeline | Mean Robustness Score |\n"
    out += "| :--- | :---: |\n"
    
    for row in grouped.itertuples():
        out += f"| {row.pipeline} | {row.robustness_score:.4f} |\n"
        
    return out

def generate_all_tables(results_path=None, output_path=None):
    if results_path is None:
        raise ValueError("Pass a results_path explicitly; historical results/tables are never overwritten implicitly")
    df = _load_data(results_path)
    out_path = Path(output_path) if output_path else Path(results_path).with_name("q1_tables_corrected.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Q1 Journal Tables (Advanced Statistics)\n\n")
        f.write(generate_table_1_predictive_performance(df) + "\n\n")
        f.write(generate_table_2_robustness(df) + "\n\n")
        f.write(generate_table_3_computational_cost(df) + "\n\n")
        f.write(generate_table_4_memory_and_features(df) + "\n\n")
        f.write(generate_table_5_statistical_tests(df) + "\n\n")
        f.write(generate_table_6_friedman_test(df) + "\n\n")
        f.write(generate_table_7_win_tie_loss(df) + "\n\n")
        f.write(generate_table_8_feature_stability(df) + "\n\n")
        f.write(generate_table_9_overfitting_gap(df) + "\n\n")
        f.write(generate_table_10_robustness_score(df) + "\n\n")
        
    print(f"Generated primary-condition tables in {out_path.absolute()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    generate_all_tables(args.results, args.output)
