"""Generate advanced publication-quality tables for Q1 journal submission."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as ss

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
    return df

def generate_table_1_predictive_performance(df: pd.DataFrame):
    if df.empty or "roc_auc" not in df.columns: return "*(No data)*\n"
    grouped = df.groupby("pipeline").agg({
        "roc_auc": "mean",
        "accuracy": "mean",
        "f1": "mean"
    }).reset_index().sort_values("roc_auc", ascending=False)
    
    out = "## Table 1: Overall Predictive Performance\n\n"
    out += "| Rank | Pipeline | Mean ROC-AUC | Mean Accuracy | Mean F1-Score |\n"
    out += "| :---: | :--- | :---: | :---: | :---: |\n"
    
    for idx, row in enumerate(grouped.itertuples(), 1):
        roc_str = f"**{row.roc_auc:.4f}**" if idx == 1 else f"{row.roc_auc:.4f}"
        acc_str = f"**{row.accuracy:.4f}**" if row.accuracy == grouped["accuracy"].max() else f"{row.accuracy:.4f}"
        f1_str = f"**{row.f1:.4f}**" if row.f1 == grouped["f1"].max() else f"{row.f1:.4f}"
        out += f"| {idx} | {row.pipeline} | {roc_str} | {acc_str} | {f1_str} |\n"
    return out

def generate_table_2_robustness(df: pd.DataFrame):
    if df.empty or "roc_auc" not in df.columns: return "*(No data)*\n"
    out = "## Table 2: Robustness Under Distribution Shift\n\n"
    
    shifts = ["clean", "missing_0.1", "missing_0.3", "gaussian_0.1", "gaussian_0.3"]
    pipelines = ["Raw", "AutoFE_Baseline", "AutoFE_NoMultiply"]
    
    out += "| Shift Condition | " + " | ".join(pipelines) + " |\n"
    out += "| :--- | " + " | ".join([":---:"] * len(pipelines)) + " |\n"
    
    for shift in shifts:
        row_str = f"| {shift} | "
        for p in pipelines:
            val = df[(df["condition"] == shift) & (df["pipeline"] == p)]["roc_auc"].mean()
            if np.isnan(val): val = 0
            row_str += f"{val:.4f} | "
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
    
    out = "## Table 5: Statistical Significance (Wilcoxon Test & Effect Size)\n\n"
    out += "| Comparison | N Pairs | p-value | Cohen's d | Meaning |\n"
    out += "| :--- | :---: | :---: | :---: | :--- |\n"
    
    comparisons = [("Raw", "AutoFE_Baseline"), ("Raw", "AutoFE_NoMultiply"), ("AutoFE_Baseline", "AutoFE_NoMultiply")]
    
    for p1, p2 in comparisons:
        df1 = df[df["pipeline"] == p1].groupby(["dataset", "seed", "fold", "condition", "model"])["roc_auc"].mean()
        df2 = df[df["pipeline"] == p2].groupby(["dataset", "seed", "fold", "condition", "model"])["roc_auc"].mean()
        paired = pd.DataFrame({p1: df1, p2: df2}).dropna()
        
        if len(paired) < 5: continue
        
        stat, p_val = ss.wilcoxon(paired[p1], paired[p2])
        
        # Cohen's d
        mean_diff = paired[p1].mean() - paired[p2].mean()
        pooled_std = np.sqrt((paired[p1].std()**2 + paired[p2].std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        if abs(cohens_d) < 0.2: effect = "Negligible"
        elif abs(cohens_d) < 0.5: effect = "Small"
        elif abs(cohens_d) < 0.8: effect = "Medium"
        else: effect = "Large"
        
        out += f"| {p1} vs {p2} | {len(paired)} | {p_val:.2e} | {cohens_d:+.3f} | {effect} |\n"
        
    return out

def generate_table_6_friedman_test(df: pd.DataFrame):
    if df.empty or "roc_auc" not in df.columns: return ""
    out = "## Table 6: Friedman Omnibus Test\n\n"
    
    pivot = df.groupby(["dataset", "seed", "fold", "condition", "pipeline"])["roc_auc"].mean().reset_index()
    pivot = pivot.pivot(index=["dataset", "seed", "fold", "condition"], columns="pipeline", values="roc_auc").dropna()
    
    if len(pivot) < 10: return "*(Not enough paired datasets for Friedman)*\n"
    
    stat, p_val = ss.friedmanchisquare(*[pivot[c] for c in pivot.columns])
    out += f"**Null Hypothesis:** All feature engineering pipelines perform equally.\n\n"
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
    
    comparisons = [("Raw", "AutoFE_Baseline"), ("Raw", "AutoFE_NoMultiply")]
    threshold = 0.001
    
    for p1, p2 in comparisons:
        df1 = df[df["pipeline"] == p1].groupby(["dataset", "seed", "fold", "condition", "model"])["roc_auc"].mean()
        df2 = df[df["pipeline"] == p2].groupby(["dataset", "seed", "fold", "condition", "model"])["roc_auc"].mean()
        paired = pd.DataFrame({p1: df1, p2: df2}).dropna()
        if paired.empty: continue
        
        wins = (paired[p1] > paired[p2] + threshold).sum()
        losses = (paired[p1] < paired[p2] - threshold).sum()
        ties = len(paired) - wins - losses
        out += f"| {p1} vs {p2} | {wins} | {ties} | {losses} |\n"
        
    return out

def generate_table_8_feature_stability(df: pd.DataFrame):
    if df.empty or "wasserstein" not in df.columns: return "*(No data)*\n"
    out = "## Table 8: Feature Distribution Stability (Shifted Data)\n\n"
    out += "*Measures how much the generated feature distributions diverge under domain shift.*\n\n"
    
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
    if df.empty or "train_auc" not in df.columns or "test_auc" not in df.columns: return "*(No data)*\n"
    out = "## Table 9: Overfitting Analysis (Train AUC - Test AUC)\n\n"
    
    df["overfitting_gap"] = df["train_auc"] - df["test_auc"]
    
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

def generate_all_tables():
    df = _load_data()
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "q1_tables_v2.md"
    
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
        
    print(f"Generated 10 advanced tables in {out_path.absolute()}")

if __name__ == "__main__":
    generate_all_tables()
