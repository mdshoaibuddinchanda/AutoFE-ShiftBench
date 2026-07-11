import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import friedmanchisquare
from pathlib import Path
import os

def load_results(jsonl_path="reports/tables/results_stream.jsonl"):
    records = []
    if not os.path.exists(jsonl_path):
        return pd.DataFrame()
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(records)

def plot_cd_diagram(ranks, cd, title="Critical Difference Diagram"):
    """Custom CD diagram using matplotlib."""
    sorted_ranks = ranks.sort_values(ascending=True)
    names = sorted_ranks.index.tolist()
    r = sorted_ranks.values
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    
    # Draw horizontal axis
    ax.hlines(0, min(r) - 0.5, max(r) + 0.5, color='black', linewidth=1)
    
    # Draw points
    ax.scatter(r, np.zeros_like(r), color='black', zorder=5)
    
    # Add labels
    for i, (name, rank) in enumerate(zip(names, r)):
        y_pos = 0.1 + (i % 3) * 0.1  # stagger labels
        ax.annotate(f"{name}\n{rank:.2f}", (rank, 0), xytext=(rank, y_pos),
                    ha='center', va='bottom', arrowprops=dict(arrowstyle="-", color='gray'))
                    
    # Draw CD bar
    ax.hlines(-0.1, min(r), min(r) + cd, color='red', linewidth=3)
    ax.text(min(r) + cd/2, -0.15, f"CD = {cd:.2f}", ha='center', va='top', color='red')
    
    ax.axis('off')
    plt.tight_layout()
    
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "cd_diagram.png", dpi=300)
    print("CD diagram saved to reports/figures/cd_diagram.png")

def main():
    df = load_results()
    if df.empty:
        print("No results found.")
        return
        
    df = df[df["status"] == "success"]
    
    # 1. Robustness Index (Average F1 across all conditions per dataset/model/pipeline)
    print("--- ROBUSTNESS INDEX ---")
    ri = df.groupby(["dataset", "model", "pipeline"])["f1_macro"].mean().reset_index()
    ri_wide = ri.pivot_table(index=["dataset", "model"], columns="pipeline", values="f1_macro")
    ri_wide["Delta"] = ri_wide["AutoFE"] - ri_wide["Raw"]
    print(ri_wide.groupby("model")["Delta"].mean().sort_values(ascending=False))
    
    # 2. Win/Tie/Loss Matrix
    print("\n--- WIN / TIE / LOSS (AutoFE vs Raw) ---")
    # Compare per dataset, fold, condition, model
    pairs = df.pivot_table(index=["dataset", "seed", "fold", "condition", "model"], 
                           columns="pipeline", values="f1_macro").dropna()
    
    threshold = 0.001
    wins = (pairs["AutoFE"] > pairs["Raw"] + threshold).sum()
    ties = (np.abs(pairs["AutoFE"] - pairs["Raw"]) <= threshold).sum()
    losses = (pairs["AutoFE"] < pairs["Raw"] - threshold).sum()
    print(f"Wins: {wins}, Ties: {ties}, Losses: {losses}")
    
    # 3. Average Rank & Friedman Test
    print("\n--- RANKING ANALYSIS ---")
    # Rank models within each (dataset, fold, condition) group
    # We combine model + pipeline as the "method"
    df["method"] = df["model"] + "_" + df["pipeline"]
    
    rank_df = df.pivot_table(index=["dataset", "seed", "fold", "condition"], 
                             columns="method", values="f1_macro").dropna()
    
    if len(rank_df) == 0:
        print("Not enough complete blocks for ranking analysis yet.")
        return
        
    # Ranks (1 is best, so ascending=False)
    ranks = rank_df.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean()
    print("Average Ranks:\n", avg_ranks.sort_values())
    
    # Friedman Test
    stat, p = friedmanchisquare(*[rank_df[c] for c in rank_df.columns])
    print(f"\nFriedman Test: statistic={stat:.2f}, p-value={p:.2e}")
    
    # CD (Critical Difference)
    k = len(rank_df.columns)
    N = len(rank_df)
    # q_alpha for alpha=0.05, approx 3.0 to 3.5 depending on k. We use a proxy 3.2 for k~20
    q_alpha = 3.2 
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))
    print(f"Critical Difference (approx alpha=0.05): {cd:.4f}")
    
    plot_cd_diagram(avg_ranks, cd)

if __name__ == "__main__":
    main()
