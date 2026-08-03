import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.ensemble import RandomForestRegressor

# Dataset Taxonomy Mapping
DOMAIN_MAP = {
    "bank-marketing": "Finance",
    "credit-g": "Finance",
    "default-of-credit-card-clients": "Finance",
    "breast-cancer-wisconsin": "Healthcare",
    "heart-disease": "Healthcare",
    "diabetes": "Healthcare",
    "haberman": "Healthcare",
    "adult": "Demographics",
    "aps_failure": "Industry",
    "electricity": "Tech/Utility",
    "PhishingWebsites": "Cybersecurity",
    "jm1": "Software Engineering",
    "covertype": "Agriculture/Nature",
    "dry-bean-dataset": "Agriculture/Nature",
    "mushroom": "Agriculture/Nature",
    "magic-telescope": "Astronomy",
    "ionosphere": "Tech/Physics",
    "sonar": "Tech/Physics",
    "spambase": "Tech/Cyber",
    "wine-quality-red": "Industry/Food"
}

def load_data():
    # Load Results
    records = []
    results_path = "reports/tables/results_stream.jsonl"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
    df_results = pd.DataFrame(records)
    
    # Load Meta-features
    meta_records = []
    for meta_file in Path("data/raw").glob("*_meta.json"):
        dataset_name = meta_file.name.replace("_meta.json", "")
        with open(meta_file, "r") as f:
            meta = json.load(f)
            meta["dataset"] = dataset_name
            meta["domain"] = DOMAIN_MAP.get(dataset_name, "Other")
            meta_records.append(meta)
    df_meta = pd.DataFrame(meta_records)
    
    return df_results, df_meta

def plot_domain_analysis(df_joined):
    """Plot average AutoFE improvement by domain."""
    domain_gains = df_joined.groupby("domain")["delta_f1"].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    domain_gains.plot(kind="barh", color="skyblue")
    plt.title("Average AutoFE Robustness Gain ($\Delta$ F1) by Domain")
    plt.xlabel("Average $\Delta$ F1 (AutoFE - Raw)")
    plt.tight_layout()
    
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "domain_analysis.png", dpi=300)
    print("Domain analysis saved to reports/figures/domain_analysis.png")

def plot_shap_analysis(X, y):
    """Train Random Forest and compute SHAP values."""
    try:
        import shap
    except ImportError:
        print("SHAP package not installed. Skipping SHAP analysis. Install with: pip install shap")
        return
        
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    
    out_dir = Path("reports/figures")
    
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary: Predictors of AutoFE Gain")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=300)
    plt.close()
    
    # Feature Importance Plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Meta-feature Importance for AutoFE Gain")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=300)
    plt.close()
    
    print("SHAP analysis plots saved to reports/figures/")

def main():
    df_results, df_meta = load_data()
    if df_results.empty or df_meta.empty:
        print("Insufficient data for structural analysis.")
        return
        
    df_results = df_results[df_results["status"] == "success"]
    
    # Compute average performance per dataset and pipeline
    perf = df_results.groupby(["dataset", "pipeline"])["f1_macro"].mean().unstack()
    if "AutoFE" not in perf.columns or "Raw" not in perf.columns:
        print("Missing pipeline data.")
        return
        
    perf["delta_f1"] = perf["AutoFE"] - perf["Raw"]
    perf = perf.reset_index()
    
    # Join with meta features
    df_joined = pd.merge(perf, df_meta, on="dataset", how="inner")
    
    # 1. Domain Analysis
    plot_domain_analysis(df_joined)
    
    # 2. Random Forest + SHAP
    # Select numeric meta-features
    meta_cols = [c for c in df_meta.columns if pd.api.types.is_numeric_dtype(df_meta[c])]
    X = df_joined[meta_cols].fillna(0)
    y = df_joined["delta_f1"]
    
    if len(X) >= 5: # Need a minimum number of datasets
        plot_shap_analysis(X, y)
    else:
        print(f"Not enough datasets ({len(X)}) to train SHAP Random Forest. Need >= 5.")

if __name__ == "__main__":
    main()
