import json
from pathlib import Path
import yaml

def get_dataset_stats():
    config_path = Path("config/dataset_list.yaml")
    raw_dir = Path("data/raw")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    datasets = config.get("datasets", [])
    
    stats_list = []
    for ds in datasets:
        meta_file = raw_dir / f"{ds}_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            
            n_samples = int(meta.get("number_of_samples", 0))
            n_features = int(meta.get("number_of_features", 0))
            n_classes = int(meta.get("number_of_classes", 0))
            missing_pct = meta.get("missing_percent", 0.0)
            cat_pct = meta.get("categorical_percent", 0.0)
            num_pct = meta.get("numerical_percent", 0.0)
            
            task_type = "Binary" if n_classes == 2 else ("Multiclass" if n_classes > 2 else "Unknown")
            
            stats_list.append({
                "Dataset": ds,
                "Instances": n_samples,
                "Features": n_features,
                "Classes": n_classes,
                "Task": task_type,
                "Missing (%)": f"{missing_pct:.1f}%",
                "Numerical (%)": f"{num_pct:.1f}%",
                "Categorical (%)": f"{cat_pct:.1f}%",
            })
        else:
            stats_list.append({
                "Dataset": ds,
                "Instances": "Missing",
                "Features": "-",
                "Classes": "-",
                "Task": "-",
                "Missing (%)": "-",
                "Numerical (%)": "-",
                "Categorical (%)": "-",
            })
            
    # Print markdown table
    print("| # | Dataset | Instances | Features | Classes | Task | Missing | Numerical | Categorical |")
    print("|:---|:---|---:|---:|---:|:---|---:|---:|---:|")
    for i, row in enumerate(stats_list, 1):
        print(f"| {i} | **{row['Dataset']}** | {row['Instances']} | {row['Features']} | {row['Classes']} | {row['Task']} | {row['Missing (%)']} | {row['Numerical (%)']} | {row['Categorical (%)']} |")

if __name__ == "__main__":
    get_dataset_stats()
