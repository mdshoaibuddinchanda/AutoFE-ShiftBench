"""Audit all five frozen seeds before fixing group-aware AUC denominators."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset_names
from src.group_splits import assess_group_fold_feasibility, canonical_feature_group_ids
from src.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456, 789, 2025)


def main() -> None:
    records = []
    for name in load_dataset_names(ROOT / "config" / "dataset_list.yaml"):
        csv_path = ROOT / "data" / "raw" / f"{name}.csv"
        sidecar_path = csv_path.with_name(f"{name}_meta.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path)
        target = sidecar.get("target_column", "target")
        if target not in frame:
            raise ValueError(f"{name}: missing target column {target}")
        groups = canonical_feature_group_ids(frame.drop(columns=[target]))
        statuses = {}
        for seed in SEEDS:
            status = assess_group_fold_feasibility(groups, frame[target], 5, seed)
            statuses[str(seed)] = {
                "split_feasible": status["split_feasible"],
                "auc_status": status["auc_status"],
                "reasons": status["reasons"],
                "fold_class_support": status["fold_class_support"],
            }
        records.append({
            "dataset": name, "csv_sha256": file_sha256(csv_path),
            "sidecar_sha256": file_sha256(sidecar_path), "seeds": statuses,
        })
        print(f"{name}: " + ", ".join(
            f"{seed}={statuses[str(seed)]['auc_status']}" for seed in SEEDS
        ), flush=True)
    output = {
        "artifact_type": "reviewer1_all_seed_group_auc_audit",
        "seeds": list(SEEDS), "n_splits": 5, "records": records,
        "auc_supported_dataset_seed_pairs": sum(
            row["seeds"][str(seed)]["auc_status"] == "supported"
            for row in records for seed in SEEDS
        ),
        "auc_skipped_dataset_seed_pairs": sum(
            row["seeds"][str(seed)]["auc_status"] != "supported"
            for row in records for seed in SEEDS
        ),
    }
    path = ROOT / "provenance" / "group_seed_grid_audit_v1.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}; skipped dataset-seed pairs={output['auc_skipped_dataset_seed_pairs']}")


if __name__ == "__main__":
    main()
