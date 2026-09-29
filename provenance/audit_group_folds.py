"""Audit row-level overlap and group-aware fold feasibility for all datasets."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_dataset_names
from src.group_splits import (
    assess_group_fold_feasibility,
    canonical_feature_rows,
    assert_no_group_hash_collisions,
    summarize_groups,
)
from src.provenance import file_sha256, stable_digest
from src.splitters import get_stratified_splits


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "provenance"


def audit_one(name: str, *, n_splits: int = 5, seed: int = 42) -> dict:
    csv_path = RAW / f"{name}.csv"
    sidecar_path = RAW / f"{name}_meta.json"
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(csv_path)
    target = meta.get("target_column")
    if target not in frame.columns:
        target = "target_label" if "target_label" in frame.columns else "target"
    X = frame.drop(columns=[target])
    y = frame[target]
    canonical_keys = canonical_feature_rows(X)
    # Build canonical rows once; calling canonical_feature_group_ids separately
    # would retain a second 100k-row JSON list on the largest datasets.
    group_ids = np.asarray([hashlib.sha256(key.encode("utf-8")).hexdigest() for key in canonical_keys], dtype=object)
    assert_no_group_hash_collisions(group_ids, canonical_keys)
    summary = summarize_groups(group_ids, y)
    conflict_ids = list(summary.pop("conflicting_label_group_ids", []))
    summary["conflicting_label_group_id_count"] = len(conflict_ids)
    summary["conflicting_label_group_id_sha256"] = stable_digest(conflict_ids)
    summary["conflicting_label_group_id_examples"] = conflict_ids[:5]
    row_splits = get_stratified_splits(X, y, n_splits, seed, target_column=target)
    row_overlap = []
    for fold, (train, test) in enumerate(row_splits, start=1):
        train_groups = set(group_ids[train].tolist())
        matches = int(sum(group_ids[index] in train_groups for index in test))
        row_overlap.append({
            "fold": fold,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_rows_with_training_feature_match": matches,
            "test_rows_with_training_feature_match_percent": round(100 * matches / max(len(test), 1), 6),
        })
    group_status = assess_group_fold_feasibility(group_ids, y, n_splits, seed)
    status_conflict_ids = list(group_status.pop("conflicting_label_group_ids", []))
    group_status["conflicting_label_group_id_count"] = len(status_conflict_ids)
    group_status["conflicting_label_group_id_sha256"] = stable_digest(status_conflict_ids)
    group_status["conflicting_label_group_id_examples"] = status_conflict_ids[:5]
    return {
        "dataset": name,
        "csv_path": csv_path.relative_to(ROOT).as_posix(),
        "csv_sha256": file_sha256(csv_path),
        "sidecar_sha256": file_sha256(sidecar_path),
        "target_column": target,
        "rows": int(len(frame)),
        "features": int(X.shape[1]),
        "canonicalization": {
            "schema_and_column_order_included": True,
            "missing_values_unified": True,
            "numeric_1_equals_numeric_1_0": True,
            "text_1_distinct_from_numeric_1": True,
            "canonical_key_sha256": stable_digest(canonical_keys),
            "group_id_sha256": stable_digest(group_ids.tolist()),
        },
        "group_summary": summary,
        "row_level_overlap": row_overlap,
        "row_level_test_match_percent_overall": round(
            100 * sum(item["test_rows_with_training_feature_match"] for item in row_overlap)
            / max(sum(item["test_rows"] for item in row_overlap), 1),
            6,
        ),
        "group_aware_fold_status": group_status,
    }


def main() -> None:
    names = load_dataset_names(ROOT / "config" / "dataset_list.yaml")
    records = [audit_one(name) for name in names]
    payload = {
        "audit_date_local": "2026-09-29 Asia/Calcutta",
        "environment": r"D:\Conda\p12",
        "n_splits": 5,
        "seed": 42,
        "records": records,
        "dataset_count": len(records),
        "group_aware_split_feasible_count": sum(r["group_aware_fold_status"]["split_feasible"] for r in records),
        "group_aware_class_support_count": sum(r["group_aware_fold_status"]["class_support_status"] == "supported" for r in records),
        "group_aware_auc_count": sum(r["group_aware_fold_status"]["auc_status"] == "supported" for r in records),
    }
    (OUT / "group_fold_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Group-aware fold feasibility audit",
        "",
        "This audit uses the existing `p12` environment, five folds, seed 42, and the canonical target-excluded raw-feature grouping in `src/group_splits.py`.",
        "",
        f"Datasets: {len(records)}; group split construction feasible: {payload['group_aware_split_feasible_count']}; all-fold class support: {payload['group_aware_class_support_count']}; all-fold ROC-AUC support: {payload['group_aware_auc_count']}.",
        "",
        "| Dataset | Rows | Groups | Conflicting-label groups | Row-level test rows matching train X | Group split | Class support | AUC support | Reasons |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in records:
        status = row["group_aware_fold_status"]
        lines.append(
            f"| `{row['dataset']}` | {row['rows']:,} | {row['group_summary']['n_groups']:,} | {row['group_summary']['conflicting_label_group_count']:,} | {row['row_level_test_match_percent_overall']}% | {status['split_feasible']} | {status['class_support_status']} | {status['auc_status']} | {', '.join(status['reasons']) or '—'} |"
        )
    lines.extend(["", "Group-aware values are feasibility diagnostics, not corrected performance estimates. A dataset is not silently substituted with row-level folds when group-aware AUC support is unavailable."])
    (OUT / "group_fold_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "datasets": len(records),
        "split_feasible": payload["group_aware_split_feasible_count"],
        "class_support": payload["group_aware_class_support_count"],
        "auc_support": payload["group_aware_auc_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
