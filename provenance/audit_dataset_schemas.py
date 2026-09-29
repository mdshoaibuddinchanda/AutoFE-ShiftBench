"""Generate a structural audit of every configured raw benchmark CSV."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.data_loader import inspect_target_proxy_candidates, load_dataset_names
from src.provenance import file_sha256
from src.splitters import get_stratified_splits


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "provenance"


def inspect_dataset(name: str) -> dict:
    csv_path = DATA_DIR / f"{name}.csv"
    meta_path = DATA_DIR / f"{name}_meta.json"
    if not csv_path.is_file() or not meta_path.is_file():
        return {
            "dataset": name,
            "status": "missing",
            "csv_exists": csv_path.is_file(),
            "metadata_exists": meta_path.is_file(),
        }

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_header = next(csv.reader(stream), [])
    frame = pd.read_csv(csv_path)
    target = meta.get("target_column")
    if target not in frame.columns:
        target = "target_label" if "target_label" in frame.columns else (
            "target" if "target" in frame.columns else None
        )
    if target is None:
        return {"dataset": name, "status": "target_missing", "columns": list(frame.columns)}

    y = frame[target]
    features = frame.drop(columns=[target])
    target_text = y.astype("string").reset_index(drop=True)
    class_count = int(y.nunique(dropna=False))
    target_values = y.astype("string").value_counts(dropna=False).head(12)
    columns = []
    missing_columns = []
    exact_numeric = []
    perfect_mappings = []
    identifier_name_candidates = []
    high_cardinality_text_fields = []
    constant_features = []
    all_missing_features = []
    hashes: dict[int, list[int]] = defaultdict(list)

    for position, column in enumerate(features.columns):
        series = features.iloc[:, position]
        column_name = str(column)
        unique_count = int(series.nunique(dropna=False))
        missing_count = int(series.isna().sum())
        unique_fraction = unique_count / max(len(frame), 1)

        if series.astype("string").reset_index(drop=True).equals(target_text):
            # The project screen below will report this exact copy.
            pass
        else:
            numeric_feature = pd.to_numeric(series, errors="coerce")
            numeric_target = pd.to_numeric(y, errors="coerce")
            all_feature_values_numeric = numeric_feature.notna().equals(series.notna())
            all_target_values_numeric = numeric_target.notna().equals(y.notna())
            if (
                all_feature_values_numeric
                and all_target_values_numeric
                and numeric_feature.reset_index(drop=True).equals(numeric_target.reset_index(drop=True))
            ):
                exact_numeric.append(column_name)

        if unique_count <= 1:
            constant_features.append(column_name)
        if missing_count == len(frame):
            all_missing_features.append(column_name)
        if missing_count:
            missing_columns.append({
                "column": column_name,
                "count": missing_count,
                "percent": round(100 * missing_count / max(len(frame), 1), 4),
            })

        tokens = set(re.findall(r"[a-z0-9]+", column_name.casefold()))
        if tokens.intersection({"id", "key", "index", "record", "row"}):
            identifier_name_candidates.append(column_name)
        if (
            (pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype))
            and len(frame) >= 100
            and unique_fraction >= 0.8
        ):
            high_cardinality_text_fields.append({
                "column": column_name,
                "unique_values": unique_count,
                "unique_fraction": round(unique_fraction, 6),
            })

        if class_count > 1 and unique_count > 1 and (
            unique_count <= max(50, 4 * class_count) or unique_fraction <= 0.2
        ):
            pairs = pd.DataFrame({
                "feature": series.astype("string").fillna("<MISSING>").to_numpy(),
                "target": target_text.fillna("<MISSING>").to_numpy(),
            })
            target_counts = pairs.groupby("feature", dropna=False)["target"].nunique(dropna=False)
            if len(target_counts) and int(target_counts.max()) == 1:
                perfect_mappings.append({
                    "column": column_name,
                    "feature_values": unique_count,
                    "target_classes": class_count,
                    "review_reason": "each observed feature value maps to one target class; this may be legitimate or a proxy",
                })

        hash_value = int(pd.util.hash_pandas_object(series, index=False, categorize=True).sum())
        hashes[hash_value].append(position)
        columns.append({
            "name": column_name,
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percent": round(100 * missing_count / max(len(frame), 1), 6),
            "unique_values_including_missing": unique_count,
            "sample_values": [str(value)[:100] for value in series.dropna().drop_duplicates().head(3)],
        })

    duplicate_feature_groups = []
    for positions in hashes.values():
        if len(positions) < 2:
            continue
        pending = list(positions)
        while pending:
            first = pending.pop(0)
            group = [first]
            for other in pending[:]:
                if features.iloc[:, first].reset_index(drop=True).equals(
                    features.iloc[:, other].reset_index(drop=True)
                ):
                    group.append(other)
                    pending.remove(other)
            if len(group) > 1:
                duplicate_feature_groups.append([str(features.columns[i]) for i in group])

    try:
        proxy_review = inspect_target_proxy_candidates(
            features,
            y,
            target_column=target,
            source_target_name=meta.get("source_target_name"),
        )
    except Exception as error:
        proxy_review = {"error": f"{type(error).__name__}: {error}"}

    feature_hashes = pd.util.hash_pandas_object(features, index=False, categorize=True)
    hash_counts = feature_hashes.value_counts()
    repeated_hashes = hash_counts[hash_counts > 1]
    target_by_hash = pd.DataFrame({
        "feature_hash": feature_hashes.to_numpy(),
        "target": y.astype("string").fillna("<MISSING>").to_numpy(),
    }).groupby("feature_hash", sort=False)["target"]
    target_counts_by_hash = target_by_hash.nunique(dropna=False)
    conflicting_hashes = target_counts_by_hash[target_counts_by_hash > 1].index
    conflicting_hash_sizes = hash_counts.loc[hash_counts.index.intersection(conflicting_hashes)]
    duplicate_test_rows_with_train_match = 0
    test_rows_in_overlap_audit = 0
    overlap_by_fold = []
    try:
        for train_indices, test_indices in get_stratified_splits(
            features, y, n_splits=5, seed=42, target_column=target
        ):
            train_hashes = set(feature_hashes.iloc[train_indices].tolist())
            matched = sum(value in train_hashes for value in feature_hashes.iloc[test_indices])
            duplicate_test_rows_with_train_match += matched
            test_rows_in_overlap_audit += len(test_indices)
            overlap_by_fold.append(round(100 * matched / max(len(test_indices), 1), 4))
    except ValueError as error:
        overlap_by_fold = [f"not computed: {error}"]

    csv_hash = file_sha256(csv_path)
    metadata_columns = meta.get("schema_columns")
    return {
        "dataset": name,
        "status": "ok",
        "file": csv_path.relative_to(ROOT).as_posix(),
        "metadata_file": meta_path.relative_to(ROOT).as_posix(),
        "file_bytes": csv_path.stat().st_size,
        "csv_sha256": csv_hash,
        "metadata_sha256_matches": meta.get("saved_csv_sha256") == csv_hash,
        "openml_or_uci_identity": meta.get("dataset_identity", {}),
        "row_cap": 100000,
        "rows": int(len(frame)),
        "feature_count": int(features.shape[1]),
        "target_column": target,
        "source_target_name": meta.get("source_target_name"),
        "target_dtype": str(y.dtype),
        "target_missing_count": int(y.isna().sum()),
        "target_class_count_including_missing": class_count,
        "target_top_values": [
            {"value": str(value), "count": int(count)}
            for value, count in target_values.items()
        ],
        "total_feature_missing_cells": int(features.isna().sum().sum()),
        "features_with_missing_values": missing_columns,
        "duplicate_rows_including_target": int(frame.duplicated().sum()),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "feature_vectors_in_repeated_groups": int(repeated_hashes.sum()),
        "unique_repeated_feature_vectors": int(len(repeated_hashes)),
        "repeated_feature_vectors_with_conflicting_targets": int(len(conflicting_hashes)),
        "rows_in_conflicting_target_feature_groups": int(conflicting_hash_sizes.sum()),
        "five_fold_seed42_test_rows_with_train_feature_match_percent": round(
            100 * duplicate_test_rows_with_train_match / max(test_rows_in_overlap_audit, 1), 4
        ),
        "five_fold_seed42_test_match_percent_by_fold": overlap_by_fold,
        "raw_duplicate_header_names": sorted({value for value in raw_header if raw_header.count(value) > 1}),
        "metadata_schema_columns_match_csv": metadata_columns == list(frame.columns),
        "metadata_schema_columns": metadata_columns,
        "exact_target_copies": proxy_review.get("exact_target_copies_for_manual_review", []),
        "numeric_equivalent_target_copies": exact_numeric,
        "target_like_feature_names": proxy_review.get("target_like_names_for_manual_review", []),
        "perfect_single_feature_target_mappings": perfect_mappings,
        "constant_features": constant_features,
        "all_missing_features": all_missing_features,
        "duplicate_feature_column_groups": duplicate_feature_groups,
        "identifier_name_candidates": identifier_name_candidates,
        "high_cardinality_text_fields": high_cardinality_text_fields,
        "columns": columns,
        "meta_feature_compute_notes": {
            "number_of_samples_matches": meta.get("number_of_samples") == float(len(frame)),
            "number_of_features_matches": meta.get("number_of_features") == float(features.shape[1]),
            "average_mutual_information": meta.get("average_mutual_information"),
            "intrinsic_dimension": meta.get("intrinsic_dimension"),
        },
    }


def main() -> None:
    names = load_dataset_names(ROOT / "config" / "dataset_list.yaml")
    records = [inspect_dataset(name) for name in names]
    ok_records = [row for row in records if row["status"] == "ok"]
    payload = {
        "audit_date_local": "2026-09-29 Asia/Calcutta",
        "repository": "AutoFE-ShiftBench",
        "environment": r"D:\Conda\p12",
        "method": (
            "Read every configured CSV; verify its JSON sidecar checksum and schema; report types, missingness, "
            "target distribution, duplicates, constants, identifier-name and high-cardinality text candidates, "
            "exact target copies, target-like names, and perfect single-feature mappings. Findings are review "
            "candidates, not automatic exclusions."
        ),
        "dataset_count_configured": len(names),
        "dataset_count_ok": len(ok_records),
        "dataset_count_missing_or_error": len(records) - len(ok_records),
        "records": records,
    }
    (OUT_DIR / "dataset_schema_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def show(values: list[str]) -> str:
        return ", ".join(f"`{value}`" for value in values) if values else "—"

    verified = sum(
        row["metadata_sha256_matches"] and row["metadata_schema_columns_match_csv"]
        for row in ok_records
    )
    lines = [
        "# Downloaded dataset schema and integrity audit",
        "",
        "Audit date: 2026-09-29 (Asia/Calcutta). Environment: existing Conda `p12` at `D:\\Conda\\p12`.",
        "",
        f"Downloaded and inspected **{len(ok_records)} of {len(names)}** configured benchmark datasets. The downloader caps each saved dataset at 100,000 rows. SHA-256 and sidecar schema checks passed for **{verified} of {len(ok_records)}** dataset/sidecar pairs.",
        "",
        "This is a structural screen. Exact target matches, target-like names, perfect one-column target mappings, and identifier-like fields are candidates for human review and are never removed automatically. Column names and values alone cannot establish semantic leakage.",
        "",
        "The audit also replayed the current row-level stratified five-fold splitter with seed 42 and checked whether each test row had an identical feature vector in training. This measures split overlap in the saved data; it does not estimate score inflation. These repeated rows remain in the data because their frequencies are part of the source distribution.",
        "",
        "| Dataset | Rows | Features | Target (classes) | Missing feature cells | Repeated X rows | Conflicting X groups | Test rows with train X match, 5-fold seed 42 | Exact copies | Target-like names | Perfect mappings | Identifier-name candidates | High-cardinality text fields |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in ok_records:
        exact = row["exact_target_copies"] + row["numeric_equivalent_target_copies"]
        lines.append(
            f"| `{row['dataset']}` | {row['rows']:,} | {row['feature_count']} | `{row['target_column']}` ({row['target_class_count_including_missing']}) | {row['total_feature_missing_cells']:,} | {row['feature_vectors_in_repeated_groups']:,} | {row['repeated_feature_vectors_with_conflicting_targets']:,} | {row['five_fold_seed42_test_rows_with_train_feature_match_percent']}% | {show(exact)} | {show(row['target_like_feature_names'])} | {show([item['column'] for item in row['perfect_single_feature_target_mappings']])} | {show(row['identifier_name_candidates'])} | {show([item['column'] for item in row['high_cardinality_text_fields']])} |"
        )

    lines.extend(["", "## Per-dataset schema", ""])
    for row in ok_records:
        identity = row["openml_or_uci_identity"]
        source_id = identity.get("openml_id") or identity.get("uci_id") or "not recorded"
        provider = identity.get("provider", "unknown")
        lines.extend([
            f"### {row['dataset']}",
            "",
            f"Source: {provider}, id `{source_id}`; target `{row['target_column']}` from `{row['source_target_name']}`; {row['rows']:,} saved rows, {row['feature_count']} features, {row['target_class_count_including_missing']} target classes; CSV checksum verified: {row['metadata_sha256_matches']}; sidecar schema verified: {row['metadata_schema_columns_match_csv']}.",
            "",
            "Columns (`name`: dtype, missing %, unique values):",
            "",
            "; ".join(
                f"`{column['name']}`: {column['dtype']}, {column['missing_percent']}%, {column['unique_values_including_missing']} unique"
                for column in row["columns"]
            ) or "(none)",
            "",
        ])
        if row["features_with_missing_values"]:
            lines.append("Missing-value fields: " + ", ".join(
                f"`{item['column']}` ({item['count']}, {item['percent']}%)"
                for item in row["features_with_missing_values"]
            ) + ".")
            lines.append("")
        lines.append("Target frequency sample: " + ", ".join(
            f"`{item['value']}`={item['count']}" for item in row["target_top_values"]
        ) + ".")
        lines.append("")
        lines.append(
            f"Repeated rows: {row['duplicate_rows_including_target']:,} including target and {row['duplicate_feature_rows']:,} by features alone; constant features: {show(row['constant_features'])}; duplicate feature groups: {row['duplicate_feature_column_groups'] or '—'}; all-missing features: {show(row['all_missing_features'])}."
        )
        lines.append("")
    (OUT_DIR / "dataset_schema_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"audited={len(ok_records)}/{len(names)}; verified={verified}/{len(ok_records)}")
    print("exact_target_copies=", [r["dataset"] for r in ok_records if r["exact_target_copies"] or r["numeric_equivalent_target_copies"]])
    print("target_like_names=", [r["dataset"] for r in ok_records if r["target_like_feature_names"]])
    print("perfect_mappings=", [r["dataset"] for r in ok_records if r["perfect_single_feature_target_mappings"]])
    print("constant_feature_sets=", [(r["dataset"], r["constant_features"]) for r in ok_records if r["constant_features"]])
    print("duplicate_row_sets=", [(r["dataset"], r["duplicate_rows_including_target"]) for r in ok_records if r["duplicate_rows_including_target"]])


if __name__ == "__main__":
    main()
