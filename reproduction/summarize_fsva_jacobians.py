"""Summarize locally available saved FSVA Jacobian sidecars.

This audit utility reads diagnostic JSON and neighboring history JSONL files only.
It never starts benchmark tasks or modifies source data, caches, or model results.
Run with the project Python environment:
    python reproduction/summarize_fsva_jacobians.py
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CACHE_SUFFIX = Path(
    "predictor_only_geometry_v3_integrity_metrics_fsva_v2"
) / "sha256_canonical_json_u32_v1"
ROOTS = [
    ROOT / "data" / "cache" / CACHE_SUFFIX,
    ROOT / "reports" / "full_benchmark_20261003" / "cache" / CACHE_SUFFIX,
]
JSON_OUT = ROOT / "reports" / "fsva_jacobian_measured_summary_20261122.json"
CSV_OUT = ROOT / "reports" / "fsva_jacobian_pipeline_summary_20261122.csv"


def quantiles(values: list[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)],
        dtype=float,
    )
    if not finite.size:
        return {"n": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(finite.size),
        "min": float(np.min(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    unique: dict[str, tuple[Path, dict]] = {}
    path_count = 0
    for root in ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*_meta_fsva.json"):
            path_count += 1
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest not in unique:
                unique[digest] = (path, json.loads(raw))

    if not unique:
        raise FileNotFoundError("No saved FSVA Jacobian sidecars found in the declared cache roots.")

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    operator_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    validation_counts: collections.Counter = collections.Counter()
    history_count = 0
    history_bytes = 0
    for path, diagnostic in unique.values():
        task = diagnostic.get("task", {})
        jacobian = diagnostic.get("jacobian", {})
        selected = jacobian.get("selected", {})
        raw_control = jacobian.get("raw_control", {})
        history = Path(str(path).replace("_meta_fsva.json", "_meta_history.jsonl"))
        if history.is_file():
            history_count += 1
            history_bytes += history.stat().st_size

        status = diagnostic.get("derivative_validation", {}).get("status", "missing")
        validation_counts[status] += 1
        pipeline = task.get("pipeline", "unknown")
        operator_set_id = task.get("operator_set_id", "unknown")
        selected_norm = selected.get("dimension_adjusted_frobenius")
        raw_norm = raw_control.get("dimension_adjusted_frobenius")
        ratio = (
            selected_norm / raw_norm
            if isinstance(selected_norm, (int, float))
            and isinstance(raw_norm, (int, float))
            and raw_norm != 0
            else None
        )
        grouped[pipeline].append(
            {
                "dataset": task.get("dataset"),
                "operator_set_id": operator_set_id,
                "selected_norm": selected_norm,
                "raw_norm": raw_norm,
                "ratio": ratio,
                "selected_frobenius": selected.get("frobenius_norm"),
            }
        )
        operator_counts[pipeline][operator_set_id] += 1

    by_pipeline = {}
    for pipeline, records in sorted(grouped.items()):
        by_pipeline[pipeline] = {
            "artifacts": len(records),
            "datasets": len({record["dataset"] for record in records}),
            "operator_set_id_counts": dict(operator_counts[pipeline]),
            "selected_dimension_adjusted_frobenius": quantiles(
                [record["selected_norm"] for record in records]
            ),
            "raw_control_dimension_adjusted_frobenius": quantiles(
                [record["raw_norm"] for record in records]
            ),
            "selected_to_raw_control_ratio": quantiles(
                [record["ratio"] for record in records]
            ),
        }

    all_autofe_ratios = [
        record["ratio"]
        for pipeline, records in grouped.items()
        if pipeline.startswith("AutoFE_")
        for record in records
    ]
    summary = {
        "local_paths_total": path_count,
        "unique_content_sidecars": len(unique),
        "duplicate_copies": path_count - len(unique),
        "history_present_unique": history_count,
        "history_total_bytes_unique": history_bytes,
        "derivative_validation_status_counts": dict(validation_counts),
        "measured_norm_definition": "dimension_adjusted_frobenius = frobenius_norm / sqrt(number of selected outputs)",
        "paired_ratio_definition": "selected dimension_adjusted_frobenius / paired raw_control dimension_adjusted_frobenius",
        "by_pipeline": by_pipeline,
        "autofe_all_selected_to_raw_control_ratio": quantiles(all_autofe_ratios),
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    fields = [
        "pipeline",
        "artifacts",
        "datasets",
        "operator_set_ids",
        "selected_dimension_adjusted_frobenius_p50",
        "selected_dimension_adjusted_frobenius_p95",
        "selected_to_raw_control_ratio_p50",
        "selected_to_raw_control_ratio_p95",
        "selected_to_raw_control_ratio_max",
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pipeline, summary_row in by_pipeline.items():
            selected = summary_row["selected_dimension_adjusted_frobenius"]
            ratio = summary_row["selected_to_raw_control_ratio"]
            writer.writerow(
                {
                    "pipeline": pipeline,
                    "artifacts": summary_row["artifacts"],
                    "datasets": summary_row["datasets"],
                    "operator_set_ids": json.dumps(
                        summary_row["operator_set_id_counts"], sort_keys=True
                    ),
                    "selected_dimension_adjusted_frobenius_p50": selected["p50"],
                    "selected_dimension_adjusted_frobenius_p95": selected["p95"],
                    "selected_to_raw_control_ratio_p50": ratio["p50"],
                    "selected_to_raw_control_ratio_p95": ratio["p95"],
                    "selected_to_raw_control_ratio_max": ratio["max"],
                }
            )
    print(json.dumps({"json": str(JSON_OUT), "csv": str(CSV_OUT), **{k: summary[k] for k in (
        "local_paths_total", "unique_content_sidecars", "duplicate_copies",
        "history_present_unique", "history_total_bytes_unique", "derivative_validation_status_counts"
    )}}, indent=2))


if __name__ == "__main__":
    main()
