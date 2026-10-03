"""Freeze scientific definitions without acquiring data or executing the grid."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialise(value):
    if isinstance(value, dict):
        return {str(k): serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialise(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_parameter": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "get_params"):
        return {"class": type(value).__module__ + "." + type(value).__name__, "parameters": serialise(value.get_params(deep=False))}
    return str(value)


def freeze(output: Path) -> dict:
    from src import pipeline_runner as runner
    from src.data_loader import load_dataset_names
    from src.dataset_statistics import AnalysisConfig
    from src.sensitivity_analysis import SensitivityConfig
    from src.fsva import FSVA_SCHEMA_VERSION, DEFAULT_PERTURBATION_MAGNITUDES
    from src.model import build_model
    from src.protocol import EVALUATION_PROTOCOL_VERSION
    from src.seeding import SEED_SCHEME_VERSION
    names = load_dataset_names(ROOT / "config/dataset_list.yaml")
    sources = [Path(p) for p in subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()]
    definitions = ["src/seeding.py", "src/splitters.py", "src/shift_generator.py", "src/preprocessing.py", "src/operator_registry.py", "src/feature_engineering.py", "src/feature_selection.py", "src/evaluation.py", "src/fsva.py", "src/model.py"]
    data = []
    for name in names:
        path = ROOT / "data/raw" / (name + ".csv")
        row = {"name": name, "path": path.relative_to(ROOT).as_posix(), "availability": "available" if path.exists() else "unavailable", "source_version": "unverified_until_exact_source_metadata_exists", "parsing": "pandas.read_csv defaults", "row_policy": "download cap 100000, pandas sample random_state=42 without replacement", "target_policy": "target_label then target"}
        if path.exists():
            import pandas as pd
            frame = pd.read_csv(path)
            row.update({"sha256": sha256(path), "bytes": path.stat().st_size, "rows": len(frame), "schema": {str(c): str(t) for c,t in frame.dtypes.items()}, "coordinates": list(frame.columns)})
        data.append(row)
    contract = {
        "schema": "scientific_contract_v1", "role": "pre_repair_baseline",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol": EVALUATION_PROTOCOL_VERSION, "seed_scheme": SEED_SCHEME_VERSION,
        "root_seeds": [42,123,456,789,2025], "folds": [1,2,3,4,5], "datasets": data,
        "conditions": [list(c) for c in runner.SHIFT_FAMILIES],
        "pipelines": {k: asdict(v) for k,v in runner.PIPELINE_CONFIGS.items()},
        "estimators": {name: {"cpu": serialise(build_model(name, random_state=42, use_gpu=False)), "declared_device": "GPU" if name in runner.GPU_MODELS else "CPU"} for name in runner.CPU_MODELS + runner.GPU_MODELS},
        "precompute_tasks": len(names)*5*5*len(runner.SHIFT_FAMILIES),
        "model_tasks": len(names)*5*5*len(runner.SHIFT_FAMILIES)*len(runner.PIPELINE_NAMES)*(len(runner.CPU_MODELS)+len(runner.GPU_MODELS)),
        "diagnostics": {"enabled_default": False, "schema": FSVA_SCHEMA_VERSION, "max_rows": 128, "finite_difference_rows": 32, "epsilon": 1e-6, "magnitudes": list(DEFAULT_PERTURBATION_MAGNITUDES), "norm": "Frobenius/L2", "history": "all evaluated candidates plus excluded base events", "seed_definition": "stable_seed('fsva_diagnostic', dataset/split_policy/seed/fold/condition)"},
        "analysis": AnalysisConfig().to_dict(), "sensitivity": SensitivityConfig().to_dict(),
        "scientific_definitions": {p: (ROOT/p).read_text(encoding="utf-8") for p in definitions},
        "source_hashes": {p.as_posix(): sha256(ROOT/p) for p in sources if (ROOT/p).is_file()},
        "environment": {"python": sys.version, "platform": platform.platform(), "supported_environment": "Conda P12"},
        "unavailable_inputs": "No exact row/corruption/feature identities can be certified for absent raw datasets. Controlled fixtures record them separately.",
        "equivalence_acceptance": {"identity_fields": "exact: task membership, seeds, rows, corruption arrays, vocabularies, candidates, scores/ties/selections, labels, validity, linkage", "same_kernel": "exact numerical equality", "rearranged_reductions": {"float64_norms": {"rtol": 1e-12, "atol": 1e-12}, "restriction": "no changes to selections, clipping, predictions, validity or inference decisions"}},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Frozen contract already exists: {output}")
    output.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = freeze(args.output)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "precompute_tasks": value["precompute_tasks"], "model_tasks": value["model_tasks"]}))
