"""Freeze the separate clean-condition mechanism-history runs; never launch them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.provenance import code_fingerprint, file_sha256


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("haberman", "sonar")
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def _pilot_result(policy: str, dataset: str) -> dict:
    path = (ROOT / "corrected_runs" / "four_dataset_pilot"
            / f"pilot-{policy}-001" / "results.jsonl")
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get("dataset") == dataset and row.get("seed") == 42
                    and row.get("fold") == 1 and row.get("condition") == "clean"
                    and row.get("pipeline") == "AutoFE_Baseline"
                    and row.get("model") == "logistic_regression"
                    and row.get("status") == "success"):
                return row
    raise RuntimeError(f"No completed pilot matrix evidence for {policy}/{dataset}")


def main() -> None:
    if Path(sys.executable).resolve() != Path(r"D:\Conda\p12\python.exe").resolve():
        raise RuntimeError("Freeze in the existing p12 environment")
    primary = json.loads((ROOT / "provenance" / "reviewer1_launch_scope_v2.json").read_text(encoding="utf-8"))
    fingerprint = code_fingerprint(ROOT)
    if fingerprint != primary["code_fingerprint"]:
        raise RuntimeError("Mechanism source differs from frozen corrected benchmark")
    mechanism_script = ROOT / "provenance" / "run_mechanism_history.py"
    analysis_script = ROOT / "provenance" / "analyze_mechanism_history.py"
    smoke = []
    sonar_bytes = []
    for policy in ("row_level", "group_aware"):
        run_dir = ROOT / "corrected_runs" / f"mechanism-smoke-{'row' if policy == 'row_level' else 'group'}-003"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["status"] != "complete" or manifest["counts"]["complete"] != 2
                or manifest["configuration"]["mechanism_script_sha256"] != file_sha256(mechanism_script)
                or manifest["configuration"]["numerical_thread_environment"] != THREAD_ENV):
            raise RuntimeError(f"Bounded mechanism smoke identity/count mismatch: {policy}")
        for dataset in DATASETS:
            task = next(
                json.loads(path.read_text(encoding="utf-8"))
                for path in (run_dir / "tasks").rglob("status.json")
                if json.loads(path.read_text(encoding="utf-8"))["dataset"] == dataset
            )
            pilot = _pilot_result(policy, dataset)
            parity = (
                task["train_matrix_sha256"] == pilot["train_matrix_sha256"]
                and task["test_matrix_sha256"] == pilot["test_matrix_sha256"]
            )
            if not parity:
                raise RuntimeError(f"Mechanism/runner matrix mismatch: {policy}/{dataset}")
            smoke.append({
                "split_policy": policy, "dataset": dataset,
                "candidate_count": task["candidate_count"],
                "matrix_hash_parity_with_runner_pilot": parity,
                "history_sha256": task["history_sha256"],
                "jacobian_sha256": task["jacobian_sha256"],
            })
            if dataset == "sonar":
                task_path = next(path for path in (run_dir / "tasks").rglob("status.json")
                                 if json.loads(path.read_text(encoding="utf-8"))["dataset"] == dataset)
                sonar_bytes.append((task_path.parent / task["history_path"]).stat().st_size
                                   + (task_path.parent / task["jacobian_path"]).stat().st_size)
    commands = [
        {"run_id": f"r1-v2-mechanism-{label}", "split_policy": policy,
         "intended_tasks": 625, "prespecified_group_auc_skips": 50 if policy == "group_aware" else 0,
         "command_argv": [str(Path(sys.executable).resolve()), "-m",
                          "provenance.run_mechanism_history", "--run-id",
                          f"r1-v2-mechanism-{label}", "--split-policy", policy]}
        for label, policy in (("row", "row_level"), ("group", "group_aware"))
    ]
    payload = {
        "artifact_type": "reviewer1_clean_arithmetic_mechanism_scope_v1",
        "status": "ready", "full_mechanism_runs_started": False,
        "code_fingerprint": fingerprint,
        "dataset_hash_digest": primary["dataset_hash_digest"],
        "mechanism_script_sha256": file_sha256(mechanism_script),
        "association_analysis_script_sha256": file_sha256(analysis_script),
        "required_numerical_thread_environment": THREAD_ENV,
        "datasets": 25, "seeds": 5, "folds": 5,
        "condition": "clean", "pipeline": "AutoFE_Baseline",
        "commands": commands,
        "totals": {"intended_tasks": 1250, "prespecified_group_auc_skips": 50,
                   "candidate_history_tasks": 1200},
        "measurement": {
            "candidate_selection_score_scope": "train_only",
            "jacobian_definition": "analytic arithmetic derivative on preprocessed training parents",
            "input_scale": "population standard deviation of each parent on the training fold; zero/nonfinite scales undefined",
            "norm": "scaled_l2", "sample_rows_per_fold_max": 256,
            "sampling": "deterministic without replacement from training rows",
            "exposure": "dataset mean of fold median selected-candidate finite scaled Jacobian norms",
            "performance_outcome": "dataset mean paired clean AutoFE_Baseline minus Raw ROC-AUC across 25 seed-folds and ten classifiers",
            "association": "separate row/group Spearman rho; 10,000 two-sided permutations and 10,000 dataset bootstraps; exploratory, not causal",
        },
        "storage": {
            "sonar_history_plus_jacobian_bytes_per_task": int(max(sonar_bytes)),
            "sonar_extrapolation_gib_for_1200_tasks": round(max(sonar_bytes) * 1200 / 1024**3, 3),
            "reserved_gib": 1,
            "note": "Measured compressed Sonar files; actual candidate counts and filesystem overhead vary by dataset. This separate run replaces the unsupported 79.66 GiB full-grid estimate for its limited clean-condition scope.",
        },
        "bounded_evidence": smoke,
        "scientific_limit": "The association samples clean-condition baseline candidate behavior only. It cannot establish a causal mechanism or condition-specific Jacobian effects.",
    }
    output = ROOT / "provenance" / "mechanism_history_scope_v1.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": payload["status"],
                      "totals": payload["totals"], "storage": payload["storage"]}, indent=2))


if __name__ == "__main__":
    main()
