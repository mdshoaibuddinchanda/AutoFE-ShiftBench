"""Write the seven-run Reviewer #1 scope and identity manifest; never launch it."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from src.data_loader import load_dataset_names
from src.pipeline_runner import PIPELINE_CONFIGS
from src.provenance import code_fingerprint, current_git_commit, file_sha256, stable_digest
from src.stats_analysis import PRIMARY_CONDITIONS


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456, 789, 2025)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}
PIPELINES = (
    "Raw", "AutoFE_Baseline", "Raw_CapMatched", "AutoFE_MI", "AutoFE_Random",
    "AutoFE_NoMultiply", "AutoFE_Isolate_Add", "AutoFE_Isolate_Subtract",
    "AutoFE_Isolate_Multiply", "AutoFE_Isolate_Divide", "AutoFE_LeaveOut_Add",
    "AutoFE_LeaveOut_Subtract", "AutoFE_LeaveOut_Multiply", "AutoFE_LeaveOut_Divide",
)
MODELS = (
    "logistic_regression", "random_forest", "extra_trees", "linear_svm", "knn",
    "gaussian_nb", "mlp", "lightgbm", "xgboost", "catboost",
)
RUNS = (
    ("r1-v2-primary-row", "primary", "row_level", 10),
    ("r1-v2-primary-group", "primary", "group_aware", 10),
    ("r1-v2-domain-row", "transductive_domain_partition", "row_level", 2),
    ("r1-v2-availability-row", "feature_availability_ablation", "row_level", 1),
    ("r1-v2-availability-group", "feature_availability_ablation", "group_aware", 1),
    ("r1-v2-relabel-row", "majority_label_relabeling", "row_level", 1),
    ("r1-v2-relabel-group", "majority_label_relabeling", "group_aware", 1),
)
RUN_CONDITIONS = {
    "primary": sorted(PRIMARY_CONDITIONS),
    "transductive_domain_partition": ["covariate_partition", "population_partition"],
    "feature_availability_ablation": ["feature_availability_ablation_0.20"],
    "majority_label_relabeling": ["majority_label_relabeling"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('2', '3', '4'), default='4')
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path(r"D:\Conda\p12\python.exe").resolve():
        raise RuntimeError("Freeze with the existing D:\\Conda\\p12\\python.exe")
    if len(PIPELINES) != 14 or set(PIPELINES).difference(PIPELINE_CONFIGS):
        raise RuntimeError("Frozen pipeline list differs from the configured operator matrix")
    names = load_dataset_names(ROOT / "config" / "dataset_list.yaml")
    if len(names) != 25 or len(set(names)) != 25:
        raise RuntimeError("Expected 25 unique configured datasets")
    group_audit_path = ROOT / "provenance" / "group_seed_grid_audit_v1.json"
    group_audit = json.loads(group_audit_path.read_text(encoding="utf-8"))
    by_name = {row["dataset"]: row for row in group_audit["records"]}
    if set(by_name) != set(names) or tuple(group_audit["seeds"]) != SEEDS:
        raise RuntimeError("All-seed group audit does not cover the frozen dataset and seed grid")
    datasets = []
    ineligible = []
    for name in names:
        csv = ROOT / "data" / "raw" / f"{name}.csv"
        meta = csv.with_name(f"{name}_meta.json")
        csv_hash, meta_hash = file_sha256(csv), file_sha256(meta)
        audited = by_name[name]
        if (audited["csv_sha256"], audited["sidecar_sha256"]) != (csv_hash, meta_hash):
            raise RuntimeError(f"Dataset changed after group seed audit: {name}")
        bad_seeds = [seed for seed in SEEDS if audited["seeds"][str(seed)]["auc_status"] != "supported"]
        if bad_seeds:
            ineligible.append(name)
        datasets.append({"name": name, "csv_sha256": csv_hash, "sidecar_sha256": meta_hash,
                         "group_auc_infeasible_seeds": bad_seeds})
    if len(ineligible) != 2 or set(ineligible) != {"wine-quality-red", "kddcup99"}:
        raise RuntimeError(f"Frozen 23-dataset AUC denominator changed: {ineligible}")

    command_base = [
        str(Path(sys.executable).resolve()), "-m", "src.pipeline_runner",
        "--cache-policy", "bounded", "--cache-max-gib", "8", "--durable-scheduler",
        "--manifest-policy", "compact", "--scheduler-lease-seconds", "600",
        "--workers", "4", "--pipelines", *PIPELINES, "--models", *MODELS,
    ]
    resource_plan = None
    if args.version == '4':
        from src.resource_policy import ResourceSettings, detect_hardware, probe_gpu_models, resolve_plan
        hardware = detect_hardware(ROOT)
        resource_plan = resolve_plan(ResourceSettings(), hardware, MODELS,
                                     gpu_probe=probe_gpu_models(hardware['gpu_devices']))
        command_base = [str(Path(sys.executable).resolve()), '-m', 'src.pipeline_runner',
                        '--cache-policy', 'bounded', '--durable-scheduler', '--manifest-policy', 'compact',
                        '--scheduler-lease-seconds', '600', '--resource-policy', 'adaptive',
                        '--reserve-cpus', '2', '--ram-target-fraction', '0.8',
                        '--vram-target-fraction', '0.8', '--gpu-policy', 'auto',
                        '--pipelines', *PIPELINES, '--models', *MODELS]
    else:
        command_base.extend(['--resource-policy', 'manual'])
    runs = []
    for run_id, scope, policy, condition_count in RUNS:
        run_id = run_id.replace('r1-v2-', f'r1-v{args.version}-')
        if len(RUN_CONDITIONS[scope]) != condition_count:
            raise RuntimeError(f"Condition list differs from frozen run count: {run_id}")
        intended = len(names) * len(SEEDS) * 5 * condition_count * len(PIPELINES) * len(MODELS)
        skipped = len(ineligible) * len(SEEDS) * 5 * condition_count * len(PIPELINES) * len(MODELS) if policy == "group_aware" else 0
        runs.append({
            "run_id": run_id, "scope": scope, "split_policy": policy,
            "conditions": condition_count, "intended_cells": intended,
            "condition_names": RUN_CONDITIONS[scope],
            "prespecified_group_auc_skips": skipped,
            "auc_eligible_before_condition_specific_skips": intended - skipped,
            "required_environment": THREAD_ENV,
            "command_argv": [*command_base, "--run-id", run_id, "--scope", scope,
                             "--split-policy", policy],
        })
    free_bytes = shutil.disk_usage(ROOT).free
    host_path = ROOT / "provenance" / "launch_host_measurement_v2.json"
    payload = {
        "artifact_type": f"reviewer1_launch_scope_v{args.version}",
        "status": "FROZEN_SCOPE_HOST_CALIBRATION_PENDING",
        "full_campaign_started": False,
        "code_commit_at_freeze": current_git_commit(ROOT),
        "code_fingerprint": code_fingerprint(ROOT),
        "analysis_sha256": {name: file_sha256(ROOT / name) for name in (
            "src/reviewer1_analysis.py", "src/stats_analysis.py", "src/generate_tables.py",
            "src/plotting_q1.py", "provenance/generate_corrected_assets.py",
            "provenance/large_dataset_recovery_v1.py",
            "provenance/large_dataset_calibration_group_v3.py",
            "provenance/verify_reviewer1_launch_v2.py",
            "provenance/reviewer1_condition_crosswalk.md",
            "provenance/final_optimization_verification.py",
            "provenance/run_mechanism_history.py",
            "provenance/analyze_mechanism_history.py",
            "provenance/adaptive_resource_verification.py",
            "provenance/verify_adaptive_launch.py",
        )},
        "dataset_list_sha256": file_sha256(ROOT / "config" / "dataset_list.yaml"),
        "group_seed_audit_sha256": file_sha256(group_audit_path),
        "dataset_hash_digest": stable_digest(datasets),
        "datasets": datasets,
        "group_auc_ineligible_datasets": ineligible,
        "all_seed_auc_rule": "A dataset is excluded from group-aware AUC for all configured cells when any configured seed lacks all-fold AUC support; its group folds and skip reasons remain in results.",
        "seeds": list(SEEDS), "folds": [1, 2, 3, 4, 5],
        "primary_conditions": sorted(PRIMARY_CONDITIONS),
        "pipelines": list(PIPELINES), "models": list(MODELS),
        "required_numerical_thread_environment": THREAD_ENV,
        "runs": runs,
        "totals": {
            "run_count": len(runs),
            "intended_cells": sum(item["intended_cells"] for item in runs),
            "prespecified_group_auc_skips": sum(item["prespecified_group_auc_skips"] for item in runs),
            "auc_eligible_before_condition_specific_skips": sum(item["auc_eligible_before_condition_specific_skips"] for item in runs),
        },
        "storage": {
            "free_bytes_at_freeze": free_bytes,
            "bounded_live_cache_cap_gib_per_run": 8,
            "cache_publication_temporary_allowance_gib": 8,
            "prior_results_checkpoints_projection_gib_primary": 20.87,
            "scaled_results_checkpoints_projection_gib_all_seven_runs": round(20.87 * 2_275_000 / 1_750_000, 2) + (1 if args.version == '3' else 0),
            "added_timing_metadata_reservation_gib": 1 if args.version == '3' else 0,
            "results_projection_basis": "Historical 27.13GiB seven-run projection plus1GiB allowance for new measured timing fields in JSONL, result envelopes and SQLite mirrors; full-grid manifests disable bounded cache audit lists. Not a guaranteed upper bound; verifier doubles this projection and checks live free space before every run.",
            "separate_mechanism_history_reservation_gib": 1,
            "prior_candidate_history_projection_gib": 79.66,
            "candidate_history_enabled_in_commands": False,
            "candidate_history_note": "The historical 79.66 GiB estimate used 612,500 assumed history tasks; the full runner does not currently write candidate histories. A separate measured mechanism-history plan is required before using that estimate as a reservation.",
        },
        "host_probe_path": host_path.relative_to(ROOT).as_posix() if host_path.exists() else None,
        "host_probe_sha256": file_sha256(host_path) if host_path.exists() else None,
        "runtime": {
            "prior_four_worker_primary_projection_days": 180.56,
            "prior_four_worker_all_seven_run_projection_days": 229.69,
            "basis": "older bounded linear scenario, not a host-calibrated estimate; ten-day limit waived by user",
            "representative_both_policy_large_dataset_calibration": "PENDING",
        },
        "optimization_verification_path": "provenance/final_optimization_verification_v1.json" if args.version == '3' else None,
        "recovery_evidence_path": "provenance/large_dataset_recovery_v3.json" if args.version == '3' else "provenance/large_dataset_recovery_v1.json",
        "mechanism_scope_path": "provenance/mechanism_history_scope_v2.json" if args.version == '3' else "provenance/mechanism_history_scope_v1.json",
        "unresolved_gates": [
            "Confirm intended execution host and remeasure there if different from the recorded host",
            "Measure representative large-dataset throughput, peak RAM, bounded cache and interruption/resume on that host",
            "Prespecify and measure candidate-history mechanism subrun if Jacobian associations are required",
        ],
    }
    if resource_plan is not None:
        cache_gib = resource_plan['cache_max_bytes']/1024**3
        payload.update(resource_plan=resource_plan,
                       adaptive_verification_path='provenance/adaptive_resource_verification_v1.json',
                       optimization_verification_path=None,
                       recovery_evidence_path='provenance/adaptive_resource_verification_v1.json',
                       mechanism_scope_path='provenance/mechanism_history_scope_v3.json')
        payload['storage'].update(
            bounded_live_cache_cap_gib_per_run=cache_gib,
            cache_publication_temporary_allowance_gib=cache_gib,
            scaled_results_checkpoints_projection_gib_all_seven_runs=31.13,
            added_timing_metadata_reservation_gib=1,
            added_adaptive_metadata_and_parameter_catalog_reservation_gib=3,
            results_projection_basis='Historical 27.13 GiB plus 1 GiB measured-timing and 3 GiB resource/parameter-reference allowances; conservative planning estimate, not a guaranteed upper bound. Gate requires twice this volume plus live cache, staging cache and separate mechanism history.')
        payload['runtime'].update(
            basis='Older 229.69-day fixed-four-worker scenario retained as historical; adaptive backend/concurrency changes require bounded v4 timings. No validated full-campaign ETA; ten-day limit waived.',
            adaptive_runtime_evidence='provenance/adaptive_resource_verification_v1.json')
    output = ROOT / "provenance" / f"reviewer1_launch_scope_v{args.version}.json"
    if output.exists():
        raise FileExistsError(f'Preserve prior freeze: {output}')
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "totals": payload["totals"],
                      "group_auc_ineligible_datasets": ineligible}, indent=2))


if __name__ == "__main__":
    main()
