"""Bounded before/after verification; never run the corrected campaign.

Use the existing p12 with a separate baseline source checkout. All numerical
settings, datasets, folds, classifiers and feature parameters are held fixed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
THREADS = {name: "1" for name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
PIPELINES = (
    "Raw", "AutoFE_Baseline", "Raw_CapMatched", "AutoFE_MI", "AutoFE_Random",
    "AutoFE_NoMultiply", "AutoFE_Isolate_Add", "AutoFE_Isolate_Subtract",
    "AutoFE_Isolate_Multiply", "AutoFE_Isolate_Divide", "AutoFE_LeaveOut_Add",
    "AutoFE_LeaveOut_Subtract", "AutoFE_LeaveOut_Multiply", "AutoFE_LeaveOut_Divide")
MODELS = (
    "logistic_regression", "random_forest", "extra_trees", "linear_svm", "knn",
    "gaussian_nb", "mlp", "lightgbm", "xgboost", "catboost")
FIELDS = ("train_matrix_sha256", "test_matrix_sha256", "prediction_sha256",
          "roc_auc", "train_auc", "pr_auc", "log_loss", "brier_score", "accuracy",
          "balanced_accuracy", "f1", "precision", "recall", "mcc", "ks_stat", "wasserstein",
          "operator_candidate_counts", "operator_configuration", "n_original", "n_retained",
          "n_generated", "n_train", "n_test", "model_backend")


def cell(row):
    return tuple(row.get(field) for field in (
        "dataset", "seed", "fold", "condition", "pipeline", "model", "split_policy"))


def rows(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["status"] == "success":
                if cell(row) in result:
                    raise ValueError("Duplicate successful cell")
                result[cell(row)] = row
    return result


def verify_saved_report(report):
    """Read-only revalidation of saved per-run identity and all scientific fields."""
    from src.provenance import code_fingerprint, file_sha256
    fingerprint = code_fingerprint(ROOT)
    evidence = []
    for measurement in report['measurements']:
        run_dir = ROOT / 'corrected_runs' / 'final_optimization' / measurement['run_id']
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        expected_source = report['baseline_code_fingerprint'] if measurement['label'] == 'baseline' else fingerprint
        config = manifest.get('configuration', {})
        large = measurement['label'] == 'optimized_covertype'
        expected_datasets = ['covertype'] if large else ['sonar', 'heart-disease', 'haberman', 'ionosphere']
        expected_pipelines = ['Raw'] if large else list(PIPELINES)
        if (measurement['code_fingerprint'] != expected_source
                or manifest.get('code_fingerprint') != expected_source
                or manifest.get('status') != 'complete'
                or manifest.get('expected_tasks') != measurement['expected']
                or manifest.get('counts_by_status', {}).get('success') != measurement['expected']
                or manifest.get('configuration', {}).get('split_policy') != measurement['policy']
                or manifest.get('configuration', {}).get('numerical_thread_environment') != THREADS
                or manifest.get('configuration', {}).get('workers') != 4):
            raise ValueError(f'Mixed or incomplete saved run evidence: {run_dir}')
        if (set(config.get('datasets', [])) != set(expected_datasets)
                or config.get('pipelines') != expected_pipelines or config.get('models') != list(MODELS)
                or config.get('seeds') != [42] or config.get('folds') != [1]
                or config.get('n_splits') != 5 or config.get('use_gpu') is not False
                or config.get('conditions') != [['clean', 0.0]]
                or config.get('scheduler_lease_seconds') != 600
                or config.get('cache_policy') != 'bounded'
                or config.get('cache_max_bytes') != 8 * 1024**3):
            raise ValueError(f'Saved diagnostic design differs: {run_dir}')
        evidence.append({'run_id': measurement['run_id'],
                         'manifest_sha256': file_sha256(run_dir / 'manifest.json'),
                         'results_sha256': file_sha256(run_dir / 'results.jsonl')})
    for entry in report['small_dataset_parity']:
        policy = entry['policy']
        baseline = next(m for m in report['measurements'] if m['label'] == 'baseline' and m['policy'] == policy)
        optimized = next(m for m in report['measurements'] if m['label'] == 'optimized' and m['policy'] == policy)
        a = rows(ROOT / 'corrected_runs' / 'final_optimization' / baseline['run_id'] / 'results.jsonl')
        b = rows(ROOT / 'corrected_runs' / 'final_optimization' / optimized['run_id'] / 'results.jsonl')
        baseline_config = json.loads((ROOT / 'corrected_runs' / 'final_optimization' / baseline['run_id'] / 'manifest.json').read_text())['configuration']
        optimized_config = json.loads((ROOT / 'corrected_runs' / 'final_optimization' / optimized['run_id'] / 'manifest.json').read_text())['configuration']
        if baseline_config != optimized_config:
            raise ValueError('Before/after scientific or execution configuration differs')
        if len(a) != 560 or set(a) != set(b):
            raise ValueError('Saved small comparison coverage differs')
        mismatch = [{'cell': key, 'field': field} for key in a for field in FIELDS
                    if a[key].get(field) != b[key].get(field)]
        if mismatch:
            raise ValueError(f'Saved scientific parity failed: {mismatch[:5]}')
    for entry in report['large_dataset_parity']:
        policy = entry['policy']
        old_version = '002' if policy == 'row_level' else '003'
        current = rows(ROOT / 'corrected_runs' / 'final_optimization' / entry['run_id'] / 'results.jsonl')
        previous = rows(ROOT / 'corrected_runs' / 'large_calibration' /
                        f'calibration-{policy}-{old_version}' / 'results.jsonl')
        if len(current) != 10:
            raise ValueError('Saved large comparison coverage differs')
        mismatch = [{'cell': key, 'field': field} for key in current for field in FIELDS
                    if current[key].get(field) != previous[key].get(field)]
        if mismatch:
            raise ValueError(f'Saved large scientific parity failed: {mismatch[:5]}')
    if report['code_fingerprint'] != fingerprint:
        raise ValueError('Saved report source differs')
    return {**report, 'scientific_fields': list(FIELDS), 'per_run_evidence': evidence,
            'verification_script_sha256': file_sha256(Path(__file__))}


def _worker(args):
    sys.path.insert(0, str(args.source_root))
    from src.pipeline_runner import run_experiment
    from src.provenance import atomic_write_json
    datasets = ("covertype",) if args.large else (
        "sonar", "heart-disease", "haberman", "ionosphere")
    pipelines = ("Raw",) if args.large else PIPELINES
    peaks = {"process_tree_rss_bytes": 0}
    stop = threading.Event()

    def monitor():
        parent = psutil.Process()
        while not stop.wait(0.1):
            total = 0
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except psutil.Error:
                    continue
            peaks["process_tree_rss_bytes"] = max(peaks["process_tree_rss_bytes"], total)

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        manifest = run_experiment(
            {name: ROOT / "data" / "raw" / f"{name}.csv" for name in datasets},
            output_root=args.output_root, run_id=args.run_id,
            seeds=[42], folds=[1], n_splits=5, conditions=(("clean", 0.0),),
            pipelines=pipelines, models=MODELS, workers=4, use_gpu=False,
            split_policy=args.policy, cache_policy="bounded", cache_max_bytes=8 * 1024**3,
            durable_scheduler=True, cache_audit=True, scheduler_lease_seconds=600.0)
    finally:
        stop.set()
        sampler.join()
    elapsed = time.perf_counter() - started
    expected = len(datasets) * len(pipelines) * len(MODELS)
    if manifest["status"] != "complete" or manifest["counts_by_status"]["success"] != expected:
        raise ValueError("Bounded verification did not finish all cells")
    atomic_write_json(args.output_root / args.run_id / "measurement.json", {
        "elapsed_s": elapsed, "expected": expected, "success": expected, **peaks,
        "cache_high_water": manifest.get("cache_storage", {}).get("high_water"),
        "code_fingerprint": manifest["code_fingerprint"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--version", default="001")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "corrected_runs" / "final_optimization")
    parser.add_argument("--run-id")
    parser.add_argument("--policy", choices=("row_level", "group_aware"))
    parser.add_argument("--large", action="store_true")
    parser.add_argument('--verify-report', type=Path,
                        help='Revalidate saved diagnostic identities and all metrics; does not fit models')
    parser.add_argument('--reuse-small-evidence', action='store_true',
                        help='Use only existing terminal same-design small diagnostic runs')
    parser.add_argument('--large-version', default='002')
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path(r"D:\Conda\p12\python.exe").resolve():
        raise RuntimeError("Use existing p12")
    os.environ.update(THREADS)
    if args.worker:
        _worker(args)
        return
    if args.verify_report:
        sys.path.insert(0, str(ROOT))
        from src.provenance import atomic_write_json
        report = verify_saved_report(json.loads(args.verify_report.read_text()))
        atomic_write_json(args.verify_report, report)
        print(json.dumps({'status': report['status'], 'per_run_evidence': report['per_run_evidence']}))
        return
    if not args.baseline_root or not (args.baseline_root / "src" / "pipeline_runner.py").is_file():
        parser.error("--baseline-root must name the preserved baseline checkout")
    sys.path.insert(0, str(ROOT))
    from src.provenance import atomic_write_json, code_fingerprint, current_git_commit, file_sha256
    measurements, parity, large_parity = [], [], []
    for policy in ("row_level", "group_aware"):
        runs = []
        for label, source in (("baseline", args.baseline_root), ("optimized", ROOT)):
            run_id = f"optimization-{label}-{policy}-{args.version}"
            run_dir = args.output_root / run_id
            reuse = args.reuse_small_evidence and (run_dir / 'measurement.json').is_file()
            if run_dir.exists() and not reuse:
                raise FileExistsError(f"Preserve diagnostic evidence; choose a new version: {run_dir}")
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--source-root", str(source), "--output-root", str(args.output_root),
                       "--run-id", run_id, "--policy", policy]
            args.output_root.mkdir(parents=True, exist_ok=True)
            if not reuse:
                with (args.output_root / f"{run_id}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=source, env=os.environ.copy(), stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            measurement = json.loads((run_dir / "measurement.json").read_text())
            measurements.append({"label": label, "policy": policy, "run_id": run_id, **measurement})
            runs.append(rows(run_dir / "results.jsonl"))
        if set(runs[0]) != set(runs[1]):
            raise ValueError("Before/after task coverage differs")
        mismatches = [{"cell": key, "field": field} for key in runs[0] for field in FIELDS
                      if runs[0][key].get(field) != runs[1][key].get(field)]
        parity.append({"policy": policy, "compared_cells": len(runs[0]), "mismatches": mismatches,
                       "elapsed_ratio_baseline_over_optimized": measurements[-2]["elapsed_s"] / measurements[-1]["elapsed_s"]})
        if mismatches:
            raise ValueError(f"Scientific parity failed: {mismatches[:5]}")
    # The prior 120-cell calibrations remain historical timing evidence. Check
    # the second large dataset on new source without replacing their identity.
    for policy, old_version in (("row_level", "002"), ("group_aware", "003")):
        run_id = f"optimization-covertype-raw-{policy}-{args.large_version}"
        run_dir = args.output_root / run_id
        if run_dir.exists():
            raise FileExistsError(run_dir)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--large",
                   "--output-root", str(args.output_root), "--run-id", run_id, "--policy", policy]
        with (args.output_root / f"{run_id}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        current = rows(run_dir / "results.jsonl")
        previous = rows(ROOT / "corrected_runs" / "large_calibration" /
                        f"calibration-{policy}-{old_version}" / "results.jsonl")
        mismatch = [{"cell": key, "field": field} for key in current for field in FIELDS
                    if current[key].get(field) != previous[key].get(field)]
        large_parity.append({"policy": policy, "run_id": run_id, "compared_cells": len(current),
                             "mismatches": mismatch})
        if mismatch:
            raise ValueError(f"Large dataset scientific parity failed: {mismatch[:5]}")
        measurements.append({"label": "optimized_covertype", "policy": policy, "run_id": run_id,
                             **json.loads((run_dir / "measurement.json").read_text())})
    payload = {
        "artifact_type": "final_optimization_verification_v1", "status": "passed",
        "code_fingerprint": code_fingerprint(ROOT),
        "baseline_commit": current_git_commit(args.baseline_root),
        "baseline_code_fingerprint": code_fingerprint(args.baseline_root),
        "numerical_thread_environment": THREADS, "workers": 4,
        "scientific_fields": list(FIELDS), "small_dataset_parity": parity,
        "large_dataset_parity": large_parity, "measurements": measurements,
        "data_hashes": {name: file_sha256(ROOT / "data" / "raw" / f"{name}.csv") for name in
                        ("sonar", "heart-disease", "haberman", "ionosphere", "covertype")},
        "interrupted_diagnostic": 'optimization-covertype-row_level-001:19/20completed; deliberately interrupted before the expensive AutoFE linear-SVM fit finished; historical fit about50minutes. Both policies use complete Raw ten-model checks; AutoFE all-ten large-data coverage is provided separately by Airlines recovery. Full campaign retains all tasks.',
        "limits": "Bounded one-seed/one-fold clean workload, not a full-grid speedup or statistical result. Baseline ran first; OS cache and host variation may affect timing. RSS sums process resident memory and may double count shared pages. Historical calibration has other conditions/ablations; no timing ratio is inferred for its Covertype subset. Fresh Covertype verification is Raw only; expensive AutoFE linear-SVM pair was not rerun.",
        "scientific_status": "PENDING CORRECTED RUN"}
    payload = verify_saved_report(payload)
    atomic_write_json(ROOT / "provenance" / f"final_optimization_verification_v{int(args.version)}.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
