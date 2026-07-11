"""Check benchmark progress by counting human-readable cache files per dataset.

Usage:
    python -m src.check_progress

Cache layout:
    data/cache/{dataset}/{pipeline}_s{seed}_f{fold}_{condition}_train.pkl

Each dataset has 7 pipelines × 5 seeds × 5 folds × 14 conditions = 2,450 train caches.
"""

from pathlib import Path
import yaml


def check_progress():
    cache_root = Path("data/cache")
    config_path = Path("config/dataset_list.yaml")

    if not config_path.exists():
        print("ERROR: config/dataset_list.yaml not found.")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)
    datasets = config.get("datasets", [])

    # Expected counts per dataset
    n_pipelines = 7
    n_seeds = 5
    n_folds = 5
    n_conditions = 14
    expected_per_dataset = n_pipelines * n_seeds * n_folds * n_conditions  # 2,450

    print("=" * 70)
    print("  AutoFE-ShiftBench — Progress Report")
    print("=" * 70)
    print(f"  Expected caches per dataset: {expected_per_dataset}")
    print(f"  Total datasets: {len(datasets)}")
    print(f"  Total expected: {expected_per_dataset * len(datasets):,}")
    print("-" * 70)
    print(f"  {'Dataset':<35} {'Cached':>8} {'Expected':>10} {'Progress':>10}")
    print("-" * 70)

    total_cached = 0
    total_expected = 0

    for ds in datasets:
        ds_dir = cache_root / ds
        if ds_dir.exists():
            # Count _train.pkl files (one per pipeline/unit combo)
            cached = sum(1 for _ in ds_dir.glob("*_train.pkl"))
        else:
            cached = 0

        pct = (cached / expected_per_dataset * 100) if expected_per_dataset > 0 else 0

        # Visual bar
        bar_len = 20
        filled = int(bar_len * cached / expected_per_dataset) if expected_per_dataset > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        status = "✓ DONE" if cached >= expected_per_dataset else f"{pct:5.1f}%"
        print(f"  {ds:<35} {cached:>8} / {expected_per_dataset:<8} {bar} {status}")

        total_cached += cached
        total_expected += expected_per_dataset

    print("-" * 70)
    total_pct = (total_cached / total_expected * 100) if total_expected > 0 else 0
    print(f"  {'TOTAL':<35} {total_cached:>8} / {total_expected:<8}          {total_pct:.1f}%")
    print("=" * 70)

    # Also check results_stream.jsonl
    results_file = Path("reports/tables/results_stream.jsonl")
    if results_file.exists():
        with open(results_file) as f:
            n_results = sum(1 for _ in f)
        size_mb = results_file.stat().st_size / (1024 * 1024)
        print(f"\n  Phase 2 results: {n_results:,} rows ({size_mb:.1f} MB)")
    else:
        print(f"\n  Phase 2 results: Not started yet (no results_stream.jsonl)")

    # Check error logs
    for phase in ["phase1", "phase2"]:
        err_file = Path(f"reports/worker_logs/{phase}_error.log")
        if err_file.exists():
            size_kb = err_file.stat().st_size / 1024
            print(f"  {phase} errors: {size_kb:.0f} KB")

    print()


if __name__ == "__main__":
    check_progress()
