"""Check benchmark progress by counting human-readable cache files per dataset.

Usage:
    python -m src.check_progress

Cache layout:
    data/cache/{protocol}/{seed_scheme}/{dataset}/{pipeline}_s{seed}_f{fold}_{condition}_train.pkl

Each dataset has 14 configured pipelines × 5 seeds × 5 folds × 14 conditions = 4,900 train caches.
"""

from pathlib import Path
import yaml

from src.protocol import cache_root, results_ledger_path
from src.task_manifest import ManifestStore


def check_progress():
    current_cache_root = cache_root()
    config_path = Path("config/dataset_list.yaml")

    if not config_path.exists():
        print("ERROR: config/dataset_list.yaml not found.")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)
    datasets = config.get("datasets", [])

    manifest_dir = Path("reports/manifests")
    manifest_dbs = sorted(manifest_dir.glob("*.db"), key=lambda path: path.stat().st_mtime, reverse=True) if manifest_dir.exists() else []
    if manifest_dbs:
        manifest_db = manifest_dbs[0]
        store = ManifestStore(manifest_db,read_only=True)
        with store._connect() as connection:
            run_row = connection.execute("SELECT run_id, status, updated_at FROM runs ORDER BY updated_at DESC LIMIT 1").fetchone()
        if run_row is not None:
            run_id = run_row["run_id"]
            print(f"  Active manifest: {manifest_db} (run {run_id}, {run_row['status']})")
            print(f"  Task states: {store.state_counts(run_id)}")
            print(f"  Attempt states: {store.attempt_counts(run_id)}")
            print("  Cache and diagnostic artifact counts below are separate from completed model-task counts.")

    # Expected counts per dataset
    n_pipelines = 14
    n_seeds = 5
    n_folds = 5
    n_conditions = 14
    expected_per_dataset = n_pipelines * n_seeds * n_folds * n_conditions  # 4,900 with current 14-pipeline configuration

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
        ds_dir = current_cache_root / ds
        if ds_dir.exists():
            # Count _train.pkl files (one per pipeline/unit combo)
            cached = sum(1 for _ in ds_dir.rglob("*_train.pkl"))
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

    # Also check the active, versioned result ledger.
    results_file = results_ledger_path()
    if results_file.exists():
        with open(results_file) as f:
            n_results = sum(1 for _ in f)
        size_mb = results_file.stat().st_size / (1024 * 1024)
        print(f"\n  Phase 2 results: {n_results:,} rows ({size_mb:.1f} MB)")
    else:
        print(f"\n  Phase 2 results: Not started yet (no {results_file.name})")

    # Check error logs
    for phase in ["phase1", "phase2"]:
        err_file = Path(f"reports/worker_logs/{phase}_error.log")
        if err_file.exists():
            size_kb = err_file.stat().st_size / 1024
            print(f"  {phase} errors: {size_kb:.0f} KB")

    print()


if __name__ == "__main__":
    check_progress()
