"""Repository entry point: download datasets and run the full benchmark.

Usage:
    python main.py                    # Full benchmark (all 25 datasets)
    python main.py --max-datasets 2   # Smoke test on 2 datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--max-datasets", type=int)
    args, runner_args = parser.parse_known_args()

    from src.data_loader import download_datasets_from_list, load_dataset_names

    # Step 1: Download all datasets
    print("\n" + "=" * 60)
    print("  Step 1/2: Downloading configured benchmark datasets")
    print("=" * 60 + "\n")

    raw_dir = Path(args.data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Honor the same dataset cap as the runner when deciding what to download.
    all_datasets = load_dataset_names("config/dataset_list.yaml")
    if args.max_datasets is not None:
        if args.max_datasets <= 0:
            parser.error("--max-datasets must be greater than zero")
        all_datasets = all_datasets[:args.max_datasets]

    missing = [d for d in all_datasets if not (raw_dir / f"{d}.csv").exists()]
    if missing:
        print(f"  Downloading {len(missing)} missing datasets: {missing}")
        download_datasets_from_list(output_dir=raw_dir, dataset_names=missing)
    else:
        print(f"  All {len(all_datasets)} datasets already downloaded. Skipping.")

    # Step 2: Run the benchmark
    print("\n" + "=" * 60)
    print("  Step 2/2: Running benchmark")
    print("=" * 60 + "\n")

    from src.pipeline_runner import main as benchmark_main
    forwarded = ["--data-dir", str(raw_dir)]
    if args.max_datasets is not None:
        forwarded.extend(["--max-datasets", str(args.max_datasets)])
    benchmark_main([*forwarded, *runner_args])


if __name__ == "__main__":
    main()
