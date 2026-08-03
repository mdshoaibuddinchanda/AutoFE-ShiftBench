"""Repository entry point: download datasets and run the full benchmark.

Usage:
    python main.py                    # Full benchmark (all 25 datasets)
    python main.py --max-datasets 2   # Smoke test on 2 datasets
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    from src.data_loader import download_datasets_from_list

    # Step 1: Download all datasets
    print("\n" + "=" * 60)
    print("  Step 1/2: Downloading datasets from OpenML")
    print("=" * 60 + "\n")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Only download datasets that are missing
    import yaml
    with open("config/dataset_list.yaml") as f:
        config = yaml.safe_load(f)
    all_datasets = config.get("datasets", [])

    missing = [d for d in all_datasets if not (raw_dir / f"{d}.csv").exists()]
    if missing:
        print(f"  Downloading {len(missing)} missing datasets: {missing}")
        download_datasets_from_list()
    else:
        print(f"  All {len(all_datasets)} datasets already downloaded. Skipping.")

    # Step 2: Run the benchmark
    print("\n" + "=" * 60)
    print("  Step 2/2: Running benchmark")
    print("=" * 60 + "\n")

    from src.pipeline_runner import main as benchmark_main
    benchmark_main()


if __name__ == "__main__":
    main()
