"""Entry point for running AutoFE-ShiftBench pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.pipeline_runner import PipelineConfig, run_pipeline


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load YAML configuration and convert it to PipelineConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a YAML mapping")

    return PipelineConfig(**config_data)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run AutoFE-ShiftBench pipeline")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to a YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """Run pipeline from configuration."""
    args = parse_args()
    pipeline_config = load_config(args.config)
    metrics = run_pipeline(pipeline_config)
    print("Pipeline completed. Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.6f}")


if __name__ == "__main__":
    main()
