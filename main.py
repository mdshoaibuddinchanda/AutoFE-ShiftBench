"""Repository entry point forwarding to the benchmark runner CLI."""

from __future__ import annotations

from src.pipeline_runner import main as benchmark_main


def main() -> None:
    """Run the full benchmark runner CLI."""
    benchmark_main()


if __name__ == "__main__":
    main()
