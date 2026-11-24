"""Run with the runner's selection and stopping flags.

Acquire explicitly selected data separately using
``python -m src.data_loader --datasets haberman``.
"""

from __future__ import annotations

def main() -> None:
    from src.pipeline_runner import main as benchmark_main
    benchmark_main()


if __name__ == "__main__":
    main()
