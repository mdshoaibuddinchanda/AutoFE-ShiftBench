"""CLI wrapper for read-only reviewer sensitivity analyses."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.reviewer_sensitivity import main


if __name__ == "__main__":
    main()
