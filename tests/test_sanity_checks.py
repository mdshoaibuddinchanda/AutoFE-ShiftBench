"""Deterministic replacement for the former exploratory sanity-check notebook."""

from __future__ import annotations

import pandas as pd


def test_csv_sanity_check_reports_shape_and_missing_values(tmp_path):
    """A small CSV can be inspected before it is passed to the benchmark."""
    path = tmp_path / "sample.csv"
    expected = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, None],
            "feature_b": ["x", "y", "x"],
            "target": [0, 1, 0],
        }
    )
    expected.to_csv(path, index=False)

    frame = pd.read_csv(path)

    assert frame.shape == (3, 3)
    assert list(frame.columns) == ["feature_a", "feature_b", "target"]
    assert frame.isna().sum().to_dict() == {
        "feature_a": 1,
        "feature_b": 0,
        "target": 0,
    }
