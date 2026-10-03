from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import data_loader


def test_random_100k_population_cap_is_exact_and_repeatable(tmp_path, monkeypatch):
    rows = 100_007
    features = pd.DataFrame({'source_row': np.arange(rows)})
    target = pd.Series(np.arange(rows) % 2)
    source = SimpleNamespace(data=features, target=target,
        details={'id': '1169', 'version': '1'}, target_names=['target'])
    monkeypatch.setattr(data_loader, '_fetch_openml_with_fallbacks', lambda *a, **k: (source, '1169'))
    monkeypatch.setattr(data_loader, 'compute_meta_features', lambda *a, **k: {})
    output = data_loader.download_openml_dataset('airlines', tmp_path, data_id=1169)
    actual = data_loader.load_csv_dataset(output)
    expected = features.sample(n=100_000, random_state=42).reset_index(drop=True)
    assert len(actual) == 100_000
    np.testing.assert_array_equal(actual['source_row'], expected['source_row'])
    np.testing.assert_array_equal(actual['target'], expected['source_row'] % 2)
    assert data_loader.download_openml_dataset('airlines', tmp_path, data_id=1169) == output


def test_oversized_manual_csv_is_rejected_without_editing(tmp_path):
    path = tmp_path / 'large.csv'
    pd.DataFrame({'x': np.arange(100_001), 'target': 0}).to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='100000'):
        data_loader.load_csv_dataset(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('cap', [0, -1, 100_001])
def test_acquisition_cannot_raise_hard_cap(tmp_path, cap):
    with pytest.raises(ValueError, match='100000'):
        data_loader.download_openml_dataset('airlines', tmp_path, max_rows=cap)
