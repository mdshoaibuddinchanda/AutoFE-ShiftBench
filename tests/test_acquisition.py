from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from src import data_loader as loader


def fixture():
    return SimpleNamespace(data=pd.DataFrame({'x':np.arange(30)}),target=pd.Series([0]*29+[1]),details={'id':'999','version':'1','md5_checksum':'mock'},target_names=['class'])


def test_acquisition_records_source_and_exact_capped_population(tmp_path,monkeypatch):
    monkeypatch.setattr(loader,'_fetch_openml_with_fallbacks',lambda *a,**k:(fixture(),'999'))
    monkeypatch.setattr(loader,'compute_meta_features',lambda *a,**k:{})
    path=loader.download_openml_dataset('mock',tmp_path,max_rows=12,random_state=42)
    expected=pd.DataFrame({'x':np.arange(30,dtype=np.int64),'target':[0]*29+[1]}).sample(n=12,random_state=42).reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_csv(path),expected)
    meta=json.loads(path.with_name('mock_meta.json').read_text())
    assert meta['source_provenance']['data_id'] == '999'
    assert meta['source_provenance']['version'] == '1'
    assert meta['split_feasibility']['status'] == 'infeasible_capped_population'
    before=path.read_bytes()
    monkeypatch.setattr(loader,'_fetch_openml_with_fallbacks',lambda *a,**k:pytest.fail('existing file reacquired'))
    assert loader.download_openml_dataset('mock',tmp_path,max_rows=12) == path
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        loader.download_openml_dataset('mock',tmp_path,max_rows=12,data_id=1000)


def test_selected_requests_only_and_failures_propagate(tmp_path,monkeypatch):
    config=tmp_path/'datasets.yaml'
    config.write_text('datasets: [a,b,c]\n')
    seen=[]
    def acquire(dataset_name,**kwargs):
        seen.append(dataset_name)
        raise RuntimeError('selected unavailable')
    monkeypatch.setattr(loader,'download_openml_dataset',acquire)
    with pytest.raises(ValueError):
        loader.download_datasets_from_list(config,tmp_path)
    with pytest.raises(RuntimeError):
        loader.download_datasets_from_list(config,tmp_path,selected_datasets=['b','b'])
    assert seen == ['b']


def test_integer_openml_request_not_retried_with_irrelevant_versions(monkeypatch):
    import sklearn.datasets
    seen=[]
    def fetch(**kwargs):
        seen.append(kwargs)
        raise RuntimeError('offline')
    monkeypatch.setattr(sklearn.datasets,'fetch_openml',fetch)
    with pytest.raises(RuntimeError):
        loader._fetch_openml_with_fallbacks('mock',data_id=999)
    assert len(seen) == 1
def test_repository_entry_point_delegates_without_acquisition(monkeypatch):
    import main
    from src import pipeline_runner,data_loader
    calls=[]
    monkeypatch.setattr(pipeline_runner,'main',lambda:calls.append('runner'))
    monkeypatch.setattr(data_loader,'download_datasets_from_list',lambda *a,**k:(_ for _ in ()).throw(AssertionError('implicit acquisition')))
    main.main()
    assert calls == ['runner']


def test_existing_ambiguous_metadata_is_preserved_and_rejected(tmp_path,monkeypatch):
    path=tmp_path/'mock.csv';path.write_bytes(b'x,target\n1,a\n2,b\n')
    (tmp_path/'mock_meta.json').write_text('{}')
    monkeypatch.setattr(loader,'_fetch_openml_with_fallbacks',lambda *a,**k:pytest.fail('existing bytes overwritten'))
    original=path.read_bytes()
    with pytest.raises(ValueError,match='incomplete source'):loader.download_openml_dataset('mock',tmp_path)
    assert path.read_bytes()==original


def test_existing_population_policy_must_match_requested_seed(tmp_path,monkeypatch):
    monkeypatch.setattr(loader,'_fetch_openml_with_fallbacks',lambda *a,**k:(fixture(),'999'))
    monkeypatch.setattr(loader,'compute_meta_features',lambda *a,**k:{})
    path=loader.download_openml_dataset('mock',tmp_path,max_rows=12,random_state=42)
    original=path.read_bytes()
    with pytest.raises(ValueError,match='row policy'):loader.download_openml_dataset('mock',tmp_path,max_rows=12,random_state=43)
    assert path.read_bytes()==original


def test_conflicting_duplicate_requests_rejected_before_acquisition(tmp_path,monkeypatch):
    config=tmp_path/'datasets.yaml';config.write_text('datasets:\n - {name: a, data_id: 1}\n - {name: a, data_id: 2}\n')
    monkeypatch.setattr(loader,'download_openml_dataset',lambda *a,**k:pytest.fail('ambiguous request fetched'))
    with pytest.raises(ValueError,match='Conflicting exact'):loader.download_datasets_from_list(config,tmp_path,selected_datasets=['a'])
