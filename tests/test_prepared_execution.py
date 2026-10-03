import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import pipeline_runner as runner
from src.task_manifest import ManifestStore,build_task_records,ExecutionConfig
from src.coordinator import execute_manifest


def test_real_manifest_workers_use_complete_prepared_inputs_and_retain_evidence(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('reports/worker_logs').mkdir(parents=True)
    data=Path('d.csv')
    pd.DataFrame({'x':np.linspace(-2,3,60),'z':np.sin(np.arange(60)),'target':np.tile(['a','b'],30)}).to_csv(data,index=False)
    records=build_task_records(['d'],[42],[1],[('clean',0),('gaussian_noise',.05)],['Raw','AutoFE_Baseline'],['logistic_regression','gaussian_nb'],data_paths={'d':data},pipeline_identity=runner.pipeline_identity_token)
    config=ExecutionConfig(max_workers=2,task_timeout_seconds=60)
    store=ManifestStore('m.db')
    store.create_run('r',{'execution':config.to_dict(),'diagnostics_enabled':True,'diagnostic_max_rows':12,'durability_protocol':'sqlite_result_outbox_v2',
        'code_identity':{'worktree_fingerprint_sha256':'controlled_source'},'environment_identity':{'fixture':'P12'}},records)
    result=execute_manifest(store,'r','results.jsonl',config)
    assert result['status'] == 'completed_successfully'
    assert result['state_counts']['intended'] == 10
    assert result['state_counts']['completed'] == 10
    assert result['attempt_counts'] == {'completed':10}
    rows=[json.loads(line) for line in Path('results.jsonl').read_text().splitlines()]
    assert len(rows) == 8
    assert len({row['scientific_task_id'] for row in rows}) == 8
    for row in rows:
        assert row['diagnostic_status'] == 'diagnostic_complete'
        assert row['held_out_distribution_distance']['wasserstein'] == 0
        assert all(Path(item['path']).exists() for item in row['artifacts'])
        assert any(item['role']=='split_indices' for item in row['artifacts'])
        if row['pipeline']=='AutoFE_Baseline': assert row['n_generated'] > 0
    # A completed resume dispatches no new attempt; lost export reconstructs from SQLite.
    original=Path('results.jsonl').read_bytes()
    Path('results.jsonl').write_bytes(b'')
    resumed=execute_manifest(store,'r','results.jsonl',config)
    assert resumed['model_attempts_dispatched_this_invocation'] == 0
    assert Path('results.jsonl').read_bytes() != b''
    assert {json.dumps(row,sort_keys=True) for row in rows} == {json.dumps(json.loads(line),sort_keys=True) for line in Path('results.jsonl').read_text().splitlines()}


def test_diagnostic_cache_hit_never_recomputes_shared_diagnostics(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner,'PIPELINE_CONFIGS',{'Raw':runner.PIPELINE_CONFIGS['Raw']})
    x=pd.DataFrame({'x':[1.,2.,3.,4.]})
    kwargs={'diagnostics_enabled':True,'diagnostic_config':{'max_rows':4,'random_state':42}}
    first=runner._run_pipeline_generation(x,x,np.array([0,1,0,1]),'d','stratified',42,1,'clean',**kwargs)
    monkeypatch.setattr(runner,'compute_jacobian_diagnostic',lambda *a,**k:pytest.fail('shared diagnostic recomputed'))
    second=runner._run_pipeline_generation(x,x,np.array([0,1,0,1]),'d','stratified',42,1,'clean',**kwargs)
    assert second[1]['Raw']['dfs_cache_hit']
    assert first[1]['Raw']['artifacts'] == second[1]['Raw']['artifacts']
