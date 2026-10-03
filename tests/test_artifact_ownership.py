import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import pipeline_runner as runner
from src.artifact_integrity import file_sha256
from src.task_manifest import ManifestStore,build_task_records,ExecutionConfig
from src.coordinator import execute_manifest


def simultaneous_creation(directory,queue):
    import os
    os.chdir(directory)
    runner.PIPELINE_CONFIGS={'Raw':runner.PIPELINE_CONFIGS['Raw']}
    original=runner.expand_features_with_dfs
    def counted(*args,**kwargs):
        with Path('creations.txt').open('a') as handle: handle.write('created\n')
        return original(*args,**kwargs)
    runner.expand_features_with_dfs=counted
    x=pd.DataFrame({'x':[1.,2.,3.,4.]})
    _,meta=runner._run_pipeline_generation(x,x,np.array([0,1,0,1]),'d','stratified',42,1,'clean',diagnostics_enabled=True,diagnostic_config={'max_rows':4,'random_state':42})
    queue.put(meta['Raw']['artifacts'])


def test_concurrent_creation_has_one_owner_and_complete_identical_artifacts(tmp_path):
    context=mp.get_context('spawn')
    queue=context.Queue()
    processes=[context.Process(target=simultaneous_creation,args=(str(tmp_path),queue)) for _ in range(2)]
    for process in processes: process.start()
    outcomes=[queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(15)
        assert process.exitcode == 0
        process.close()
    queue.close();queue.join_thread()
    assert (tmp_path/'creations.txt').read_text().splitlines() == ['created']
    assert outcomes[0] == outcomes[1]


def test_missing_diagnostic_repairs_dependency_without_changing_original_bytes(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner,'PIPELINE_CONFIGS',{'Raw':runner.PIPELINE_CONFIGS['Raw']})
    Path('reports/worker_logs').mkdir(parents=True)
    data=Path('d.csv')
    pd.DataFrame({'x':np.arange(60),'target':np.tile(['a','b'],30)}).to_csv(data,index=False)
    records=build_task_records(['d'],[42],[1],[('clean',0)],['Raw'],['logistic_regression'],data_paths={'d':data},pipeline_identity=runner.pipeline_identity_token)
    config=ExecutionConfig(max_workers=1,task_timeout_seconds=60)
    store=ManifestStore('m.db')
    store.create_run('r',{'execution':config.to_dict(),'diagnostics_enabled':True,'diagnostic_max_rows':4,'durability_protocol':'sqlite_result_outbox_v2'},records)
    from src.seeding import stable_seed
    from src.fsva import DEFAULT_PERTURBATION_MAGNITUDES
    task={**records[0],'dataset_name':'d','manifest_db':'m.db','run_id':'r','diagnostics_enabled':True,
        'diagnostic_config':{'max_rows':4,'random_state':stable_seed('fsva_diagnostic',{'dataset':'d','split_policy':'stratified','seed':42,'fold':1,'condition':'clean'}),'magnitudes':list(DEFAULT_PERTURBATION_MAGNITUDES)}}
    assert runner.precompute_unit(task) == 'd'
    before=store.durable_payload('r',records[0]['scientific_task_id'])
    unit=pd.read_pickle(before['prepared_descriptor']['path'])
    item=unit['metadata']['Raw']['artifacts']['fsva']
    original=Path(item['path']).read_bytes()
    Path(item['path']).unlink()
    outcome=execute_manifest(store,'r','results.jsonl',config)
    assert outcome['status'] == 'completed_successfully'
    assert Path(item['path']).read_bytes() == original
    assert store.durable_payload('r',records[0]['scientific_task_id'])['prepared_descriptor'] == before['prepared_descriptor']
    assert len(store.snapshot('r')['superseded_results']) == 1
    assert outcome['attempt_counts'] == {'completed':3}


def test_preflight_stop_does_not_launch_or_falsely_complete(tmp_path):
    store=ManifestStore(tmp_path/'m.db')
    records=build_task_records(['d'],[42],[1],[('clean',0)],['Raw'],['logistic_regression'])
    config=ExecutionConfig()
    store.create_run('r',{'execution':config.to_dict()},records)
    outcome=execute_manifest(store,'r',tmp_path/'results.jsonl',config,stop_requested=lambda:True)
    # No completed units means the main supervision loop also checks the signal.
    assert outcome['status']=='stopped'
    assert outcome['model_attempts_dispatched_this_invocation']==0
    assert store.attempt_counts('r')=={}


def test_hash_verification_is_interruptible(tmp_path):
    path=tmp_path/'large.bin'
    path.write_bytes(b'x'*(3*1024*1024))
    calls=[]
    def stop():
        calls.append(True)
        raise TimeoutError('deadline')
    with pytest.raises(TimeoutError):
        file_sha256(path,check_cancel=stop)
    assert len(calls)==1
