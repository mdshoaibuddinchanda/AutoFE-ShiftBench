import os
import time
from src.coordinator import execute_manifest,final_status
from src.task_manifest import ManifestStore,build_task_records,ExecutionConfig


def sleeping_worker(task,queue):
    time.sleep(30)


def crash_writer(queue,ledger,db,run):
    os._exit(17)


def fast_worker(task,queue):
    store=ManifestStore(task['manifest_db'])
    if task['stage']=='precompute':
        store.commit_result(task['run_id'],task['scientific_task_id'],task['attempt_id'],{'stage':'precompute'})
    else:
        queue.put({**{key:task[key] for key in ('run_id','scientific_task_id','attempt_id','seed','fold','condition','pipeline','model','split_policy')},'dataset':task['dataset'],'status':'success','roc_auc':.5})


def crashing_first_attempt(task,queue):
    if task['stage']=='precompute' and task['attempt_count']==0:os._exit(23)
    fast_worker(task,queue)


def fixture(tmp_path,monkeypatch,config):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'reports/worker_logs').mkdir(parents=True)
    data=tmp_path/'d.csv'
    data.write_text('x,target\n1,a\n2,b\n')
    records=build_task_records(['d'],[42],[1,2],[('clean',0)],['Raw'],['logistic_regression'],data_paths={'d':data})
    store=ManifestStore(tmp_path/'m.db')
    store.create_run('r',{'execution':config.to_dict(),'durability_protocol':'sqlite_result_outbox_v2'},records)
    return store,records


def test_timeouts_during_precompute_are_real(tmp_path,monkeypatch):
    config=ExecutionConfig(max_workers=2,task_timeout_seconds=.5)
    store,records=fixture(tmp_path,monkeypatch,config)
    result=execute_manifest(store,'r',tmp_path/'results.jsonl',config,entry=sleeping_worker)
    assert result['state_counts']['timeout'] == 2
    assert result['state_counts']['skipped'] == 2
    assert result['status'] == 'finished_with_errors'


def test_run_deadline_interrupts_blocking_precompute(tmp_path,monkeypatch):
    config=ExecutionConfig(max_workers=1,run_wall_time_seconds=.5,max_attempts=2)
    store,_=fixture(tmp_path,monkeypatch,config)
    start=time.monotonic()
    result=execute_manifest(store,'r',tmp_path/'results.jsonl',config,entry=sleeping_worker)
    assert time.monotonic()-start < 15
    assert result['status'] == 'stopped'
    assert result['state_counts']['running'] == 0
    assert result['state_counts']['pending'] == 4


def test_stop_budget_reserves_exact_model_launches(tmp_path,monkeypatch):
    config=ExecutionConfig(max_workers=2,stop_after_tasks=1)
    store,_=fixture(tmp_path,monkeypatch,config)
    result=execute_manifest(store,'r',tmp_path/'results.jsonl',config,entry=fast_worker)
    assert result['model_attempts_dispatched_this_invocation'] == 1
    assert result['status'] == 'stopped'
    assert result['state_counts']['completed'] == 3
    assert result['state_counts']['pending'] == 1


def test_writer_failure_fences_active_work(tmp_path,monkeypatch):
    config=ExecutionConfig(max_workers=1,max_attempts=2)
    store,_=fixture(tmp_path,monkeypatch,config)
    result=execute_manifest(store,'r',tmp_path/'results.jsonl',config,entry=sleeping_worker,writer_target=crash_writer)
    assert result['status'] == 'failed'
    assert result['state_counts']['running'] == 0
    assert result['state_counts']['pending'] == 4


def test_pending_work_is_never_successful():
    counts={key:0 for key in ('pending','running','failed','timeout','skipped')}
    counts['pending']=1
    assert final_status(counts) == 'unfinished'


def test_reused_worker_acknowledges_each_owned_attempt(tmp_path,monkeypatch):
    config=ExecutionConfig(max_workers=1,worker_max_tasks=8)
    store,_=fixture(tmp_path,monkeypatch,config)
    result=execute_manifest(store,'r',tmp_path/'results.jsonl',config,entry=fast_worker)
    assert result['status']=='completed_successfully'
    assert result['state_counts']['completed']==4
    assert result['attempt_counts']=={'completed':4}
    assert result['workers_spawned']==1
    assert [row['tasks_executed'] for row in result['worker_resource_samples']]==[1,2,3,4]
    assert len({row['pid'] for row in result['worker_resource_samples']})==1


def test_recycling_and_crashed_retry_preserve_exact_accounting(tmp_path,monkeypatch):
    config=ExecutionConfig(max_workers=1,max_attempts=2,worker_max_tasks=2)
    store,_=fixture(tmp_path,monkeypatch,config)
    result=execute_manifest(store,'r',tmp_path/'results.jsonl',config,entry=crashing_first_attempt)
    assert result['status']=='completed_successfully'
    assert result['state_counts']['completed']==4
    assert result['attempt_counts']=={'completed':4,'failed':2}
    assert result['workers_spawned']>=3
    assert result['workers_recycled']>=1


def test_worker_launch_failure_records_attempt_and_closes_supervision(tmp_path,monkeypatch):
    import multiprocessing.context
    import pytest
    config=ExecutionConfig(max_workers=1,max_attempts=2)
    store,_=fixture(tmp_path,monkeypatch,config)
    original=multiprocessing.context.SpawnProcess.start
    def start(process):
        if process._target.__name__=='worker_service':raise OSError('injected worker launch failure')
        return original(process)
    monkeypatch.setattr(multiprocessing.context.SpawnProcess,'start',start)
    with pytest.raises(OSError):execute_manifest(store,'r',tmp_path/'results.jsonl',config,entry=fast_worker)
    assert store.state_counts('r')['running']==0
    assert store.attempt_counts('r')=={'failed':1}
    assert store.run_config('r')['execution']['max_attempts']==2


def test_manager_startup_failure_records_failed_run_without_claims(tmp_path,monkeypatch):
    import multiprocessing.context
    import pytest
    config=ExecutionConfig(max_workers=1)
    store,_=fixture(tmp_path,monkeypatch,config)
    def fail(*args,**kwargs):raise OSError('injected manager startup failure')
    monkeypatch.setattr(multiprocessing.context.SpawnContext,'Manager',fail)
    with pytest.raises(OSError):execute_manifest(store,'r',tmp_path/'results.jsonl',config,entry=fast_worker)
    assert store.attempt_counts('r')=={}
    assert store.snapshot('r')['run']['status']=='failed'
