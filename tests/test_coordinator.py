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
