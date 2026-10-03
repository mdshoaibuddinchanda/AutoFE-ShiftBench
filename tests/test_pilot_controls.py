from pathlib import Path
from dataclasses import replace
import os
import time
import pytest

from src.artifact_integrity import atomic_bytes
from src.protocol import cache_root, EVALUATION_PROTOCOL_VERSION
from src.resource_limits import ResourceLimitError
from src.task_manifest import ExecutionConfig, build_task_records, ManifestStore
from src.coordinator import execute_manifest
from tests.test_coordinator import fixture, fast_worker


def test_cache_isolation_preserves_protocol(monkeypatch,tmp_path):
    monkeypatch.setenv('AUTOFE_CACHE_BASE',str(tmp_path/'pilot'))
    assert cache_root().is_relative_to(tmp_path/'pilot')
    assert EVALUATION_PROTOCOL_VERSION in cache_root().parts


def test_exact_pilot_selection_and_full_accounting():
    from reproduction.real_data_preflight import declared_records
    with pytest.raises(ValueError,match='Unknown explicitly'):
        declared_records(['known'],units=[('unknown',42,1,'clean',0)])


def test_planned_scientific_skip_is_no_attempt_and_no_cleanup_failure(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    records=build_task_records(['d'],[42],[1],[('clean',0)],['Raw'],['knn'],
        planned_infeasible={('d',42,1,'clean'):'scientific_infeasibility:rare_class'})
    assert len(records)==2 and all(r['initial_state']=='skipped' for r in records)
    assert records[1]['depends_on']==[records[0]['scientific_task_id']]
    store=ManifestStore('m.db'); config=ExecutionConfig(cache_policy='rolling',min_free_bytes=0)
    store.create_run('r',{'execution':config.to_dict()},records)
    result=execute_manifest(store,'r','results.jsonl',config)
    assert result['status']=='completed_successfully'
    assert result['planned_skipped_tasks']==2 and result['attempt_counts']=={}
    assert result['cache_stop_reason'] is None and not result['cache_evictions']
    from src.sensitivity_analysis import _coverage_rows, SensitivityConfig
    rows,_=_coverage_rows(store.snapshot('r'),{},SensitivityConfig())
    assert rows[0]['eligibility_status']=='planned_infeasible' and not rows[0]['eligible']


def slow_success_worker(task,queue):
    time.sleep(.25)
    fast_worker(task,queue)


def test_graceful_stop_drains_owned_attempt_then_resume(tmp_path,monkeypatch):
    config=ExecutionConfig(max_workers=1,task_timeout_seconds=30,min_free_bytes=0)
    store,_=fixture(tmp_path,monkeypatch,config)
    def stop(): return store.state_counts('r')['running']>0
    result=execute_manifest(store,'r','results.jsonl',config,stop_requested=stop,entry=slow_success_worker)
    assert result['graceful_stop_drained'] and result['status']=='stopped'
    assert result['state_counts']['completed']==1 and result['state_counts']['pending']==3
    assert store.attempt_counts('r')=={'completed':1}
    assert execute_manifest(store,'r','results.jsonl',config,entry=fast_worker)['status']=='completed_successfully'
    assert execute_manifest(store,'r','results.jsonl',config,entry=fast_worker)['model_attempts_dispatched_this_invocation']==0


def test_atomic_write_reserve_checks_payload_before_any_temp(tmp_path,monkeypatch):
    import shutil
    monkeypatch.setenv('AUTOFE_MIN_FREE_BYTES','100')
    monkeypatch.setenv('AUTOFE_DISK_WRITE_LOCK',str(tmp_path/'write.lock'))
    monkeypatch.setattr(shutil,'disk_usage',lambda p: type('Disk',(),{'free':110})())
    target=tmp_path/'old'; target.write_bytes(b'preserved')
    with pytest.raises(ResourceLimitError,match='would breach'):
        atomic_bytes(target,b'x'*11)
    assert target.read_bytes()==b'preserved' and not list(tmp_path.glob('.publish_*'))
    atomic_bytes(target,b'x'*10)
    assert target.read_bytes()==b'x'*10


@pytest.mark.parametrize('reason',['resource_limit','unsupported_gpu'])
def test_device_resource_failures_keep_their_class(tmp_path,reason):
    store=ManifestStore(tmp_path/'m.db'); records=build_task_records(['d'],[42],[1],[('clean',0)],['Raw'],['knn'])
    store.create_run('r',{'execution':ExecutionConfig().to_dict()},records)
    task=records[0]['scientific_task_id']; attempt=store.claim_task('r',task)
    store.record_failure('r',task,attempt,failure_class=reason,retry=True)
    assert store.get_task('r',task)['outcome_reason']==reason
    assert store.get_task('r',task)['state']=='failed'


def test_dense_budget_validation():
    assert replace(ExecutionConfig(),dense_budget_bytes=4*1024**3).dense_budget_bytes==4*1024**3
    with pytest.raises(ValueError): ExecutionConfig(dense_budget_bytes=0)
