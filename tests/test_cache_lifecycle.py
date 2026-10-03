import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest

from src import pipeline_runner as runner
from src.artifact_integrity import file_sha256
from src.cache_lifecycle import (CacheCheckFailed, CacheNotReady, DISPOSABLE_ROLES,
    checked_cache_path, evict_unit, verify_eviction, snapshot_eviction_evidence)
from src.coordinator import execute_manifest
from src.evaluation import METRIC_RANGES, METRIC_SEMANTICS_VERSION
from src.protocol import cache_root
from src.task_manifest import ExecutionConfig, ManifestStore, build_task_records


def fixture(tmp_path, monkeypatch, *, conditions=None, stop_after=None, pipelines=None, models=None):
    monkeypatch.chdir(tmp_path)
    Path('reports/worker_logs').mkdir(parents=True)
    data = Path('d.csv')
    pd.DataFrame({'x': np.linspace(-2, 3, 60), 'z': np.sin(np.arange(60)),
                  'target': np.tile(['a', 'b'], 30)}).to_csv(data, index=False)
    config = ExecutionConfig(max_workers=2, cache_policy='rolling', min_free_bytes=0,
        task_timeout_seconds=60, stop_after_tasks=stop_after)
    records = build_task_records(['d'], [42], [1], conditions or [('clean', 0)],
        pipelines or ['Raw'], models or ['logistic_regression', 'gaussian_nb'], data_paths={'d': data},
        pipeline_identity=runner.pipeline_identity_token)
    store = ManifestStore('m.db')
    store.create_run('r', {'execution': config.to_dict(), 'diagnostics_enabled': True,
        'diagnostic_max_rows': 12, 'code_identity': {'worktree_fingerprint_sha256': 'fixture'},
        'environment_identity': {'fixture': 'P12'}}, records)
    return store, records, config


def prepare_and_commit(store, records, *, leave_pending=False, no_export=False, bad_metric=False, missing_input=False):
    parent = next(record for record in records if record['stage'] == 'precompute')
    task = {**parent, 'dataset_name': 'd', 'data_path': Path('d.csv'),
            'manifest_db': str(store.db_path), 'run_id': 'r', 'diagnostics_enabled': True,
            'diagnostic_config': {'max_rows': 12, 'random_state': 42}}
    runner.precompute_unit(task)
    assert store.get_task('r', parent['scientific_task_id'])['state'] == 'completed'
    prepared = store.durable_payload('r', parent['scientific_task_id'])
    for i, record in enumerate(r for r in records if r['stage'] == 'model'):
        if leave_pending and i == 1: continue
        attempt = store.claim_task('r', record['scientific_task_id'])
        result = {**{key: record[key] for key in ('dataset', 'seed', 'fold', 'condition', 'pipeline', 'model', 'split_policy')},
            'run_id': 'r', 'scientific_task_id': record['scientific_task_id'], 'attempt_id': attempt,
            'dataset_fingerprint': record['data_identity']['fingerprint'], 'status': 'success',
            'metric_status': {name: 'complete' for name in METRIC_RANGES},
            'metric_semantics_version': METRIC_SEMANTICS_VERSION,
            **{name: .5 for name in METRIC_RANGES},
            'artifacts': [item for item in prepared['artifacts'] if item.get('pipeline') in (None, record['pipeline'])
                and (record['model'] not in runner.GPU_MODELS or item['role'] not in ('model_train_numeric_matrix', 'model_test_numeric_matrix'))]}
        if bad_metric: result['accuracy'] = None
        if missing_input: result['artifacts'] = prepared['artifacts'][1:]
        store.commit_result('r', record['scientific_task_id'], attempt, result)
    store.export_durable_results('r', 'results.jsonl')
    if not no_export: store.index_ledger_exports('r', 'results.jsonl')
    return parent['scientific_task_id'], prepared


def matrices(prepared):
    return [Path(item['path']) for item in prepared['artifacts'] if item['role'] in DISPOSABLE_ROLES]


def test_verified_eviction_keeps_evidence_and_is_idempotent(tmp_path, monkeypatch):
    store, records, _ = fixture(tmp_path, monkeypatch)
    unit, prepared = prepare_and_commit(store, records)
    result = evict_unit(store, 'r', unit, 'results.jsonl', cache_root())
    assert result['deleted_files'] == 4
    assert all(not path.exists() for path in matrices(prepared))
    assert all(Path(item['path']).exists() for item in prepared['artifacts'] if item['role'] not in DISPOSABLE_ROLES)
    assert verify_eviction(store, 'r', unit)['result_hashes']
    assert snapshot_eviction_evidence(store.snapshot('r'), tmp_path)
    assert evict_unit(store, 'r', unit, 'results.jsonl', cache_root())['status'] == 'completed'


@pytest.mark.parametrize('failure', ['pending', 'unexported', 'bad_metric', 'live_worker', 'corrupt_matrix', 'corrupt_ledger', 'missing_history', 'missing_input'])
def test_failed_gate_never_deletes_matrices(tmp_path, monkeypatch, failure):
    store, records, _ = fixture(tmp_path, monkeypatch)
    unit, prepared = prepare_and_commit(store, records, leave_pending=failure == 'pending',
        no_export=failure == 'unexported', bad_metric=failure == 'bad_metric', missing_input=failure == 'missing_input')
    kwargs = {}
    if failure == 'live_worker':
        process = psutil.Process()
        kwargs['exited_processes'] = [{'pid': process.pid, 'create_time': process.create_time()}]
    if failure == 'corrupt_matrix': matrices(prepared)[0].write_bytes(b'corrupt')
    if failure == 'corrupt_ledger': Path('results.jsonl').write_bytes(b'{}\n')
    if failure == 'missing_history':
        Path(next(item['path'] for item in prepared['artifacts'] if item['role'] == 'candidate_history')).unlink()
    with pytest.raises((CacheCheckFailed, OSError, ValueError)):
        evict_unit(store, 'r', unit, 'results.jsonl', cache_root(), **kwargs)
    assert all(path.exists() for path in matrices(prepared))
    assert store.cache_eviction('r', unit) is None


@pytest.mark.skipif(os.name != 'nt', reason='Windows exclusive-file/mapping semantics')
def test_windows_open_mapping_blocks_entire_batch_then_retry_succeeds(tmp_path, monkeypatch):
    store, records, _ = fixture(tmp_path, monkeypatch)
    unit, prepared = prepare_and_commit(store, records)
    target = next(path for path in matrices(prepared) if path.suffix == '.npy')
    mapped = np.load(target, mmap_mode='r')
    try:
        with pytest.raises(CacheCheckFailed):
            evict_unit(store, 'r', unit, 'results.jsonl', cache_root())
        assert all(path.exists() for path in matrices(prepared))
        assert store.cache_eviction('r', unit)['state'] == 'intent'
    finally:
        mapped._mmap.close()
    assert evict_unit(store, 'r', unit, 'results.jsonl', cache_root())['status'] == 'completed'


def test_outside_path_and_hardlink_are_rejected(tmp_path, monkeypatch):
    _, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(CacheCheckFailed): checked_cache_path('d.csv', cache_root())
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    original = root / 'a'
    original.write_bytes(b'keep')
    os.link(original, root / 'b')
    with pytest.raises(CacheCheckFailed): checked_cache_path(original, root)


def test_tampered_receipt_is_not_accepted_as_missing_input_evidence(tmp_path, monkeypatch):
    store, records, _ = fixture(tmp_path, monkeypatch)
    unit, prepared = prepare_and_commit(store, records)
    result = evict_unit(store, 'r', unit, 'results.jsonl', cache_root())
    path = Path(result['receipt_path'])
    value = json.loads(path.read_text())
    value['result_hashes'] = {}
    path.write_text(json.dumps(value))
    with pytest.raises(CacheCheckFailed): verify_eviction(store, 'r', unit)
    with pytest.raises(CacheCheckFailed): snapshot_eviction_evidence(store.snapshot('r'), tmp_path)


def test_incompatible_numerical_result_cannot_authorize_cleanup(tmp_path,monkeypatch):
    from src.numerical_validation import NUMERICAL_VALIDATION_VERSION
    store,records,_ = fixture(tmp_path,monkeypatch)
    unit,prepared = prepare_and_commit(store,records)
    # A legacy/unversioned synthetic result must not be relabeled as compatible.
    with store._connect() as connection:
        row = connection.execute("SELECT config_json FROM runs WHERE run_id='r'").fetchone()
        config = {**json.loads(row[0]),'numerical_validation_version':NUMERICAL_VALIDATION_VERSION}
        connection.execute("UPDATE runs SET config_json=? WHERE run_id='r'",(json.dumps(config),))
    with pytest.raises(CacheCheckFailed,match='Numerical validation'):
        evict_unit(store,'r',unit,'results.jsonl',cache_root())
    assert all(path.exists() for path in matrices(prepared))
    assert store.cache_eviction('r',unit) is None


def test_real_rolling_batches_resume_without_regeneration(tmp_path, monkeypatch):
    store, records, config = fixture(tmp_path, monkeypatch,
        conditions=[('clean', 0), ('gaussian_noise', .05)], pipelines=['Raw', 'AutoFE_Baseline'])
    outcome = execute_manifest(store, 'r', 'results.jsonl', config)
    assert outcome['status'] == 'completed_successfully', outcome
    assert outcome['state_counts']['completed'] == 10
    assert len(outcome['cache_evictions']) == 2
    assert len(Path('results.jsonl').read_text().splitlines()) == 8
    for row in store.completed_precompute_results('r'):
        prepared = json.loads(row['payload_json'])
        assert all(not path.exists() for path in matrices(prepared))
        verify_eviction(store, 'r', row['scientific_task_id'])
    from src.provenance_evidence import actual_lineage
    graph = actual_lineage(store.snapshot('r'), [], tmp_path)
    assert graph['status'] == 'valid', graph['unverified_dependencies']
    assert any(edge['relationship'] == 'verified_input_before_eviction' for edge in graph['edges'])
    resumed = execute_manifest(store, 'r', 'results.jsonl', config)
    assert resumed['status'] == 'completed_successfully'
    assert resumed['model_attempts_dispatched_this_invocation'] == 0
    assert store.attempt_counts('r') == {'completed': 10}


def test_stop_mid_batch_retains_inputs(tmp_path, monkeypatch):
    store, records, config = fixture(tmp_path, monkeypatch, stop_after=1)
    outcome = execute_manifest(store, 'r', 'results.jsonl', config)
    assert outcome['status'] == 'stopped'
    assert outcome['model_attempts_dispatched_this_invocation'] == 1
    assert not outcome['cache_evictions']
    prepared = json.loads(store.completed_precompute_results('r')[0]['payload_json'])
    assert all(path.exists() for path in matrices(prepared))


def test_capacity_guard_stops_before_preparation(tmp_path, monkeypatch):
    store, records, config = fixture(tmp_path, monkeypatch)
    from dataclasses import replace
    config = replace(config, min_free_bytes=10**18)
    outcome = execute_manifest(store, 'r', 'results.jsonl', config)
    assert outcome['status'] == 'stopped'
    assert outcome['cache_stop_reason']
    assert store.attempt_counts('r') == {}


def test_new_run_exactly_rebuilds_evicted_inputs(tmp_path, monkeypatch):
    store, records, config = fixture(tmp_path, monkeypatch)
    assert execute_manifest(store, 'r', 'results.jsonl', config)['status'] == 'completed_successfully'
    parent = next(row for row in records if row['stage'] == 'precompute')
    original = store.durable_payload('r', parent['scientific_task_id'])['prepared_descriptor']
    store.create_run('r2', store.run_config('r'), records)
    second = execute_manifest(store, 'r2', 'results.jsonl', config)
    assert second['status'] == 'completed_successfully', second
    assert store.durable_payload('r2', parent['scientific_task_id'])['prepared_descriptor'] == original
    assert second['model_attempts_dispatched_this_invocation'] == 2
    assert len(Path('results.jsonl').read_text().splitlines()) == 4


def test_crash_after_deletion_recovers_intent_without_training(tmp_path, monkeypatch):
    store, records, config = fixture(tmp_path, monkeypatch)
    unit, prepared = prepare_and_commit(store, records)
    record = store.record_cache_eviction
    def interrupt(run, unit_id, state, receipt, path):
        if state == 'completed': raise OSError('injected crash before completion publication')
        return record(run, unit_id, state, receipt, path)
    monkeypatch.setattr(store, 'record_cache_eviction', interrupt)
    with pytest.raises(OSError): evict_unit(store, 'r', unit, 'results.jsonl', cache_root())
    assert store.cache_eviction('r', unit)['state'] == 'intent'
    assert all(not path.exists() for path in matrices(prepared))
    monkeypatch.setattr(store, 'record_cache_eviction', record)
    resumed = execute_manifest(store, 'r', 'results.jsonl', config)
    assert resumed['status'] == 'completed_successfully', resumed
    assert resumed['model_attempts_dispatched_this_invocation'] == 0
    assert store.cache_eviction('r', unit)['state'] == 'completed'


@pytest.mark.skipif(os.name != 'nt', reason='Windows file sharing semantics')
def test_windows_open_regular_file_blocks_batch_without_deleting_other_files(tmp_path, monkeypatch):
    store, records, _ = fixture(tmp_path, monkeypatch)
    unit, prepared = prepare_and_commit(store, records)
    with matrices(prepared)[-1].open('rb'):
        with pytest.raises(CacheCheckFailed):
            evict_unit(store, 'r', unit, 'results.jsonl', cache_root())
        assert all(path.exists() for path in matrices(prepared))
    assert evict_unit(store, 'r', unit, 'results.jsonl', cache_root())['status'] == 'completed'


def test_second_coordinator_cannot_claim_tasks_while_cache_is_in_use(tmp_path, monkeypatch):
    store, _, config = fixture(tmp_path, monkeypatch)
    from src.artifact_integrity import artifact_lock
    with artifact_lock(cache_root() / '.coordinator.lock'):
        with pytest.raises(TimeoutError): execute_manifest(store, 'r', 'results.jsonl', config)
    assert store.attempt_counts('r') == {}


def test_gpu_input_evidence_uses_selected_pickles_without_numeric_arrays(tmp_path, monkeypatch):
    # Verify GPU artifact-role gating without requiring GPU hardware or a fit.
    store, records, _ = fixture(tmp_path, monkeypatch, models=['catboost'])
    unit, prepared = prepare_and_commit(store, records)
    assert evict_unit(store, 'r', unit, 'results.jsonl', cache_root())['status'] == 'completed'
    assert all(not path.exists() for path in matrices(prepared))


def test_crashed_coordinator_live_worker_registration_blocks_resume(tmp_path, monkeypatch):
    store, _, config = fixture(tmp_path, monkeypatch)
    from src.cache_lifecycle import register_worker
    register_worker(cache_root(), psutil.Process(), run_id='previous_run', unit_id='previous_unit')
    with pytest.raises(CacheCheckFailed, match='still exists'):
        execute_manifest(store, 'r', 'results.jsonl', config)
    assert store.attempt_counts('r') == {}
