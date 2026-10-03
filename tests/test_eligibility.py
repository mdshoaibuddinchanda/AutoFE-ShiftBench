import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.artifact_integrity import atomic_json, file_sha256
from src.eligibility import SCHEMA, SPLIT_SOURCES, load_verified_eligibility, unit_identity, split_dependency_versions
from src.protocol import EVALUATION_PROTOCOL_VERSION, cache_root
from src.seeding import SEED_SCHEME_VERSION
from src.task_manifest import ExecutionConfig, ManifestStore, build_task_records
from src.coordinator import execute_manifest, task_entry


def bundle_fixture(tmp_path):
    identities = {name: {'fingerprint': name, 'bytes_sha256': name} for name in 'abc'}
    for source in SPLIT_SOURCES:
        path = tmp_path/source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (tmp_path/'data/raw').mkdir(parents=True)
    for name in identities:
        (tmp_path/'data/raw'/f'{name}_meta.json').write_text('{}')
    config = {'datasets':list(identities),'seeds':[42],'folds':[1],
              'conditions':[['clean',0.]],'pipelines':['Raw'],'models':['logistic_regression']}
    units = [{'dataset':n,'seed':42,'fold':1,'condition':'clean','shift_family':'clean',
              'severity':0.,'policy':'stratified','unit_identity':unit_identity(n,42,1,'clean',0.,identities[n]),
              'status':'scientifically_infeasible' if n == 'b' else 'split_eligible',
              'reason':'SplitInfeasibleError: fixture class count' if n == 'b' else None} for n in identities]
    value = {'schema':SCHEMA,'protocol_version':EVALUATION_PROTOCOL_VERSION,
        'seed_scheme_version':SEED_SCHEME_VERSION,'n_splits':5,'configuration':config,
        'split_dependency_versions':split_dependency_versions(),
        'dataset_identities':identities,'metadata_sha256':{n:file_sha256(tmp_path/'data/raw'/f'{n}_meta.json') for n in identities},
        'split_source_sha256':{p:file_sha256(tmp_path/p) for p in SPLIT_SOURCES},'units':units}
    return identities,value


def load_bundle(tmp_path, identities, value, **overrides):
    path = tmp_path/'eligibility.json'
    atomic_json(path,value)
    args = dict(datasets=list(identities),data_identities=identities,seeds=[42],folds=[1],
                conditions=[('clean',0.)],repo_root=tmp_path)
    args.update(overrides)
    return load_verified_eligibility(path,file_sha256(path),**args)


def test_verified_mapping_accounts_exact_units(tmp_path):
    identities,value = bundle_fixture(tmp_path)
    planned,summary = load_bundle(tmp_path,identities,value)
    assert planned == {('b',42,1,'clean'):'scientific_infeasibility:SplitInfeasibleError: fixture class count'}
    assert summary['units'] == 3 and summary['split_eligible'] == 2 and summary['planned_infeasible'] == 1


@pytest.mark.parametrize('change', ['missing','duplicate','identity','policy','protocol','seed',
                                  'fold','reason','status','source','metadata','population','condition','environment'])
def test_stale_or_incompatible_mapping_rejected(tmp_path, change):
    identities,value = bundle_fixture(tmp_path)
    if change == 'missing': value['units'].pop()
    elif change == 'duplicate': value['units'].append(copy.deepcopy(value['units'][0]))
    elif change == 'identity': value['units'][0]['unit_identity'] = 'stale'
    elif change == 'policy': value['units'][0]['policy'] = 'random'
    elif change == 'protocol': value['protocol_version'] = 'old'
    elif change == 'seed': value['seed_scheme_version'] = 'old'
    elif change == 'fold': value['n_splits'] = 3
    elif change == 'reason': value['units'][1]['reason'] = 'MetricInputError: bad distribution'
    elif change == 'status': value['units'][0]['status'] = 'failed'
    elif change == 'source': (tmp_path/SPLIT_SOURCES[0]).write_text('changed')
    elif change == 'metadata': (tmp_path/'data/raw/a_meta.json').write_text('{"different":true}')
    elif change == 'population': identities = {**identities,'a':{'fingerprint':'different'}}
    elif change == 'environment': value['split_dependency_versions']['scikit-learn'] = 'different'
    overrides = {'conditions':[('missingness',.1)]} if change == 'condition' else {}
    with pytest.raises(ValueError): load_bundle(tmp_path,identities,value,**overrides)


def test_pinned_bytes_reject_tampering(tmp_path):
    identities,value = bundle_fixture(tmp_path)
    path = tmp_path/'eligibility.json'; atomic_json(path,value); digest = file_sha256(path)
    with path.open('a') as handle: handle.write(' ')
    with pytest.raises(ValueError,match='tampered'):
        load_verified_eligibility(path,digest,datasets=list(identities),data_identities=identities,
            seeds=[42],folds=[1],conditions=[('clean',0.)],repo_root=tmp_path)


def test_ordinary_cli_persists_planned_membership_before_dispatch(tmp_path,monkeypatch):
    import sys
    from src import pipeline_runner as runner
    from src.artifact_integrity import dataset_identity
    monkeypatch.chdir(tmp_path)
    _,value = bundle_fixture(tmp_path)
    identities = {}
    for name in 'abc':
        path = tmp_path/'data/raw'/f'{name}.csv'
        pd.DataFrame({'x':np.arange(20),'target':np.tile(['a','b'],10)}).to_csv(path,index=False)
        identities[name] = dataset_identity(path)
    value['dataset_identities'] = identities
    for row in value['units']:
        row['unit_identity'] = unit_identity(row['dataset'],42,1,'clean',0.,identities[row['dataset']])
    path = tmp_path/'eligibility.json';atomic_json(path,value)
    monkeypatch.setattr(runner,'load_dataset_names',lambda _:list('abc'))
    monkeypatch.setattr(sys,'argv',['runner','--dry-run-manifest','--max-seeds','1','--max-folds','1',
        '--max-conditions','1','--pipelines','Raw','--models','logistic_regression','--run-id','fixture',
        '--eligibility',str(path),'--eligibility-sha256',file_sha256(path),
        '--manifest-db',str(tmp_path/'manifest.db'),'--manifest-path',str(tmp_path/'manifest.jsonl')])
    runner.main()
    store = ManifestStore(tmp_path/'manifest.db',read_only=True)
    assert store.state_counts('fixture') == {'intended':6,'pending':4,'running':0,'completed':0,
                                           'failed':0,'timeout':0,'skipped':2}
    assert store.attempt_counts('fixture') == {}
    assert store.run_config('fixture')['eligibility']['planned_infeasible'] == 1


def unexpected_entry(task, queue):
    if task['dataset'] == 'c' and task['stage'] == 'model':
        raise RuntimeError('declared unexpected validation failure fixture')
    return task_entry(task,queue)


@pytest.mark.parametrize('unexpected', [False,True])
def test_rolling_planned_skip_advances_but_failure_retains_on_resume(tmp_path,monkeypatch,unexpected):
    monkeypatch.chdir(tmp_path)
    Path('reports/worker_logs').mkdir(parents=True)
    paths = {}
    for name in 'abcd':
        paths[name] = Path(name+'.csv')
        pd.DataFrame({'x':np.linspace(-2,3,60),'z':np.sin(np.arange(60)),
                      'target':np.tile(['a','b'],30)}).to_csv(paths[name],index=False)
    records = build_task_records(list(paths),[42],[1],[('clean',0)],['Raw'],['logistic_regression'],
        data_paths=paths, planned_infeasible={('b',42,1,'clean'):'scientific_infeasibility:SplitInfeasibleError: fixture'})
    config = ExecutionConfig(max_workers=1,cache_policy='rolling',min_free_bytes=0,task_timeout_seconds=60)
    store = ManifestStore('manifest.db');store.create_run('r',{'execution':config.to_dict()},records)
    outcome = execute_manifest(store,'r','ledger.jsonl',config,entry=unexpected_entry if unexpected else task_entry)
    snapshot = store.snapshot('r')
    b_tasks = [t for t in snapshot['tasks'] if json.loads(t['payload_json'])['dataset'] == 'b']
    assert len(b_tasks) == 2 and all(t['state'] == 'skipped' and t['attempt_count'] == 0 for t in b_tasks)
    b_unit = next(t['scientific_task_id'] for t in b_tasks if t['stage'] == 'precompute')
    assert store.cache_eviction('r',b_unit) is None
    before = len(snapshot['attempts'])
    resumed = execute_manifest(store,'r','ledger.jsonl',config,entry=unexpected_entry if unexpected else task_entry)
    assert len(store.snapshot('r')['attempts']) == before
    if unexpected:
        assert outcome['status'] != 'completed_successfully'
        c_unit = next(r['scientific_task_id'] for r in records if r['dataset'] == 'c' and r['stage'] == 'precompute')
        assert store.cache_eviction('r',c_unit) is None
        assert list(cache_root().rglob('*_train.pkl'))
        assert all(t['state'] == 'pending' for t in snapshot['tasks'] if json.loads(t['payload_json'])['dataset'] == 'd')
        assert resumed['model_attempts_dispatched_this_invocation'] == 0
    else:
        assert outcome['status'] == resumed['status'] == 'completed_successfully'
        assert len(outcome['cache_evictions']) == 3
        assert outcome['planned_skipped_tasks'] == 2
