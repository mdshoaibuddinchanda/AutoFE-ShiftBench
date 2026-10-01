from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.array_transport import ArrayTransport, task_arrays


def payload():
    x = np.asfortranarray(np.arange(60,dtype=np.float32).reshape(20,3))
    return dict(model='gaussian_nb', seed=42, xtr=x, xte=x[:10].copy(order='F'),
                y_train_enc=np.tile([0,1],10),y_test_enc=np.tile([0,1],5),classes=np.array(['a','b']))


def test_published_once_copy_on_write_preserves_bytes_and_layout(tmp_path):
    manager=ArrayTransport(tmp_path/'maps',budget_bytes=100_000,threshold_bytes=1)
    original=payload()
    a,mode=manager.payload(original,'group');b,mode2=manager.payload(original,'group')
    assert mode==mode2=='mapped' and manager.stats['groups_published']==1
    assert all(n not in a for n in ['xtr','xte','y_train_enc','y_test_enc'])
    with task_arrays(a) as opened:
        np.testing.assert_array_equal(opened['xtr'],original['xtr'])
        assert opened['xtr'].flags.f_contiguous
        opened['xtr'][:]=999;opened['y_train_enc'][:]=999
    with task_arrays(b) as opened:
        np.testing.assert_array_equal(opened['xtr'],original['xtr'])
        np.testing.assert_array_equal(opened['y_train_enc'],original['y_train_enc'])
    manager.release('group')
    assert not list(manager.root.rglob('*.npy'))
    manager.close()


def test_small_and_over_budget_work_uses_pickle(tmp_path):
    manager=ArrayTransport(tmp_path/'maps',budget_bytes=10_000,threshold_bytes=100_000)
    assert manager.payload(payload(),'a')[1]=='pickle'
    assert manager.payload(payload(),'b',mode='mapped')[1]=='pickle' # header allowance exceeds budget
    assert manager.stats['groups_published']==0
    manager.close()


def test_mapped_shape_mutation_rejected_and_handles_closed(tmp_path):
    manager=ArrayTransport(tmp_path/'maps',budget_bytes=100_000,threshold_bytes=1)
    p,_=manager.payload(payload(),'a')
    p['_mapped_arrays']['xtr']['shape']=[999,3]
    with pytest.raises(ValueError,match='shape/dtype'):
        with task_arrays(p):pass
    manager.close()
    assert not list(manager.root.rglob('*.npy'))


def test_cleanup_cannot_escape_root(tmp_path):
    manager=ArrayTransport(tmp_path/'maps',budget_bytes=100_000,threshold_bytes=1)
    protected=tmp_path/'keep';protected.mkdir();(protected/'keep.npy').write_bytes(b'keep')
    with pytest.raises(ValueError,match='escaped'):
        manager._remove_epoch(protected)
    assert (protected/'keep.npy').read_bytes()==b'keep'
    manager.close()


def test_resume_accumulates_saved_counters_and_peak(tmp_path):
    prior=dict(groups_published=3,mapped_task_submissions=10,serialized_task_submissions=5,
               cleanup_deferred=2,peak_disk_bytes=80_000)
    manager=ArrayTransport(tmp_path/'maps',budget_bytes=100_000,threshold_bytes=1,prior_stats=prior)
    manager.payload(payload(),'a')
    assert manager.stats['groups_published']==4 and manager.stats['mapped_task_submissions']==11
    assert manager.stats['serialized_task_submissions']==5 and manager.stats['cleanup_deferred']==2
    assert manager.stats['peak_disk_bytes']==80_000
    manager.close()


def test_stale_coordinator_files_are_regenerated_not_reused(tmp_path):
    script='''
import os
import numpy as np
from pathlib import Path
from src.array_transport import ArrayTransport
m=ArrayTransport(Path(__import__('sys').argv[1]),budget_bytes=100000,threshold_bytes=1)
x=np.arange(60,dtype=np.float32).reshape(20,3)
m.payload(dict(xtr=x,xte=x,y_train_enc=np.zeros(20,dtype=int),y_test_enc=np.zeros(20,dtype=int)), 'a')
os._exit(77)
'''
    root=tmp_path/'maps'
    result=subprocess.run([sys.executable,'-c',script,str(root)],cwd=Path(__file__).parents[1])
    assert result.returncode==77 and list(root.rglob('*.npy'))
    manager=ArrayTransport(root,budget_bytes=100_000,threshold_bytes=1)
    assert not list(root.rglob('*.npy'))
    manager.close()


def test_spawned_runner_mapped_vs_pickled_exact_parity_and_reclamation(tmp_path):
    import pandas as pd
    from src.pipeline_runner import run_experiment
    from provenance.final_optimization_verification import rows,FIELDS
    rng=np.random.default_rng(3)
    frame=pd.DataFrame(rng.normal(size=(240,5)),columns=list('abcde'))
    frame['target']=(frame.a+frame.b>0).astype(int)
    path=tmp_path/'data.csv';frame.to_csv(path,index=False)
    common=dict(data_paths={'data':path},output_root=tmp_path,workers=2,seeds=[42],folds=[1],
        n_splits=3,pipelines=('Raw','AutoFE_Baseline'),models=('gaussian_nb','logistic_regression'),
        conditions=(('clean',0.),('gaussian_noise',.05)),durable_scheduler=True,cache_policy='bounded',cache_max_bytes=1_000_000)
    mapped=run_experiment(**common,run_id='mapped',array_transport='mapped')
    pickled=run_experiment(**common,run_id='pickled',array_transport='pickle')
    assert mapped['counts_by_status']['success']==pickled['counts_by_status']['success']==8
    assert mapped['array_transport_usage']['groups_published']==4
    assert mapped['array_transport_usage']['mapped_task_submissions']==8
    assert not list((tmp_path/'mapped/array_transport').rglob('*.npy'))
    a,b=rows(tmp_path/'mapped/results.jsonl'),rows(tmp_path/'pickled/results.jsonl')
    assert all({k:r.get(k) for k in FIELDS}=={k:b[key].get(k) for k in FIELDS} for key,r in a.items())
    assert all(r['array_transport']=='mapped' for r in a.values())
