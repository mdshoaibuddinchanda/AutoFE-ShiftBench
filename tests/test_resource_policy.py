from dataclasses import asdict
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from src import resource_policy as resource
from src import pipeline_runner as runner


def hardware(cpus=8, ram=32, gpu=True):
    return dict(allowed_cpu_ids=list(range(cpus)),physical_cpus=max(1,cpus//2),
                logical_cpus=cpus,ram_total_bytes=ram*resource.GIB,
                disk_total_bytes=512*resource.GIB,cuda_visible_devices=None,
                gpu_devices=[dict(device_index=0,physical_index=0,uuid='gpu-0',name='testGPU',
                                  total_bytes=4*resource.GIB)] if gpu else [])


def test_detect_hardware_accepts_a_path_on_existing_host(tmp_path, monkeypatch):
    monkeypatch.setattr(resource, 'gpu_inventory', lambda: [])
    machine = resource.detect_hardware(tmp_path/'not_created')
    assert machine['allowed_cpu_ids']
    assert machine['ram_total_bytes'] > 0 and machine['disk_total_bytes'] > 0


@pytest.mark.parametrize('cpus,ram,workers',[(8,32,6),(16,64,14),(4,8,2),(2,4,1)])
def test_dynamic_host_limits_preserve_cpu_and_ram_headroom(cpus,ram,workers):
    plan=resource.resolve_plan(resource.ResourceSettings(),hardware(cpus,ram),('knn',))
    assert plan['worker_ceiling']==workers
    assert plan['ram_budget_bytes']==int(ram*resource.GIB*.8)
    assert plan['ram_reserve_bytes']==int(ram*resource.GIB*.2)
    assert plan['cache_max_bytes']==int(min(ram*resource.GIB*.4,512*resource.GIB*.025))
    assert len(plan['worker_cpu_ids'])==workers


def test_low_worker_override_and_gpu_probe_do_not_change_scientific_models():
    probe={'xgboost':dict(supported=True),'catboost':dict(supported=False,reason='probe failure')}
    plan=resource.resolve_plan(resource.ResourceSettings(),hardware(),('xgboost','catboost','knn'),
                               gpu_probe=probe,worker_override=3)
    assert plan['worker_ceiling']==3
    assert plan['backend_by_model']==dict(xgboost='gpu',catboost='cpu',knn='cpu')
    with pytest.raises(ValueError,match='GPU backend'):
        resource.resolve_plan(resource.ResourceSettings(gpu_policy='require'),hardware(),('catboost',),gpu_probe=probe)


@pytest.mark.parametrize('settings',[resource.ResourceSettings(reserve_cpus=0),
    resource.ResourceSettings(ram_target_fraction=1),resource.ResourceSettings(vram_target_fraction=0),
    resource.ResourceSettings(gpu_policy='invalid')])
def test_invalid_resource_settings_rejected(settings):
    with pytest.raises(ValueError):settings.validate()


def test_ram_pressure_blocks_admission_and_recovers(monkeypatch):
    controller=resource.ResourceAdmission(resource.resolve_plan(resource.ResourceSettings(gpu_policy='cpu'),hardware(),('knn',)))
    state=dict(available_bytes=2*resource.GIB,resident_bytes=resource.GIB)
    monkeypatch.setattr(controller,'sample',lambda **kwargs:state)
    assert controller.reservation('knn',1024,[]) is None
    state['available_bytes']=20*resource.GIB
    reservation=controller.reservation('knn',1024,[])
    assert reservation['gpu_device'] is None
    assert controller.telemetry['ram_waits']==1
    controller.completed('knn',dict(worker_peak_rss_bytes=2*resource.GIB))
    assert controller.model_peak_bytes['knn']==2*resource.GIB


def test_gpu_vram_headroom_and_single_slot_per_device(monkeypatch):
    plan=resource.resolve_plan(resource.ResourceSettings(),hardware(),('xgboost',),gpu_probe={'xgboost':dict(supported=True)})
    controller=resource.ResourceAdmission(plan)
    monkeypatch.setattr(controller,'sample',lambda **kwargs:dict(available_bytes=20*resource.GIB,resident_bytes=resource.GIB))
    gpu={**hardware()['gpu_devices'][0],'free_bytes':3*resource.GIB}
    monkeypatch.setattr(resource,'gpu_inventory',lambda:[gpu])
    reservation=controller.reservation('xgboost',1024,[])
    assert reservation['gpu_device']==0
    assert 0<reservation['gpu_ram_part']<.8
    assert controller.reservation('xgboost',1024,[reservation]) is None
    gpu['free_bytes']=512*1024**2
    assert controller.reservation('xgboost',1024,[]) is None
    assert controller.telemetry['gpu_waits']==2


def test_static_gpu_capacity_uses_recorded_cpu_fallback_only_in_auto(monkeypatch):
    plan=resource.resolve_plan(resource.ResourceSettings(),hardware(),('xgboost',),gpu_probe={'xgboost':dict(supported=True)})
    controller=resource.ResourceAdmission(plan)
    monkeypatch.setattr(controller,'sample',lambda **kwargs:dict(available_bytes=25*resource.GIB,resident_bytes=resource.GIB))
    reservation=controller.reservation('xgboost',resource.GIB,[],comparison_matrix_bytes=2*resource.GIB)
    assert reservation['backend']=='cpu' and reservation['gpu_device'] is None
    assert 'static matrix estimate' in reservation['backend_reason']
    plan['settings']['gpu_policy']='require'
    with pytest.raises(MemoryError,match='VRAM budget'):
        controller.reservation('xgboost',resource.GIB,[],comparison_matrix_bytes=2*resource.GIB)


def test_gpu_capacity_backend_is_shared_across_different_pipeline_widths(monkeypatch):
    plan=resource.resolve_plan(resource.ResourceSettings(),hardware(),('xgboost',),gpu_probe={'xgboost':dict(supported=True)})
    controller=resource.ResourceAdmission(plan)
    monkeypatch.setattr(controller,'sample',lambda **kwargs:dict(available_bytes=25*resource.GIB,resident_bytes=resource.GIB))
    small=controller.reservation('xgboost',1024,[],comparison_matrix_bytes=2*resource.GIB)
    large=controller.reservation('xgboost',resource.GIB//2,[],comparison_matrix_bytes=2*resource.GIB)
    assert small['backend']==large['backend']=='cpu'
    assert small['comparison_matrix_bytes']==large['comparison_matrix_bytes']


def test_preparation_pressure_and_resident_reservations_are_not_double_counted(monkeypatch):
    controller=resource.ResourceAdmission(resource.resolve_plan(resource.ResourceSettings(gpu_policy='cpu'),hardware(),('knn',)))
    state=dict(available_bytes=12*resource.GIB,resident_bytes=8*resource.GIB,child_resident_bytes=6*resource.GIB)
    monkeypatch.setattr(controller,'sample',lambda **kwargs:state)
    assert controller.preparation_allowed(2*resource.GIB,[dict(ram_bytes=6*resource.GIB)])
    state['available_bytes']=7*resource.GIB
    assert not controller.preparation_allowed(2*resource.GIB,[])


def test_cuda_visibility_maps_physical_to_logical_devices(monkeypatch):
    class Result:
        returncode=0
        stdout='0, gpu0, card0, 4096, 3000, 616.64\n1, gpu1, card1, 8192, 7000, 616.64\n'
    monkeypatch.setattr(resource.subprocess,'run',lambda *args,**kwargs:Result())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1,0')
    devices=resource.gpu_inventory()
    assert [(d['physical_index'],d['device_index']) for d in devices]==[(1,0),(0,1)]


def test_gpu_fits_share_one_dedicated_worker_pool(tmp_path,monkeypatch):
    """Mock routing; real installed GPU fits are checked by the bounded proof."""
    for name in runner.NUMERICAL_THREAD_ENV:monkeypatch.setenv(name,'1')
    machine=hardware(cpus=4)
    monkeypatch.setattr(runner,'detect_hardware',lambda root:machine)
    monkeypatch.setattr(runner,'probe_gpu_models',lambda devices:{m:dict(supported=True) for m in ('xgboost','catboost')})
    monkeypatch.setattr(resource,'gpu_inventory',lambda:[{**machine['gpu_devices'][0],'free_bytes':3*resource.GIB}])
    monkeypatch.setattr(resource.ResourceAdmission,'sample',lambda self,**kwargs:dict(available_bytes=20*resource.GIB,resident_bytes=resource.GIB))
    pools=[];threads=[]
    def pool(**options):
        pools.append(options['max_workers'])
        return ThreadPoolExecutor(max_workers=options['max_workers'])
    monkeypatch.setattr(runner,'ProcessPoolExecutor',pool)
    original=runner._fit_and_score_worker
    def fake_gpu_fit(payload):
        threads.append(threading.get_ident())
        result=original({**payload,'use_gpu':False,'gpu_device':None,'gpu_ram_part':None})
        result.update(model_backend='gpu',gpu_device=payload['gpu_device'],gpu_ram_part=payload['gpu_ram_part'])
        return result
    monkeypatch.setattr(runner,'_fit_and_score_worker',fake_gpu_fit)
    rng=np.random.default_rng(42);frame=pd.DataFrame(rng.normal(size=(150,4)),columns=list('abcd'))
    frame['target']=(frame.a>0).astype(int)
    path=tmp_path/'data.csv';frame.to_csv(path,index=False)
    manifest=runner.run_experiment({'test':path},run_id='gpu-pool',output_root=tmp_path,
        seeds=[42],folds=[1],n_splits=3,conditions=(('clean',0.0),),
        pipelines=('Raw','AutoFE_Baseline'),models=('xgboost','catboost'),
        resource_policy='adaptive',cache_policy='bounded',durable_scheduler=True)
    assert manifest['counts_by_status']['success']==4
    assert pools==[2,1] and len(set(threads))==1 and len(threads)==4


def test_adaptive_cpu_has_exact_scientific_parity_and_resumes_same_plan(tmp_path,monkeypatch):
    for name in runner.NUMERICAL_THREAD_ENV:monkeypatch.setenv(name,'1')
    machine=hardware(cpus=4,ram=32,gpu=False)
    monkeypatch.setattr(runner,'detect_hardware',lambda root:machine)
    rng=np.random.default_rng(42)
    frame=pd.DataFrame(rng.normal(size=(150,4)),columns=list('abcd'))
    frame['target']=(frame.a>0).astype(int)
    path=tmp_path/'data.csv';frame.to_csv(path,index=False)
    common=dict(data_paths={'test':path},output_root=tmp_path,seeds=[42],folds=[1],
                n_splits=3,conditions=(('clean',0.0),),pipelines=('Raw','AutoFE_Baseline'),
                models=('gaussian_nb','logistic_regression'),cache_policy='bounded',durable_scheduler=True)
    fixed=runner.run_experiment(**common,run_id='fixed',workers=1)
    adaptive=runner.run_experiment(**common,run_id='adaptive',resource_policy='adaptive',gpu_policy='cpu')
    assert adaptive['status']=='complete' and adaptive['counts_by_status']['success']==4
    assert adaptive['configuration']['workers']==2
    assert adaptive['configuration']['cache_max_bytes']==int(machine['disk_total_bytes']*.025)
    def fields(name):
        rows=map(json.loads,(tmp_path/name/'results.jsonl').read_text().splitlines())
        return {(r['pipeline'],r['model']):(r['roc_auc'],r['f1'],r['prediction_sha256'],r['train_matrix_sha256'],r['test_matrix_sha256']) for r in rows}
    assert fields('fixed')==fields('adaptive')
    assert adaptive['resource_usage']['admissions']==4
    from src.provenance import stable_digest
    for line in (tmp_path/'adaptive'/'results.jsonl').read_text().splitlines():
        row=json.loads(line)
        parameters=json.loads((tmp_path/'adaptive'/row['model_parameters_path']).read_text())
        assert stable_digest(parameters)==row['model_parameters_fingerprint']
        assert row['worker_peak_rss_bytes']>0 and parameters['model']==row['model']
    assert len(list((tmp_path/'adaptive'/'model_parameters').glob('*.json')))==2
    result_rows=[json.loads(line) for line in (tmp_path/'adaptive'/'results.jsonl').read_text().splitlines()]
    assert len({r['resource_comparison_matrix_bytes'] for r in result_rows})==1
    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps({'resource_plan':adaptive['configuration']['resource_plan']}))
    monkeypatch.setattr(runner,'probe_gpu_models',lambda devices: (_ for _ in ()).throw(AssertionError('Frozen plan must not be re-probed')))
    frozen=runner.run_experiment(**common,run_id='frozen',resource_policy='adaptive',gpu_policy='cpu',resource_profile=profile)
    assert fields('frozen')==fields('adaptive')
    assert frozen['configuration']['resource_profile_sha256']
    resumed=runner.run_experiment(**common,run_id='adaptive',resource_policy='adaptive',gpu_policy='cpu')
    assert resumed['configuration_fingerprint']==adaptive['configuration_fingerprint']
    machine['ram_total_bytes']=64*resource.GIB
    with pytest.raises(ValueError,match='host/policy changed'):
        runner.run_experiment(**common,run_id='adaptive',resource_policy='adaptive',gpu_policy='cpu')
