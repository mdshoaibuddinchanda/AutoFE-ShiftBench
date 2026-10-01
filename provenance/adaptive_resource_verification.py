"""Bounded adaptive execution evidence; never launch the corrected campaign."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from provenance.final_optimization_verification import FIELDS, MODELS, PIPELINES, THREADS, rows
from provenance.large_dataset_recovery_v1 import (
    _stop_process_tree, _wait_for_successes, _successes, _check_terminal,
)
from src.provenance import atomic_write_json, code_fingerprint, file_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'corrected_runs'/'adaptive_verification'
FEATURE_FIELDS = ('train_matrix_sha256','test_matrix_sha256','operator_candidate_counts',
                  'operator_configuration','n_original','n_retained','n_generated','n_train','n_test')
GPU_MODELS = ('xgboost','catboost')


def worker(kind, policy, run_id, scope_path, transport='auto'):
    from src.pipeline_runner import run_experiment
    datasets = ('sonar','heart-disease','haberman','ionosphere') if kind == 'small' else ('airlines' if kind=='transport' else kind,)
    pipelines = PIPELINES if kind == 'small' else ('Raw',) if kind in ('covertype','transport') else ('Raw','AutoFE_Baseline')
    selected_models=('gaussian_nb','logistic_regression') if kind=='transport' else GPU_MODELS if kind=='airlines' else MODELS
    started=time.monotonic()
    manifest=run_experiment(
        {name:ROOT/'data'/'raw'/f'{name}.csv' for name in datasets},
        output_root=OUTPUT,run_id=run_id,seeds=[42],folds=[1],n_splits=5,
        conditions=(('clean',0.0),),pipelines=pipelines,models=selected_models,
        split_policy=policy,cache_policy='bounded',durable_scheduler=True,
        scheduler_lease_seconds=600,cache_audit=True,
        resource_policy='adaptive',gpu_policy='cpu' if kind=='small' else 'auto',
        resource_profile=None if kind=='small' else scope_path,array_transport=transport)
    expected=len(datasets)*len(pipelines)*len(selected_models)
    if manifest['status']!='complete' or manifest['counts_by_status']['success']!=expected:
        raise ValueError(f'Adaptive diagnostic not complete: {run_id}')
    atomic_write_json(OUTPUT/run_id/'measurement.json',dict(
        elapsed_s=time.monotonic()-started,expected=expected,success=expected,
        code_fingerprint=manifest['code_fingerprint'],resource_usage=manifest['resource_usage']))


def start(kind,policy,run_id,phase,scope_path,transport='auto'):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    environment={**os.environ,**THREADS}
    command=[sys.executable,'-m','provenance.adaptive_resource_verification',
             '--worker','--kind',kind,'--policy',policy,'--run-id',run_id,
             '--scope',str(scope_path),'--transport',transport]
    with (OUTPUT/f'{run_id}.{phase}.log').open('w',encoding='utf-8') as log:
        return subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)


def compare(current,previous,allow_gpu):
    actual,baseline=rows(current/'results.jsonl'),rows(previous/'results.jsonl')
    expected=set(baseline) if len(actual)==560 else set(actual)
    if not actual or set(actual)!=expected or not set(actual).issubset(baseline):
        raise ValueError('Diagnostic comparison cell coverage differs')
    gpu=0
    for key,row in actual.items():
        is_gpu=row['model_backend']=='gpu'
        if is_gpu:
            if not allow_gpu or row['model'] not in {'xgboost','catboost'} or row.get('gpu_device') is None:
                raise ValueError('Unexpected GPU backend/device')
            if not 0 < row['gpu_ram_part'] <= .8:
                raise ValueError('GPU resource fraction missing or invalid')
            gpu+=1
        fields=FEATURE_FIELDS if is_gpu else FIELDS
        mismatches=[field for field in fields if row.get(field)!=baseline[key].get(field)]
        if mismatches:raise ValueError(f'Adaptive scientific parity failed: {key}, {mismatches}')
    return dict(compared_cells=len(actual),exact_cpu_cells=len(actual)-gpu,
                gpu_feature_only_cells=gpu,mismatches=[],
                cpu_fields=list(FIELDS),gpu_fields=list(FEATURE_FIELDS),
                gpu_metrics='Backend changed; CPU/GPU metric equality is not asserted')


def evidence(kind,policy,run_id,version,scope_path):
    run_dir=OUTPUT/run_id
    if run_dir.exists():raise FileExistsError(f'Preserve existing evidence: {run_dir}')
    started=time.monotonic()
    before={}
    transport='mapped' if kind=='airlines' else 'auto'
    if kind=='airlines':
        first=start(kind,policy,run_id,'forced_stop',scope_path,transport)
        try:_wait_for_successes(first,run_dir,expected=4)
        finally:_stop_process_tree(first)
        before=_successes(run_dir)
        if not 0<len(before)<4:raise ValueError('Forced stop missed an in-progress run')
    process=start(kind,policy,run_id,'resume' if before else 'run',scope_path,transport)
    try:
        process.wait(timeout=7200)
        if process.returncode:raise RuntimeError(f'Adaptive worker failed: {run_id}; see its log')
    finally:_stop_process_tree(process)
    baseline=(ROOT/'corrected_runs'/'final_optimization'/f'optimization-optimized-{policy}-001'
              if kind=='small' else ROOT/'corrected_runs'/'large_calibration'/
              f'calibration-{policy}-{"002" if policy=="row_level" else "003"}')
    result=dict(kind=kind,policy=policy,run_id=run_id,baseline_dir=baseline.relative_to(ROOT).as_posix(),
                elapsed_s_including_restart=time.monotonic()-started,
                **compare(run_dir,baseline,kind!='small'))
    if before:
        result['pre_crash_committed_rows']=before
        result['recovery']=_check_terminal(run_dir,before,code_fingerprint(ROOT),expected=4)
    return result


def transport_evidence(policy, version, scope_path):
    """Real-data exact CPU comparison, including mapped input recovery."""
    label='row' if policy=='row_level' else 'group'
    results=[]
    for transport in ('pickle','mapped'):
        run_id=f'adaptive-transport-{transport}-{label}-{version}'
        run_dir=OUTPUT/run_id
        if run_dir.exists():raise FileExistsError(f'Preserve existing evidence: {run_dir}')
        started=time.monotonic();before={}
        if transport=='mapped':
            first=start('transport',policy,run_id,'forced_stop',scope_path,transport)
            try:_wait_for_successes(first,run_dir,expected=2)
            finally:_stop_process_tree(first)
            before=_successes(run_dir)
            if len(before)!=1:raise ValueError('Mapped recovery missed the one-success boundary')
        process=start('transport',policy,run_id,'resume' if before else 'run',scope_path,transport)
        try:
            process.wait(timeout=7200)
            if process.returncode:raise RuntimeError(f'Transport worker failed: {run_id}')
        finally:_stop_process_tree(process)
        manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
        result=dict(kind='transport',policy=policy,transport=transport,run_id=run_id,
                    elapsed_s_including_restart=time.monotonic()-started,
                    array_transport_usage=manifest['array_transport_usage'])
        if before:
            result['pre_crash_committed_rows']=before
            result['recovery']=_check_terminal(run_dir,before,code_fingerprint(ROOT),expected=2)
        results.append(result)
    comparison=compare(OUTPUT/results[1]['run_id'],OUTPUT/results[0]['run_id'],False)
    results[1]['mapped_vs_pickle_comparison']=comparison
    results[1]['pickle_reference_run_id']=results[0]['run_id']
    historical=ROOT/'corrected_runs'/'large_calibration'/f'calibration-{policy}-{"002" if policy=="row_level" else "003"}'
    for result in results:
        result.update(compare(OUTPUT/result['run_id'],historical,False))
        result['baseline_dir']=historical.relative_to(ROOT).as_posix()
    return results


def verify_saved_report(report):
    from src.provenance import stable_digest
    if report['code_fingerprint']!=code_fingerprint(ROOT):raise ValueError('Adaptive evidence source changed')
    if report['verification_script_sha256']!=file_sha256(Path(__file__)):raise ValueError('Adaptive verifier changed')
    if report['host'].casefold()!=platform.node().casefold():raise ValueError('Adaptive evidence belongs to another host')
    expected={(kind,policy) for kind in ('small','covertype','airlines') for policy in ('row_level','group_aware')}
    if {(r['kind'],r['policy']) for r in report['runs']}!=expected or len(report['runs'])!=6:
        raise ValueError('Adaptive evidence lacks full bounded coverage')
    for relative,digest in report['artifact_sha256'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or file_sha256(path)!=digest:raise ValueError(f'Adaptive evidence changed: {relative}')
    for result in report['runs']:
        run_dir=OUTPUT/result['run_id']
        manifest=json.loads((run_dir/'manifest.json').read_text())
        config=manifest['configuration']
        plan=config['resource_plan']
        count={'small':560,'covertype':10,'airlines':4}[result['kind']]
        datasets=['sonar','heart-disease','haberman','ionosphere'] if result['kind']=='small' else [result['kind']]
        pipelines=list(PIPELINES) if result['kind']=='small' else ['Raw'] if result['kind']=='covertype' else ['Raw','AutoFE_Baseline']
        if (manifest['status']!='complete' or manifest['expected_tasks']!=count
                or manifest['counts_by_status']['success']!=count
                or manifest['code_fingerprint']!=report['code_fingerprint']
                or config['split_policy']!=result['policy'] or config['resource_policy']!='adaptive'
                or config['numerical_thread_environment']!=THREADS
                or config['models']!=list(GPU_MODELS if result['kind']=='airlines' else MODELS)
                or config['seeds']!=[42] or config['folds']!=[1]
                or config['datasets']!=datasets or config['pipelines']!=pipelines
                or config['conditions']!=[['clean',0.0]] or config['scheduler_lease_seconds']!=600
                or config['cache_policy']!='bounded' or config['workers']!=plan['worker_ceiling']
                or plan['settings']!={**report['resource_plan']['settings'],
                                      'gpu_policy':'cpu' if result['kind']=='small' else 'auto'}
                or plan['hardware']!=report['resource_plan']['hardware']
                or config['cache_max_bytes']!=plan['cache_max_bytes']):
            raise ValueError('Adaptive run design/source/host mismatch')
        expected_transport='mapped' if result['kind']=='airlines' else 'auto'
        if config.get('array_transport')!=expected_transport or config.get('reuse_preprocessing') is not True:
            raise ValueError('Runtime reuse policy missing from current proof')
        if result['kind']!='small' and (plan!=report['resource_plan']
                or config.get('resource_profile_sha256')!=report['scope_sha256']):
            raise ValueError('Adaptive run did not use the frozen capability profile')
        rechecked=compare(run_dir,ROOT/result['baseline_dir'],result['kind']!='small')
        if any(rechecked[k]!=result[k] for k in rechecked):raise ValueError('Saved comparison changed')
        for row in rows(run_dir/'results.jsonl').values():
            if result['kind']=='airlines' and row.get('array_transport')!='mapped':
                raise ValueError('GPU/native mapped input was not exercised')
            path=run_dir/row['model_parameters_path']
            record=json.loads(path.read_text())
            if stable_digest(record)!=row['model_parameters_fingerprint'] or record['model']!=row['model']:
                raise ValueError('Resolved model parameter record changed')
        common_backends = {}
        gpu_worker_pids = {}
        for row in rows(run_dir/'results.jsonl').values():
            key=tuple(row[field] for field in ('dataset','seed','fold','condition','model'))
            signature=(row['model_backend'],row.get('resource_comparison_matrix_bytes'))
            if signature[1] is None or common_backends.setdefault(key,signature)!=signature:
                raise ValueError('Backend/capacity bound differs within a pipeline comparison')
            if row['model_backend']=='gpu':
                gpu_worker_pids.setdefault(row['gpu_device'],set()).add(row['worker_pid'])
        # A forced restart gets a new process; each uninterrupted segment
        # still uses one device process. Other runs must use exactly one.
        if result['kind']!='airlines' and any(len(pids)!=1 for pids in gpu_worker_pids.values()):
            raise ValueError('GPU tasks did not share a persistent device worker')
        if result['kind']=='airlines':
            before={key:tuple(value) for key,value in result['pre_crash_committed_rows'].items()}
            rechecked_recovery=_check_terminal(run_dir,before,report['code_fingerprint'],expected=4)
            if not 0<len(before)<4 or rechecked_recovery!=result['recovery']:
                raise ValueError('Adaptive committed-cell recovery failed')
        for entry in manifest['datasets']:
            if file_sha256(ROOT/'data'/'raw'/f"{entry['dataset']}.csv")!=entry['source_csv_sha256']:
                raise ValueError('Diagnostic data changed')
    transports=report.get('transport_runs',[])
    if {(r['transport'],r['policy']) for r in transports}!={(m,p) for m in ('mapped','pickle') for p in ('row_level','group_aware')} or len(transports)!=4:
        raise ValueError('Missing real-data mapped/serialized CPU and recovery proof')
    for result in transports:
        run_dir=OUTPUT/result['run_id']
        manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
        config=manifest['configuration']
        if (manifest['status']!='complete' or manifest['expected_tasks']!=2
                or manifest['counts_by_status']['success']!=2
                or manifest['code_fingerprint']!=report['code_fingerprint']
                or config['datasets']!=['airlines'] or config['pipelines']!=['Raw']
                or config['models']!=['gaussian_nb','logistic_regression']
                or config['seeds']!=[42] or config['folds']!=[1] or config['conditions']!=[['clean',0.0]]
                or config['split_policy']!=result['policy'] or config['numerical_thread_environment']!=THREADS
                or config['resource_plan']!=report['resource_plan']
                or config['resource_profile_sha256']!=report['scope_sha256']
                or config['array_transport']!=result['transport'] or not config['reuse_preprocessing']
                or manifest['array_transport_usage']!=result['array_transport_usage']):
            raise ValueError('Mapped real-data proof design/identity changed')
        rechecked=compare(run_dir,ROOT/result['baseline_dir'],False)
        if any(rechecked[k]!=result[k] for k in rechecked):raise ValueError('Mapped CPU parity changed')
        if result['transport']=='mapped':
            if compare(run_dir,OUTPUT/result['pickle_reference_run_id'],False)!=result['mapped_vs_pickle_comparison']:
                raise ValueError('Mapped versus serialized CPU comparison changed')
            before={key:tuple(value) for key,value in result['pre_crash_committed_rows'].items()}
            if len(before)!=1 or _check_terminal(run_dir,before,report['code_fingerprint'],expected=2)!=result['recovery']:
                raise ValueError('Mapped committed-cell recovery changed')
            if manifest['array_transport_usage']['mapped_task_submissions']<2 or list((run_dir/'array_transport').rglob('*.npy')):
                raise ValueError('Mapped inputs were unused or not reclaimed')
        for row in rows(run_dir/'results.jsonl').values():
            if row['array_transport']!=result['transport'] or row['model_backend']!='cpu':
                raise ValueError('Unexpected transport/backend in proof')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--kind',choices=['small','covertype','airlines','transport'])
    parser.add_argument('--policy',choices=['row_level','group_aware'])
    parser.add_argument('--run-id');parser.add_argument('--version',default='004')
    parser.add_argument('--transport',choices=['auto','mapped','pickle'],default='auto')
    parser.add_argument('--scope',type=Path,default=ROOT/'provenance'/'reviewer1_launch_scope_v5.json')
    parser.add_argument('--verify-report',type=Path)
    args=parser.parse_args()
    os.environ.update(THREADS)
    if args.worker:
        worker(args.kind,args.policy,args.run_id,args.scope,args.transport);return
    if args.verify_report:
        verify_saved_report(json.loads(args.verify_report.read_text()));print('Saved adaptive evidence verified');return
    if Path(sys.executable).resolve()!=Path(r'D:\Conda\p12\python.exe').resolve():raise RuntimeError('Use existing p12')
    scope=json.loads(args.scope.read_text())
    if scope['code_fingerprint']!=code_fingerprint(ROOT):raise ValueError('Adaptive freeze source changed')
    results=[]
    for kind in ('small','covertype','airlines'):
        for policy in ('row_level','group_aware'):
            suffix='row' if policy=='row_level' else 'group'
            run_id=f'adaptive-{kind}-{suffix}-{args.version}'
            results.append(evidence(kind,policy,run_id,args.version,args.scope))
            print(json.dumps(results[-1]),flush=True)
    transports=[]
    for policy in ('row_level','group_aware'):
        transports.extend(transport_evidence(policy,args.version,args.scope))
        print(json.dumps(transports[-2:]),flush=True)
    report=dict(artifact_type='adaptive_resource_verification_v2',status='passed',
                scientific_use='bounded execution verification; PENDING CORRECTED RUN',
                host=platform.node(),python_executable=str(Path(sys.executable).resolve()),
                code_fingerprint=code_fingerprint(ROOT),scope_sha256=file_sha256(args.scope),
                resource_plan=scope['resource_plan'],runs=results,transport_runs=transports,
                verification_script_sha256=file_sha256(Path(__file__)),artifact_sha256={})
    for result in [*results,*transports]:
        run_dir=OUTPUT/result['run_id'];baseline=ROOT/result['baseline_dir']
        files=[run_dir/'manifest.json',run_dir/'results.jsonl',baseline/'manifest.json',baseline/'results.jsonl',
               *sorted((run_dir/'model_parameters').glob('*.json'))]
        if result['kind'] in ('airlines','transport'):files.append(run_dir/'scheduler.sqlite')
        for path in files:report['artifact_sha256'][path.relative_to(ROOT).as_posix()]=file_sha256(path)
    verify_saved_report(report)
    output=ROOT/scope['adaptive_verification_path']
    if output.exists():raise FileExistsError('Preserve prior adaptive report')
    atomic_write_json(output,report)
    print(json.dumps({'report':str(output),'status':'passed'}))


if __name__=='__main__':main()
