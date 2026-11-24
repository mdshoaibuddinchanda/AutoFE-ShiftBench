"""Bounded, explicit real-data preflight; never dispatches the full grid.

Run from the repository with P12: python -B -m reproduction.real_data_preflight
{audit,freeze,pilot,report} --output reports/real_data_preflight_20261003
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import psutil
import yaml

from src.artifact_integrity import atomic_json, dataset_identity, file_sha256, fingerprint
from src.data_loader import OPENML_NAME_CANDIDATES, load_csv_dataset, load_dataset_names
from src.pipeline_runner import (PIPELINE_NAMES, PIPELINE_CONFIGS, CPU_MODELS, GPU_MODELS,
    SHIFT_FAMILIES, pipeline_identity_token, split_predictors_and_target, split_policy_for_condition)
from src.feature_engineering import CAP_POLICY_VERSION
from src.fsva import FSVA_SCHEMA_VERSION, DEFAULT_PERTURBATION_MAGNITUDES
from src.protocol import EVALUATION_PROTOCOL_VERSION, cache_root
from src.seeding import SEED_SCHEME_VERSION, split_seed, corruption_seed
from src.shift_generator import apply_perturbation
from src.splitters import get_splits, SplitInfeasibleError
from src.task_manifest import ExecutionConfig, ManifestStore, build_task_records, run_id_for
from src.provenance import collect_code_identity, collect_environment_identity

SEEDS = [42, 123, 456, 789, 2025]
# Metadata selection, fixed before accuracy/model timings exist.
PILOT_UNITS = [('haberman',42,1,'clean',0.), ('dry-bean-dataset',42,1,'clean',0.),
               ('aps_failure',42,1,'clean',0.), ('dry-bean-dataset',42,1,'gaussian_noise',.05),
               ('dry-bean-dataset',123,1,'clean',0.)]
LIMITS = {'model_attempts':840, 'pilot_model_attempts':700, 'verification_reserve':140,
          'execution_seconds':14400, 'workers':2, 'gpu_workers':1, 'task_seconds':1800,
          'cache_bytes':100*1024**3, 'min_free_bytes':50*1024**3,
          'dense_budget_bytes':4*1024**3}


def now(): return datetime.now(timezone.utc).isoformat()


def footprint(path):
    sizes = {'total':0, 'disposable':0, 'temporary':0, 'files':0}
    for base, _, files in os.walk(path):
        for name in files:
            item = Path(base)/name
            try: size = item.stat().st_size
            except FileNotFoundError: continue
            sizes['total'] += size; sizes['files'] += 1
            if name.endswith(('_train.pkl','_test.pkl','.model_float32.npy')): sizes['disposable'] += size
            if name.startswith('.publish_') or '.tmp_' in name or '.export_' in name: sizes['temporary'] += size
    sizes['retained'] = sizes['total']-sizes['disposable']-sizes['temporary']
    return sizes


def declared_records(names, *, units=None, identities=None, planned=None):
    paths = {n:Path('data/raw')/(n+'.csv') for n in names}
    metadata = {'diagnostics_enabled':True, 'max_rows':128, 'schema':FSVA_SCHEMA_VERSION,
                'magnitudes':list(DEFAULT_PERTURBATION_MAGNITUDES)}
    common = dict(pipeline_identity=pipeline_identity_token, data_paths=paths, data_identities=identities,
        pipeline_metadata=lambda n:{'operator_set_id':PIPELINE_CONFIGS[n].operator_set_id,
            'cap_policy_version':CAP_POLICY_VERSION if PIPELINE_CONFIGS[n].max_features is not None else 'none_v1'},
        precompute_metadata=metadata, planned_infeasible=planned)
    if units is None:
        return build_task_records(names, SEEDS, range(1,6), SHIFT_FAMILIES, PIPELINE_NAMES, CPU_MODELS+GPU_MODELS, **common)
    records = []
    for n,s,f,fam,sev in units:
        if n not in names: raise ValueError('Unknown explicitly selected dataset: '+n)
        records.extend(build_task_records([n],[s],[f],[(fam,sev)],PIPELINE_NAMES,CPU_MODELS+GPU_MODELS,**common))
    if len({r['scientific_task_id'] for r in records}) != len(records): raise ValueError('Duplicate pilot units')
    return records


def original_source(name, meta, offline_cache):
    """Parse retained provider bytes; network forbidden, originals never modified."""
    source=meta['source_provenance']; details=source['details']
    if source['provider']=='UCI':
        from scipy.io import arff
        path=Path(details['source_archive_path'])
        assert file_sha256(path)==details['source_archive_sha256']
        with zipfile.ZipFile(path) as z:
            member=next(n for n in z.namelist() if n.endswith('Dry_Bean_Dataset.arff'))
            rows,_=arff.loadarff(io.StringIO(z.read(member).decode('utf-8-sig')))
        frame=pd.DataFrame(rows); y=frame.pop('Class').map(lambda x:x.decode('utf-8'))
        return frame,y,{'path':str(path),'sha256':file_sha256(path),'member':member,'source_details_match':True}
    from sklearn.datasets import fetch_openml
    def no_network(*a,**k): raise RuntimeError('Offline source verification refuses network')
    with patch('sklearn.datasets._openml.urlopen', no_network):
        dataset=fetch_openml(data_id=int(source['data_id']),data_home=str(offline_cache),as_frame=True,parser='auto')
    x,y=dataset.data,dataset.target
    if y is None and name=='heart-disease' and 'target' in dataset.frame:
        x=dataset.frame.drop(columns=['target']); y=dataset.frame['target']
    if x is None or y is None: raise ValueError('Retained provider source lacks the declared target')
    for key in ('id','version','name','default_target_attribute','md5_checksum'):
        if str(dataset.details.get(key)) != str(details.get(key)): raise ValueError('Provider evidence mismatch: '+key)
    api=offline_cache/'openml/openml.org/api/v1/json/data'/(source['data_id']+'.gz')
    raw=offline_cache/'openml/openml.org/data/v1/download'/(str(details['file_id'])+'.gz')
    md5=hashlib.md5()
    with gzip.open(raw,'rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): md5.update(block)
    if md5.hexdigest()!=details['md5_checksum']: raise ValueError('Retained raw provider checksum differs')
    return x,y,{'api_metadata_path':str(api),'api_metadata_sha256':file_sha256(api),
                'raw_response_path':str(raw),'gzip_sha256':file_sha256(raw),
                'decompressed_md5':md5.hexdigest(),'source_details_match':True}


def audit(out):
    out.mkdir(parents=True,exist_ok=True)
    declarations=yaml.safe_load(Path('config/dataset_list.yaml').read_text())['datasets']
    names=load_dataset_names('config/dataset_list.yaml')
    offline=out/'offline_source_copy'
    # sklearn retries may remove a corrupt cached response; parse only a copy.
    if not offline.exists(): shutil.copytree('data/openml_download_cache',offline)
    env=collect_environment_identity()
    from threadpoolctl import threadpool_info
    atomic_json(out/'environment.json',{'created_utc':now(),'code':collect_code_identity(), 'environment':env,
        'python':sys.executable,'native_pools':threadpool_info(),
        'thread_environment':{n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'physical_cpus':psutil.cpu_count(logical=False),'logical_cpus':psutil.cpu_count(),
        'ram':dict(psutil.virtual_memory()._asdict()),'disk':dict(shutil.disk_usage('.')._asdict()),
        'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),
        'bulk_cache':footprint('data/cache'),'reports':footprint('reports'),
        'roots':{'workspace':str(Path.cwd()),'raw':'data/raw','bulk_cache':'data/cache','pilot':str(out)},
        'native_thread_policy':'unchanged inherited native limits; rejected one-thread rewrite disabled'})
    atomic_json(out/'declared_limits.json',{'created_utc':now(),'limits':LIMITS,'units':PILOT_UNITS,
        'pipelines':PIPELINE_NAMES,'models':CPU_MODELS+GPU_MODELS,'diagnostics':{'max_rows':128,'fd_rows':32,'magnitudes':list(DEFAULT_PERTURBATION_MAGNITUDES)},
        'selection_reason':{'haberman':'small numeric control, 306 rows/3 predictors','dry-bean-dataset':'medium seven-class numeric, 13611/16',
                            'aps_failure':'large wide missing-value numeric, 76000/170'},
        'model_fits_before_declaration':0,'extra_pilot_row_cap':None,'full_grid_dispatch':False})
    results=[]; eligibility=[]
    for declaration in declarations:
        request={'name':declaration,'provider':'OpenML','version':1} if isinstance(declaration,str) else declaration
        name=request['name']; path=Path('data/raw')/(name+'.csv'); mpath=path.with_name(name+'_meta.json')
        frame=load_csv_dataset(path); meta=json.loads(mpath.read_text()); x,y,target=split_predictors_and_target(frame)
        source=meta['source_provenance']; selection=meta['row_selection']; identity=dataset_identity(path,frame=frame)
        record={'dataset':name,'request':request,'identity':identity,'metadata_sha256':file_sha256(mpath),
            'provider':source['provider'],'source_id':source['data_id'],'source_version':source['version'],
            'resolved_request':source['resolved_request'],'source_name':source['details']['name'],
            'source_target':source['target_names'],'target':target,'rows':len(frame),'features':x.shape[1],
            'dtypes':{str(k):str(v) for k,v in frame.dtypes.items()},
            'missing':{str(k):int(v) for k,v in frame.isna().sum().items()},
            'nonfinite_numeric':{str(c):int(np.isinf(frame[c].to_numpy()).sum()) for c in frame.select_dtypes('number')},
            'class_counts':{str(k):int(v) for k,v in y.value_counts(dropna=False).items()},
            'source_rows':selection['source_rows'],'sampling_policy':selection['policy'],'cap':selection['max_rows'],'cap_seed':selection['random_state'],
            'positions_count':len(selection['selected_source_positions']),'positions_sha256':fingerprint(selection['selected_source_positions']),
            'duplicate_predictor_rows':int(x.duplicated().sum()),'predictor_target_isolation':target not in x,
            'metafeature_status':meta.get('metafeature_status')}
        try:
            assert source['provider']==request.get('provider','OpenML')
            assert int(source['version'])==request.get('version',1)
            if request.get('data_id'): assert str(source['data_id'])==str(request['data_id'])
            elif source['provider']=='OpenML':
                candidates=OPENML_NAME_CANDIDATES.get(name,[name]); assert source['resolved_request'] in list(map(str,candidates))
                if source['resolved_request'].isdigit(): assert source['data_id']==source['resolved_request']
                else: assert source['details']['name'].lower()==source['resolved_request'].lower()
            assert identity==meta['dataset_identity']
            assert selection['max_rows']==100000 and selection['random_state']==42
            count=selection['source_rows']
            expected=pd.Series(np.arange(count)).sample(n=100000,random_state=42).to_numpy() if count>100000 else np.arange(count)
            assert np.array_equal(expected,selection['selected_source_positions']) and len(expected)==len(frame)
            original,labels,evidence=original_source(name,meta,offline)
            origin_fingerprint=fingerprint({'columns':list(original.columns),'dtypes':list(map(str,original.dtypes)),
                'hashes':pd.util.hash_pandas_object(original,index=True).astype(str).tolist(),
                'target_hashes':pd.util.hash_pandas_object(pd.Series(labels),index=True).astype(str).tolist()})
            assert origin_fingerprint==source['original_source_frame_fingerprint']
            combined=original.reset_index(drop=True).copy(); combined[target]=pd.Series(labels).reset_index(drop=True)
            selected=combined.iloc[expected].reset_index(drop=True)
            assert hashlib.sha256(selected.to_csv(index=False).encode()).hexdigest()==identity['bytes_sha256']
            original_counts={str(k):int(v) for k,v in labels.value_counts(dropna=False).items() if v>0}
            record.update(source_verification='verified_retained_provider_bytes_and_exact_sample_csv',source_evidence=evidence,
                original_class_counts=original_counts,missing_source_classes=sorted(set(original_counts)-set(record['class_counts'])))
        except Exception as error:
            record.update(source_verification='unresolved',source_error=type(error).__name__+': '+str(error))
        results.append(record)
        atomic_json(out/'datasets.json',results)
        print('AUDIT',name,record['source_verification'],len(frame),flush=True)
        hashes=pd.util.hash_pandas_object(x,index=False).to_numpy()
        for seed in SEEDS:
            for policy in ('stratified','covariate_shift','population_shift'):
                started=time.perf_counter()
                try:
                    splits=get_splits(x,y,policy,n_splits=5,seed=split_seed(name,policy,seed,n_splits=5))
                except (SplitInfeasibleError,ValueError) as error:
                    for family,severity in SHIFT_FAMILIES:
                        if split_policy_for_condition(family)!=policy: continue
                        condition=family if severity==0 else f'{family}_{severity}'
                        for fold in range(1,6):
                            eligibility.append({'dataset':name,'seed':seed,'fold':fold,'condition':condition,'policy':policy,
                                'status':'scientifically_infeasible','reason':type(error).__name__+': '+str(error)})
                    atomic_json(out/'eligibility.json',eligibility)
                    continue
                for fold,(train,test) in enumerate(splits,1):
                    for family,severity in SHIFT_FAMILIES:
                        if split_policy_for_condition(family)!=policy: continue
                        condition=family if severity==0 else f'{family}_{severity}'
                        ty=y.iloc[train].copy()
                        if family in ('label_noise','class_prior_shift'):
                            _,ty=apply_perturbation(pd.DataFrame(index=ty.index),ty,shift_family=family,severity=severity,
                                random_state=corruption_seed(name,policy,seed,fold,condition))
                        train_classes=set(ty.dropna().astype(str)); test_classes=set(y.iloc[test].dropna().astype(str))
                        missing=sorted(test_classes-train_classes)
                        status='scientifically_infeasible' if missing or len(train_classes)<2 else 'split_eligible'
                        reason='held_out_classes_absent_from_training:'+','.join(missing) if missing else ('single_training_class' if len(train_classes)<2 else None)
                        eligibility.append({'dataset':name,'seed':seed,'fold':fold,'condition':condition,'policy':policy,
                            'status':status,'reason':reason,'train_rows':len(train),'test_rows':len(test),
                            'train_classes':sorted(train_classes),'test_classes':sorted(test_classes),
                            'train_class_counts':{str(k):int(v) for k,v in ty.value_counts().items()},
                            'test_class_counts':{str(k):int(v) for k,v in y.iloc[test].value_counts().items()},
                            'auc_status':'complete_possible' if train_classes==test_classes else 'undefined_missing_held_out_class',
                            'duplicate_predictor_hash_overlap':int(np.isin(hashes[test],hashes[train]).sum()),
                            'geometry_seconds_shared_for_five_folds':time.perf_counter()-started,
                            'split_indices_sha256':fingerprint({'train':train.tolist(),'test':test.tolist()})})
                atomic_json(out/'eligibility.json',eligibility)
        del frame,x,y
    atomic_json(out/'audit_summary.json',{'datasets':len(results),'sources_verified':sum(r['source_verification'].startswith('verified') for r in results),
        'intended_precompute_units':len(eligibility),'eligible_units':sum(e['status']=='split_eligible' for e in eligibility),
        'scientifically_infeasible_units':sum(e['status']=='scientifically_infeasible' for e in eligibility),
        'duplicate_limitation':'Exact pandas duplicate rows; fold overlaps use 64-bit predictor hashes, not group independence.',
        'metric_limits':'AUC/PR-AUC can be explicitly undefined on missing held-out classes; estimator feasibility additionally depends on training counts/fit.'})


def freeze(out):
    data=json.loads((out/'datasets.json').read_text()); elig=json.loads((out/'eligibility.json').read_text())
    names=load_dataset_names('config/dataset_list.yaml'); identities={r['dataset']:r['identity'] for r in data}
    planned={(e['dataset'],e['seed'],e['fold'],e['condition']):'scientific_infeasibility:'+e['reason'] for e in elig if e['status']=='scientifically_infeasible'}
    config={'code_identity':collect_code_identity(),'environment_identity':collect_environment_identity(),
        'protocol_version':EVALUATION_PROTOCOL_VERSION,'seed_scheme_version':SEED_SCHEME_VERSION,
        'diagnostics_enabled':True,'diagnostic_max_rows':128,'full_grid_nonexecuting':True,
        'datasets':names,'seeds':SEEDS,'folds':list(range(1,6)),'conditions':SHIFT_FAMILIES,
        'pipelines':PIPELINE_NAMES,'models':CPU_MODELS+GPU_MODELS,'dataset_identities':identities,
        'eligibility_sha256':file_sha256(out/'eligibility.json')}
    path=out/'full_intended_manifest.jsonl.gz'
    if path.exists(): raise FileExistsError('Preserve the existing frozen full manifest')
    counts={'precompute':0,'model':0,'planned_skipped_precompute':0,'planned_skipped_model':0}
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=1,newline='\n') as f:
        for name in names:
            for record in declared_records([name],identities=identities,planned=planned):
                f.write(json.dumps(record,sort_keys=True,separators=(',',':'))+'\n')
                counts[record['stage']]+=1
                if record['planned_skip_reason']: counts['planned_skipped_'+record['stage']]+=1
            print('FREEZE',name,counts['model'],flush=True)
    atomic_json(out/'full_manifest_config.json',config)
    atomic_json(out/'full_manifest_receipt.json',{'path':str(path),'sha256':file_sha256(path),'bytes':path.stat().st_size,'counts':counts,'dispatched':0})
    for name,_,_,_,_ in PILOT_UNITS:
        row=next(r for r in data if r['dataset']==name)
        if not row['source_verification'].startswith('verified'): raise ValueError('Unverified selected pilot source: '+name)
    records=declared_records(names,units=PILOT_UNITS,identities=identities,planned=planned)
    if any(r['planned_skip_reason'] for r in records): raise ValueError('Selected pilot contains an infeasible unit')
    execution=ExecutionConfig(max_workers=2,task_timeout_seconds=1800,run_wall_time_seconds=14400,
        stop_after_tasks=700,cache_policy='rolling',cache_max_bytes=LIMITS['cache_bytes'],
        min_free_bytes=LIMITS['min_free_bytes'],dense_budget_bytes=LIMITS['dense_budget_bytes'])
    pilot_config={**config,'full_grid_nonexecuting':False,'scope':'PILOT calibration only','units':PILOT_UNITS,
        'execution':execution.to_dict(),'limits':LIMITS,'cache_base':str((out/'cache').resolve()),
        'datasets':list(dict.fromkeys(u[0] for u in PILOT_UNITS))}
    run=run_id_for(pilot_config); pilot_config['declared_run_id']=run
    store=ManifestStore(out/'pilot.db',manifest_path=out/'pilot_manifest.jsonl')
    store.create_run(run,pilot_config,records)
    atomic_json(out/'pilot_config.json',pilot_config)
    atomic_json(out/'budget_ledger.json',{'started_utc':None,'execution_seconds':0,'invocations':[],
        'model_attempts':0,'verification_model_fits':0,'limits':LIMITS})
    print('PILOT DECLARED',run,len(records),flush=True)


def pilot(out, interrupt=False):
    from src.coordinator import execute_manifest
    from reproduction.pilot_measurement import profiled_entry, profiled_writer, install_parent_profile, monitor
    config=json.loads((out/'pilot_config.json').read_text()); budget=json.loads((out/'budget_ledger.json').read_text())
    if budget.get('inflight'): raise RuntimeError('Unreconciled invocation; preserve and reconcile budget before resume')
    remaining=LIMITS['execution_seconds']-budget['execution_seconds']
    attempts=LIMITS['model_attempts']-budget['model_attempts']-budget.get('verification_model_fits',0)
    if remaining<=0 or attempts<=0: raise RuntimeError('Predeclared envelope exhausted')
    run=config['declared_run_id']; store=ManifestStore(out/'pilot.db',initialize=False)
    for name,digest in config['code_identity']['source_content_hashes'].items():
        if name.endswith('.py') and file_sha256(name)!=digest:
            raise RuntimeError('Executed Python source differs from the frozen pilot: '+name)
    launches_left=min(attempts,LIMITS['pilot_model_attempts']-budget['model_attempts'])
    if launches_left<=0 and store.state_counts(run)['pending']:
        raise RuntimeError('Pilot attempt envelope exhausted; preserve pending coverage')
    execution=ExecutionConfig(**{**config['execution'],'run_wall_time_seconds':remaining,
        'stop_after_tasks':max(1,launches_left)})
    os.environ['AUTOFE_CACHE_BASE']=config['cache_base']; os.environ['AUTOFE_PROFILE_DIR']=str((out/'profiles').resolve())
    Path('reports/worker_logs').mkdir(parents=True,exist_ok=True)
    budget['inflight']={'utc':now(),'remaining_seconds':remaining,'remaining_attempts':attempts}
    if budget['started_utc'] is None: budget['started_utc']=now()
    atomic_json(out/'budget_ledger.json',budget)
    started=time.monotonic(); interrupted=False
    def stop():
        nonlocal interrupted
        if (out/'STOP').exists(): return True
        if interrupt and not interrupted:
            with store._connect() as c:
                running=c.execute("SELECT COUNT(*) FROM tasks WHERE run_id=? AND stage='model' AND state='running'",(run,)).fetchone()[0]
                completed=c.execute("SELECT COUNT(*) FROM tasks WHERE run_id=? AND stage='model' AND state='completed'",(run,)).fetchone()[0]
            if running and completed>=3:
                interrupted=True; return True
        return interrupted
    install_parent_profile()
    # Preserve the production writer-ready handshake; this wrapper accepts the
    # same five arguments and delegates to the production writer in its child.
    import src.pipeline_runner as runner
    runner.writer_process=profiled_writer
    outcome={}; error=None
    try:
        with monitor(out,store,run):
            outcome=execute_manifest(store,run,out/'pilot_results.jsonl',execution,stop_requested=stop,entry=profiled_entry)
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        raise
    finally:
        elapsed=time.monotonic()-started
        budget.pop('inflight',None); budget['execution_seconds']+=elapsed
        with store._connect() as c:
            budget['model_attempts']=c.execute("SELECT COUNT(*) FROM attempts WHERE run_id=? AND stage='model'",(run,)).fetchone()[0]
        receipt={'utc':now(),'elapsed_seconds':elapsed,'declared_graceful_interrupt':interrupt,
                 'interrupt_triggered':interrupted,'outcome':outcome,'error':error}
        budget['invocations'].append(receipt); atomic_json(out/'budget_ledger.json',budget)
        atomic_json(out/f'invocation_{len(budget["invocations"]):02d}.json',receipt)
    print(json.dumps({'seconds':elapsed,'model_attempts_total':budget['model_attempts'],'outcome':outcome['status'],'states':outcome['state_counts']}),flush=True)


def report(out):
    from src.dataset_statistics import analyze_ledger, AnalysisConfig, write_analysis_bundle
    from src.sensitivity_analysis import analyze_sensitivity, SensitivityConfig, write_sensitivity_bundle
    from src.provenance import build_provenance_package, verify_provenance
    config=json.loads((out/'pilot_config.json').read_text()); run=config['declared_run_id']; ledger=out/'pilot_results.jsonl'
    store=ManifestStore(out/'pilot.db',read_only=True)
    snapshot=store.snapshot(run); atomic_json(out/'final_snapshot.json',snapshot)
    results=[json.loads(r['payload_json']) for r in snapshot['durable_results'] if json.loads(r['payload_json']).get('stage')!='precompute']
    pd.DataFrame(results).to_csv(out/'model_timings.csv',index=False)
    analysis=analyze_ledger(ledger,AnalysisConfig(run_id=run))
    write_analysis_bundle(analysis,out/'pilot_paired_analysis')
    sensitivity=analyze_sensitivity(out/'pilot.db',ledger,run,config=SensitivityConfig(expected_protocol_version=EVALUATION_PROTOCOL_VERSION,expected_seed_scheme_version=SEED_SCHEME_VERSION))
    write_sensitivity_bundle(sensitivity,out/'pilot_sensitivity')
    package=build_provenance_package(output_dir=out/'provenance',repo_root='.',dataset_list_path='config/dataset_list.yaml',
        manifest_db=out/'pilot.db',manifest_path=out/'pilot_manifest.jsonl',ledger_path=ledger,run_id=run,
        analysis_dirs=[out/'pilot_paired_analysis',out/'pilot_sensitivity'])
    verification=verify_provenance(repo_root='.',dataset_list_path='config/dataset_list.yaml',manifest_db=out/'pilot.db',ledger_path=ledger,run_id=run,package_dir=package)
    atomic_json(out/'provenance_verification.json',verification)
    print(json.dumps({'model_rows':len(results),'provenance':verification['overall_status'],'states':store.state_counts(run),'attempts':store.attempt_counts(run)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('audit','freeze','pilot','report')); p.add_argument('--output',type=Path,default=Path('reports/real_data_preflight_20261003'))
    p.add_argument('--interrupt',action='store_true',help='Predeclared one-time graceful interruption after >=3 model results and an active model')
    args=p.parse_args()
    if args.phase=='pilot': pilot(args.output,args.interrupt)
    else: globals()[args.phase](args.output)


if __name__=='__main__': main()
