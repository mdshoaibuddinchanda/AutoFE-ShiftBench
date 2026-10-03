"""Opt-in inclusive/exclusive spans and process/storage sampling for the pilot.

No thread limits, estimator parameters, RNG, arrays, or scientific metadata are
changed. Per-process sidecars keep timing outside immutable scientific inputs.
"""
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import psutil

_installed=False
_context={}
_stack=[]


def emit(value):
    root=Path(os.environ['AUTOFE_PROFILE_DIR']); root.mkdir(parents=True,exist_ok=True)
    with (root/f'process_{os.getpid()}.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'pid':os.getpid(),'utc_epoch':time.time(),**_context,**value},allow_nan=False)+'\n')


def timed(function, stage):
    if getattr(function,'_pilot_timed',False): return function
    @functools.wraps(function)
    def wrapper(*args,**kwargs):
        start=time.perf_counter(); cpu=time.process_time(); frame={'child':0.}; _stack.append(frame)
        proc=psutil.Process(); before=proc.io_counters(); extra={}
        if stage=='atomic_publish' and len(args)>1 and isinstance(args[1],bytes): extra['payload_bytes']=len(args[1])
        if stage=='candidate_scoring_selection':
            cfg=kwargs.get('config'); extra['pipeline_display']=getattr(cfg,'display_identity',None)
        if stage=='fit': emit({'event':'fit_start','stage':stage})
        error=None
        try: return function(*args,**kwargs)
        except BaseException as exc:
            error=type(exc).__name__+': '+str(exc); raise
        finally:
            elapsed=time.perf_counter()-start; after=proc.io_counters(); _stack.pop()
            if _stack: _stack[-1]['child']+=elapsed
            emit({'event':'span','stage':stage,'inclusive_seconds':elapsed,
                  'exclusive_seconds':max(0.,elapsed-frame['child']),'cpu_seconds':time.process_time()-cpu,
                  'rss_end_bytes':proc.memory_info().rss,
                  'io_read_bytes':after.read_bytes-before.read_bytes,'io_write_bytes':after.write_bytes-before.write_bytes,
                  'error':error,**extra})
    wrapper._pilot_timed=True
    return wrapper


def install_parent_profile():
    from src import cache_lifecycle, prepared_inputs, task_manifest
    for module,name,stage in [(cache_lifecycle,'evict_unit','rolling_cleanup'),
        (cache_lifecycle,'verify_eviction','receipt_verification'),
        (cache_lifecycle,'verify_export_bytes','export_readback'),
        (prepared_inputs,'verify_unit','prepared_integrity'),
        (task_manifest.ManifestStore,'export_durable_results','export_recovery')]:
        setattr(module,name,timed(getattr(module,name),stage))


def install_worker_profile():
    global _installed
    if _installed:return
    from src import pipeline_runner as r, prepared_inputs as p, feature_engineering as fe
    from sklearn.compose import ColumnTransformer
    pairs=[(r,'load_csv_dataset','dataset_load'),(r,'get_splits','split_geometry'),
        (r,'apply_training_condition','training_corruption'),(r,'expand_features_with_dfs','candidate_scoring_selection'),
        (fe,'_generate_expressions','candidate_generation'),(fe,'_score_candidates','MI_scoring'),
        (r,'compute_jacobian_diagnostic','FSVA_jacobian'),(r,'compute_empirical_amplification','FSVA_amplification'),
        (r,'validate_jacobian_finite_difference','FSVA_finite_difference'),(r,'compute_distribution_distance','distribution_distances'),
        (r,'compute_classification_metrics','classification_metrics'),(r,'load_pipeline','prepared_input_load'),
        (r,'publish_unit','descriptor_numeric_publish'),(p,'_model_arrays','numeric_preparation'),
        (p,'_publish_array','numeric_atomic_publish'),(r,'atomic_bytes','atomic_publish'),
        (r,'file_sha256','artifact_hash'),(p,'file_sha256','prepared_hash'),
        (ColumnTransformer,'fit_transform','training_preprocessing'),(ColumnTransformer,'transform','preprocessing_transform')]
    for module,name,stage in pairs: setattr(module,name,timed(getattr(module,name),stage))
    from src.model_prediction import predict_labels_and_probabilities
    import src.model_prediction as pred
    pred.predict_labels_and_probabilities=timed(predict_labels_and_probabilities,'prediction')
    original=r.build_model
    def build(*args,**kwargs):
        model=original(*args,**kwargs); model.fit=timed(model.fit,'fit'); return model
    r.build_model=build
    _installed=True


def profiled_entry(task,queue):
    global _context
    _context={k:task.get(k) for k in ('scientific_task_id','attempt_id','dataset','seed','fold','condition','pipeline','model','stage')}
    install_worker_profile()
    from src.coordinator import task_entry
    emit({'event':'task_start'})
    return timed(task_entry,'task_total')(task,queue)


def profiled_writer(queue,ledger,db,run,ready_event=None):
    global _context
    _context={'role':'writer','run_id':run}
    from src.task_manifest import ManifestStore
    from src.pipeline_runner import writer_process
    ManifestStore.commit_result=timed(ManifestStore.commit_result,'authoritative_commit')
    ManifestStore.note_ledger_export=timed(ManifestStore.note_ledger_export,'fsynced_export_ack')
    ManifestStore.export_durable_results=timed(ManifestStore.export_durable_results,'writer_export_recovery')
    writer_process(queue,ledger,db,run,ready_event)


@contextmanager
def monitor(out,store,run):
    from reproduction.real_data_preflight import footprint
    from src.protocol import cache_root
    from src.artifact_integrity import atomic_json
    done=threading.Event(); root=cache_root(); out=Path(out); processes={}
    def sample():
        with (out/'resource_samples.jsonl').open('a',encoding='utf-8') as f:
            tick=0
            while not done.is_set():
                children=psutil.Process().children(recursive=True); values=[]
                for child in children:
                    try:
                        proc=processes.setdefault((child.pid,child.create_time()),child)
                        values.append({'pid':proc.pid,'created':proc.create_time(),'rss':proc.memory_info().rss,
                            'cpu_percent':proc.cpu_percent(),'cpu_seconds':sum(proc.cpu_times()[:2]),
                            'io':dict(proc.io_counters()._asdict()),'threads':proc.num_threads()})
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                row={'utc_epoch':time.time(),'run_id':run,'machine_memory':dict(psutil.virtual_memory()._asdict()),
                     'machine_cpu_percent':psutil.cpu_percent(),'workers_and_writer':values,
                     'disk_free_bytes':__import__('shutil').disk_usage(out).free}
                if tick%2==0:
                    row['cache']=footprint(root)
                    try:
                        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                        row['gpu']=gpu.stdout.strip()
                    except (OSError,subprocess.TimeoutExpired): row['gpu']='unmeasured'
                    atomic_json(out/'live_status.json',{'utc_epoch':row['utc_epoch'],'counts':store.state_counts(run),'attempts':store.attempt_counts(run),
                        'cache':row['cache'],'disk_free_bytes':row['disk_free_bytes'],'gpu':row['gpu']})
                f.write(json.dumps(row)+'\n'); f.flush(); tick+=1
                done.wait(1.)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try: yield
    finally:
        done.set();thread.join(5)
