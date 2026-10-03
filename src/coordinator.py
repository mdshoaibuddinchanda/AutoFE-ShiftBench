"""Dependency-ready process supervision with explicit attempt and deadline units."""
from __future__ import annotations
import multiprocessing as mp
import time
import queue as queue_module
from pathlib import Path
import psutil
from src.task_manifest import ManifestStore

SCHEDULER_VERSION='dependency_ready_supervised_attempts_v1'


def task_entry(task,queue):
    import os
    os.environ['AUTOFE_DENSE_BUDGET_BYTES']=str(task['dense_budget_bytes'])
    from src.pipeline_runner import precompute_unit,_train_process_entry
    if task['stage'] == 'precompute':
        precompute_unit(task)
    else:
        _train_process_entry(task,queue)


def final_status(counts,*,stopped=False,writer_failed=False):
    if writer_failed: return 'failed'
    if stopped: return 'stopped'
    if counts['pending'] or counts['running']: return 'unfinished'
    if counts['failed'] or counts['timeout'] or counts['skipped']: return 'finished_with_errors'
    return 'completed_successfully'


def execute_manifest(store,run_id,ledger,config,*,stop_requested=lambda:False,entry=task_entry,writer_target=None):
    from src.pipeline_runner import writer_process,_record_manifest_failure
    context=mp.get_context('spawn')
    manager=context.Manager()
    queue=manager.Queue(maxsize=max(2,config.max_workers*2))
    writer=context.Process(target=writer_target or writer_process,args=(queue,ledger,str(store.db_path),run_id))
    active={}
    started=time.monotonic()
    dispatched=0
    stopped=False
    writer_failed=False
    deadline=None if config.run_wall_time_seconds is None else started+config.run_wall_time_seconds
    store.resume(run_id)
    writer.start()
    try:
        while True:
            now=time.monotonic()
            if not writer.is_alive():
                writer_failed=True
                break
            if stop_requested() or (deadline is not None and now >= deadline):
                stopped=True
                break # hard run deadline/signal; attempts stay explicitly recoverable
            for task_id,(process,task,launched,exited_at) in list(active.items()):
                state=store.get_task(run_id,task_id)
                timed_out=config.task_timeout_seconds is not None and now-launched >= config.task_timeout_seconds
                if timed_out and state['state'] == 'running':
                    if process.is_alive():
                        process.terminate()
                    process.join(5)
                    _record_manifest_failure(task,task['attempt_id'],failure_class='timeout',retry=task['retry'])
                elif not process.is_alive() and state['state'] == 'running':
                    if process.exitcode != 0:
                        _record_manifest_failure(task,task['attempt_id'],failure_class='worker_exception',exception=RuntimeError(f'Worker exit {process.exitcode}'),retry=task['retry'])
                    elif exited_at is None:
                        active[task_id]=(process,task,launched,now)
                        continue # writer may still be committing its bounded queue
                    elif now-exited_at > 10:
                        _record_manifest_failure(task,task['attempt_id'],failure_class='result_write_failure',exception=RuntimeError('Worker exited without an authoritative commit'),retry=task['retry'])
                    else:
                        continue
                elif process.is_alive():
                    continue
                process.join()
                process.close()
                del active[task_id]
            budget_reached=config.stop_after_tasks is not None and dispatched >= config.stop_after_tasks
            if budget_reached and not active:
                stopped=True
                break
            if not budget_reached and len(active) < config.max_workers and psutil.virtual_memory().percent < 85:
                gpu_active=any(task['stage']=='model' and task.get('model') in ('xgboost','catboost') for _,task,_,_ in active.values())
                for record in store.ready_tasks(run_id,limit=max(64,config.max_workers*2)):
                    if len(active) >= config.max_workers: break
                    if record['scientific_task_id'] in active: continue
                    if record['stage'] == 'model' and config.stop_after_tasks is not None and dispatched >= config.stop_after_tasks: break
                    gpu=record['stage']=='model' and record.get('model') in ('xgboost','catboost')
                    if gpu and gpu_active: continue
                    attempt=store.claim_task(run_id,record['scientific_task_id'],worker_id='supervised-parent',timeout_seconds=config.task_timeout_seconds)
                    if attempt is None: continue
                    run_config=store.run_config(run_id)
                    from src.seeding import stable_seed
                    from src.fsva import DEFAULT_PERTURBATION_MAGNITUDES
                    task={**record,'dataset_name':record['dataset'],'data_path':Path(record['data_path']),
                        'dense_budget_bytes':min(1024**3,int(psutil.virtual_memory().available*.4/config.max_workers)),
                        'manifest_db':str(store.db_path),'run_id':run_id,'attempt_id':attempt,
                        'task_timeout_seconds':config.task_timeout_seconds,'retry':record['attempt_count']+1 < config.max_attempts,
                        'diagnostics_enabled':run_config.get('diagnostics_enabled',False),
                        'diagnostic_config':{'max_rows':run_config.get('diagnostic_max_rows',128),'random_state':stable_seed('fsva_diagnostic',{
                            'dataset':record['dataset'],'split_policy':record['split_policy'],'seed':record['seed'],'fold':record['fold'],'condition':record['condition']}),
                            'magnitudes':list(DEFAULT_PERTURBATION_MAGNITUDES)}}
                    process=context.Process(target=entry,args=(task,queue))
                    process.start()
                    active[record['scientific_task_id']]=(process,task,time.monotonic(),None)
                    if record['stage'] == 'model': dispatched+=1
                    gpu_active=gpu_active or gpu
            if not active:
                counts=store.state_counts(run_id)
                if not counts['pending'] or not store.ready_tasks(run_id,limit=1): break
                if psutil.virtual_memory().percent >= 85:
                    # Resource backpressure cannot become an unbounded wait.
                    stopped=True
                    break
            time.sleep(.02)
    except BaseException:
        writer_failed=True
        raise
    finally:
        for process,task,_,_ in active.values():
            if process.is_alive(): process.terminate()
            process.join(5)
            state=store.get_task(run_id,task['scientific_task_id'])
            if state and state['state']=='running':
                _record_manifest_failure(task,task['attempt_id'],failure_class='coordinator_crash',retry=True)
            process.close()
        if writer.is_alive():
            try:
                queue.put('DONE',timeout=5)
            except queue_module.Full:
                writer_failed=True
            writer.join(15)
            if writer.is_alive():
                writer.terminate()
                writer.join(5)
                writer_failed=True
        writer_failed=writer_failed or writer.exitcode != 0
        writer.close()
        manager.shutdown()
        # SQLite commits survive a failed export or writer. Reconstruct independently.
        store.export_durable_results(run_id,ledger)
        status=final_status(store.state_counts(run_id),stopped=stopped,writer_failed=writer_failed)
        store.set_run_status(run_id,status)
    return {'status':status,'state_counts':store.state_counts(run_id),'attempt_counts':store.attempt_counts(run_id),
        'model_attempts_dispatched_this_invocation':dispatched,'stop_limit_unit':'model attempt launches including retries',
        'run_deadline_policy':'hard during precompute/model; bounded cleanup/export follows',
        'cache_retention_policy':'retain all resumable inputs and committed diagnostic evidence; no automatic eviction'}
