"""Dependency-ready process supervision with explicit attempt and deadline units."""
from __future__ import annotations
import multiprocessing as mp
import time
import queue as queue_module
from pathlib import Path
import psutil
from collections import deque
from src.task_manifest import ManifestStore

SCHEDULER_VERSION='dependency_ready_verified_rolling_v3'


def task_entry(task,queue):
    import os
    os.environ['AUTOFE_DENSE_BUDGET_BYTES']=str(task['dense_budget_bytes'])
    from src.pipeline_runner import precompute_unit,_train_process_entry
    if task['stage'] == 'precompute':
        precompute_unit(task)
    else:
        _train_process_entry(task,queue)


def worker_service(commands,control,results,entry,max_tasks,rss_growth):
    """One owned task at a time; deadline enforcement stays in the parent."""
    import os
    import gc
    baseline=None
    for index in range(max_tasks):
        task=commands.get()
        if task is None:return
        entry(task,results)
        gc.collect()
        rss=psutil.Process().memory_info().rss
        if baseline is None:baseline=rss
        recycle=index+1>=max_tasks or rss-baseline>rss_growth or task.get('model') in ('xgboost','catboost')
        control.put({'scientific_task_id':task['scientific_task_id'],'attempt_id':task['attempt_id'],
            'pid':os.getpid(),'rss_bytes':rss,'baseline_rss_bytes':baseline,'tasks_executed':index+1,'recycle':recycle,
            'descendants':[{'pid':child.pid,'create_time':child.create_time()} for child in psutil.Process().children(recursive=True)]})
        if recycle:return


def final_status(counts,*,stopped=False,writer_failed=False):
    if writer_failed: return 'failed'
    if stopped: return 'stopped'
    if counts['pending'] or counts['running']: return 'unfinished'
    if counts['failed'] or counts['timeout'] or counts['skipped']: return 'finished_with_errors'
    return 'completed_successfully'


def execute_manifest(store,run_id,ledger,config,*,stop_requested=lambda:False,entry=task_entry,writer_target=None):
    from src.artifact_integrity import artifact_lock
    from src.protocol import cache_root
    # All production coordinators, including retain mode, take this lease.
    # A second benchmark cannot open or regenerate shared cache files during
    # another run's deletion barrier. OS ownership dies with the coordinator.
    with artifact_lock(cache_root() / '.coordinator.lock', timeout_seconds=1):
        from src.cache_lifecycle import verify_registered_workers
        verify_registered_workers(cache_root())
        import os
        settings = {'AUTOFE_MIN_FREE_BYTES': str(config.min_free_bytes),
                    'AUTOFE_DISK_WRITE_LOCK': str((cache_root() / '.disk-write.lock').resolve())}
        previous = {key: os.environ.get(key) for key in settings}
        os.environ.update(settings)
        try:
            return _execute_manifest(store, run_id, ledger, config,
                stop_requested=stop_requested, entry=entry, writer_target=writer_target)
        finally:
            for key, value in previous.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value


def _execute_manifest(store,run_id,ledger,config,*,stop_requested,entry,writer_target):
    from src.pipeline_runner import writer_process,_record_manifest_failure
    from src.prepared_inputs import verify_unit
    from src.protocol import cache_root
    from src.cache_lifecycle import (CacheCheckFailed, CacheNotReady, cache_bytes, capacity_problem,
        completed_unit_rows, evict_unit, verify_eviction, register_worker, POLICY_VERSION)
    rolling = config.cache_policy == 'rolling'
    root = cache_root().resolve()
    cache_stop_reason = None
    evictions = []
    units = store.preparation_units(run_id) if rolling else []
    unit_cursor = 0
    current_unit = None
    unit_processes = {}
    measured_cache = cache_bytes(root) if rolling else None
    last_capacity_check = time.monotonic()
    export_wait_started = None
    repair_descriptors={}
    import json
    started=time.monotonic()
    deadline=None if config.run_wall_time_seconds is None else started+config.run_wall_time_seconds
    def check_cancel():
        if stop_requested() or (deadline is not None and time.monotonic() >= deadline):
            raise TimeoutError('Run stopped during prepared-artifact verification')
    try:
        for row in store.completed_precompute_results(run_id):
            check_cancel()
            descriptor=json.loads(row['payload_json']).get('prepared_descriptor')
            if descriptor is None: continue # controlled/legacy evidence remains explicitly unverified
            receipt = store.cache_eviction(run_id, row['scientific_task_id'])
            if receipt and receipt['state'] in ('intent', 'completed'):
                verify_eviction(store, run_id, row['scientific_task_id'], allow_intent=True)
                continue # Intentional eviction is not corruption or an exact-repair attempt.
            try:
                verify_unit(descriptor,check_cancel=check_cancel)
            except (OSError,ValueError,KeyError) as error:
                if isinstance(error,TimeoutError): raise
                store.reopen_precompute_for_artifact_repair(run_id,row['scientific_task_id'],max_repairs=config.max_artifact_repairs_per_unit)
                repair_descriptors[row['scientific_task_id']]=descriptor
    except TimeoutError:
        store.set_run_status(run_id,'stopped')
        return {'status':'stopped','state_counts':store.state_counts(run_id),'attempt_counts':store.attempt_counts(run_id),
            'model_attempts_dispatched_this_invocation':0,'stop_reason':'artifact verification deadline/signal'}
    except BaseException:
        store.set_run_status(run_id,'failed')
        raise
    store.resume(run_id)
    run_config=store.run_config(run_id)
    context=mp.get_context('spawn')
    manager=None
    try:
        manager=context.Manager()
        queue=manager.Queue(maxsize=max(2,config.max_workers*2))
        control=manager.Queue(maxsize=max(2,config.max_workers*2))
        writer_ready=context.Event()
        writer_args=(queue,ledger,str(store.db_path),run_id)
        if writer_target is None:writer_args=(*writer_args,writer_ready)
        else:writer_ready.set()
        writer=context.Process(target=writer_target or writer_process,args=writer_args)
    except BaseException:
        if manager is not None:manager.shutdown()
        store.set_run_status(run_id,'failed')
        raise
    active={}
    workers={}
    acknowledged={}
    resource_samples=deque(maxlen=128)
    spawned=recycled=0
    def dispose(process):
        if process.is_alive():process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        if process.is_alive():
            raise CacheCheckFailed('Worker did not exit; no cleanup permitted')
        workers.pop(process.pid,None)
        process.close()

    def release_idle_workers():
        release_started=time.monotonic()
        released_count=len(workers)
        for worker in list(workers.values()):
            process = worker['process']
            if not worker['idle']:
                raise CacheCheckFailed('Active worker at deletion barrier')
            if process.is_alive():
                for child in psutil.Process(process.pid).children(recursive=True):
                    owner = register_worker(root, child, run_id=run_id, unit_id=current_unit)
                    unit_processes[(child.pid, owner['create_time'])] = owner
                worker['commands'].put(None)
                process.join(10)
            if process.is_alive():
                # Do not use termination as proof that a healthy unit is safe.
                raise CacheCheckFailed('Worker failed graceful release; cache retained')
            workers.pop(process.pid, None)
            process.close()
        from src.performance import event
        event('worker_release',time.monotonic()-release_started,run_id=run_id,unit_id=current_unit,workers=released_count)
    dispatched=0
    stopped=False
    writer_failed=False
    graceful_requested=False
    try:
        writer.start()
        while True:
            now=time.monotonic()
            if not writer.is_alive():
                writer_failed=True
                break
            if deadline is not None and now >= deadline:
                stopped=True
                break # hard run deadline/signal; attempts stay explicitly recoverable
            if stop_requested():
                if not config.graceful_stop:
                    stopped=True
                    break
                graceful_requested=True
            if not writer_ready.is_set():
                time.sleep(.02)
                continue # no model deadlines start while initial export is recovering
            if rolling:
                now_check = time.monotonic()
                if now_check - last_capacity_check >= 10:
                    measured_cache = cache_bytes(root)
                    last_capacity_check = now_check
                problem = capacity_problem(root, config, measured_bytes=None)
                if problem:
                    cache_stop_reason = problem
                    stopped = True
                    break
            while True:
                try:message=control.get_nowait()
                except queue_module.Empty:break
                tid=message['scientific_task_id']
                if tid in active and active[tid][0].pid==message['pid'] and active[tid][1]['attempt_id']==message['attempt_id']:
                    acknowledged[tid]=message
                    resource_samples.append(message)
                    for child in message.get('descendants', []):
                        unit_processes[(child['pid'], child['create_time'])] = child
                        register_worker(root, child, run_id=run_id, unit_id=current_unit)
            states=store.task_states(run_id,active)
            for task_id,(process,task,launched,exited_at) in list(active.items()):
                state={'state':states[task_id]}
                timed_out=config.task_timeout_seconds is not None and now-launched >= config.task_timeout_seconds
                ack=acknowledged.get(task_id)
                if timed_out and (ack is None or state['state']=='running'):
                    if process.is_alive():
                        process.terminate()
                    process.join(5)
                    if state['state']=='running':_record_manifest_failure(task,task['attempt_id'],failure_class='timeout',retry=task['retry'])
                elif (ack is not None or not process.is_alive()) and state['state'] == 'running':
                    if not process.is_alive() and process.exitcode != 0:
                        _record_manifest_failure(task,task['attempt_id'],failure_class='worker_exception',exception=RuntimeError(f'Worker exit {process.exitcode}'),retry=task['retry'])
                    elif exited_at is None:
                        active[task_id]=(process,task,launched,now)
                        continue # writer may still be committing its bounded queue
                    elif now-exited_at > 10:
                        _record_manifest_failure(task,task['attempt_id'],failure_class='result_write_failure',exception=RuntimeError('Worker exited without an authoritative commit'),retry=task['retry'])
                    else:
                        continue
                elif ack is None and process.is_alive():
                    continue
                if process.is_alive() and ack is not None and not ack['recycle']:
                    workers[process.pid]['idle']=True
                else:
                    if ack and ack['recycle']:recycled+=1
                    dispose(process)
                acknowledged.pop(task_id,None)
                del active[task_id]
            budget_reached=config.stop_after_tasks is not None and dispatched >= config.stop_after_tasks
            for worker in list(workers.values()):
                if worker['idle'] and not worker['process'].is_alive():dispose(worker['process'])
            if graceful_requested and not active:
                stopped=True
                break # retain unfinished unit; acknowledged results survive resume
            if rolling and not active:
                if current_unit is not None:
                    snapshot = store.unit_snapshot(run_id, current_unit)
                    if any(row['state'] in ('failed', 'timeout', 'skipped') for row in snapshot):
                        cache_stop_reason = 'Unit has failed, timed-out, or skipped work; cache retained'
                        stopped = True
                        break
                    if all(row['state'] == 'completed' for row in snapshot):
                        try:
                            completed_unit_rows(store, run_id, current_unit)
                        except CacheNotReady:
                            if export_wait_started is None: export_wait_started = time.monotonic()
                            if time.monotonic() - export_wait_started > 30:
                                raise CacheCheckFailed('Writer did not confirm saved results within 30 seconds; cache retained')
                            time.sleep(.02)
                            continue
                        export_wait_started = None
                        release_idle_workers()
                        evictions.append(evict_unit(store, run_id, current_unit, ledger, root,
                            exited_processes=list(unit_processes.values())))
                        current_unit = None
                        unit_processes.clear()
                        measured_cache = cache_bytes(root)
                if current_unit is None and not budget_reached:
                    while unit_cursor < len(units):
                        candidate = units[unit_cursor]['scientific_task_id']
                        unit_cursor += 1
                        task = store.get_task(run_id, candidate)
                        if task['planned_skip_reason']:
                            continue
                        receipt = store.cache_eviction(run_id, candidate)
                        if receipt and receipt['state'] == 'completed':
                            verify_eviction(store, run_id, candidate)
                            continue
                        snapshot = store.unit_snapshot(run_id, candidate)
                        if any(row['state'] in ('failed', 'timeout', 'skipped') for row in snapshot):
                            cache_stop_reason = 'Unfinished/failed unit needs recovery; no cache deleted'
                            stopped = True
                            break
                        problem = capacity_problem(root, config, measured_bytes=measured_cache)
                        if problem:
                            cache_stop_reason = problem
                            stopped = True
                            break
                        current_unit = candidate
                        break
                    if stopped: break
                    if current_unit is None: break
            if budget_reached and not active:
                stopped=True
                break
            if not budget_reached and not graceful_requested and len(active) < config.max_workers and psutil.virtual_memory().percent < 85:
                gpu_active=any(task['stage']=='model' and task.get('model') in ('xgboost','catboost') for _,task,_,_ in active.values())
                for record in store.ready_tasks(run_id,limit=max(64,config.max_workers*2),unit_id=current_unit if rolling else None):
                    if len(active) >= config.max_workers: break
                    if record['scientific_task_id'] in active: continue
                    if record['stage'] == 'model' and config.stop_after_tasks is not None and dispatched >= config.stop_after_tasks: break
                    gpu=record['stage']=='model' and record.get('model') in ('xgboost','catboost')
                    if gpu and gpu_active: continue
                    attempt=store.claim_task(run_id,record['scientific_task_id'],worker_id='supervised-parent',timeout_seconds=config.task_timeout_seconds)
                    if attempt is None: continue
                    from src.seeding import stable_seed
                    from src.fsva import DEFAULT_PERTURBATION_MAGNITUDES
                    task={**record,'dataset_name':record['dataset'],'data_path':Path(record['data_path']),
                        'repair_descriptor':repair_descriptors.get(record['scientific_task_id']),
                        'dense_budget_bytes':min(config.dense_budget_bytes,int(psutil.virtual_memory().available*.4/config.max_workers)),
                        'manifest_db':str(store.db_path),'run_id':run_id,'attempt_id':attempt,
                        'task_timeout_seconds':config.task_timeout_seconds,'retry':record['attempt_count']+1 < config.max_attempts,
                        'diagnostics_enabled':run_config.get('diagnostics_enabled',False),
                        'diagnostic_config':{'max_rows':run_config.get('diagnostic_max_rows',128),'random_state':stable_seed('fsva_diagnostic',{
                            'dataset':record['dataset'],'split_policy':record['split_policy'],'seed':record['seed'],'fold':record['fold'],'condition':record['condition']}),
                            'magnitudes':list(DEFAULT_PERTURBATION_MAGNITUDES)}}
                    if record['stage'] == 'model': dispatched+=1
                    process=None
                    try:
                        idle=next((value for value in workers.values() if value['idle'] and value['process'].is_alive()),None)
                        if idle is None:
                            commands=manager.Queue(maxsize=1)
                            process=context.Process(target=worker_service,args=(commands,control,queue,entry,config.worker_max_tasks,config.worker_rss_growth_bytes))
                            spawn_started=time.monotonic()
                            process.start()
                            from src.performance import event
                            event('worker_spawn_call',time.monotonic()-spawn_started,run_id=run_id,unit_id=current_unit,worker_pid=process.pid)
                            idle={'process':process,'commands':commands,'idle':True}
                            workers[process.pid]=idle
                            owner = register_worker(root, psutil.Process(process.pid), run_id=run_id, unit_id=current_unit)
                            if rolling:
                                created = owner['create_time']
                                unit_processes[(process.pid, created)] = {'pid': process.pid, 'create_time': created}
                            spawned+=1
                        process=idle['process']
                        idle['idle']=False
                        active[record['scientific_task_id']]=(process,task,time.monotonic(),None)
                        idle['commands'].put(task)
                    except BaseException as error:
                        if record['scientific_task_id'] not in active:
                            _record_manifest_failure(task,task['attempt_id'],failure_class='worker_exception',exception=error,retry=task['retry'])
                            if process is not None and process.pid is None:process.close()
                        raise
                    gpu_active=gpu_active or gpu
            if not active:
                counts=store.state_counts(run_id)
                if not counts['pending'] or not store.ready_tasks(run_id,limit=1,unit_id=current_unit if rolling else None):
                    if rolling and current_unit is not None:
                        # A newly completed unit must pass its barrier on the next loop.
                        if all(row['state'] == 'completed' for row in store.unit_snapshot(run_id, current_unit)):
                            continue
                    break
                if psutil.virtual_memory().percent >= 85:
                    # Resource backpressure cannot become an unbounded wait.
                    stopped=True
                    break
            time.sleep(.02)
    except CacheCheckFailed as error:
        cache_stop_reason = str(error)
        stopped = True
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
        for worker in list(workers.values()):dispose(worker['process'])
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
        final_counts=store.state_counts(run_id)
        with store._connect() as connection:
            planned=connection.execute("SELECT COUNT(*) FROM tasks WHERE run_id=? AND state='skipped' AND planned_skip_reason IS NOT NULL",(run_id,)).fetchone()[0]
        effective_counts={**final_counts,'skipped':final_counts['skipped']-planned}
        status=final_status(effective_counts,stopped=stopped,writer_failed=writer_failed)
        store.set_run_status(run_id,status)
    return {'status':status,'state_counts':store.state_counts(run_id),'attempt_counts':store.attempt_counts(run_id),
        'model_attempts_dispatched_this_invocation':dispatched,'stop_limit_unit':'model attempt launches including retries',
        'workers_spawned':spawned,'workers_recycled':recycled,'worker_resource_samples':list(resource_samples),
        'run_deadline_policy':'hard during precompute/model; bounded cleanup/export follows',
        'cache_retention_policy':POLICY_VERSION if rolling else 'retain all resumable inputs and committed diagnostic evidence; no automatic eviction',
        'cache_evictions':evictions, 'cache_stop_reason':cache_stop_reason,
        'planned_skipped_tasks':planned, 'graceful_stop_drained':graceful_requested}
