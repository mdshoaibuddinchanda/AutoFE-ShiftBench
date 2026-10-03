"""Two bounded model-only repeats using already verified prepared dependencies."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--prepared-run-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.source_root.resolve();prepared=args.prepared_run_directory.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);sys.path.insert(0,str(root));os.chdir(prepared)
    import psutil
    from src.task_manifest import ManifestStore,ExecutionConfig
    from src.coordinator import execute_manifest
    original=ManifestStore('production.db',read_only=True).snapshot('production')
    records=[json.loads(row['payload_json']) for row in original['tasks']]
    parent=next(record for record in records if record['stage']=='precompute')
    evidence=json.loads(next(row['payload_json'] for row in original['durable_results'] if row['scientific_task_id']==parent['scientific_task_id']))
    measurements=[];process=psutil.Process()
    for repeat in range(2):
        store=ManifestStore(output/f'repeat_{repeat}.db');run_id=f'warm_{repeat}';config=ExecutionConfig(max_workers=2,task_timeout_seconds=60)
        store.create_run(run_id,{'execution':config.to_dict(),'durability_protocol':'sqlite_result_outbox_v2'},records)
        attempt=store.claim_task(run_id,parent['scientific_task_id'])
        store.commit_result(run_id,parent['scientific_task_id'],attempt,{**evidence,'run_id':run_id,'attempt_id':attempt})
        done=threading.Event();samples=[]
        def poll():
            while not done.is_set():
                rss=process.memory_info().rss
                for child in process.children(recursive=True):
                    try:rss+=child.memory_info().rss
                    except psutil.Error:pass
                samples.append(rss);done.wait(.02)
        monitor=threading.Thread(target=poll);monitor.start();start=time.perf_counter()
        outcome=execute_manifest(store,run_id,output/f'repeat_{repeat}.jsonl',config)
        elapsed=time.perf_counter()-start;done.set();monitor.join()
        if outcome['status']!='completed_successfully':raise RuntimeError(outcome)
        measurements.append({'repeat':repeat,'seconds':elapsed,'model_tasks_executed':4,'new_precompute_execution':0,'import_spawn_included':True,
            'peak_aggregate_rss_sampled_bytes':max(samples),'rss_sampling_interval_seconds':.02,'outcome':outcome})
    result={'schema':'bounded_warm_model_execution_v1','synthetic_only':True,'source_root':str(root),'measurements':measurements,
        'precompute_evidence_policy':'Each controlled run starts with an explicitly committed verified identical prepared dependency; no precompute worker runs.'}
    (output/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'seconds':[row['seconds'] for row in measurements]}))


if __name__=='__main__':main()
