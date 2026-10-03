"""Bounded synthetic reference/profile harness; never acquires real datasets.

Run this same file with --source-root pointing to a retained corrected checkout.
Outputs, complete candidate histories and inputs stay under ignored reports/.
"""
from __future__ import annotations
import argparse
import cProfile
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--production',action='store_true')
    parser.add_argument('--pipelines',default='Raw,AutoFE_Baseline',help='Comma-separated identities or all; full diagnostic budgets retained')
    args=parser.parse_args()
    root=args.source_root.resolve()
    output=args.output.resolve()
    if output.exists(): raise FileExistsError('Retained measurements must not be overwritten')
    output.mkdir(parents=True)
    sys.path.insert(0,str(root))
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    start=time.perf_counter()
    import numpy as np
    import pandas as pd
    import psutil
    from src import pipeline_runner as runner
    from src.artifact_integrity import file_sha256,frame_identity
    from src.feature_engineering import expand_features_with_dfs,DFSConfig
    from src.fsva import compute_empirical_amplification,compute_jacobian_diagnostic,validate_jacobian_finite_difference
    from src.operator_registry import expression_from_dict,raw_expression
    from src.dataset_statistics import AnalysisConfig
    from src.reporting import report_inputs
    from src.stats_analysis import cliffs_delta
    from src.task_manifest import ManifestStore,ExecutionConfig,build_task_records
    from src.coordinator import execute_manifest
    from tests.test_dataset_statistics import _record
    from threadpoolctl import threadpool_info
    selected_pipelines=list(runner.PIPELINE_CONFIGS) if args.pipelines=='all' else args.pipelines.split(',')
    if not selected_pipelines or set(selected_pipelines)-set(runner.PIPELINE_CONFIGS):raise ValueError('Unknown pipeline selection')
    result={'schema':'bounded_corrected_comparison_v1','synthetic_only':True,
        'import_seconds':time.perf_counter()-start,'source_root':str(root),'stages':[],
        'threadpools':threadpool_info(),'fixture':{'rows':480,'numeric_columns':8,'category_levels':4,'pipelines':selected_pipelines,
        'diagnostic_rows':128,'finite_difference_rows':32,'magnitudes':[.001,.01,.05],'analysis':AnalysisConfig().to_dict()}}
    process=psutil.Process()
    def measured(stage,call):
        samples=[]
        done=threading.Event()
        before=process.io_counters()
        def poll():
            while not done.is_set():
                total=process.memory_info().rss
                for child in process.children(recursive=True):
                    try: total+=child.memory_info().rss
                    except psutil.Error: pass
                samples.append(total)
                done.wait(.02)
        monitor=threading.Thread(target=poll,daemon=True);monitor.start()
        start=time.perf_counter()
        try: value=call()
        finally:
            elapsed=time.perf_counter()-start
            done.set();monitor.join()
        after=process.io_counters()
        result['stages'].append({'stage':stage,'seconds':elapsed,'peak_aggregate_rss_sampled_bytes':max(samples,default=0),
            'rss_sampling_interval_seconds':.02,'process_read_bytes':after.read_bytes-before.read_bytes,'process_write_bytes':after.write_bytes-before.write_bytes})
        return value
    def science(prepared):
        pipelines,ytr,yte,encoder,metadata,_=prepared
        fields=('selected_feature_identities','selected_feature_expressions','feature_metadata','selection_history','candidate_count','base_feature_count','eligible_base_feature_count','num_original','num_generated','num_selected','retained_raw_count','retained_generated_count','actual_estimator_input_dimension','selection_seed','split_seed','corruption_seed','training_distribution_distance','held_out_distribution_distance','diagnostic_status','dependency_signature','discrete_base_features')
        selected={}
        for name in result['fixture']['pipelines']:
            meta=metadata[name]
            selected[name]={'train':pipelines[name][0],'test':pipelines[name][1],'metadata':{key:meta[key] for key in fields if key in meta}}
            if meta.get('diagnostic_path'):
                selected[name]['diagnostics']=json.loads(Path(meta['diagnostic_path']).read_text())
        return {'pipelines':selected,'labels_train':ytr,'labels_test':yte,'classes':encoder.classes_}
    for repeat in range(args.repeats):
        directory=output/f'repeat_{repeat}'
        directory.mkdir();os.chdir(directory)
        Path('reports/worker_logs').mkdir(parents=True)
        rng=np.random.default_rng(7261)
        frame=pd.DataFrame(rng.normal(size=(480,8)),columns=[f'x{i}' for i in range(8)])
        frame['category']=np.array(['a','b','c','d'])[np.arange(480)%4]
        frame['target']=np.where(frame.x0+frame.x1*.3>0,'positive','negative')
        frame.to_csv('synthetic.csv',index=False)
        result.setdefault('data_sha256',file_sha256('synthetic.csv'))
        kwargs={'data_path':Path('synthetic.csv'),'dataset_name':'synthetic','seed':42,'fold':1,'condition':'gaussian_noise_0.05','shift_family':'gaussian_noise','severity':.05,
            'diagnostics_enabled':True,'diagnostic_config':{'max_rows':128,'random_state':1973}}
        if 'selected_pipelines' in inspect.signature(runner.get_data_splits).parameters:
            kwargs['selected_pipelines']=result['fixture']['pipelines']
        counts={}
        wrapped={}
        for name in ('load_csv_dataset','_build_preprocessor','expand_features_with_dfs','compute_jacobian_diagnostic','compute_empirical_amplification','validate_jacobian_finite_difference'):
            original=getattr(runner,name)
            wrapped[name]=original
            def counted(*a,_name=name,_call=original,**kw):
                counts[_name]=counts.get(_name,0)+1
                return _call(*a,**kw)
            setattr(runner,name,counted)
        cold=measured('cold_preparation',lambda:runner.get_data_splits(**kwargs))
        result['stages'][-1]['call_counts']=dict(counts)
        result['stages'][-1]['artifact_bytes']=sum(p.stat().st_size for p in Path('data').rglob('*') if p.is_file())
        with Path('science.pkl').open('wb') as handle: pickle.dump(science(cold),handle)
        counts.clear()
        warm=measured('warm_preparation',lambda:runner.get_data_splits(**kwargs))
        result['stages'][-1]['call_counts']=dict(counts)
        for name,original in wrapped.items(): setattr(runner,name,original)
        # Candidate scaling and all diagnostics are separate from end-to-end cold work.
        scale=pd.DataFrame(np.random.default_rng(44).normal(size=(1600,16)),columns=[f'z{i}' for i in range(16)])
        feature=measured('candidate_scale',lambda:expand_features_with_dfs(scale.iloc[:1200],scale.iloc[1200:],np.arange(1200)%2,DFSConfig(monitor_ram=False)))
        with Path('candidate_scale.pkl').open('wb') as handle: pickle.dump(feature,handle)
        expr=[expression_from_dict(x['expression']) for x in feature[2]['selected_feature_expressions']]
        raw=[raw_expression(c) for c in scale]
        diag=measured('diagnostics',lambda:{'jacobian':compute_jacobian_diagnostic(scale,expr,raw_control_expressions=raw,max_rows=128,random_state=1973),
            'amplification':compute_empirical_amplification(scale,expr,raw_control_expressions=raw,max_rows=128,random_state=1973),
            'finite_difference':validate_jacobian_finite_difference(scale,expr,max_rows=32,random_state=1973)})
        Path('diagnostics.json').write_text(json.dumps(diag,sort_keys=True),encoding='utf-8')
        rows=[_record(f'd{i}',seed,pipeline,.65+(.03*np.sin(i) if pipeline=='AutoFE_Baseline' else 0)) for i in range(25) for seed in range(20) for pipeline in ('Raw','AutoFE_Baseline')]
        ledger=Path('analysis.jsonl');ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        report=measured('analysis',lambda:report_inputs(ledger,output_dir='analysis_output'))
        with Path('analysis.pkl').open('wb') as handle: pickle.dump(report,handle)
        delta=measured('cliffs_delta_scale',lambda:cliffs_delta(np.arange(3000),np.arange(3000)[::-1]+.5))
        result['stages'][-1]['value']=delta
        records=build_task_records(['synthetic'],[42],[1],[('clean',0)],['Raw','AutoFE_Baseline'],['logistic_regression','gaussian_nb'],data_paths={'synthetic':Path('synthetic.csv')},pipeline_identity=runner.pipeline_identity_token)
        store=ManifestStore('persistence.db');store.create_run('persist',{},records[1:])
        def persistence():
            for record in records[1:]:
                tid=record['scientific_task_id']
                # This persistence-only fixture declares no preparation dependencies.
                with store._connect() as connection:
                    payload=json.loads(connection.execute('SELECT payload_json FROM tasks WHERE scientific_task_id=?',(tid,)).fetchone()[0]);payload['depends_on']=[]
                    connection.execute('UPDATE tasks SET payload_json=? WHERE scientific_task_id=?',(json.dumps(payload),tid))
                attempt=store.claim_task('persist',tid)
                store.commit_result('persist',tid,attempt,{'run_id':'persist','scientific_task_id':tid,'attempt_id':attempt,'roc_auc':.7})
            return store.export_durable_results('persist','persist.jsonl')
        measured('persistence',persistence)
        if args.production and repeat == 0:
            config=ExecutionConfig(max_workers=2,task_timeout_seconds=60)
            store=ManifestStore('production.db');store.create_run('production',{'execution':config.to_dict(),'diagnostics_enabled':True,'diagnostic_max_rows':128,'durability_protocol':'sqlite_result_outbox_v2'},records)
            outcome=measured('production_workers',lambda:execute_manifest(store,'production','production.jsonl',config))
            result['stages'][-1]['outcome']=outcome
            if outcome['status'] != 'completed_successfully': raise RuntimeError(outcome)
            resumed=measured('resume',lambda:execute_manifest(store,'production','production.jsonl',config))
            result['stages'][-1]['outcome']=resumed
        if repeat == 0:
            profile=cProfile.Profile()
            profile.runcall(lambda:expand_features_with_dfs(scale.iloc[:1200],scale.iloc[1200:],np.arange(1200)%2,DFSConfig(monitor_ram=False)))
            profile.dump_stats(str(output/'candidate_profile.prof'))
    result['source_hashes']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src').glob('*.py')}
    (output/'measurements.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'output':str(output),'stages':[(x['stage'],round(x['seconds'],3)) for x in result['stages']]}))


if __name__=='__main__': main()
