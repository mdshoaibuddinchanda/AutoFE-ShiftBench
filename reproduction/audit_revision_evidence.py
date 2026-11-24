"""Read-only KBS revision evidence audit. Does not fit models or mutate input files."""
from __future__ import annotations
import collections, hashlib, itertools, json, math, os, re, sqlite3, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT=ROOT/'revision_evidence_audit.json'
SUMMARY=ROOT/'reports/revision_evidence_audit_summary.json'
if OUT.exists() or SUMMARY.exists():
    raise FileExistsError('Refusing to overwrite an existing evidence audit.')
start=time.monotonic()
def log(message): print(message,flush=True)
def compact(v): return json.dumps(v,ensure_ascii=True,separators=(',',':'),allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
def finite(v):
    return isinstance(v,(int,float)) and math.isfinite(v)
def normal(path): return str(path).replace('\\','/').lower()
def idof(table,index,value):
    token=compact(value)
    if token not in index:
        index[token]=len(table);table.append(value)
    return index[token]
ep=ROOT/'reports/benchmark_runs/preflight_eligibility.json'
elig=json.loads(ep.read_text(encoding='utf-8-sig'))
cfg=elig['configuration']
datasets=cfg['datasets'];seeds=cfg['seeds'];folds=cfg['folds'];pipelines=cfg['pipelines'];models=cfg['models']
conditions=[f if float(v)==0 else f'{f}_{float(v)}' for f,v in cfg['conditions']]
dsidx={v:i for i,v in enumerate(datasets)}
si={v:i for i,v in enumerate(seeds)};fi={v:i for i,v in enumerate(folds)}
ci={v:i for i,v in enumerate(conditions)};pi={v:i for i,v in enumerate(pipelines)};mi={v:i for i,v in enumerate(models)}
def config_index(d,s,f,c,p,m):
    return (((((dsidx[d]*len(seeds)+si[s])*len(folds)+fi[f])*len(conditions)+ci[c])*len(pipelines)+pi[p])*len(models)+mi[m])
unit_elig={(x['dataset'],int(x['seed']),int(x['fold']),x['condition']):x for x in elig['units']}
original_scope=[d for d in datasets if d not in {'aps_failure','covertype','kddcup99'}]
original_pipelines=['Raw','Raw_Variance','Raw_MI','AutoFE_Baseline','AutoFE_MI','AutoFE_Random','AutoFE_NoMultiply']

log('Reading the declared grid, immutable source hashes, and existing FSVA validation reports.')
inventory=[]
for p in [ROOT/'reports/benchmark_runs/combined_results_20261122.jsonl',
          ROOT/'reports/full_benchmark_20261003/results.jsonl',
          ROOT/'results.zip',ep,ROOT/'reports/full_benchmark_20261003/manifest.db',
          ROOT/'reports/reviewer_experimental_audit_20261122.json',
          ROOT/'reports/fsva_revalidation_20261122.json',
          ROOT/'KNOSYS-D-26-17927/manuscript.tex',ROOT/'paper/manuscript.tex']:
    if p.is_file():
        st=p.stat();inventory.append({'path':str(p.relative_to(ROOT)),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,
                                    'role':'audit_input','sha256':sha(p) if st.st_size<50_000_000 else 'computed_during_scan_or_not_hashed_see_limitations'})
source_hashes={p:sha(ROOT/p) for p in ['src/splitters.py','src/pipeline_runner.py','src/preprocessing.py','src/feature_engineering.py','src/fsva.py','src/evaluation.py','src/shift_generator.py','src/data_loader.py','src/seeding.py','src/protocol.py']}
raw_checks={}
for ds,ident in elig['dataset_identities'].items():
    p=ROOT/'data/raw'/f'{ds}.csv'
    raw_checks[ds]={'path':str(p.relative_to(ROOT)),'local_file_present':p.is_file(),
                    'expected_fingerprint':ident['fingerprint'],'expected_rows':ident['rows']}
    if p.is_file():
        raw_checks[ds].update({'observed_bytes_sha256':sha(p),'expected_bytes_sha256':ident['bytes_sha256'],
                              'bytes_match':p.stat().st_size==ident['bytes'],'sha256_match':sha(p)==ident['bytes_sha256']})
fsva_prior=json.loads((ROOT/'reports/fsva_revalidation_20261122.json').read_text(encoding='utf-8'))
prior_stats=json.loads((ROOT/'reports/reviewer_experimental_audit_20261122.json').read_text(encoding='utf-8'))
prov,prov_index=[],{}
units,unit_index=[],{}
split_evidence,split_index=[],{}
diagnostic_refs,diagnostic_index=[],{}
metric_states,metric_state_index=[],{}
observations=collections.defaultdict(list)
dataset_counts=collections.Counter();run_counts=collections.Counter();protocol_counts=collections.Counter()
auc_reasons=collections.defaultdict(collections.Counter)
dimension_stats=collections.defaultdict(list)
pair_groups=collections.defaultdict(dict)
parse_errors=[];unknown=[];metric_conflicts=[];membership_conflicts=[]
unit_conflicts=[];status_counts=collections.Counter();identity_missing=collections.Counter()
ledger=ROOT/'reports/benchmark_runs/combined_results_20261122.jsonl'
stat0=ledger.stat();ledger_hash=hashlib.sha256()
log(f'Scanning merged ledger: {stat0.st_size/1e9:.2f} GB. No benchmark work is launched.')
row_count=0;offset=0
with ledger.open('rb',buffering=8*1024*1024) as f:
    for line_number,line in enumerate(f,1):
        ledger_hash.update(line)
        current=offset;offset+=len(line)
        if not line.strip():continue
        try:r=json.loads(line)
        except Exception as exc:
            parse_errors.append({'line':line_number,'byte_offset':current,'error':str(exc)});continue
        row_count+=1
        d=r.get('dataset');s=r.get('seed');fold=r.get('fold');condition=r.get('condition');pipe=r.get('pipeline');model=r.get('model')
        try:key=config_index(d,s,fold,condition,pipe,model)
        except (KeyError,TypeError):
            unknown.append({'line':line_number,'configuration':[d,s,fold,condition,pipe,model]});continue
        profile={k:r.get(k) for k in ['evaluation_protocol_version','seed_scheme_version','source_code_fingerprint',
                  'environment_fingerprint','metric_semantics_version','numerical_validation_version','condition_semantics_version']}
        prid=idof(prov,prov_index,profile)
        for k,v in profile.items():
            if v is None:identity_missing[k]+=1
        arts={a.get('role'):a for a in r.get('artifacts',[]) if isinstance(a,dict)}
        split=arts.get('split_indices',{})
        split_profile={
            'dataset':d,'dataset_fingerprint':r.get('dataset_fingerprint'),'split_policy':r.get('split_policy'),
            'seed':s,'fold':fold,'split_seed':r.get('split_seed'),
            'split_pickle_path':split.get('path'),'split_pickle_sha256':split.get('sha256'),
            'dependency_signature':split.get('dependency_signature'),
            'recorded_test_rows':r.get('n_test'),'recorded_train_rows':r.get('n_train')}
        spid=idof(split_evidence,split_index,split_profile)
        diag=arts.get('fsva_diagnostic',{});history=arts.get('candidate_history',{})
        ref={'diagnostic_path':diag.get('path') or r.get('diagnostic_path'),
             'expected_diagnostic_sha256':diag.get('sha256'),
             'history_path':history.get('path') or r.get('history_path'),
             'expected_history_sha256':history.get('sha256'),
             'recorded_diagnostic_status':r.get('diagnostic_status')}
        did=idof(diagnostic_refs,diagnostic_index,ref)
        sel=r.get('selected_feature_identities',[])
        selhash=hashlib.sha256(compact(sel).encode()).hexdigest()
        unit={
            'dataset':d,'seed':s,'fold':fold,'condition':condition,'pipeline':pipe,'provenance_id':prid,
            'dataset_fingerprint':r.get('dataset_fingerprint'),'split_evidence_id':spid,
            'diagnostic_ref_id':did,'pipeline_identity':r.get('pipeline_identity'),
            'corruption_seed':r.get('corruption_seed'),'selection_seed':r.get('selection_seed'),
            'operator_set_id':r.get('operator_set_id'),'operator_set':r.get('operator_set'),
            'requested_base_cap':r.get('requested_base_cap'),'requested_cap':r.get('requested_cap'),
            'eligible_base_features':r.get('eligible_base_feature_count'),
            'actual_estimator_input_dimension':r.get('actual_estimator_input_dimension'),
            'baseline_kind':r.get('baseline_kind'),'candidate_count':r.get('candidate_count'),
            'selected_feature_identities_sha256':selhash,
            'fitted_preprocessor_sha256':arts.get('fitted_preprocessor',{}).get('sha256'),
            'feature_metadata_path':arts.get('feature_metadata',{}).get('path'),
            'feature_metadata_sha256':arts.get('feature_metadata',{}).get('sha256')}
        uid=idof(units,unit_index,unit)
        msid=idof(metric_states,metric_state_index,r.get('metric_status',{}))
        vals=[r.get('roc_auc') if finite(r.get('roc_auc')) else None,
              r.get('balanced_accuracy') if finite(r.get('balanced_accuracy')) else None]
        # A record retains its original source offset, identity, status, and metric values.
        obs=[line_number,current,prid,uid,msid,r.get('run_id'),r.get('scientific_task_id'),r.get('status'),*vals,r.get('model_seed')]
        observations[key].append(obs)
        dataset_counts[d]+=1;run_counts[r.get('run_id')]+=1
        protocol_counts[r.get('evaluation_protocol_version')]+=1;status_counts[r.get('status')]+=1
        if vals[0] is None:auc_reasons[d][(r.get('metric_status') or {}).get('roc_auc','missing_reason')]+=1
        dimension=r.get('actual_estimator_input_dimension')
        if finite(dimension):dimension_stats[(d,pipe)].append(dimension)
        pg=(d,s,fold,condition,model,prid,r.get('dataset_fingerprint'))
        signature=(split.get('sha256'),r.get('split_seed'),r.get('corruption_seed'),r.get('n_test'))
        pair_groups[pg].setdefault(pipe,set()).add(signature)
        if row_count%50000==0:
            log(f'Ledger {row_count:,} rows, {offset/1e9:.2f}/{stat0.st_size/1e9:.2f} GB, {(time.monotonic()-start)/60:.1f} min.')
stat1=ledger.stat()
if (stat0.st_size,stat0.st_mtime_ns)!=(stat1.st_size,stat1.st_mtime_ns):
    raise RuntimeError('Ledger changed during audit; report publication refused.')
log(f'Ledger scan complete: {row_count:,} rows; checking duplicates and paired split identities.')
dupconfigs=0;extraobs=0;provenance_overlap=0
for key,values in observations.items():
    if len(values)>1:
        dupconfigs+=1;extraobs+=len(values)-1
        profiles={v[2] for v in values}
        if len(profiles)>1:provenance_overlap+=1
        # Preserve variation across environments; identify exact-identity metric disagreement separately.
        byprofile=collections.defaultdict(list)
        for v in values:byprofile[v[2]].append(v)
        for prid,subset in byprofile.items():
            if len({(v[7],v[8],v[9],v[10]) for v in subset})>1:
                metric_conflicts.append({'configuration_index':key,'provenance_id':prid,
                                        'source_lines':[v[0] for v in subset],'metric_values':[[v[8],v[9],v[10]] for v in subset]})
pair_summary=collections.Counter();pair_examples=[]
for group,by_pipe in pair_groups.items():
    pair_summary['evaluation_context_groups']+=1
    sigs=set().union(*by_pipe.values())
    if any(v[0] is None for v in sigs):
        pair_summary['groups_missing_recorded_split_hash']+=1
    elif len(sigs)>1:
        pair_summary['groups_with_mismatched_split_or_corruption_identity']+=1
        if len(pair_examples)<100:pair_examples.append({'context':list(group),'per_pipeline':{k:[list(x) for x in v] for k,v in by_pipe.items()}})
    else:pair_summary['groups_with_consistent_recorded_split_hash_seed_and_test_count']+=1
    if len(by_pipe)==len(pipelines):pair_summary['all_14_pipelines_present']+=1
    else:pair_summary['fewer_than_14_pipelines_present']+=1
# Local splits are verified by content hash once per unique file. Hash equality plus fold
# establishes identical serialized fold definitions; external files remain a recorded claim.
split_path_checks={}
for sp in split_evidence:
    path=sp.get('split_pickle_path')
    if not path:continue
    if path in split_path_checks:continue
    p=Path(path)
    if p.is_file():
        actual=sha(p);split_path_checks[path]={'availability':'local','observed_sha256':actual,
                                            'matches_recorded_sha256':actual==sp.get('split_pickle_sha256')}
    else:split_path_checks[path]={'availability':'not_locally_available','hash_verified':False}
log('Reading local Jacobian sidecars and checking their selection-history availability.')
local_diagnostics=[];local_by_path={};local_by_sha=collections.defaultdict(list)
norm_by=collections.defaultdict(list);histcounts=collections.Counter();diagcounts=collections.Counter()
roots=[ROOT/'data/cache/predictor_only_geometry_v3_integrity_metrics_fsva_v2/sha256_canonical_json_u32_v1',
       ROOT/'reports/full_benchmark_20261003/cache/predictor_only_geometry_v3_integrity_metrics_fsva_v2/sha256_canonical_json_u32_v1']
for root in roots:
    for p in root.rglob('*_meta_fsva.json'):
        raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        try:diag=json.loads(raw)
        except Exception:
            diagcounts['unreadable']+=1;continue
        task=diag.get('task',{});jac=diag.get('jacobian',{});selected=jac.get('selected',{})
        history=Path(str(p).replace('_meta_fsva.json','_meta_history.jsonl'))
        hp=history.is_file()
        histcounts['present' if hp else 'missing']+=1
        if hp:histcounts['bytes_present']+=history.stat().st_size
        per_output=jac.get('per_output',[])
        l2=[x.get('mean_l2') for x in per_output if finite(x.get('mean_l2'))]
        row={'path':str(p.relative_to(ROOT)),'sha256':digest,'task':task,
             'jacobian_selected':selected,'jacobian_raw_control':jac.get('raw_control'),
             'per_output_entries':len(per_output),
             'per_output_mean_l2_quantiles':dict(zip(['min','p50','p95','p99','max'],map(float,np.quantile(l2,[0,.5,.95,.99,1])))) if l2 else None,
             'input_coordinates_count':len(jac.get('input_coordinates',[])),
             'sample_identity':jac.get('sample_identity'),
             'original_derivative_validation':diag.get('derivative_validation'),
             'empirical_amplification':diag.get('empirical_amplification'),
             'history_path':str(history.relative_to(ROOT)),'history_present':hp,
             'history_bytes':history.stat().st_size if hp else None,
             'history_content_validation':'presence_only; candidate-level parsing/hash validation not performed in this audit'}
        lid=len(local_diagnostics);local_diagnostics.append(row)
        local_by_path[normal(p.resolve())]=lid;local_by_sha[digest].append(lid)
        diagcounts[diag.get('diagnostic_status','missing')]+=1
        norm_by[(task.get('dataset'),task.get('pipeline'))].append(row)
        if lid and lid%5000==0:log(f'Jacobian sidecars read: {lid:,}.')
ref_counts=collections.Counter()
for ref in diagnostic_refs:
    path=ref.get('diagnostic_path'); lid=local_by_path.get(normal(path)) if path else None
    expected=ref.get('expected_diagnostic_sha256')
    if lid is not None:
        status='local_path_and_hash_verified' if expected and local_diagnostics[lid]['sha256']==expected else 'local_file_hash_missing_or_mismatched'
        ref.update({'availability':status,'local_diagnostic_id':lid})
    elif expected and expected in local_by_sha:
        ref.update({'availability':'identical_content_available_at_another_local_path','local_diagnostic_id':local_by_sha[expected][0]})
    else:
        ref['availability']='referenced_but_not_locally_available'
    ref_counts[ref['availability']]+=1
norm_summary=[]
for (ds,pipe),values in sorted(norm_by.items()):
    row={'dataset':ds,'pipeline':pipe,'artifacts':len(values)}
    for field in ['frobenius_norm','dimension_adjusted_frobenius','mean_output_l2','max_output_l2']:
        numbers=[v['jacobian_selected'].get(field) for v in values]
        numbers=[v for v in numbers if finite(v)]
        row[field]=dict(zip(['min','p50','p95','max'],map(float,np.quantile(numbers,[0,.5,.95,1])))) if numbers else None
    norm_summary.append(row)

log('Building explicit completion accounting for every declared configuration.')
dataset_summary={}
missing_by_ds=collections.Counter();infeasible_by_ds=collections.Counter()
seed_scope=collections.defaultdict(collections.Counter)
expected_total=len(datasets)*len(seeds)*len(folds)*len(conditions)*len(pipelines)*len(models)
for d in datasets:
    observed=sum(len(v) for k,v in observations.items() if k//(expected_total//len(datasets))==dsidx[d])
    eligible_units=[u for u in elig['units'] if u['dataset']==d and u['status']=='split_eligible']
    observed_keys={k for k in observations if k//(expected_total//len(datasets))==dsidx[d]}
    eligible_configs=len(eligible_units)*len(pipelines)*len(models)
    dataset_summary[d]={'observed_rows':observed,'observed_unique_configurations':len(observed_keys),
                        'declared_configurations':expected_total//len(datasets),
                        'eligible_configurations':eligible_configs,
                        'planned_infeasible_configurations':expected_total//len(datasets)-eligible_configs,
                        'missing_eligible_configurations':0,'observed_in_infeasible_configurations':0,
                        'roc_auc_missing_reasons':dict(auc_reasons[d]),'seeds':{}}
for d,s,f,c in itertools.product(datasets,seeds,folds,conditions):
    e=unit_elig[(d,s,f,c)]
    for p,m in itertools.product(pipelines,models):
        k=config_index(d,s,f,c,p,m);obs=observations.get(k,[])
        if e['status']=='scientifically_infeasible':
            infeasible_by_ds[d]+=1
            if obs:dataset_summary[d]['observed_in_infeasible_configurations']+=1
        elif not obs:
            missing_by_ds[d]+=1
        else:
            seed_scope[(d,s)]['observed_unique_eligible']+=1
for d in datasets:
    dataset_summary[d]['missing_eligible_configurations']=missing_by_ds[d]
    for s in seeds:
        declared=sum(u['status']=='split_eligible' for u in elig['units'] if u['dataset']==d and u['seed']==s)*len(pipelines)*len(models)
        actual=seed_scope[(d,s)]['observed_unique_eligible']
        dataset_summary[d]['seeds'][str(s)]={'eligible':declared,'observed_unique':actual,'missing_eligible':declared-actual,
            'status':'no_eligible_tasks' if declared==0 else ('eligible_grid_complete' if actual==declared else 'incomplete')}
# SQLite is opened in read-only mode; its pending tasks describe that machine/run only.
manifest_summary={}
db=ROOT/'reports/full_benchmark_20261003/manifest.db'
with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as con:
    manifest_summary['runs']=[]
    for rid,protocol,seedver,state,config in con.execute('select run_id,protocol_version,seed_scheme_version,status,config_json from runs'):
        c=json.loads(config);code=c.get('code_identity',{})
        manifest_summary['runs'].append({'run_id':rid,'protocol':protocol,'seed_scheme':seedver,'recorded_run_state':state,
            'commit':code.get('commit'),'branch':code.get('branch'),'worktree_fingerprint_sha256':code.get('worktree_fingerprint_sha256'),
            'configured_datasets':c.get('datasets'),'pipelines':c.get('pipelines'),'seeds':c.get('seeds')})
    manifest_summary['task_states']=[list(x) for x in con.execute('select run_id,stage,state,count(*) from tasks group by run_id,stage,state')]
    manifest_summary['planned_skip_reasons']=[list(x) for x in con.execute("select planned_skip_reason,count(*) from tasks where state='skipped' group by planned_skip_reason")]
    manifest_summary['scope_note']='This database belongs to the local first-machine run. Its pending tasks cannot establish that merged results from other machines are incomplete.'
raw_has_outcome_fields={}
for ds in datasets:
    cols=elig['dataset_identities'][ds]['coordinates']
    flagged=[x for x in cols if ds=='titanic' and x in ['boat','body']]
    if flagged:raw_has_outcome_fields[ds]=flagged

limitations=[
 'Original seven-pipeline manuscript numbers are preserved as reported claims. No independently identified original seven-pipeline task ledger was found among the inspected result sources; all supplied result source samples belong to the corrected protocol. A subset of corrected pipeline names is not the historical experiment.',
 'Identical split hashes, fold, corruption seeds and test counts are checked within identical protocol/source/environment/dataset profiles. External split files have only recorded hashes unless locally available. Matching row counts alone is not proof of identical held-out data.',
 'Source/environment fingerprints vary. Results are not declared bitwise interchangeable between different profiles; hardware runtime comparisons require separate treatment.',
 'Selection-history completeness means file availability here. This audit does not parse every candidate event or verify all history hashes; no full theoretical FSVA index or covariance-weighted amplification is claimed.',
 'The supplied Covertype data are retained separately. Eligible-grid and single-seed completion are verified rather than assumed from successful-row counts.',
 'Explicit split infeasibility and undefined metrics are distinct from execution failure or computational stopping.',
 'No benchmark, model training, estimator prediction Jacobian, manuscript revision, source repair, or dataset download is performed by this audit.',
 'Titanic boat/body are present as predictor columns; their temporal availability is not established by a target-column-exclusion test. They require a dedicated outcome-proxy review before an unrestricted leakage-free claim.',
]
findings=[
 {'severity':'high','issue':'Historical and corrected experiments are not reconciled by pipeline name alone','evidence':'Original manuscript reports 7 pipelines, 538972 primary rows, rare-class KFold fallback, and absent Jacobian histories; corrected result protocol has 14 pipelines and explicit infeasibility.','action':'Preserve original 22-dataset claims as historical. Treat new records as a distinct revision experiment until a historical source ledger and compatibility evidence are available.'},
 {'severity':'high','issue':'Potential outcome-proxy predictors in Titanic','evidence':raw_has_outcome_fields,'action':'Review and isolate Titanic in sensitivity analysis; do not claim target-column removal excludes all target information.'},
 {'severity':'high','issue':'Complete FSVA theoretical index is not measured','evidence':'Saved Jacobians are for frozen preprocessed feature mappings on sampled rows. Covariance-weighted amplification, estimation variance, search penalty, and bias reduction are separate terms.','action':'Report Jacobian and empirical perturbation diagnostics as direct partial mechanism measurements.'},
 {'severity':'medium','issue':'Predictor-only covariate/population geometry is transductive','evidence':'Current splitters accept predictors only; scaling and PCA/KMeans fit all predictor rows for partition construction.','action':'Exclude these two conditions from principal corrected comparisons and report separately.'},
 {'severity':'medium','issue':'Original Raw baseline is base-cap20 and synthesized outputs may use cap100','evidence':'Raw_Full and Raw_Capped are new corrected controls; same maximum feature cap does not force identical realized dimensions.','action':'Analyze paired controls using declared cap and actual dimensionality; report rather than conceal dimensional differences.'},
 {'severity':'medium','issue':'Primary scope differs between original manuscript and previous audit','evidence':{'original_manuscript_datasets':22,'previous_audit_datasets_balanced_accuracy':24,'previous_audit_datasets_roc_auc':23},'action':'Recalculate original-scope and extension-scope comparisons separately.'},
]
summary={
 'schema_version':'revision_evidence_audit_v1','date_local':'2026-11-22 Asia/Kolkata',
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'audit_mode':'read_only_inputs_new_audit_outputs_only',
 'overall_status':'evidence_inventory_completed_methodological_reconciliation_has_open_items',
 'scope':{'unit_of_record':'one configuration (dataset,seed,fold,condition,pipeline,model) with all observed source rows retained',
          'expected_configurations':expected_total,'observed_rows':row_count,'observed_unique_configurations':len(observations),
          'primary_scope':'historical_22_dataset_scope_preserved','historical_22_datasets':original_scope,
          'historical_7_pipelines':original_pipelines,'corrected_14_pipelines':pipelines,
          'merged_result_protocol_counts':dict(protocol_counts),'merged_row_status_counts':dict(status_counts),
          'observed_rows_by_run':dict(run_counts),'source_ledger_sha256':ledger_hash.hexdigest(),
          'ledger_unchanged_during_scan':True,'original_historical_ledger_independently_available':False},
 'integrity':{'unparseable_rows':parse_errors,'rows_outside_declared_grid':unknown,
              'duplicate_configuration_groups':dupconfigs,'observations_beyond_one_per_config':extraobs,
              'multiple_provenance_profiles_on_same_config':provenance_overlap,
              'within_same_profile_metric_conflicts':metric_conflicts,
              'missing_identity_field_counts':dict(identity_missing)},
 'completion':{'by_dataset':dataset_summary,'missing_eligible_total':sum(missing_by_ds.values()),
               'scientifically_infeasible_total':sum(infeasible_by_ds.values()),
               'covertype_policy':'retain observed seed42 and other partial seeds separately; do not insert into historical22 primary analysis',
               'legacy_22_manuscript_claims':{'completed_primary_rows':538972,'total_success_rows':560002,'intended_tasks':612500,'independently_recomputed':False}},
 'pairing':{'counts':dict(pair_summary),'mismatch_examples':pair_examples,
            'local_split_files':dict(collections.Counter(v['availability'] for v in split_path_checks.values())),
            'local_split_hash_mismatches':[p for p,x in split_path_checks.items() if x.get('matches_recorded_sha256') is False],
            'evidence_level':'recorded_serialized_split_hash_plus_fold_and_corruption_seed; local_file_content_hash_where_available'},
 'leakage_access':{
    'target_exclusion':'pipeline_runner.split_predictors_and_target removes declared target before split geometry',
    'stratification':'labels permitted solely for class balancing; rare-class infeasibility is explicit in corrected protocol',
    'imputation_scaling_encoding':'fit on corrupted training features only',
    'base_feature_selection_variance':'training predictors only',
    'mutual_information_selection':'uses training labels; does not directly use held-out labels',
    'model_fit':'corrupted training labels',
    'covariate_population_partitioning':'all predictor rows used for unsupervised geometry; no target argument; transductive',
    'duplicate_predictor_overlap':'available in eligibility units; not proof of entity independence',
    'outcome_proxy_predictor_review_required':raw_has_outcome_fields,
    'claim_limit':'Current-source inspection and content identities support the corrected pipeline; old/source-mismatched executions are not retroactively repaired.'},
 'fsva':{'local_artifacts':len(local_diagnostics),'diagnostic_status_counts':dict(diagcounts),
         'selection_history_availability':dict(histcounts),'referenced_artifact_availability':dict(ref_counts),
         'full_prior_warning_revalidation_census':fsva_prior['artifact_census'],
         'prior_derivative_reproduction':fsva_prior['original_fixed_step_reproduction'],
         'prior_residual_probe':fsva_prior.get('targeted_residual_step_sensitivity'),
         'measured_quantity':'frozen selected-feature mapping Jacobian on sampled preprocessed corrupted-training rows',
         'theoretical_covariance_weighted_amplification_measured':False,
         'jacobian_norm_summaries':norm_summary},
 'methodological_findings':findings,'limitations':limitations,
 'source_inventory':inventory,'audit_source_hashes':source_hashes,'raw_dataset_checks':raw_checks,
 'local_manifest':manifest_summary,
 'record_schema':{'observation_columns':['source_line','source_byte_offset','provenance_id','feature_unit_id','metric_state_id','run_id','scientific_task_id','recorded_status','roc_auc','balanced_accuracy','model_seed'],
                  'configuration_statuses':['observed_success','observed_other_status','planned_scientific_infeasibility','missing_eligible_result'],
                  'references':'Integer ids index top-level provenance_profiles, feature_units, split_evidence, diagnostic_references, metric_states, local_fsva_artifacts arrays.',
                  'compatibility':'Within identical recorded provenance only; split pairing checked separately. No automatic compatibility with original7.'},
}
SUMMARY.parent.mkdir(parents=True,exist_ok=True)
tmp=OUT.with_suffix('.json.tmp')
if tmp.exists():raise FileExistsError(tmp)
log('Writing the requested one-record-per-configuration JSON report.')
with tmp.open('w',encoding='utf-8',newline='\n') as out:
    out.write('{"summary":'+compact(summary))
    for name,value in [('provenance_profiles',prov),('feature_units',units),('split_evidence',split_evidence),
                       ('diagnostic_references',diagnostic_refs),('metric_states',metric_states),
                       ('local_fsva_artifacts',local_diagnostics)]:
        out.write(',\n'+compact(name)+':'+compact(value))
    out.write(',\n"experimental_configurations":[\n')
    first=True
    for d,s,f,c,p,m in itertools.product(datasets,seeds,folds,conditions,pipelines,models):
        k=config_index(d,s,f,c,p,m);obs=observations.get(k,[])
        e=unit_elig[(d,s,f,c)]
        successful=any(v[7]=='success' for v in obs)
        status='observed_success' if successful else ('observed_other_status' if obs else ('planned_scientific_infeasibility' if e['status']=='scientifically_infeasible' else 'missing_eligible_result'))
        record={'dataset':d,'seed':s,'fold':f,'condition':c,'pipeline':p,'model':m,
                'completion_status':status,'eligibility_unit_id':e.get('unit_identity'),
                'observations':obs,
                'compatibility':'corrected_protocol_only_not_established_against_historical7' if obs else 'no_observed_result'}
        if e.get('reason'):record['limitation']=e['reason']
        if d=='covertype':record['scope']='separate_supporting_partial_dataset'
        if not first:out.write(',\n')
        out.write(compact(record));first=False
    out.write('],\n"eligibility_units":'+compact(elig['units'])+'}\n')
tmp.replace(OUT)
SUMMARY.write_text(json.dumps(summary,ensure_ascii=True,indent=2,allow_nan=False),encoding='utf-8')
log('Audit written: '+str(OUT))
log('SUMMARY '+compact({'observed_rows':row_count,'observed_unique_configurations':len(observations),'expected':expected_total,
                        'duplicates':dupconfigs,'metric_conflicts':len(metric_conflicts),'planned_skips':sum(infeasible_by_ds.values()),
                        'missing_eligible':sum(missing_by_ds.values()),'local_fsva':len(local_diagnostics),
                        'pairing':dict(pair_summary),'minutes':round((time.monotonic()-start)/60,2)}))

