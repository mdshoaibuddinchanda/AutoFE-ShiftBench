"""Compare retained scientific evidence; field-specific norm tolerance frozen first."""
import argparse
from dataclasses import is_dataclass,asdict
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

NORMS={'frobenius_norm','dimension_adjusted_frobenius','mean_output_l2','max_output_l2','mean_l2'}


def equal(reference,actual,path='root',tolerated=None):
    tolerated=[] if tolerated is None else tolerated
    if isinstance(reference,pd.DataFrame):
        pd.testing.assert_frame_equal(reference,actual,check_exact=True)
    elif isinstance(reference,np.ndarray):
        np.testing.assert_array_equal(reference,actual,strict=True)
    elif is_dataclass(reference):
        equal(reference.__dict__,actual.__dict__,path,tolerated)
    elif isinstance(reference,dict):
        assert reference.keys()==actual.keys(),path
        for key in reference: equal(reference[key],actual[key],path+'.'+str(key),tolerated)
    elif isinstance(reference,(list,tuple)):
        assert len(reference)==len(actual),path
        for i,(a,b) in enumerate(zip(reference,actual)): equal(a,b,path+f'[{i}]',tolerated)
    elif isinstance(reference,float) and np.isnan(reference):
        assert np.isnan(actual),path
    elif reference != actual:
        if path.rsplit('.',1)[-1] in NORMS:
            assert np.isclose(reference,actual,rtol=1e-12,atol=1e-12,equal_nan=True),(path,reference,actual)
            tolerated.append({'path':path,'reference':reference,'actual':actual,'absolute_difference':abs(reference-actual)})
        else: raise AssertionError((path,reference,actual))
    return tolerated


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('optimized',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    checks=[]
    tolerated=[]
    for directory in sorted(args.reference.glob('repeat_*')):
        other=args.optimized/directory.name
        for filename in ('science.pkl','candidate_scale.pkl','analysis.pkl','diagnostics.json'):
            def read(path):
                if path.suffix=='.json': return json.loads(path.read_text())
                with path.open('rb') as handle:return pickle.load(handle)
            tolerated.extend(equal(read(directory/filename),read(other/filename),directory.name+'/'+filename))
            checks.append({'repeat':directory.name,'artifact':filename,'status':'scientific_equivalence_verified'})
        reference_splits=list((directory/'data').rglob('splits_*.pkl'))
        for split in reference_splits:
            counterpart=other/split.relative_to(directory)
            equal(read(split),read(counterpart),'split_indices')
    ref=json.loads((args.reference/'measurements.json').read_text())
    opt=json.loads((args.optimized/'measurements.json').read_text())
    assert ref['data_sha256']==opt['data_sha256']
    assert ref['fixture']==opt['fixture']
    model_fields=('scientific_task_id','dataset','split_policy','seed','fold','condition','severity','pipeline','pipeline_identity','model','status','selection_seed','split_seed','corruption_seed','estimator_seed','diagnostic_status','metric_semantics_version','metric_statuses','accuracy','balanced_accuracy','precision','recall','f1','f1_macro','mcc','roc_auc','pr_auc','log_loss','brier_score','n_original','n_generated','n_selected','candidate_count','actual_estimator_input_dimension','selected_feature_identities','training_distribution_distance','held_out_distribution_distance','device_evidence')
    def model_rows(path):
        if not path.exists(): return None
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        return {row['scientific_task_id']:{key:row[key] for key in model_fields if key in row} for row in rows}
    reference_models=model_rows(args.reference/'repeat_0/production.jsonl')
    optimized_models=model_rows(args.optimized/'repeat_0/production.jsonl')
    equal(reference_models,optimized_models,'actual_model_results')
    if reference_models is not None:
        checks.append({'artifact':'production.jsonl','actual_model_rows':len(reference_models),'status':'scientific_equivalence_verified'})
    result={'schema':'corrected_equivalence_v1','checks':checks,'same_data_and_configuration':True,'norm_differences_within_prefrozen_tolerance':tolerated,
        'exact_fields':'all selected matrix values/dtypes/order/attrs, labels, candidate universes/scores/ties/history, seeds, sampled inputs, statuses, analysis tables/config and split indices',
        'physical_artifacts':'Prepared descriptor inventory omits unrequested pipelines. Timing/RSS fields and content-addressed file hashes can differ; each execution verifies its actual artifacts. Required scientific dependency signatures and identities remain compared.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'checks':len(checks),'tolerated_norm_differences':len(tolerated),'status':'passed'}))


if __name__=='__main__':main()
