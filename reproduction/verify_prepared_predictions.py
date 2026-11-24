"""Replay exact executed inputs/seeds to compare legacy and prepared predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model import build_model
from src.model_prediction import predict_labels_and_probabilities
from src.prepared_inputs import historical_model_array,load_pipeline
from src.task_manifest import ManifestStore
from src.artifact_integrity import array_identity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',type=Path);parser.add_argument('optimized',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.reference=args.reference.resolve();args.optimized=args.optimized.resolve();args.output=args.output.resolve()
    if args.output.exists():raise FileExistsError(args.output)
    rows=list(map(json.loads,(args.reference/'production.jsonl').read_text().splitlines()))
    def descriptor(directory):
        store=ManifestStore(directory/'production.db',read_only=True)
        return next(value['prepared_descriptor'] for row in store.snapshot('production')['durable_results'] if 'prepared_descriptor' in (value:=json.loads(row['payload_json'])))
    before_descriptor=descriptor(args.reference);after_descriptor=descriptor(args.optimized)
    # Descriptor paths were recorded relative to the execution working directory.
    import os
    checks=[]
    for row in rows:
        pipeline=row['pipeline'];name=row['model'];seed=row['model_seed']
        os.chdir(args.reference)
        reference=load_pipeline(before_descriptor,pipeline,model_numeric=False)
        os.chdir(args.optimized)
        prepared=load_pipeline(after_descriptor,pipeline,model_numeric=True)
        arrays=prepared[0][pipeline]
        previous=tuple(historical_model_array(frame) for frame in reference[0][pipeline])
        np.testing.assert_array_equal(reference[1],prepared[1],strict=True)
        np.testing.assert_array_equal(reference[2],prepared[2],strict=True)
        for left,right in zip(previous,arrays):
            np.testing.assert_array_equal(left,right,strict=True)
            assert array_identity(left)==array_identity(right)
        first=build_model(name,random_state=seed,use_gpu=False)
        second=build_model(name,random_state=seed,use_gpu=False)
        first.fit(previous[0],reference[1]);second.fit(arrays[0],prepared[1])
        partitions=[]
        for part,left,right in zip(('train','test'),previous,arrays):
            expected=(first.predict(left),first.predict_proba(left))
            actual=predict_labels_and_probabilities(second,right)
            for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b,strict=True)
            partitions.append({'partition':part,'prediction_identity':array_identity(actual[0]),'probability_identity':array_identity(actual[1]),'exact':True})
        for array in arrays:array._mmap.close()
        checks.append({'pipeline':pipeline,'model':name,'estimator_seed':seed,'partitions':partitions})
    result={'schema':'executed_seed_prepared_prediction_replay_v1','checks':checks,'scientific_equivalence':'exact labels/probabilities/inputs on replayed actual production seeds and prepared dependencies','native_threads':'inherited corrected-reference limits'}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'actual_seed_model_replays':len(checks),'status':'passed'}))


if __name__=='__main__':main()
