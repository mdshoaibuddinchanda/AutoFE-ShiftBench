"""Bounded actual CPU factory fits; reject thread changes that alter outputs."""
import json
import argparse
import pickle
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline_runner import CPU_MODELS
from src.model import build_model
from threadpoolctl import threadpool_limits,threadpool_info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-science',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Retain earlier measurements')
    results=[]
    fixtures=[('small',np.random.default_rng(113).normal(size=(192,24)).astype(np.float32),np.arange(192)%3,CPU_MODELS,42),
        ('larger',np.random.default_rng(113).normal(size=(1024,96)).astype(np.float32),np.arange(1024)%3,CPU_MODELS,42)]
    if args.reference_science:
        with args.reference_science.open('rb') as handle:reference=pickle.load(handle)
        ledger=args.reference_science.parent/'production.jsonl'
        row=next(row for row in map(json.loads,ledger.read_text().splitlines()) if row['pipeline']=='AutoFE_Baseline' and row['model']=='logistic_regression')
        from src.prepared_inputs import historical_model_array
        fixtures.append(('retained_gaussian_autofe',historical_model_array(reference['pipelines']['AutoFE_Baseline']['train']),reference['labels_train'],['logistic_regression'],row['model_seed']))
    for fixture,x,y,names,seed in fixtures:
      for name in names:
        variants=[]
        for threads in (8,1):
            model=build_model(name,random_state=seed,use_gpu=False)
            with threadpool_limits(limits=threads):
                start=time.perf_counter();model.fit(x,y)
                prediction=model.predict(x)
                probability=model.predict_proba(x)
                variants.append((prediction,probability,time.perf_counter()-start))
        results.append({'fixture':fixture,'shape':list(x.shape),'model':name,'seconds_threads_8':variants[0][2],'seconds_threads_1':variants[1][2],
            'prediction_exact':np.array_equal(variants[0][0],variants[1][0]),'probability_exact':np.array_equal(variants[0][1],variants[1][1]),
            'maximum_probability_difference':float(np.max(np.abs(variants[0][1]-variants[1][1])))})
    output={'schema':'native_thread_equivalence_probe_v2','results':results,'restored_threadpools':threadpool_info(),
        'policy':'Inherited reference limits retained: actual AutoFE production outputs rejected global one-thread rewrite.'}
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(results))


if __name__=='__main__':main()
