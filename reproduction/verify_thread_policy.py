"""Bounded actual CPU factory fits; reject thread changes that alter outputs."""
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline_runner import CPU_MODELS
from src.model import build_model
from threadpoolctl import threadpool_limits,threadpool_info


def main():
    x=np.random.default_rng(113).normal(size=(192,24)).astype(np.float32)
    y=np.arange(192)%3
    results=[]
    for name in CPU_MODELS:
        variants=[]
        for threads in (8,1):
            with threadpool_limits(limits=threads):
                model=build_model(name,random_state=42,use_gpu=False)
                start=time.perf_counter();model.fit(x,y)
                prediction=model.predict(x)
                probability=model.predict_proba(x)
                variants.append((prediction,probability,time.perf_counter()-start))
        results.append({'model':name,'seconds_threads_8':variants[0][2],'seconds_threads_1':variants[1][2],
            'prediction_exact':np.array_equal(variants[0][0],variants[1][0]),'probability_exact':np.array_equal(variants[0][1],variants[1][1]),
            'maximum_probability_difference':float(np.max(np.abs(variants[0][1]-variants[1][1])))})
    output={'schema':'native_thread_equivalence_probe_v1','synthetic_shape':[192,24],'results':results,'restored_threadpools':threadpool_info(),
        'policy':'Do not enable changed native reduction threads globally unless exact required outputs are supported.'}
    Path('provenance/critical_repair/thread_policy_probe.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(results))


if __name__=='__main__':main()
