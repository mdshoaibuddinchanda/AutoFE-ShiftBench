"""Same wide synthetic FSVA fixture for archived reference and current source."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.source_root.resolve()))
    import numpy as np
    import pandas as pd
    import psutil
    from src.fsva import compute_jacobian_diagnostic,compute_empirical_amplification,validate_jacobian_finite_difference
    from src.operator_registry import raw_expression,op_expression
    args.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(397)
    frame=pd.DataFrame(rng.normal(size=(160,128)),columns=[f'x{i}' for i in range(128)])
    expr=[raw_expression(c) for c in frame.columns[:32]]+[op_expression('multiply_numeric',raw_expression(f'x{i}'),raw_expression(f'x{i+1}')) for i in range(64)]
    raw=[raw_expression(c) for c in frame]
    measurements=[]
    for i in range(2):
        done=threading.Event();samples=[]
        def poll():
            while not done.is_set():samples.append(psutil.Process().memory_info().rss);done.wait(.005)
        t=threading.Thread(target=poll);t.start();start=time.perf_counter()
        try:
            results={'jacobian':compute_jacobian_diagnostic(frame,expr,raw_control_expressions=raw,max_rows=128,random_state=947),
                'amplification':compute_empirical_amplification(frame,expr,raw_control_expressions=raw,max_rows=128,random_state=947),
                'finite_difference':validate_jacobian_finite_difference(frame,expr,max_rows=32,random_state=947)}
        finally:elapsed=time.perf_counter()-start;done.set();t.join()
        (args.output/f'diagnostics_{i}.json').write_text(json.dumps(results,sort_keys=True),encoding='utf-8')
        measurements.append({'seconds':elapsed,'sampled_peak_rss_bytes':max(samples),'rows':160,'inputs':128,'outputs':96,'jacobian_rows':128,'finite_difference_rows':32,'magnitudes':[.001,.01,.05]})
    (args.output/'measurements.json').write_text(json.dumps(measurements,indent=2),encoding='utf-8')
    print(json.dumps(measurements))


if __name__=='__main__':main()
