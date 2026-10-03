"""Bounded synthetic input loading, indexed membership and scoped hashing probes."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import threading
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
PIPELINE='Raw_Full'


def load_probe(descriptor,numeric,queue):
    import psutil
    import numpy as np
    from src.prepared_inputs import load_pipeline,historical_model_array
    from src.artifact_integrity import array_identity
    process=psutil.Process();done=threading.Event();rss=[]
    def poll():
        while not done.is_set():rss.append(process.memory_info().rss);done.wait(.005)
    thread=threading.Thread(target=poll);thread.start()
    before=process.io_counters();start=time.perf_counter()
    loaded=load_pipeline(descriptor,PIPELINE,model_numeric=numeric)
    values=loaded[0][PIPELINE] if numeric else tuple(historical_model_array(x) for x in loaded[0][PIPELINE])
    elapsed=time.perf_counter()-start
    identities=[array_identity(x) for x in values]
    done.set();thread.join();after=process.io_counters()
    output={'numeric_memmap':numeric,'seconds':elapsed,'peak_rss_sampled_bytes':max(rss),'rss_sampling_interval_seconds':.005,
        'process_read_bytes':after.read_bytes-before.read_bytes,'array_identities':identities}
    if numeric:
        for value in values:value._mmap.close()
    queue.put(output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);os.chdir(output)
    import numpy as np
    import pandas as pd
    from src import pipeline_runner as runner
    from src.prepared_inputs import publish_unit
    from src.provenance_evidence import _FileHashes
    from src.provenance import file_sha256
    from src.sensitivity_analysis import _pair_keys,_selection_mask
    shape=(32768,128)
    frame=pd.DataFrame(np.random.default_rng(82).normal(size=shape),columns=[f'x{i}' for i in range(shape[1])])
    frame['target']=np.arange(shape[0])%3
    frame.to_csv('synthetic.csv',index=False)
    prepared=runner.get_data_splits('synthetic.csv','synthetic',42,1,'clean','clean',0,selected_pipelines=[PIPELINE],retain_matrices=False)
    descriptor=publish_unit('unit.pkl',prepared,input_artifacts=prepared[4][PIPELINE]['input_artifacts'])
    del frame,prepared
    context=mp.get_context('spawn');loads=[]
    for repeat in range(2):
        for numeric in (False,True):
            queue=context.Queue();process=context.Process(target=load_probe,args=(descriptor,numeric,queue))
            process.start();result=queue.get(timeout=60);process.join(10)
            if process.exitcode!=0:raise RuntimeError('Loading probe failed')
            process.close();queue.close();queue.join_thread()
            result['repeat']=repeat;loads.append(result)
        assert loads[-1]['array_identities']==loads[-2]['array_identities']
    pairs=pd.DataFrame({'contrast_id':['c']*50000,'stratum':['s']*50000,'task_key_tuple':[(f'd{i%25}',i,1,'clean',0.) for i in range(50000)]})
    selected=set(_pair_keys(pairs)[::2]);memberships=[]
    for repeat in range(2):
        start=time.perf_counter();oracle=pairs.apply(lambda row:(str(row['contrast_id']),str(row['stratum']),tuple(row['task_key_tuple'])) in selected,axis=1).to_numpy();original=time.perf_counter()-start
        start=time.perf_counter();keys=_pair_keys(pairs);mask=_selection_mask(keys,selected);indexed=time.perf_counter()-start
        np.testing.assert_array_equal(oracle,mask)
        memberships.append({'repeat':repeat,'rows':len(pairs),'row_apply_seconds':original,'indexed_seconds':indexed,
            'expanded_four_regime_bytes_estimate':int(pairs.memory_usage(deep=True).sum()*4),'key_index_bytes_approx':sum(sys.getsizeof(x) for x in keys)+sys.getsizeof(keys)})
    path=Path('hash_fixture');path.write_bytes(b'a'*65536);hashes=[]
    for repeat in range(2):
        start=time.perf_counter();reference=[file_sha256(path) for _ in range(1000)];original=time.perf_counter()-start
        cache=_FileHashes();start=time.perf_counter();actual=[cache(path) for _ in range(1000)];cache.validate();scoped=time.perf_counter()-start
        assert reference==actual
        hashes.append({'repeat':repeat,'repeated_dependencies':1000,'distinct_files':1,'uncached_seconds':original,'scoped_seconds':scoped,'hash_bytes_before':65536000,'hash_bytes_after':65536})
    result={'schema':'bounded_remaining_optimization_probes_v1','synthetic_only':True,'pipeline':PIPELINE,'matrix_shape_before_split':list(shape),'load_probes':loads,
        'membership_probes':memberships,'hash_probes':hashes,'artifact_bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        'limits':'Input-loading fixture has diagnostics disabled to isolate loading; it is not an alternative benchmark contract. RSS is sampled, timings exclude spawn/import, process I/O omits child OS page-cache accounting.'}
    (output/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
