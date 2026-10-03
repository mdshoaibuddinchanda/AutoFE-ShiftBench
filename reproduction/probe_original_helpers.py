"""Bounded archived-checkout controls; no network, no workspace artifacts."""
import os
from pathlib import Path
import tempfile
import types
import sys
import zipfile
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from src.task_manifest import _callable_entry
    from tests.test_runtime_helpers import large_result
    archive=zipfile.ZipFile(Path('reports/references/pre_repair_55cf473.zip').resolve())
    module=types.ModuleType('reference_helpers')
    sys.modules[module.__name__]=module
    old=module.__dict__
    exec(archive.read('src/task_manifest.py'),old)
    old['_callable_entry']=_callable_entry
    outcome=old['run_callable_with_timeout'](large_result,timeout_seconds=5)
    print('original large successful result:',outcome)
    class Explainer:
        def __init__(self,model): pass
        def shap_values(self,x): return np.ones((2,2,2))
    sys.modules['shap']=types.SimpleNamespace(TreeExplainer=Explainer)
    shap_ns={'__name__':'reference_shap'}
    exec(archive.read('src/shap_explainer.py'),shap_ns)
    frame=pd.DataFrame({'a':[0,1],'b':[1,0]})
    print('original SHAP output:',shap_ns['compute_shap_values'](object(),frame,frame,'random_forest'))
    loader={'__name__':'reference_acquisition'}
    exec(archive.read('src/data_loader.py'),loader)
    source=types.SimpleNamespace(data=frame,target=pd.Series([0,1]),details={'id':'999','version':'1'},target_names=['class'])
    loader['_fetch_openml_with_fallbacks']=lambda name:(source,'999')
    loader['compute_meta_features']=lambda *a,**k:{}
    previous=Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            path=loader['download_openml_dataset']('mock',directory)
            before=path.read_bytes()
            source.data=frame*9
            loader['download_openml_dataset']('mock',directory)
            print('original existing data overwritten:',before != path.read_bytes())
            print('original source metadata:',path.with_name('mock_meta.json').read_text())
        finally:
            os.chdir(previous)


if __name__ == '__main__':
    main()
