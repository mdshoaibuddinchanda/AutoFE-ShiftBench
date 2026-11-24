import sys
import types
import numpy as np
import pandas as pd

from src.task_manifest import run_callable_with_timeout


def large_result():
    return b'x'*(4*1024*1024)


def test_timeout_helper_drains_large_successful_result():
    outcome=run_callable_with_timeout(large_result,timeout_seconds=15)
    assert outcome['state'] == 'completed'
    assert len(outcome['value']) == 4*1024*1024


def test_shap_045_class_axis(monkeypatch):
    from src.shap_explainer import compute_shap_values
    values=np.array([[[1.,3.],[5.,7.]],[[3.,5.],[7.,9.]]])
    class Explainer:
        def __init__(self,model): pass
        def shap_values(self,x): return values
    monkeypatch.setitem(sys.modules,'shap',types.SimpleNamespace(TreeExplainer=Explainer))
    frame=pd.DataFrame({'a':[0,1],'b':[1,0]})
    assert compute_shap_values(object(),frame,frame,'random_forest') == {'b':7.,'a':3.}


def test_silent_gpu_fallback_is_rejected():
    import pytest
    from src.device_policy import fitted_device,UnsupportedDeviceError
    model=types.SimpleNamespace(get_booster=lambda:types.SimpleNamespace(save_config=lambda:'{"learner":{"generic_param":{"device":"cpu"}}}'))
    with pytest.raises(UnsupportedDeviceError):
        fitted_device(model,'xgboost',True)
