"""Verify executed estimator devices; no silent GPU-to-CPU substitution."""
import json
import numpy as np

DEVICE_POLICY_VERSION='actual_fit_device_no_fallback_v1'


class UnsupportedDeviceError(RuntimeError):
    pass


def fitted_device(model,model_type,requested_gpu):
    actual='cpu'
    if model_type == 'xgboost':
        actual=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    elif model_type == 'catboost':
        actual=str(model.get_all_params().get('task_type','CPU')).lower()
    if requested_gpu and not (actual.startswith('cuda') or actual == 'gpu'):
        raise UnsupportedDeviceError(f'{model_type} requested GPU but actual fitted device is {actual}')
    return {'device_policy_version':DEVICE_POLICY_VERSION,'requested_device':'gpu' if requested_gpu else 'cpu',
        'actual_device':actual,'fallback_policy':'unsupported task; never substitute estimator',
        'estimator_input_dtype':'float32 (historical model input policy)',
        'determinism':'catboost GPU reductions may be nondeterministic' if model_type == 'catboost' and requested_gpu else 'environment/kernel dependent'}


def probe_gpu(model_type):
    from src.model import build_model
    x=np.random.default_rng(31).normal(size=(24,3)).astype(np.float32)
    model=build_model(model_type,random_state=42,use_gpu=True)
    model.fit(x,np.tile([0,1],12))
    return fitted_device(model,model_type,True)
