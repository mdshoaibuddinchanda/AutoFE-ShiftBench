"""Immutable prepared-unit descriptors and verified executed-input artifacts."""
import json
import os
import tempfile
import numpy as np
from pathlib import Path
import pandas as pd
from src.artifact_integrity import artifact_lock,atomic_pickle,atomic_json,atomic_bytes,file_sha256,fingerprint,validate_feature_cache

PREPARED_SCHEMA='single_owner_prepared_inputs_v1'
MODEL_ARRAY_POLICY='historical_float32_nan_preserved_inf_1e10_v1'


def historical_model_array(frame):
    return np.nan_to_num(frame.astype(np.float32),nan=np.nan,posinf=1e10,neginf=-1e10)


def _publish_array(path,values):
    path=Path(path)
    with artifact_lock(path.with_suffix('.lock')):
        if path.exists():
            previous=None
            try:
                previous=np.load(path,mmap_mode='r',allow_pickle=False)
                matches=previous.dtype==values.dtype and previous.shape==values.shape and bool(previous.flags.f_contiguous and not previous.flags.c_contiguous)==bool(values.flags.f_contiguous and not values.flags.c_contiguous) and all(np.array_equal(np.ascontiguousarray(previous[i:i+128]).view(np.uint32),np.ascontiguousarray(values[i:i+128]).view(np.uint32)) for i in range(0,len(values),128))
                if matches:return
            except (OSError,ValueError):pass
            finally:
                if previous is not None:previous._mmap.close()
        descriptor,name=tempfile.mkstemp(prefix=path.name+'.tmp_',dir=path.parent)
        temporary=Path(name)
        try:
            with os.fdopen(descriptor,'wb') as output:
                np.save(output,values,allow_pickle=False)
                output.flush();os.fsync(output.fileno())
            os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()


def _model_arrays(path,metadata):
    output={}
    for pipeline,meta in metadata.items():
        output[pipeline]={}
        for partition in ('train','test'):
            original=meta['artifacts'][partition]
            from src.resource_limits import require_bytes
            shape=original['identity']['shape']
            require_bytes(int(np.prod(shape))*24,purpose='exact historical model-array conversion')
            values=historical_model_array(pd.read_pickle(original['path']))
            target=Path(original['path']).with_suffix('.model_float32.npy')
            _publish_array(target,values)
            output[pipeline][partition]={**artifact(target,'model_'+partition+'_numeric_matrix',meta['dependency_signature']),
                'policy':MODEL_ARRAY_POLICY,'dtype':str(values.dtype),'shape':list(values.shape),
                'fortran_order':bool(values.flags.f_contiguous and not values.flags.c_contiguous),
                'selected_feature_identities':meta['selected_feature_identities'],'source_sha256':original['sha256']}
    return output


def artifact(path,role,dependency=None):
    return {'path':str(path),'role':role,'sha256':file_sha256(path),'dependency_signature':dependency}


def unit_artifacts(descriptor):
    unit=pd.read_pickle(descriptor['path'])
    output=[descriptor,*unit['input_artifacts']]
    roles={'train':'selected_training_features','test':'selected_held_out_features','history':'candidate_history','fsva':'fsva_diagnostic'}
    for pipeline,meta in unit['metadata'].items():
        output.append({**artifact(meta['metadata_path'],'feature_metadata',meta['dependency_signature']),'pipeline':pipeline})
        output.extend({'role':roles[role],**item,'pipeline':pipeline,'dependency_signature':meta['dependency_signature']} for role,item in meta['artifacts'].items())
        output.extend({**item,'pipeline':pipeline} for item in unit.get('model_inputs',{}).get(pipeline,{}).values())
    return output


def publish_unit(path,prepared,*,input_artifacts=()):
    """Only a complete descriptor is discoverable; previous verified bytes stay."""
    path=Path(path)
    manifest=path.with_suffix('.json')
    with artifact_lock(path.with_suffix('.lock')):
        if path.exists() and manifest.exists():
            previous=json.loads(manifest.read_text(encoding='utf-8'))
            if previous['sha256'] == file_sha256(path):
                unit=pd.read_pickle(path)
                if unit.get('model_inputs'):
                    # Exact repair of model arrays is independent of runtime metadata.
                    regenerated=_model_arrays(path,unit['metadata'])
                    if regenerated!=unit['model_inputs']:
                        raise ValueError('Regenerated model arrays differ from the immutable descriptor')
                return previous
        pipelines,train_y,test_y,encoder,metadata,_=prepared
        metadata=json.loads(json.dumps({name:{**{key:value for key,value in meta.items() if key != 'selection_history'},'dfs_cache_hit':False} for name,meta in metadata.items()},sort_keys=True))
        value={'schema':PREPARED_SCHEMA,'y_train':train_y,'y_test':test_y,'label_encoder':encoder,
            'metadata':metadata,'input_artifacts':list(input_artifacts),
            'metadata_bytes':{name:Path(meta['metadata_path']).read_bytes() for name,meta in metadata.items()}}
        value['model_inputs']=_model_arrays(path,metadata)
        atomic_pickle(path,value)
        envelope={'schema':PREPARED_SCHEMA,**artifact(path,'prepared_unit_descriptor')}
        atomic_json(manifest,envelope)
        return envelope


def load_pipeline(descriptor,pipeline,*,model_numeric=False):
    opened=[]
    try:
        return _load_pipeline(descriptor,pipeline,model_numeric=model_numeric,opened_arrays=opened)
    except BaseException:
        for values in opened:values._mmap.close()
        raise


def _load_pipeline(descriptor,pipeline,*,model_numeric=False,opened_arrays=None):
    path=Path(descriptor['path'])
    if descriptor.get('schema') != PREPARED_SCHEMA or file_sha256(path) != descriptor['sha256']:
        raise ValueError('Prepared descriptor identity is incompatible/corrupt')
    unit=pd.read_pickle(path)
    meta=unit['metadata'][pipeline]
    artifacts=meta['artifacts']
    with artifact_lock(Path(meta['metadata_path']).with_suffix('.lock')):
        numeric=unit.get('model_inputs',{}).get(pipeline) if model_numeric else None
        if numeric:
            from hashlib import sha256
            if file_sha256(meta['metadata_path'])!=sha256(unit['metadata_bytes'][pipeline]).hexdigest():
                raise ValueError('Prepared feature metadata changed')
            for role in ('train','test'):
                if file_sha256(artifacts[role]['path'])!=artifacts[role]['sha256']:
                    raise ValueError('Prepared selected source bytes changed')
            validated=opened_arrays
            try:
                for role in ('train','test'):
                    spec=numeric[role]
                    if spec['policy']!=MODEL_ARRAY_POLICY or file_sha256(spec['path'])!=spec['sha256'] or spec['source_sha256']!=artifacts[role]['sha256'] or spec['selected_feature_identities']!=meta['selected_feature_identities']:
                        raise ValueError('Prepared numeric matrix policy/bytes/coordinates changed')
                    array=np.load(spec['path'],mmap_mode='r',allow_pickle=False)
                    validated.append(array)
                    if str(array.dtype)!=spec['dtype'] or list(array.shape)!=spec['shape'] or bool(array.flags.f_contiguous and not array.flags.c_contiguous)!=spec['fortran_order']:
                        raise ValueError('Prepared numeric matrix dtype/order/shape mismatch')
            except BaseException:
                for array in validated:array._mmap.close()
                raise
        else:
            validated=validate_feature_cache(artifacts['train']['path'],artifacts['test']['path'],meta['metadata_path'],meta['dependency_signature'])
        if validated is None:
            raise ValueError('Prepared feature cache is incomplete/corrupt')
        if meta['diagnostic_status'] != 'diagnostic_disabled':
            for role in ('history','fsva'):
                if file_sha256(artifacts[role]['path']) != artifacts[role]['sha256']:
                    raise ValueError('Required diagnostic evidence is missing/corrupt')
        executed=[artifact(meta['metadata_path'],'feature_metadata',meta['dependency_signature'])]
        roles={'train':'selected_training_features','test':'selected_held_out_features','history':'candidate_history','fsva':'fsva_diagnostic'}
        executed += [{'role':roles[role],**item,'dependency_signature':meta['dependency_signature']} for role,item in artifacts.items()]
        if numeric:executed.extend(numeric.values())
    meta={**meta,'dfs_cache_hit':True,'model_numeric_policy':MODEL_ARRAY_POLICY if numeric else None,'executed_artifacts':[descriptor,*unit['input_artifacts'],*executed]}
    return {pipeline:(validated[0],validated[1])},unit['y_train'],unit['y_test'],unit['label_encoder'],{pipeline:meta},None


def verify_unit(descriptor, *, check_cancel=None):
    def digest(path):
        if check_cancel is not None:
            check_cancel()
        return file_sha256(path, check_cancel=check_cancel)
    path=Path(descriptor['path'])
    if digest(path) != descriptor['sha256']:
        raise ValueError('Prepared descriptor bytes changed')
    unit=pd.read_pickle(path)
    for matrices in unit.get('model_inputs',{}).values():
        for item in matrices.values():
            if digest(item['path'])!=item['sha256']:raise ValueError('Prepared model numeric artifact missing/corrupt')
    for item in unit['input_artifacts']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Prepared input artifact missing/corrupt')
    for name,meta in unit['metadata'].items():
        for item in meta['artifacts'].values():
            if digest(item['path']) != item['sha256']:
                raise ValueError('Prepared feature/diagnostic artifact missing/corrupt')
        if digest(meta['metadata_path']) != __import__('hashlib').sha256(unit['metadata_bytes'][name]).hexdigest():
            raise ValueError('Feature metadata changed')
    return unit


def restore_metadata_after_exact_repair(descriptor):
    """Restore original immutable metadata only if regenerated science hashes match."""
    path=Path(descriptor['path'])
    if file_sha256(path) != descriptor['sha256']:
        raise ValueError('Descriptor cannot establish repair identity')
    unit=pd.read_pickle(path)
    for name,meta in unit['metadata'].items():
        if any(file_sha256(item['path']) != item['sha256'] for item in meta['artifacts'].values()):
            raise ValueError('Regeneration changed scientific artifacts; exact repair unsupported')
        with artifact_lock(Path(meta['metadata_path']).with_suffix('.lock')):
            atomic_bytes(meta['metadata_path'],unit['metadata_bytes'][name])
    verify_unit(descriptor)
