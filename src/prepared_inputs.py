"""Immutable prepared-unit descriptors and verified executed-input artifacts."""
import json
from pathlib import Path
import pandas as pd
from src.artifact_integrity import artifact_lock,atomic_pickle,atomic_json,atomic_bytes,file_sha256,fingerprint,validate_feature_cache

PREPARED_SCHEMA='single_owner_prepared_inputs_v1'


def artifact(path,role,dependency=None):
    return {'path':str(path),'role':role,'sha256':file_sha256(path),'dependency_signature':dependency}


def publish_unit(path,prepared,*,input_artifacts=()):
    """Only a complete descriptor is discoverable; previous verified bytes stay."""
    path=Path(path)
    manifest=path.with_suffix('.json')
    with artifact_lock(path.with_suffix('.lock')):
        if path.exists() and manifest.exists():
            previous=json.loads(manifest.read_text(encoding='utf-8'))
            if previous['sha256'] == file_sha256(path):
                return previous
        pipelines,train_y,test_y,encoder,metadata,_=prepared
        metadata=json.loads(json.dumps({name:{**{key:value for key,value in meta.items() if key != 'selection_history'},'dfs_cache_hit':False} for name,meta in metadata.items()},sort_keys=True))
        value={'schema':PREPARED_SCHEMA,'y_train':train_y,'y_test':test_y,'label_encoder':encoder,
            'metadata':metadata,'input_artifacts':list(input_artifacts),
            'metadata_bytes':{name:Path(meta['metadata_path']).read_bytes() for name,meta in metadata.items()}}
        atomic_pickle(path,value)
        envelope={'schema':PREPARED_SCHEMA,**artifact(path,'prepared_unit_descriptor')}
        atomic_json(manifest,envelope)
        return envelope


def load_pipeline(descriptor,pipeline):
    path=Path(descriptor['path'])
    if descriptor.get('schema') != PREPARED_SCHEMA or file_sha256(path) != descriptor['sha256']:
        raise ValueError('Prepared descriptor identity is incompatible/corrupt')
    unit=pd.read_pickle(path)
    meta=unit['metadata'][pipeline]
    artifacts=meta['artifacts']
    with artifact_lock(Path(meta['metadata_path']).with_suffix('.lock')):
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
    meta={**meta,'dfs_cache_hit':True,'executed_artifacts':[descriptor,*unit['input_artifacts'],*executed]}
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
