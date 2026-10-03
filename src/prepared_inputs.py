"""Immutable prepared-unit descriptors and verified executed-input artifacts."""
import json
from pathlib import Path
import pandas as pd
from src.artifact_integrity import artifact_lock,atomic_pickle,atomic_json,file_sha256,fingerprint,validate_feature_cache

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
        value={'schema':PREPARED_SCHEMA,'y_train':train_y,'y_test':test_y,'label_encoder':encoder,
            'metadata':metadata,'input_artifacts':list(input_artifacts)}
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
        executed += [{'role':role,**item,'dependency_signature':meta['dependency_signature']} for role,item in artifacts.items()]
    meta={**meta,'dfs_cache_hit':True,'executed_artifacts':[descriptor,*unit['input_artifacts'],*executed]}
    return {pipeline:(validated[0],validated[1])},unit['y_train'],unit['y_test'],unit['label_encoder'],{pipeline:meta},None
