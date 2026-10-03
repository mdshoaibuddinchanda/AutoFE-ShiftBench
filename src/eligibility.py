"""Consume verified split feasibility before production preparation dispatch."""
import json
from importlib.metadata import version
from pathlib import Path

from src.artifact_integrity import file_sha256
from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION
from src.task_manifest import scientific_task_id, _normalise_condition

SCHEMA = 'verified_split_eligibility_v1'
SPLIT_SOURCES = ('src/splitters.py', 'src/seeding.py', 'src/protocol.py',
                 'src/data_loader.py', 'src/shift_generator.py', 'config/dataset_list.yaml')
SPLIT_DEPENDENCIES = ('numpy', 'pandas', 'scipy', 'scikit-learn')


def split_dependency_versions():
    return {name: version(name) for name in SPLIT_DEPENDENCIES}


def unit_identity(name, seed, fold, family, severity, data_identity):
    _, _, condition = _normalise_condition((family, severity))
    return scientific_task_id({'eligibility_identity_version': SCHEMA,
        'dataset': name, 'data_identity': data_identity,
        'protocol_version': EVALUATION_PROTOCOL_VERSION, 'seed_scheme_version': SEED_SCHEME_VERSION,
        'n_splits': 5, 'seed': seed, 'fold': fold, 'shift_family': family,
        'severity': severity, 'condition': condition,
        'split_policy': family if family in ('covariate_shift', 'population_shift') else 'stratified'})


def compile_preflight_eligibility(preflight, repo_root='.'):
    """Repackage existing verified evidence; no split refit or data acquisition."""
    root = Path(repo_root); preflight = Path(preflight)
    config = json.loads((preflight / 'full_manifest_config.json').read_text())
    if file_sha256(preflight / 'eligibility.json') != config['eligibility_sha256']:
        raise ValueError('Preflight eligibility evidence hash differs from its frozen configuration')
    if config['protocol_version'] != EVALUATION_PROTOCOL_VERSION or config['seed_scheme_version'] != SEED_SCHEME_VERSION:
        raise ValueError('Preflight scientific protocol is incompatible')
    for name in SPLIT_SOURCES:
        if file_sha256(root / name) != config['code_identity']['source_content_hashes'][name]:
            raise ValueError('Split source differs from verified preflight: ' + name)
    data = json.loads((preflight / 'datasets.json').read_text())
    identities = {d['dataset']: d['identity'] for d in data}
    conditions = {_normalise_condition(c)[2]: c for c in config['conditions']}
    units = []
    for row in json.loads((preflight / 'eligibility.json').read_text()):
        family, severity = conditions[row['condition']]
        units.append({**row, 'shift_family': family, 'severity': severity,
            'unit_identity': unit_identity(row['dataset'], row['seed'], row['fold'], family, severity, identities[row['dataset']])})
    return {'schema': SCHEMA, 'protocol_version': EVALUATION_PROTOCOL_VERSION,
        'seed_scheme_version': SEED_SCHEME_VERSION, 'n_splits': 5,
        'configuration': {k: config[k] for k in ('datasets', 'seeds', 'folds', 'conditions', 'pipelines', 'models')},
        'dataset_identities': identities,
        'metadata_sha256': {d['dataset']: d['metadata_sha256'] for d in data},
        'split_dependency_versions': {name: config['environment_identity']['resolved_dependencies'][name]
                                      for name in SPLIT_DEPENDENCIES},
        'split_source_sha256': {name: file_sha256(root / name) for name in SPLIT_SOURCES},
        'source_eligibility_sha256': config['eligibility_sha256'], 'units': units}


def load_verified_eligibility(path, expected_sha256, *, datasets, data_identities,
                              seeds, folds, conditions, repo_root='.'):
    """The caller pins the file hash in its run config/launch record."""
    path = Path(path); root = Path(repo_root)
    if not expected_sha256 or file_sha256(path) != expected_sha256:
        raise ValueError('Eligibility file hash is stale or tampered')
    value = json.loads(path.read_text())
    if value.get('schema') != SCHEMA or value.get('n_splits') != 5 or value.get('protocol_version') != EVALUATION_PROTOCOL_VERSION or value.get('seed_scheme_version') != SEED_SCHEME_VERSION:
        raise ValueError('Eligibility schema/protocol/seed/fold contract incompatible')
    config = value['configuration']
    if value.get('split_dependency_versions') != split_dependency_versions():
        raise ValueError('Eligibility split dependency environment is stale')
    if not set(datasets).issubset(config['datasets']) or not set(seeds).issubset(config['seeds']) or not set(folds).issubset(config['folds']):
        raise ValueError('Requested units are outside verified eligibility configuration')
    verified_conditions = {_normalise_condition(c)[2]: tuple(c) for c in config['conditions']}
    for family, severity in conditions:
        if verified_conditions.get(_normalise_condition((family, severity))[2]) != (family, severity):
            raise ValueError('Condition differs from verified eligibility')
    for name, digest in value['split_source_sha256'].items():
        if name not in SPLIT_SOURCES or file_sha256(root / name) != digest:
            raise ValueError('Eligibility split source is stale: ' + name)
    if set(value['split_source_sha256']) != set(SPLIT_SOURCES):
        raise ValueError('Eligibility split source binding is incomplete')
    for name in datasets:
        if data_identities.get(name) != value['dataset_identities'].get(name):
            raise ValueError('Eligibility dataset identity is stale: ' + name)
        if file_sha256(root / 'data/raw' / (name + '_meta.json')) != value['metadata_sha256'].get(name):
            raise ValueError('Eligibility dataset metadata is stale: ' + name)
    expected = {(d, s, f, c) for d in config['datasets'] for s in config['seeds']
                for f in config['folds'] for c in verified_conditions}
    seen = set(); planned = {}; eligible = 0
    for row in value['units']:
        key = (row['dataset'], row['seed'], row['fold'], row['condition'])
        if key in seen or key not in expected:
            raise ValueError('Duplicate or undeclared eligibility unit')
        seen.add(key)
        family, severity = verified_conditions[row['condition']]
        if (row['shift_family'], row['severity']) != (family, severity) or row['unit_identity'] != unit_identity(row['dataset'], row['seed'], row['fold'], family, severity, value['dataset_identities'][row['dataset']]):
            raise ValueError('Eligibility unit identity mismatch')
        policy = family if family in ('covariate_shift', 'population_shift') else 'stratified'
        if row['policy'] != policy:
            raise ValueError('Eligibility split policy mismatch')
        if row['status'] == 'scientifically_infeasible':
            reason = row.get('reason')
            if not isinstance(reason, str) or not reason.startswith(('SplitInfeasibleError:', 'held_out_classes_absent_from_training:')):
                raise ValueError('Unverified scientific infeasibility reason')
            planned[key] = 'scientific_infeasibility:' + reason
        elif row['status'] == 'split_eligible' and row.get('reason') is None:
            eligible += 1
        else:
            raise ValueError('Eligibility status is not a declared scientific split outcome')
    if seen != expected:
        raise ValueError('Eligibility mapping has missing units')
    summary = {'schema': SCHEMA, 'sha256': expected_sha256, 'units': len(seen),
               'split_eligible': eligible, 'planned_infeasible': len(planned)}
    return planned, summary
