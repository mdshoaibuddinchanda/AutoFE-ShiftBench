"""Saved bounded evidence must bind its design, source and scientific output."""
import json
from itertools import product

import pytest

from provenance import final_optimization_verification as verification
from src import provenance


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(verification, 'ROOT', tmp_path)
    monkeypatch.setattr(provenance, 'code_fingerprint', lambda root: 'new-source')
    measurements, small, large = [], [], []
    for policy in ('row_level', 'group_aware'):
        for label in ('baseline', 'optimized', 'optimized_covertype'):
            is_large = label == 'optimized_covertype'
            datasets = ['covertype'] if is_large else ['sonar', 'heart-disease', 'haberman', 'ionosphere']
            pipelines = ['Raw'] if is_large else list(verification.PIPELINES)
            run_id = f'{label}-{policy}'
            directory = tmp_path / 'corrected_runs' / 'final_optimization' / run_id
            directory.mkdir(parents=True)
            config = dict(datasets=datasets, pipelines=pipelines, models=list(verification.MODELS),
                          seeds=[42], folds=[1], conditions=[['clean', 0.0]], n_splits=5,
                          use_gpu=False, workers=4, split_policy=policy,
                          numerical_thread_environment=verification.THREADS,
                          scheduler_lease_seconds=600, cache_policy='bounded', cache_max_bytes=8*1024**3)
            expected = len(datasets)*len(pipelines)*10
            source = 'old-source' if label == 'baseline' else 'new-source'
            (directory / 'manifest.json').write_text(json.dumps(dict(
                status='complete', configuration=config, code_fingerprint=source,
                expected_tasks=expected, counts_by_status={'success': expected})))
            records = [dict(dataset=d, seed=42, fold=1, condition='clean', pipeline=p,
                            model=m, split_policy=policy, status='success', roc_auc=.75,
                            log_loss=.5, f1=.7, train_matrix_sha256='train',
                            test_matrix_sha256='test', prediction_sha256='pred')
                       for d,p,m in product(datasets,pipelines,verification.MODELS)]
            ledger = ''.join(json.dumps(row)+'\n' for row in records)
            (directory / 'results.jsonl').write_text(ledger)
            if is_large:
                old_version = '002' if policy == 'row_level' else '003'
                prior = tmp_path / 'corrected_runs' / 'large_calibration' / f'calibration-{policy}-{old_version}'
                prior.mkdir(parents=True)
                (prior / 'results.jsonl').write_text(ledger)
                large.append(dict(policy=policy, run_id=run_id, compared_cells=10, mismatches=[]))
            measurements.append(dict(run_id=run_id, label=label, policy=policy,
                                     expected=expected, code_fingerprint=source))
        small.append(dict(policy=policy, compared_cells=560, mismatches=[]))
    return dict(code_fingerprint='new-source', baseline_code_fingerprint='old-source',
                measurements=measurements, small_dataset_parity=small, large_dataset_parity=large)


def test_saved_complete_same_design_evidence_verifies(tmp_path, monkeypatch):
    report = verification.verify_saved_report(_fixture(tmp_path, monkeypatch))
    assert len(report['per_run_evidence']) == 6
    assert 'log_loss' in report['scientific_fields']


def test_saved_measurements_cannot_stamp_a_new_source_over_old_runs(tmp_path, monkeypatch):
    report = _fixture(tmp_path, monkeypatch)
    next(row for row in report['measurements'] if row['label']=='optimized')['code_fingerprint']='stale'
    with pytest.raises(ValueError, match='Mixed'):
        verification.verify_saved_report(report)


def test_same_cell_count_cannot_replace_the_frozen_seed(tmp_path, monkeypatch):
    report = _fixture(tmp_path, monkeypatch)
    path=tmp_path/'corrected_runs/final_optimization/optimized-row_level/manifest.json'
    manifest=json.loads(path.read_text()); manifest['configuration']['seeds']=[123]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='design differs'):
        verification.verify_saved_report(report)


def test_probability_metric_parity_required_even_when_prediction_and_auc_match(tmp_path, monkeypatch):
    report = _fixture(tmp_path, monkeypatch)
    path=tmp_path/'corrected_runs/final_optimization/optimized-row_level/results.jsonl'
    records=list(map(json.loads,path.read_text().splitlines())); records[0]['log_loss']=.6
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))
    with pytest.raises(ValueError, match='log_loss'):
        verification.verify_saved_report(report)
