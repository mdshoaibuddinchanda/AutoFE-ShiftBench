from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src import evaluation, pipeline_runner as runner


@pytest.mark.parametrize('classes', [None, np.array(['a', 'b']), np.array(['a', 'b', 'c'])])
def test_training_auc_matches_full_metrics(classes):
    count = 3 if classes is not None and len(classes) == 3 else 2
    labels = np.tile(np.arange(count), 10)
    probability = np.full((len(labels), count), .1/(count-1))
    probability[np.arange(len(labels)), labels] = .9
    expected = evaluation.compute_classification_metrics(labels, labels, probability, classes)['roc_auc']
    assert evaluation.compute_training_roc_auc(labels, probability, classes) == expected


@pytest.mark.parametrize('probability', [None, np.empty((0, 2)), np.full((4, 2), np.nan)])
def test_training_auc_retains_undefined_values(probability):
    assert np.isnan(evaluation.compute_training_roc_auc(np.array([0, 1, 0, 1]), probability, np.array(['a', 'b'])))


def test_training_auc_retains_joint_pr_exception_boundary(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('PR failure')
    monkeypatch.setattr(evaluation, 'average_precision_score', fail)
    labels = np.array([0, 1, 0, 1])
    probability = np.array([[.8, .2], [.2, .8], [.7, .3], [.1, .9]])
    assert np.isnan(evaluation.compute_classification_metrics(labels, labels, probability)['roc_auc'])
    assert np.isnan(evaluation.compute_training_roc_auc(labels, probability))


def test_worker_does_not_request_training_hard_predictions(monkeypatch):
    calls = []
    class Model:
        def fit(self, x, y): pass
        def predict(self, x):
            calls.append(len(x))
            return (x[:, 0] > 0).astype(int)
        def predict_proba(self, x):
            p = np.where(x[:, 0] > 0, .8, .2)
            return np.column_stack([1-p, p])
        def get_params(self, deep=False): return {}
    monkeypatch.setattr(runner, 'build_model', lambda *args, **kwargs: Model())
    x = np.array([[-1], [1], [-2], [2], [-3], [3]], dtype=np.float32)
    result = runner._fit_and_score_worker(dict(model='gaussian_nb', seed=42, xtr=x,
        xte=x[:4], y_train_enc=(x[:, 0] > 0).astype(int),
        y_test_enc=(x[:4, 0] > 0).astype(int), classes=np.array(['a', 'b'])))
    assert calls == [4]
    assert result['train_auc'] == 1
    assert result['train_infer_time_s'] >= 0 and result['scoring_time_s'] >= 0


@pytest.mark.parametrize('policy', ['row_level', 'group_aware'])
def test_shared_preprocessing_is_once_per_input_and_preserves_outputs(tmp_path, monkeypatch, policy):
    rng = np.random.default_rng(18)
    frame = pd.DataFrame(rng.normal(size=(120, 5)), columns=list('abcde'))
    frame['category'] = np.where(frame.a > 0, 'red', 'blue')
    frame['target'] = (frame.a+frame.b > 0).astype(int)
    path = tmp_path/'inputs.csv'; frame.to_csv(path, index=False)
    calls = []
    original = runner._preprocess_task
    def counted(*args, **kwargs):
        calls.append(kwargs['family'])
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, '_preprocess_task', counted)
    common = dict(data_paths={'inputs':path}, output_root=tmp_path, seeds=[42], folds=[1],
                  n_splits=3, conditions=(('clean',0.),('gaussian_noise',.05)),
                  split_policy=policy, pipelines=('Raw','AutoFE_Baseline','AutoFE_LeaveOut_Multiply'),
                  models=('gaussian_nb','logistic_regression'))
    first = runner.run_experiment(**common,run_id='reused')
    assert first['counts_by_status']['success'] == 12 and len(calls) == 2
    calls.clear()
    second = runner.run_experiment(**common,run_id='unreused',reuse_preprocessing=False)
    assert second['counts_by_status']['success'] == 12 and len(calls) == 6
    from provenance.final_optimization_verification import FIELDS, rows
    a,b = rows(tmp_path/'reused/results.jsonl'),rows(tmp_path/'unreused/results.jsonl')
    assert set(a) == set(b)
    assert all({k:r.get(k) for k in FIELDS} == {k:b[key].get(k) for k in FIELDS} for key,r in a.items())
    with pytest.raises(ValueError,match='different code/configuration/runtime'):
        runner.run_experiment(**common,run_id='reused',reuse_preprocessing=False)


def test_preprocessing_input_ownership_isolated_between_variants():
    x = pd.DataFrame({'a':[-2.,-1.,1.,2.], 'b':[1.,2.,3.,4.]})
    y = pd.Series([0,0,1,1])
    prepared = runner._preprocess_task(x,y,x,family='clean',severity=0,perturbation_seed=42)
    before = prepared[0].copy(deep=True)
    result = runner._prepare_matrices(x,y,x,family='clean',severity=0,perturbation_seed=42,
        pipeline_name='Raw',config=runner.PIPELINE_CONFIGS['Raw'],preprocessed=prepared)
    result[0].iloc[:,:] = 999
    pd.testing.assert_frame_equal(prepared[0], before)
