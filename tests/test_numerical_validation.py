"""Independent represented-output fixtures and adversarial validation checks."""
from decimal import Decimal, localcontext
import hashlib

import numpy as np
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from src.evaluation import compute_classification_metrics
from src.numerical_validation import (BASELINE_MASS_ATOL, LEGACY_VALIDATION_VERSION,
    NUMERICAL_VALIDATION_VERSION, mass_error_budget, validate_probability_mass)
from src.task_manifest import ManifestConflictError, ManifestStore, build_task_records
from src.seeding import estimator_seed


MLP_ROW = [2.5346682530624776e-09, 7.732123208370467e-08,
    5.018990778538068e-10, 0.9999336004257202, 6.025740003678948e-05,
    6.235146798871938e-08, 5.789900569652673e-06]


def test_actual_mlp_preserves_entries_and_reference_metrics():
    row = np.asarray(MLP_ROW, dtype=np.float32)
    assert abs(row.sum(dtype=np.float64) - 1) == 2.095644059396662e-7
    probabilities = np.array([np.roll(row, i) for i in range(7)])
    digest = hashlib.sha256(probabilities.tobytes()).hexdigest()
    labels = np.arange(7)
    result = compute_classification_metrics(labels, probabilities.argmax(1), probabilities, labels)
    assert not result['probability_validation']['strict_legacy_pass']
    assert result['probability_validation']['baseline_mass_atol'] == 1e-7
    assert result['log_loss'] == log_loss(labels, probabilities, labels=labels)
    assert result['roc_auc'] == roc_auc_score(labels, probabilities, labels=labels,
                                             multi_class='ovr', average='macro')
    assert hashlib.sha256(probabilities.tobytes()).hexdigest() == digest
    assert probabilities.dtype == np.float32
    with pytest.raises(ValueError, match='sum to one'):
        validate_probability_mass(probabilities, LEGACY_VALIDATION_VERSION)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('columns', [2, 7, 21])
def test_bound_independently_derived_with_decimal(dtype, columns):
    with localcontext() as ctx:
        ctx.prec = 70
        u = Decimal(2) ** (-24 if dtype == np.float32 else -53)
        gamma = (columns-1)*u / (1-(columns-1)*u)
        normal = (gamma+u)/(1-gamma)
        subnormal = columns*(Decimal(2) ** (-149 if dtype == np.float32 else -1074))/2
        u64 = Decimal(2) ** -53
        gamma64 = (columns-1)*u64/(1-(columns-1)*u64)
        bound = normal+subnormal+gamma64*(1+normal+subnormal)
    budget = mass_error_budget(dtype, columns)
    assert budget['additional_rounding_allowance'] == pytest.approx(float(bound), rel=2e-15)
    assert budget['acceptance_limit'] == BASELINE_MASS_ATOL+budget['additional_rounding_allowance']
    # Material errors beyond this class/dtype-derived budget must fail.
    bad = np.full((1, columns), 1/columns, dtype=dtype)
    bad[0, 0] += 4*budget['acceptance_limit']
    with pytest.raises(ValueError, match='sum to one'):
        validate_probability_mass(bad)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_valid_binary_and_multiclass(dtype):
    for values in ([[.75,.25],[.1,.9]], [[.7,.2,.1],[.1,.2,.7]]):
        info = validate_probability_mass(np.array(values, dtype=dtype))
        assert info['strict_legacy_pass']
        assert info['version'] == NUMERICAL_VALIDATION_VERSION


@pytest.mark.parametrize('values', [[[-.01,1.01]], [[.2,np.inf]], [[np.nan,.5]],
                                    [[.5,.51]], [[.4,.4]], [[1.1,0.]]])
def test_malformed_outputs_rejected(values):
    with pytest.raises(ValueError):
        validate_probability_mass(np.array(values, dtype=np.float32))


@pytest.mark.parametrize('columns, probabilities', [([0,0], [[.5,.5],[.5,.5]]),
    ([0,3], [[.5,.5],[.5,.5]]), ([0,1,2], [[.5,.5],[.5,.5]])])
def test_wrong_duplicate_and_missing_columns(columns, probabilities):
    with pytest.raises(ValueError, match='probability'):
        compute_classification_metrics(np.array([0,1]), np.array([0,1]),
            np.array(probabilities), np.arange(3), probability_classes=np.array(columns))


def test_no_implicit_missing_class_and_undefined_cases():
    with pytest.raises(ValueError, match='dimensions'):
        compute_classification_metrics(np.array([0,1]), np.array([0,1]),
                                       np.array([[.8,.2],[.2,.8]]), np.arange(3))
    result = compute_classification_metrics(np.array([0,0]), np.array([0,0]),
                                          np.array([[.8,.2],[.9,.1]]), np.arange(2))
    assert result['metric_status']['roc_auc'] == 'undefined_missing_held_out_class'
    assert result['metric_status']['log_loss'] == 'complete'
    assert compute_classification_metrics(np.array([]), np.array([]), None,
             np.arange(2))['metric_status']['accuracy'] == 'undefined_empty_target'


def test_unsupported_precision_and_version():
    with pytest.raises(ValueError, match='float32/float64'):
        validate_probability_mass(np.array([[.5,.5]], dtype=np.float16))
    with pytest.raises(ValueError, match='version'):
        validate_probability_mass(np.array([[.5,.5]]), 'future_unknown')


def test_validation_changes_model_identity_without_changing_purpose_seed(tmp_path):
    args = (['d'], [42], [1], [('clean',0)], ['Raw'], ['mlp'])
    old = build_task_records(*args, numerical_validation_version=LEGACY_VALIDATION_VERSION)
    new = build_task_records(*args)
    assert old[0]['scientific_task_id'] == new[0]['scientific_task_id']
    assert old[1]['scientific_task_id'] != new[1]['scientific_task_id']
    seeds = [estimator_seed(r['dataset'],r['split_policy'],r['seed'],r['fold'],r['condition'],r['model'])
             for r in (old[1],new[1])]
    assert seeds[0] == seeds[1]
    store = ManifestStore(tmp_path/'manifest.db')
    store.create_run('r', {'numerical_validation_version': NUMERICAL_VALIDATION_VERSION}, new)
    attempt = store.claim_task('r', new[0]['scientific_task_id'])
    store.commit_result('r',new[0]['scientific_task_id'],attempt, {'status':'success'})
    attempt = store.claim_task('r',new[1]['scientific_task_id'])
    with pytest.raises(ManifestConflictError, match='validation'):
        store.commit_result('r',new[1]['scientific_task_id'],attempt,
                            {'numerical_validation_version':LEGACY_VALIDATION_VERSION})


def test_stale_production_claim_refuses_before_any_fit(tmp_path):
    from src.pipeline_runner import _claim_manifest_task
    records = build_task_records(['d'],[42],[1],[('clean',0)],['Raw'],['mlp'],
                                 numerical_validation_version=LEGACY_VALIDATION_VERSION)
    store = ManifestStore(tmp_path/'manifest.db');store.create_run('old',{},records)
    task = {**records[1],'manifest_db':str(store.db_path),'run_id':'old'}
    with pytest.raises(ManifestConflictError, match='Stale numerical'):
        _claim_manifest_task(task,worker_id='test')
    assert store.attempt_counts('old') == {}
