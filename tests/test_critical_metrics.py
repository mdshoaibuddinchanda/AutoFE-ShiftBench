"""Independent metric fixtures, including the production encoded-label defect."""
import numpy as np
import pytest

from src.evaluation import compute_classification_metrics


def test_encoded_targets_cannot_be_evaluated_against_original_strings():
    with pytest.raises(ValueError, match="class"):
        compute_classification_metrics(np.array([0,1,2]), np.array([0,1,2]),
            np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]), np.array(["a","b","c"]))


def test_multiclass_hand_calculation():
    result = compute_classification_metrics(np.array([0,1,2]), np.array([0,1,2]),
        np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]), np.array([0,1,2]))
    assert result["log_loss"] == pytest.approx(-np.log(.8))
    assert result["brier_score"] == pytest.approx((.2**2+.1**2+.1**2)/3)
    assert result["pr_auc"] == pytest.approx(1)
    assert result["roc_auc"] == pytest.approx(1)


def test_probability_column_order_is_explicit():
    result = compute_classification_metrics(np.array([0,1]), np.array([0,1]),
        np.array([[.2,.8],[.9,.1]]), np.array([0,1]), probability_classes=np.array([1,0]))
    assert result["log_loss"] == pytest.approx((-np.log(.8)-np.log(.9))/2)
    assert result["brier_score"] == pytest.approx((.2**2+.1**2)/2)


@pytest.mark.parametrize("probabilities", [np.array([[.5,.6],[.1,.9]]), np.array([[-.1,1.1],[.1,.9]]), np.array([[np.nan,.5],[.1,.9]]), np.array([[.8],[.9]])])
def test_invalid_probabilities_have_explicit_error_state(probabilities):
    with pytest.raises(ValueError, match="probab"):
        compute_classification_metrics(np.array([0,1]), np.array([0,1]), probabilities, np.array([0,1]))


def test_missing_held_out_class_is_explicitly_undefined():
    result = compute_classification_metrics(np.array([0,0]), np.array([0,0]), np.array([[.8,.2],[.9,.1]]), np.array([0,1]))
    assert np.isnan(result["roc_auc"])
    assert result["metric_status"]["roc_auc"].startswith("undefined")
    assert result["log_loss"] == pytest.approx((-np.log(.8)-np.log(.9))/2)


def test_absent_estimator_class_has_zero_probability_column():
    result = compute_classification_metrics(np.array([0,2]), np.array([0,2]), np.array([[.8,.2],[.1,.9]]), np.array([0,1,2]), probability_classes=np.array([0,2]))
    assert result["log_loss"] == pytest.approx((-np.log(.8)-np.log(.9))/2)
    assert np.isnan(result["roc_auc"])


def test_macro_f1_has_a_distinct_identity():
    result = compute_classification_metrics(np.array([0,0,0,1]), np.array([0,0,0,0]), np.array([[.8,.2]]*4), np.array([0,1]))
    assert result["f1"] == 0
    assert result["f1_macro"] == pytest.approx(3/7)


def test_float32_probability_mass_uses_accurate_sum_without_mutation():
    # Actual Dry Bean/XGBoost row from the bounded real-data preflight.
    row = np.array([1.4195753692547441e-06, 2.67126279140939e-06,
        3.968446890212363e-06, 0.00016821626923047006,
        9.655785788709181e-07, 0.9997901320457458,
        3.266586645622738e-05], dtype=np.float32)
    assert abs(float(row.sum()) - 1) > 1e-7
    assert abs(float(row.sum(dtype=np.float64)) - 1) < 1e-7
    probabilities = np.array([np.roll(row, i) for i in range(7)])
    before = probabilities.copy()
    classes = np.arange(7)
    predicted = probabilities.argmax(axis=1)
    result = compute_classification_metrics(classes, predicted, probabilities, classes)
    from sklearn.metrics import log_loss, roc_auc_score
    assert result['log_loss'] == log_loss(classes, probabilities, labels=classes)
    assert result['roc_auc'] == roc_auc_score(classes, probabilities, labels=classes, multi_class='ovr', average='macro')
    assert probabilities.dtype == np.float32
    np.testing.assert_array_equal(probabilities, before)


def test_probability_mass_tolerance_remains_1e_7():
    # Accurate mass still outside the original tolerance is rejected.
    probabilities = np.array([[.5, .5000002], [.25, .75]], dtype=np.float32)
    assert abs(float(probabilities[0].sum(dtype=np.float64)) - 1) > 1e-7
    with pytest.raises(ValueError, match='sum to one'):
        compute_classification_metrics(np.array([0, 1]), np.array([0, 1]), probabilities, np.array([0, 1]))
