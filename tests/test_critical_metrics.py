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
