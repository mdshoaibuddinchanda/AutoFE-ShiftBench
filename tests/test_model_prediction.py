import numpy as np
import pytest
from src.model import build_model
from src.model_prediction import predict_labels_and_probabilities


@pytest.mark.parametrize('name',['random_forest','extra_trees','linear_svm','knn','logistic_regression','gaussian_nb','mlp','lightgbm'])
def test_actual_factory_labels_and_probabilities_match_original_calls(name):
    x=np.random.default_rng(616).normal(size=(90,6)).astype(np.float32)
    y=np.arange(90)%3
    model=build_model(name,random_state=42,use_gpu=False)
    model.fit(x,y)
    expected=model.predict(x);probability=model.predict_proba(x)
    labels,actual=predict_labels_and_probabilities(model,x)
    np.testing.assert_array_equal(labels,expected,strict=True)
    np.testing.assert_array_equal(actual,probability,strict=True)


def test_supported_rule_reuses_one_probability_call_with_exact_ties(monkeypatch):
    model=build_model('random_forest',use_gpu=False)
    model.classes_=np.array([0,1])
    calls=[]
    def probabilities(values):calls.append(1);return np.array([[.5,.5],[.2,.8]])
    monkeypatch.setattr(model,'predict_proba',probabilities)
    labels,p=predict_labels_and_probabilities(model,np.zeros((2,2)))
    np.testing.assert_array_equal(labels,[0,1])
    assert calls==[1]
