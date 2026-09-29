from __future__ import annotations

import importlib.util

import pytest

from src.model import build_model


def test_gpu_routing_is_explicit_for_supported_backends():
    if importlib.util.find_spec("xgboost") is not None:
        model = build_model("xgboost", use_gpu=True)
        assert model.get_params()["device"] == "cuda"
        assert build_model("xgboost", use_gpu=False).get_params()["device"] == "cpu"
    if importlib.util.find_spec("catboost") is not None:
        model = build_model("catboost", use_gpu=True)
        assert model.get_params()["task_type"] == "GPU"
        assert build_model("catboost", use_gpu=False).get_params()["task_type"] == "CPU"


def test_cpu_only_models_remain_cpu_when_gpu_mode_is_requested():
    assert build_model("logistic_regression", use_gpu=True).__class__.__name__ == "LogisticRegression"
    assert build_model("lightgbm", use_gpu=True).get_params().get("device") != "gpu"
