import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.task_manifest import build_task_records
from src import pipeline_runner as runner


def test_same_path_different_bytes_changes_task_identity(tmp_path):
    data = tmp_path / "d.csv"
    data.write_text("x,target\n1,a\n2,b\n")
    kwargs = dict(datasets=["d"], seeds=[42], folds=[1], conditions=[("clean",0)],pipelines=["Raw"],models=["logistic_regression"],data_paths={"d":data})
    before = build_task_records(**kwargs)
    data.write_text("x,target\n9,a\n2,b\n")
    after = build_task_records(**kwargs)
    assert {r["scientific_task_id"] for r in before}.isdisjoint({r["scientific_task_id"] for r in after})


def test_changed_matrix_does_not_reuse_previous_feature_cache(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = pd.DataFrame({"x":[1.,2.,3.],"y":[2.,4.,8.]})
    first = runner._run_pipeline_generation(x,x,np.array([0,1,0]),"d","stratified",42,1,"clean")
    changed = x * 7
    second = runner._run_pipeline_generation(changed,changed,np.array([0,1,0]),"d","stratified",42,1,"clean")
    assert not second[1]["Raw"]["dfs_cache_hit"]
    assert not first[0]["Raw"][0].equals(second[0]["Raw"][0])


def test_corrupted_cache_cannot_pass_compatibility(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = pd.DataFrame({"x":[1.,2.,3.]})
    before = runner._run_pipeline_generation(x,x,np.array([0,1,0]),"d","stratified",42,1,"clean")
    files = list(Path("data/cache").rglob("*_train.pkl"))
    for path in files:
        pd.DataFrame({"wrong":[999.]}).to_pickle(path)
    after = runner._run_pipeline_generation(x,x,np.array([0,1,0]),"d","stratified",42,1,"clean")
    for name in runner.PIPELINE_NAMES:
        pd.testing.assert_frame_equal(before[0][name][0],after[0][name][0])
        assert not after[1][name]["dfs_cache_hit"]


def test_valid_cache_reuse_preserves_features_and_decisions(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = pd.DataFrame({"x":[1.,2.,3.],"y":[4.,2.,1.]})
    before = runner._run_pipeline_generation(x,x,np.array([0,1,0]),"d","stratified",42,1,"clean")
    after = runner._run_pipeline_generation(x,x,np.array([0,1,0]),"d","stratified",42,1,"clean")
    for name in runner.PIPELINE_NAMES:
        pd.testing.assert_frame_equal(before[0][name][0],after[0][name][0])
        assert before[1][name]["selected_feature_identities"] == after[1][name]["selected_feature_identities"]
        assert after[1][name]["dfs_cache_hit"]
