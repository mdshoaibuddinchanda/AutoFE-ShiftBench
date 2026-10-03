import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.dataset_statistics import AnalysisConfig,AnalysisInputError,analyze_ledger
from src.sensitivity_analysis import SensitivityConfig,_dataset_summary,snapshot_manifest_ledger
from tests.test_dataset_statistics import _record
from tests import test_sensitivity_analysis as sensitivity_tests


def ledger(tmp_path,rows):
    path=tmp_path/"results.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return path


def test_failed_right_only_row_is_supported(tmp_path):
    bundle=analyze_ledger(ledger(tmp_path,[_record("d",1,"AutoFE_Baseline",None,status="failed")]),AnalysisConfig(bootstrap_resamples=0,permutation_resamples=0))
    assert bundle.task_pairs.iloc[0]["pair_status"] == "missing_partner"
    assert bundle.summaries.iloc[0]["estimate_b_minus_a"] is None


def test_out_of_range_auc_is_not_a_valid_pair(tmp_path):
    bundle=analyze_ledger(ledger(tmp_path,[_record("d",1,"Raw",.5),_record("d",1,"AutoFE_Baseline",1.5)]))
    assert bundle.task_pairs.iloc[0]["pair_status"] != "paired"
    assert "metric_outside_declared_range" in set(bundle.exclusions["reason"])


def test_both_comparison_sides_absent_is_explicit(tmp_path):
    bundle=analyze_ledger(ledger(tmp_path,[]),AnalysisConfig(run_id="r"))
    assert bundle.task_pairs.empty and bundle.summaries.empty
    assert "empty_selected_run_ledger" in set(bundle.exclusions["reason"])


def test_disjoint_runs_cannot_be_combined(tmp_path):
    rows=[{**_record("d1",1,"Raw",.5),"run_id":"r1"},{**_record("d1",1,"AutoFE_Baseline",.7),"run_id":"r1"},
        {**_record("d2",1,"Raw",.5),"run_id":"r2"},{**_record("d2",1,"AutoFE_Baseline",.9),"run_id":"r2"}]
    path=ledger(tmp_path,rows)
    with pytest.raises(AnalysisInputError,match="run"):
        analyze_ledger(path)
    bundle=analyze_ledger(path,AnalysisConfig(run_id="r1",bootstrap_resamples=0,permutation_resamples=0))
    assert set(bundle.task_pairs["dataset"]) == {"d1"}


def test_operator_semantics_mix_is_rejected(tmp_path):
    rows=[_record("d",i,p,.5) for i in (1,2) for p in ("Raw","AutoFE_Baseline")]
    rows[0]["operator_semantics_version"]="different"
    with pytest.raises(AnalysisInputError,match="semantics"):
        analyze_ledger(ledger(tmp_path,rows))


def test_snapshot_identity_includes_durable_content(tmp_path):
    fixture=sensitivity_tests.SensitivityAnalysisTests()
    db,path,run=fixture._fixture()
    from src.task_manifest import ManifestStore
    from src.provenance import canonical_sha256
    store=ManifestStore(db)
    before=snapshot_manifest_ledger(db,path,run)[0]["snapshot_id"]
    with store._connect() as connection:
        row=connection.execute("SELECT * FROM durable_results LIMIT 1").fetchone()
        payload=json.loads(row["payload_json"])
        payload["roc_auc"] += .01
        encoded=json.dumps(payload,sort_keys=True,separators=(",",":"))
        connection.execute("UPDATE durable_results SET payload_json=?,payload_hash=? WHERE scientific_task_id=?",(encoded,canonical_sha256(payload),row["scientific_task_id"]))
    after=snapshot_manifest_ledger(db,path,run)[0]["snapshot_id"]
    assert before != after


def test_bounds_cannot_drop_an_unbounded_eligible_task():
    frame=pd.DataFrame([
        {"contrast_id":"c","stratum":"s","dataset":"d","pipeline_a":"Raw","pipeline_b":"AutoFE_Baseline","eligible":True,"pair_status":"unresolved_outcome","bound_lower":-.2,"bound_upper":.8},
        {"contrast_id":"c","stratum":"s","dataset":"d","pipeline_a":"Raw","pipeline_b":"AutoFE_Baseline","eligible":True,"pair_status":"unresolved_outcome","bound_lower":None,"bound_upper":None}])
    summaries,datasets,bounds=_dataset_summary(frame.iloc[0:0],frame,SensitivityConfig(bootstrap_resamples=0,permutation_resamples=0),regime="test",fingerprint="test")
    assert bounds[0]["bound_status"] != "complete"
    assert bounds[0]["bound_lower"] is None
