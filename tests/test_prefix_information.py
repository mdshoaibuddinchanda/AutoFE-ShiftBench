import json

import pandas as pd

from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION
from src.sensitivity_analysis import SensitivityConfig,analyze_sensitivity
from src.task_manifest import ManifestStore,build_task_records
from src.provenance import canonical_sha256


def prefix_fixture(tmp_path):
    paths={name:tmp_path/(name+".csv") for name in ("d1","d2","d3","d4")}
    for path in paths.values():
        path.write_text("x,target\n1,a\n2,b\n")
    records=build_task_records(paths,[42],[1],[("clean",0)],["Raw","AutoFE_Baseline"],["logistic_regression"],data_paths=paths,include_precompute=False)
    store=ManifestStore(tmp_path/"manifest.db")
    store.create_run("r",{"protocol_version":EVALUATION_PROTOCOL_VERSION,"seed_scheme_version":SEED_SCHEME_VERSION},records)
    rows=[]
    for record in records:
        attempt=store.claim_task("r",record["scientific_task_id"])
        value=.5 if record["pipeline"] == "Raw" else .5+.05*int(record["dataset"][-1])
        payload={"run_id":"r","scientific_task_id":record["scientific_task_id"],"attempt_id":attempt,"status":"success","roc_auc":value}
        store.commit_result("r",record["scientific_task_id"],attempt,payload)
        rows.append(payload)
    ledger=tmp_path/"results.jsonl"
    ledger.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return store,records,rows,ledger


def config():
    return SensitivityConfig(contrasts=(("c","Raw","AutoFE_Baseline"),),cutoff_fractions=(.5,),bootstrap_resamples=100,permutation_resamples=100)


def test_prefix_bounds_do_not_use_final_completed_outcomes(tmp_path):
    store,records,rows,ledger=prefix_fixture(tmp_path)
    bundle=analyze_sensitivity(store.db_path,ledger,"r",config=config())
    prefix=bundle.summaries[bundle.summaries["regime"] == "stop_prefix_0.50"].iloc[0]
    assert prefix["n_valid_pairs"] == 2
    # Two observed differences .05/.10 and two unknown differences in [-1,1].
    assert abs(prefix["bound_lower"]-(-.4625)) < 1e-12
    assert abs(prefix["bound_upper"]-.5375) < 1e-12


def test_changing_future_values_does_not_change_any_prefix_summary(tmp_path):
    store,records,rows,ledger=prefix_fixture(tmp_path)
    before=analyze_sensitivity(store.db_path,ledger,"r",config=config())
    for record,row in zip(records,rows):
        if record["dataset"] in {"d3","d4"}:
            row["roc_auc"] = .99 if record["pipeline"] == "AutoFE_Baseline" else .01
            encoded=json.dumps(row,sort_keys=True,separators=(",",":"))
            with store._connect() as connection:
                connection.execute("UPDATE durable_results SET payload_json=?,payload_hash=? WHERE scientific_task_id=?",(encoded,canonical_sha256(row),record["scientific_task_id"]))
    ledger.write_text("".join(json.dumps(row)+"\n" for row in rows))
    after=analyze_sensitivity(store.db_path,ledger,"r",config=config())
    select=lambda frame:frame[frame["regime"] == "stop_prefix_0.50"].reset_index(drop=True)
    pd.testing.assert_frame_equal(select(before.summaries),select(after.summaries))
    pd.testing.assert_frame_equal(select(before.bounds),select(after.bounds))
    pd.testing.assert_frame_equal(select(before.regime_membership),select(after.regime_membership))
    pd.testing.assert_frame_equal(before.cutoff_summaries,after.cutoff_summaries)


def test_legacy_manifest_history_is_explicitly_unsupported(tmp_path):
    store,_,_,ledger=prefix_fixture(tmp_path)
    with store._connect() as connection:
        connection.execute("DELETE FROM task_events")
    bundle=analyze_sensitivity(store.db_path,ledger,"r",config=config())
    assert bundle.cutoff_summaries.iloc[0]["status"].startswith("unsupported")
    assert not bundle.summaries["regime"].str.startswith("stop_prefix").any()


def test_common_eligibility_includes_unresolved_outcomes(tmp_path):
    store,records,_,ledger=prefix_fixture(tmp_path)
    with store._connect() as connection:
        connection.execute("DELETE FROM durable_results WHERE scientific_task_id=?",(records[-1]["scientific_task_id"],))
        connection.execute("UPDATE tasks SET state='pending',result_ref=NULL WHERE scientific_task_id=?",(records[-1]["scientific_task_id"],))
    bundle=analyze_sensitivity(store.db_path,ledger,"r",config=config())
    common=bundle.regime_membership[bundle.regime_membership["regime"] == "common_eligible_task_set"]
    assert len(common) == 4
    assert common["selected"].all()
    assert (common["pair_status"] != "paired_valid").any()
