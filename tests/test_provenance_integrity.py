import json

import pytest

from src.provenance import build_provenance_package,verify_provenance
from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION
from src.task_manifest import ManifestStore,build_task_records


def package_fixture(tmp_path):
    (tmp_path/"config").mkdir()
    (tmp_path/"data/raw").mkdir(parents=True)
    (tmp_path/"config/dataset_list.yaml").write_text("datasets: [d]\n")
    data=tmp_path/"data/raw/d.csv"
    data.write_text("x,target\n1,a\n2,b\n")
    records=build_task_records(["d"],[42],[1],[("clean",0)],["Raw"],["logistic_regression"],data_paths={"d":data},include_precompute=False)
    store=ManifestStore(tmp_path/"manifest.db")
    config={"protocol_version":EVALUATION_PROTOCOL_VERSION,"seed_scheme_version":SEED_SCHEME_VERSION}
    store.create_run("r",config,records)
    task=records[0]["scientific_task_id"]
    attempt=store.claim_task("r",task)
    payload={"run_id":"r","scientific_task_id":task,"attempt_id":attempt,"status":"success","roc_auc":.8}
    store.commit_result("r",task,attempt,payload)
    ledger=tmp_path/"results.jsonl"
    ledger.write_text(json.dumps(payload)+"\n")
    package=tmp_path/"package"
    build_provenance_package(output_dir=package,repo_root=tmp_path,manifest_db=store.db_path,ledger_path=ledger,run_id="r")
    kwargs={"repo_root":tmp_path,"manifest_db":store.db_path,"ledger_path":ledger,"run_id":"r","package_dir":package}
    return store,payload,ledger,package,kwargs


def test_changed_metric_is_invalid_against_authoritative_payload(tmp_path):
    store,payload,ledger,package,kwargs=package_fixture(tmp_path)
    ledger.write_text(json.dumps({**payload,"roc_auc":.1})+"\n")
    kwargs.pop("package_dir")
    assert verify_provenance(**kwargs)["overall_status"] == "invalid"


def test_empty_export_is_incomplete_not_valid(tmp_path):
    store,payload,ledger,package,kwargs=package_fixture(tmp_path)
    ledger.write_text("")
    kwargs.pop("package_dir")
    result=verify_provenance(**kwargs)
    assert result["overall_status"] in {"incomplete","invalid"}
    assert any(c["name"].startswith("missing_ledger_result") for c in result["checks"])


@pytest.mark.parametrize("member",["artifact_inventory.json","lineage.json","environment.json","code_identity.json"])
def test_required_package_member_cannot_disappear(tmp_path,member):
    store,payload,ledger,package,kwargs=package_fixture(tmp_path)
    (package/member).unlink()
    assert verify_provenance(**kwargs)["overall_status"] == "invalid"


def test_removed_inventory_and_self_updated_digest_do_not_pass(tmp_path):
    store,payload,ledger,package,kwargs=package_fixture(tmp_path)
    from src.provenance import canonical_sha256
    inventory=json.loads((package/"artifact_inventory.json").read_text())
    inventory["artifacts"]=[]
    (package/"artifact_inventory.json").write_text(json.dumps(inventory))
    package_json=json.loads((package/"provenance.json").read_text())
    package_json["artifact_inventory_sha256"]=canonical_sha256(inventory)
    if "component_hashes" in package_json:
        package_json["component_hashes"]["artifact_inventory.json"]=canonical_sha256(inventory)
    (package/"provenance.json").write_text(json.dumps(package_json))
    assert verify_provenance(**kwargs)["overall_status"] == "invalid"


def test_verification_never_creates_missing_database(tmp_path):
    (tmp_path/"config").mkdir()
    (tmp_path/"config/dataset_list.yaml").write_text("datasets: [d]\n")
    ledger=tmp_path/"results.jsonl"
    ledger.write_text("")
    absent=tmp_path/"missing/db.sqlite"
    verify_provenance(repo_root=tmp_path,manifest_db=absent,ledger_path=ledger,run_id="r")
    assert not absent.parent.exists()


def test_lineage_contains_actual_tasks_not_generic_nodes(tmp_path):
    store,payload,ledger,package,kwargs=package_fixture(tmp_path)
    lineage=json.loads((package/"lineage.json").read_text())
    assert payload["scientific_task_id"] in {n["id"] for n in lineage["nodes"]}


def test_untracked_source_content_changes_code_identity(tmp_path):
    import subprocess
    from src.provenance import collect_code_identity
    subprocess.run(["git","init","--quiet",str(tmp_path)],check=True)
    source=tmp_path/"app.py"
    source.write_text("value=1\n")
    before=collect_code_identity(tmp_path)
    source.write_text("value=2\n")
    after=collect_code_identity(tmp_path)
    assert before["worktree_fingerprint_sha256"] != after["worktree_fingerprint_sha256"]


def test_external_paths_do_not_collapse_to_same_basename(tmp_path):
    from src.provenance import relative_path
    assert relative_path(tmp_path/"a/file.json",tmp_path/"repo") != relative_path(tmp_path/"b/file.json",tmp_path/"repo")


@pytest.mark.parametrize('changed',[None,'run','ledger','tasks'])
def test_analysis_edges_require_recorded_matching_inputs(tmp_path,changed):
    from src.provenance_evidence import actual_lineage
    from src.artifact_integrity import file_sha256
    store,payload,ledger,package,kwargs=package_fixture(tmp_path)
    output=tmp_path/'analysis'
    output.mkdir()
    artifact=output/'summary.csv'
    artifact.write_text('effect\n0.1\n')
    config={'run_id':'r','input_fingerprint_sha256':file_sha256(ledger),
        'analysis_input_task_ids':[payload['scientific_task_id']]}
    if changed == 'run': config['run_id']='other'
    if changed == 'ledger': config['input_fingerprint_sha256']='0'*64
    if changed == 'tasks': config['analysis_input_task_ids']=['unknown']
    (output/'analysis_config.json').write_text(json.dumps(config))
    inventory=[{'path':'results.jsonl','role':'result_ledger_export','sha256':file_sha256(ledger)},
        {'path':'analysis/summary.csv','role':'analysis_output','sha256':file_sha256(artifact)}]
    graph=actual_lineage(store.snapshot('r'),inventory,tmp_path)
    edges=[edge for edge in graph['edges'] if edge['relationship']=='recorded_analysis_scientific_input']
    assert len(edges) == (1 if changed is None else 0)
    assert any(x['reason']=='analysis_dependency_unverified' for x in graph['unverified_dependencies']) == (changed is not None)


def test_sensitivity_links_exact_manifest_eligibility(tmp_path):
    from src.provenance_evidence import actual_lineage,snapshot_digest
    from src.sensitivity_analysis import analyze_sensitivity,SensitivityConfig
    store,payload,ledger,package,kwargs=package_fixture(tmp_path)
    output=tmp_path/'sensitivity'
    bundle=analyze_sensitivity(store.db_path,ledger,'r',output_dir=output,
        config=SensitivityConfig(bootstrap_resamples=0,permutation_resamples=0))
    assert bundle.config['manifest_snapshot_sha256']==snapshot_digest(store.snapshot('r'))
    inventory=[{'path':'results.jsonl','role':'result_ledger_export','sha256':bundle.config['ledger_sha256']},
        {'path':'sensitivity/bounds.csv','role':'analysis_output','sha256':'example'}]
    graph=actual_lineage(store.snapshot('r'),inventory,tmp_path)
    assert any(edge['relationship']=='recorded_analysis_eligibility_input' for edge in graph['edges'])
    config=json.loads((output/'sensitivity_config.json').read_text())
    config['manifest_snapshot_sha256']='0'*64
    (output/'sensitivity_config.json').write_text(json.dumps(config))
    graph=actual_lineage(store.snapshot('r'),inventory,tmp_path)
    assert any(row['reason']=='analysis_dependency_unverified' for row in graph['unverified_dependencies'])


def test_hash_reuse_is_scoped_fresh_and_preserves_conflicting_expectations(tmp_path,monkeypatch):
    from src import provenance as p
    from src.provenance_evidence import _FileHashes
    path=tmp_path/'artifact';path.write_bytes(b'original')
    calls=[];original=p.file_sha256
    def counted(path):calls.append(str(path));return original(path)
    monkeypatch.setattr(p,'file_sha256',counted)
    hashes=_FileHashes()
    expected=original(path)
    for _ in range(20):
        assert p._artifact(path,tmp_path,'selected',expected_sha256=expected,hash_file=hashes)['status']=='valid'
    assert len(calls)==1
    assert p._artifact(path,tmp_path,'selected',expected_sha256='0'*64,hash_file=hashes)['status']=='conflict'
    hashes.validate()
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):hashes(path)
    assert _FileHashes()(path)==original(path)


def test_conflicting_shared_dependency_is_retained_and_invalid(tmp_path):
    from src.provenance import canonical_sha256
    store,payload,ledger,package,kwargs=package_fixture(tmp_path)
    path=tmp_path/'shared';path.write_bytes(b'original')
    artifacts=[{'path':str(path),'role':'shared','sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest()},
        {'path':str(path),'role':'shared','sha256':'0'*64}]
    payload={**payload,'artifacts':artifacts}
    with store._connect() as connection:
        connection.execute('UPDATE durable_results SET payload_json=?,payload_hash=? WHERE run_id=?',
            (json.dumps(payload),canonical_sha256(payload),'r'))
    ledger.write_text(json.dumps(payload)+'\n')
    build_provenance_package(output_dir=package,repo_root=tmp_path,manifest_db=store.db_path,ledger_path=ledger,run_id='r')
    inventory=json.loads((package/'artifact_inventory.json').read_text())
    assert len([x for x in inventory['artifacts'] if x.get('role')=='shared'])==2
    assert verify_provenance(**kwargs)['overall_status']=='invalid'
