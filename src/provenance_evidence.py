"""Manifest-anchored provenance verification; inspection never initializes data."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime,timezone
from pathlib import Path

from src.artifact_integrity import atomic_json
from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION
from src.task_manifest import ManifestError,ManifestStore


def snapshot_digest(snapshot):
    from src.provenance import canonical_sha256
    return canonical_sha256({key:snapshot[key] for key in ("run","tasks","attempts","durable_results","task_events") if key in snapshot})


def _resolve(path,root):
    path=Path(path)
    return path if path.is_absolute() else root/path


def actual_lineage(snapshot,inventory,root):
    from src.provenance import LINEAGE_SCHEMA_VERSION,canonical_sha256,relative_path
    nodes,edges={},[]
    unknown=[]
    if snapshot is not None:
        results={r["scientific_task_id"]:r for r in snapshot["durable_results"]}
        task_ids={r["scientific_task_id"] for r in snapshot["tasks"]}
        for task in snapshot["tasks"]:
            payload=json.loads(task["payload_json"])
            if payload.get("data_path"):
                payload["data_path"]=relative_path(payload["data_path"],root)
            task_id=task["scientific_task_id"]
            nodes[task_id]={"id":task_id,"role":task["stage"]+"_task","scientific_identity":payload,"state":task["state"]}
            data=payload.get("data_identity",{})
            data_id=data.get("fingerprint")
            if data_id:
                data_id="data_"+data_id
                nodes[data_id]={"id":data_id,"role":"dataset_bytes","identity":data}
                edges.append({"from":data_id,"to":task_id,"relationship":"scientific_data_input"})
            elif not task.get("planned_skip_reason"):
                unknown.append({"task_id":task_id,"reason":"data_identity_unverified"})
            for dependency in payload.get("depends_on",[]):
                if dependency in task_ids:
                    edges.append({"from":dependency,"to":task_id,"relationship":"declared_precompute_dependency"})
                else:
                    unknown.append({"task_id":task_id,"reason":"missing_dependency_task","dependency":dependency})
            if task_id in results:
                durable=results[task_id]
                result_id="result_"+durable["payload_hash"]
                nodes[result_id]={"id":result_id,"role":"durable_"+task["stage"]+"_result","payload_hash":durable["payload_hash"],"attempt_id":durable["attempt_id"]}
                edges.append({"from":task_id,"to":result_id,"relationship":"transactional_attempt_result"})
                evidence=json.loads(durable["payload_json"])
                artifacts=evidence.get("artifacts",[])
                if task["stage"] == "model" and not artifacts:
                    unknown.append({"task_id":task_id,"reason":"prepared_input_artifacts_unverified"})
                for artifact in artifacts:
                    aid="artifact_"+str(artifact.get("sha256",canonical_sha256(artifact)))
                    nodes[aid]={"id":aid,"role":artifact.get("role","dependency_artifact"),"path":artifact.get("path"),"sha256":artifact.get("sha256"),"dependency_signature":artifact.get("dependency_signature")}
                    edges.append({"from":aid,"to":task_id,"relationship":"verified_executed_input"})
    for item in inventory:
        if item.get("role") == "analysis_output":
            aid="analysis_"+str(item.get("sha256"))
            nodes[aid]={"id":aid,"role":"analysis_output",**item}
            for result_node in list(nodes.values()):
                if result_node.get("role") == "durable_model_result":
                    edges.append({"from":result_node["id"],"to":aid,"relationship":"declared_run_analysis_input"})
    return {"schema_version":LINEAGE_SCHEMA_VERSION,"nodes":list(nodes.values()),"edges":edges,
        "status":"incomplete" if unknown or snapshot is None else "valid","unverified_dependencies":unknown,
        "snapshot_sha256":None if snapshot is None else snapshot_digest(snapshot)}


def build_package(*,output_dir,repo_root=".",dataset_list_path="config/dataset_list.yaml",manifest_db=None,manifest_path=None,ledger_path=None,run_id=None,analysis_dirs=()):
    from src import provenance as p
    root=Path(repo_root).resolve()
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    registry=p.build_dataset_registry(dataset_list_path,repo_root=root)
    code=p.collect_code_identity(root)
    environment=p.collect_environment_identity(root)
    snapshot=None
    store=None
    inventory=[]
    if manifest_db is not None and run_id is not None:
        db=_resolve(manifest_db,root)
        store=ManifestStore(db,read_only=True)
        snapshot=store.snapshot(run_id)
        recorded_config=json.loads(snapshot["run"]["config_json"])
        code=recorded_config.get("code_identity",code)
        environment=recorded_config.get("environment_identity",environment)
        inventory.append({"role":"manifest_sqlite","path":p.relative_path(db,root),"required":True,"status":"valid","identity_kind":"scientific_snapshot","snapshot_sha256":snapshot_digest(snapshot)})
    for path,role,required in ((manifest_path,"manifest_jsonl",False),(ledger_path,"result_ledger_export",True)):
        if path is not None:
            inventory.append(p._artifact(_resolve(path,root),root,role,required=required))
    for dataset in registry["datasets"]:
        inventory.append(p._artifact(root/dataset["path"],root,"dataset_bytes",required=False,expected_sha256=dataset.get("sha256")))
    scoped={}
    if snapshot:
        for row in snapshot["durable_results"]:
            payload=json.loads(row["payload_json"])
            for item in payload.get("artifacts",[]):
                if item.get("path"):
                    path=_resolve(item["path"],root)
                    entry=p._artifact(path,root,item.get("role","dependency_artifact"),required=True,expected_sha256=item.get("sha256"))
                    entry["dependency_signature"]=item.get("dependency_signature")
                    scoped[(entry["path"],entry["role"])]=entry
    inventory.extend(scoped.values())
    for directory in analysis_dirs:
        directory=_resolve(directory,root)
        if directory.exists():
            inventory.extend(p._artifact(path,root,"analysis_output",required=True) for path in sorted(directory.rglob("*")) if path.is_file())
    inventory_value={"schema_version":"dependency_inventory_v2","cache_inventory":{"status":"verified" if scoped else "unavailable","scope":"actual_run_dependency_artifacts","listed_file_count":len(scoped)},"artifacts":inventory}
    lineage=actual_lineage(snapshot,inventory,root)
    components={"dataset_registry.json":registry,"code_identity.json":code,"environment.json":environment,"artifact_inventory.json":inventory_value,"lineage.json":lineage}
    if snapshot:
        components["manifest_summary.json"]={"run_id":run_id,"snapshot_sha256":snapshot_digest(snapshot),"run_config":json.loads(snapshot["run"]["config_json"]),"task_count":len(snapshot["tasks"]),"attempt_count":len(snapshot["attempts"]),"durable_result_count":len(snapshot["durable_results"])}
    component_hashes={name:p.canonical_sha256(value) for name,value in components.items()}
    evidence={"schema_version":p.PROVENANCE_SCHEMA_VERSION,"run_id":run_id,"component_hashes":component_hashes,"snapshot_sha256":None if snapshot is None else snapshot_digest(snapshot)}
    anchor=None
    if manifest_db is not None and run_id is not None:
        # Packaging is a write operation; verification below only reads this anchor.
        anchor=ManifestStore(_resolve(manifest_db,root)).record_evidence_anchor(run_id,evidence)
    package={**evidence,"anchor_id":anchor,"generated_at":datetime.now(timezone.utc).isoformat(),"dataset_registry_id":registry["registry_id"],"artifact_inventory_sha256":p.canonical_sha256(inventory_value)}
    for name,value in components.items():
        atomic_json(output/name,value)
    atomic_json(output/"provenance.json",package)
    (output/"README.md").write_text("# Manifest-anchored metadata package\n\nIntegrity, source availability, lineage completeness and benchmark readiness are separate. Required components are hashed against the authoritative manifest anchor. Data and caches are referenced, not copied.\n",encoding="utf-8")
    (output/"reproduction_commands.txt").write_text("conda run -n P12 python -m src.provenance_cli verify --manifest-db <db> --ledger <ledger> --run-id <run> --package-dir <package>\n",encoding="utf-8")
    return output


def verify(*,repo_root=".",dataset_list_path="config/dataset_list.yaml",manifest_db=None,ledger_path=None,run_id=None,package_dir=None):
    from src import provenance as p
    root=Path(repo_root).resolve()
    checks=[]
    def check(name,status,**details):
        checks.append({"name":name,"status":status,**details})
    registry=p.build_dataset_registry(dataset_list_path,repo_root=root)
    for item in registry["datasets"]:
        check("dataset:"+item["name"],"valid" if item["status"] == "verified" else ("incomplete" if item["status"] == "unavailable" else "unverified"))
    snapshot=None
    reader=None
    if manifest_db is not None and run_id is not None:
        try:
            reader=ManifestStore(_resolve(manifest_db,root),read_only=True)
            snapshot=reader.snapshot(run_id)
            check("manifest_snapshot","valid",snapshot_sha256=snapshot_digest(snapshot))
            for name,actual,expected in (("protocol_compatibility",snapshot["run"]["protocol_version"],EVALUATION_PROTOCOL_VERSION),("seed_scheme_compatibility",snapshot["run"]["seed_scheme_version"],SEED_SCHEME_VERSION)):
                check(name,"valid" if actual == expected else "invalid",observed=actual,expected=expected)
            for field,expected,name in (("protocol_version",EVALUATION_PROTOCOL_VERSION,"task_protocol_compatibility"),("seed_scheme_version",SEED_SCHEME_VERSION,"task_seed_scheme_compatibility")):
                observed={json.loads(t["payload_json"]).get(field) for t in snapshot["tasks"]}
                check(name,"valid" if observed == {expected} else "invalid",observed=list(observed))
            current_data={item["name"]:item for item in registry["datasets"]}
            checked_data=set()
            for task in snapshot["tasks"]:
                payload=json.loads(task["payload_json"])
                name=payload.get("dataset")
                signature=payload.get("data_identity",{})
                key=(name,signature.get("fingerprint"))
                if key in checked_data:
                    continue
                checked_data.add(key)
                local=current_data.get(name,{})
                if not signature.get("bytes_sha256"):
                    check("executed_dataset:"+str(name),"unverified")
                elif not local.get("sha256"):
                    check("executed_dataset:"+str(name),"incomplete")
                else:
                    check("executed_dataset:"+str(name),"valid" if signature["bytes_sha256"] == local["sha256"] else "invalid")
            states={t["state"] for t in snapshot["tasks"] if not t["planned_skip_reason"]}
            check("eligible_task_completion","valid" if states == {"completed"} else "incomplete",states=sorted(states))
        except (OSError,ManifestError,sqlite3.Error,ValueError) as exc:
            check("manifest_snapshot","invalid",detail=str(exc))
    else:
        check("authoritative_manifest","unverified")
    if snapshot is not None:
        tasks={t["scientific_task_id"]:t for t in snapshot["tasks"]}
        durable={r["scientific_task_id"]:r for r in snapshot["durable_results"] if tasks.get(r["scientific_task_id"],{}).get("stage") == "model"}
        for row in snapshot["durable_results"]:
            task=tasks.get(row["scientific_task_id"])
            if task is None or task["state"] != "completed":
                check("durable_task_state:"+row["scientific_task_id"],"invalid")
        seen=set()
        if ledger_path is None or not _resolve(ledger_path,root).is_file():
            check("ledger_export","incomplete")
        else:
            ledger=_resolve(ledger_path,root)
            before=p.file_sha256(ledger)
            with ledger.open(encoding="utf-8") as handle:
                for number,line in enumerate(handle,1):
                    if not line.strip():
                        continue
                    try:
                        value=json.loads(line)
                        if not isinstance(value,dict):
                            raise ValueError("nonmapping")
                    except (ValueError,TypeError):
                        check(f"ledger_line:{number}","invalid",reason="malformed")
                        continue
                    if value.get("run_id") != run_id:
                        continue
                    tid=value.get("scientific_task_id")
                    saved=durable.get(tid)
                    if saved is None:
                        check(f"ledger_result:{tid}","invalid",reason="no_authoritative_model_result")
                        continue
                    expected=json.loads(saved["payload_json"])
                    expected.setdefault("run_id",run_id)
                    expected.setdefault("scientific_task_id",tid)
                    expected.setdefault("attempt_id",saved["attempt_id"])
                    if p.canonical_sha256(expected) != p.canonical_sha256(value):
                        check(f"ledger_payload:{tid}","invalid",reason="authoritative_payload_mismatch")
                    elif tid in seen:
                        check(f"ledger_duplicate:{tid}","incomplete",reason="identical_duplicate")
                    else:
                        check(f"ledger_payload:{tid}","valid")
                    seen.add(tid)
            if p.file_sha256(ledger) != before:
                check("ledger_read_stability","invalid")
        for tid in sorted(set(durable)-seen):
            check(f"missing_ledger_result:{tid}","incomplete")
        for tid,row in durable.items():
            if p.canonical_sha256(json.loads(row["payload_json"])) != row["payload_hash"]:
                check(f"durable_payload_hash:{tid}","invalid")
    package_status="unverified"
    if package_dir is not None:
        directory=Path(package_dir)
        components={}
        try:
            package=json.loads((directory/"provenance.json").read_text(encoding="utf-8"))
            required={"dataset_registry.json","code_identity.json","environment.json","artifact_inventory.json","lineage.json"}
            if snapshot is not None:
                required.add("manifest_summary.json")
            recorded=package.get("component_hashes",{})
            if not required.issubset(recorded):
                check("package_required_digests","invalid")
            for name in required:
                try:
                    value=json.loads((directory/name).read_text(encoding="utf-8"))
                    components[name]=value
                    check("package_component:"+name,"valid" if recorded.get(name) == p.canonical_sha256(value) else "invalid")
                except (OSError,ValueError):
                    check("package_component:"+name,"invalid",reason="missing_or_malformed")
            anchor=reader.evidence_anchor(run_id,package.get("anchor_id","")) if reader is not None else None
            expected={"schema_version":package.get("schema_version"),"run_id":package.get("run_id"),"component_hashes":recorded,"snapshot_sha256":package.get("snapshot_sha256")}
            check("authoritative_package_anchor","valid" if anchor == expected else ("invalid" if reader is not None else "unverified"))
            if snapshot is not None:
                check("package_manifest_snapshot","valid" if package.get("snapshot_sha256") == snapshot_digest(snapshot) else "incomplete",reason="package_snapshot_must_match_declared_run_state")
                run_config=json.loads(snapshot["run"]["config_json"])
                for name,key in (("code_identity.json","code_identity"),("environment.json","environment_identity")):
                    recorded_identity=run_config.get(key)
                    check("executed_identity:"+key,"unverified" if recorded_identity is None else ("valid" if recorded_identity == components.get(name) else "invalid"))
            inventory=components.get("artifact_inventory.json",{})
            if not inventory.get("artifacts"):
                check("required_artifact_inventory","invalid")
            for item in inventory.get("artifacts",[]):
                if item.get("identity_kind") == "scientific_snapshot":
                    check("manifest_inventory_identity","valid" if snapshot is not None and item.get("snapshot_sha256") == snapshot_digest(snapshot) else "incomplete")
                    continue
                if str(item.get("path","")).startswith("external/"):
                    check("external_artifact:"+item["path"],"unverified",reason="explicit_location_mapping_required")
                    continue
                path=root/item["path"]
                if not path.is_file():
                    check("artifact:"+item["path"],"incomplete" if item.get("required") else "unverified")
                elif not item.get("sha256") or p.file_sha256(path) != item["sha256"]:
                    check("artifact:"+item["path"],"invalid")
                else:
                    check("artifact:"+item["path"],"valid")
            lineage=components.get("lineage.json",{})
            if snapshot is not None:
                ids={n.get("id") for n in lineage.get("nodes",[])}
                check("lineage_task_coverage","valid" if {t["scientific_task_id"] for t in snapshot["tasks"]}.issubset(ids) else "invalid")
            check("lineage_completeness",lineage.get("status","invalid"))
            package_status="invalid" if any(c["status"] == "invalid" for c in checks if c["name"].startswith(("package","authoritative_package","required_artifact","lineage_task"))) else "valid"
        except (OSError,ValueError,KeyError,sqlite3.Error) as exc:
            package_status="invalid"
            check("package_integrity","invalid",detail=str(exc))
    statuses={c["status"] for c in checks}
    overall="invalid" if "invalid" in statuses else ("incomplete" if "incomplete" in statuses else ("unverified" if "unverified" in statuses else "valid"))
    return {"schema_version":"provenance_verification_v2","overall_status":overall,"package_integrity":package_status,
        "benchmark_readiness":"ready" if overall == "valid" and snapshot is not None else "not_certified",
        "status_definitions":{"valid":"all required authority, integrity, availability and lineage checks established","incomplete":"required evidence/work missing","unverified":"identity or optional external evidence unavailable","invalid":"conflicting, malformed, tampered or incompatible evidence"},
        "checks":checks,"counts":{status:sum(c["status"] == status for c in checks) for status in statuses},"code_identity":p.collect_code_identity(root),"registry_id":registry["registry_id"]}
