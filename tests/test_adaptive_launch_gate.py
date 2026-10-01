"""Adaptive launch snapshots must reject missing proof and changed hardware."""
import json

from provenance import verify_adaptive_launch as gate
from provenance import adaptive_resource_verification as proof
from provenance import verify_reviewer1_launch_v2 as dispatcher
from src.provenance import file_sha256


def test_adaptive_gate_requires_current_evidence_and_host(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    monkeypatch.setattr(gate,'code_fingerprint',lambda root:'source')
    monkeypatch.setattr(gate,'detect_hardware',lambda root:{'logical_cpus':8})
    monkeypatch.setattr(dispatcher,'_active_heavy_coordinators',lambda:[])
    monkeypatch.setattr(proof,'verify_saved_report',lambda report:True)
    (tmp_path/'config').mkdir();(tmp_path/'provenance').mkdir()
    (tmp_path/'config'/'dataset_list.yaml').write_text('datasets: []')
    (tmp_path/'provenance'/'group_seed_grid_audit_v1.json').write_text('{}')
    (tmp_path/'requirements.txt').write_text('')
    scope=dict(code_fingerprint='source',analysis_sha256={},datasets=[],
               dataset_list_sha256=file_sha256(tmp_path/'config'/'dataset_list.yaml'),
               group_seed_audit_sha256=file_sha256(tmp_path/'provenance'/'group_seed_grid_audit_v1.json'),
               dataset_hash_digest='data',required_numerical_thread_environment={},
               resource_plan={'hardware':{'logical_cpus':8}},
               adaptive_verification_path='provenance/evidence.json',
               mechanism_scope_path='provenance/mechanism.json',
               runs=[dict(run_id='test',scope='primary',split_policy='row_level')],
               storage=dict(scaled_results_checkpoints_projection_gib_all_seven_runs=1,
                            bounded_live_cache_cap_gib_per_run=1,
                            cache_publication_temporary_allowance_gib=1,
                            array_transport_temporary_cap_gib=2,
                            separate_mechanism_history_reservation_gib=1))
    path=tmp_path/'scope.json';path.write_text(json.dumps(scope))
    first=gate.verify('test',path)
    assert first['minimum_free_gib']==7
    assert not first['ready_for_frozen_command']
    assert not first['checks']['adaptive_cpu_gpu_large_and_recovery_evidence']
    evidence=dict(status='passed',scope_sha256=file_sha256(path),resource_plan=scope['resource_plan'])
    (tmp_path/'provenance'/'evidence.json').write_text(json.dumps(evidence))
    second=gate.verify('test',path)
    assert second['checks']['adaptive_cpu_gpu_large_and_recovery_evidence']
    assert not second['checks']['mechanism_history_scope']
    monkeypatch.setattr(gate,'detect_hardware',lambda root:{'logical_cpus':16})
    assert not gate.verify('test',path)['checks']['adaptive_hardware']
    (tmp_path/'requirements.txt').write_text('setuptools<81')
    monkeypatch.setattr(gate,'version',lambda name:'81.0.0')
    assert not gate.verify('test',path)['checks']['requirements_pins']
    monkeypatch.setattr(gate,'version',lambda name:'80.10.2')
    assert gate.verify('test',path)['checks']['requirements_pins']
