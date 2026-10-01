"""Read-only gate for the frozen adaptive campaign; never dispatch tasks."""
from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from src.provenance import code_fingerprint, file_sha256
from src.resource_policy import detect_hardware

ROOT=Path(__file__).resolve().parents[1]


def verify(run_id,scope_path):
    from provenance.adaptive_resource_verification import verify_saved_report
    from provenance.verify_reviewer1_launch_v2 import _active_heavy_coordinators
    scope=json.loads(scope_path.read_text())
    run=next(r for r in scope['runs'] if r['run_id']==run_id)
    checks=dict(existing_p12=Path(sys.executable).resolve()==Path(r'D:\Conda\p12\python.exe').resolve(),
                code_fingerprint=scope['code_fingerprint']==code_fingerprint(ROOT),
                analysis_definitions=all(file_sha256(ROOT/name)==digest for name,digest in scope['analysis_sha256'].items()),
                dataset_list=file_sha256(ROOT/'config'/'dataset_list.yaml')==scope['dataset_list_sha256'],
                group_seed_audit=file_sha256(ROOT/'provenance'/'group_seed_grid_audit_v1.json')==scope['group_seed_audit_sha256'],
                datasets=all(file_sha256(ROOT/'data'/'raw'/f"{d['name']}.csv")==d['csv_sha256']
                             and file_sha256(ROOT/'data'/'raw'/f"{d['name']}_meta.json")==d['sidecar_sha256']
                             for d in scope['datasets']),
                numerical_threads=all(os.environ.get(k)==v for k,v in scope['required_numerical_thread_environment'].items()),
                adaptive_hardware=detect_hardware(ROOT)==scope['resource_plan']['hardware'])
    active=_active_heavy_coordinators()
    checks['one_heavy_coordinator']=not active
    mismatches=[]
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if '==' not in line or line.lstrip().startswith('#'):continue
        name,pin=line.split('==',1)
        try:installed=version(name.strip())
        except PackageNotFoundError:installed=None
        if installed!=pin.strip():mismatches.append(dict(package=name,required=pin,installed=installed))
    checks['requirements_pins']=not mismatches
    storage=scope['storage']
    minimum=2*storage['scaled_results_checkpoints_projection_gib_all_seven_runs']+storage['bounded_live_cache_cap_gib_per_run']+storage['cache_publication_temporary_allowance_gib']+storage['separate_mechanism_history_reservation_gib']
    free=shutil.disk_usage(ROOT).free/1024**3
    checks['disk_margin']=free>=minimum
    report_path=ROOT/scope['adaptive_verification_path']
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    valid=False
    error=None
    try:
        valid=(report['status']=='passed' and report['scope_sha256']==file_sha256(scope_path)
               and report['resource_plan']==scope['resource_plan'] and verify_saved_report(report))
    except (OSError,ValueError,KeyError) as exc:error=str(exc)
    checks['adaptive_cpu_gpu_large_and_recovery_evidence']=bool(valid)
    mechanism_path=ROOT/scope['mechanism_scope_path']
    mechanism=json.loads(mechanism_path.read_text()) if mechanism_path.exists() else {}
    checks['mechanism_history_scope']=(
        mechanism.get('status')=='ready' and mechanism.get('code_fingerprint')==scope['code_fingerprint']
        and mechanism.get('dataset_hash_digest')==scope['dataset_hash_digest']
        and mechanism.get('mechanism_script_sha256')==file_sha256(ROOT/'provenance'/'run_mechanism_history.py')
        and mechanism.get('association_analysis_script_sha256')==file_sha256(ROOT/'provenance'/'analyze_mechanism_history.py'))
    existing=ROOT/'corrected_runs'/run_id/'manifest.json'
    if existing.exists():
        prior=json.loads(existing.read_text())
        checks['run_identity']=(prior.get('code_fingerprint')==scope['code_fingerprint']
                                and prior['configuration'].get('resource_plan')==scope['resource_plan']
                                and prior['configuration'].get('resource_profile_sha256')==file_sha256(scope_path)
                                and prior['configuration'].get('split_policy')==run['split_policy']
                                and prior.get('experiment_scope')==('primary_training_corruption' if run['scope']=='primary' else run['scope']))
    else:checks['run_identity']=not existing.parent.exists()
    return dict(run_id=run_id,ready_for_frozen_command=all(checks.values()),checks=checks,
                free_gib=round(free,2),minimum_free_gib=round(minimum,2),package_mismatches=mismatches,
                active_heavy_coordinators=active,adaptive_evidence_error=error,
                adaptive_evidence_status=report.get('status','missing'),
                mechanism_history_status=mechanism.get('status','missing'))
