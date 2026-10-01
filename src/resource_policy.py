"""Host-aware execution admission, independent of scientific feature budgets.

RAM/VRAM limits are admission estimates. Native libraries can temporarily use
more than estimated; no claim is made that Python enforces an allocator cap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil

GIB = 1024**3
GPU_MODELS = {'xgboost', 'catboost'}


@dataclass(frozen=True)
class ResourceSettings:
    reserve_cpus: int = 2
    ram_target_fraction: float = 0.8
    vram_target_fraction: float = 0.8
    gpu_policy: str = 'auto'

    def validate(self):
        if self.reserve_cpus < 1:
            raise ValueError('reserve_cpus must be at least one')
        if not 0.1 <= self.ram_target_fraction <= 0.95:
            raise ValueError('ram_target_fraction must be between 0.1 and 0.95')
        if not 0.1 <= self.vram_target_fraction <= 0.95:
            raise ValueError('vram_target_fraction must be between 0.1 and 0.95')
        if self.gpu_policy not in {'auto', 'cpu', 'require'}:
            raise ValueError('gpu_policy must be auto, cpu, or require')


def gpu_inventory() -> list[dict]:
    """Physical NVIDIA devices with stable identity and live free memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,driver_version',
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=5, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode:
            return []
        devices = []
        for line in result.stdout.splitlines():
            index, uuid, name, total, free, driver = [v.strip() for v in line.split(',')]
            devices.append(dict(physical_index=int(index), uuid=uuid, name=name,
                                total_bytes=int(total)*1024**2, free_bytes=int(free)*1024**2,
                                driver_version=driver))
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is not None:
            requested = [v.strip() for v in visible.split(',') if v.strip() and v.strip() != '-1']
            devices = [next((d for d in devices if str(d['physical_index']) == v or d['uuid'] == v), None)
                       for v in requested]
            devices = [d for d in devices if d is not None]
        return [{**device, 'device_index': index} for index, device in enumerate(devices)]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def detect_hardware(output_root: Path) -> dict:
    process = psutil.Process()
    try:
        cpus = process.cpu_affinity()
    except (AttributeError, psutil.Error):
        cpus = list(range(psutil.cpu_count() or 1))
    ancestor = output_root.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    devices = [{k: v for k, v in device.items() if k != 'free_bytes'} for device in gpu_inventory()]
    return dict(allowed_cpu_ids=cpus, physical_cpus=psutil.cpu_count(logical=False),
                logical_cpus=psutil.cpu_count(), ram_total_bytes=psutil.virtual_memory().total,
                disk_total_bytes=psutil.disk_usage(str(ancestor)).total, gpu_devices=devices,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def probe_gpu_models(devices: list[dict]) -> dict:
    """Small isolated fits test actual installed CUDA backends, not imports."""
    if not devices:
        return {model: {'supported': False, 'reason': 'No visible NVIDIA device'} for model in GPU_MODELS}
    script = r'''
import json,os,numpy as np
from src.model import build_model
rng=np.random.default_rng(42);x=rng.normal(size=(64,4)).astype('float32');y=(x[:,0]>0).astype(int)
result={}
for name in ('xgboost','catboost'):
 verified={}
 for ordinal in range(int(os.environ['AUTOFE_GPU_PROBE_COUNT'])):
  try:
   model=build_model(name,random_state=42,use_gpu=True,gpu_device=ordinal,gpu_ram_part=0.2)
   model.fit(x,y)
   if name=='xgboost':
    device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if not device.startswith('cuda'):raise RuntimeError('XGBoost silently used CPU')
   verified[str(ordinal)]={'supported':True}
  except Exception as exc:verified[str(ordinal)]={'supported':False,'reason':type(exc).__name__+': '+str(exc)[:500]}
 result[name]={'supported':any(v['supported'] for v in verified.values()),'devices':verified}
print(json.dumps(result))
'''
    environment = os.environ.copy()
    environment['AUTOFE_GPU_PROBE_COUNT'] = str(len(devices))
    environment.update({key: '1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    try:
        result = subprocess.run([sys.executable, '-c', script], env=environment,
                                capture_output=True, text=True, timeout=120*len(devices),
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return json.loads(result.stdout.splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {model: {'supported': False, 'reason': str(exc)[:500]} for model in GPU_MODELS}


def resolve_plan(settings: ResourceSettings, hardware: dict, models: tuple[str, ...],
                 *, gpu_probe: dict | None = None, worker_override: int | None = None,
                 cache_override: int | None = None) -> dict:
    settings.validate()
    cpus = hardware['allowed_cpu_ids']
    available_slots = max(1, len(cpus)-settings.reserve_cpus)
    if worker_override is not None and worker_override < 1:
        raise ValueError('worker override must be positive')
    workers = min(available_slots, worker_override) if worker_override else available_slots
    probe = gpu_probe or {}
    backend = {model: 'gpu' if settings.gpu_policy != 'cpu' and model in GPU_MODELS
               and probe.get(model, {}).get('supported') else 'cpu' for model in models}
    if settings.gpu_policy == 'require' and any(backend[m] != 'gpu' for m in models if m in GPU_MODELS):
        raise ValueError('Requested GPU backend failed capability verification')
    ram_total = hardware['ram_total_bytes']
    # A cache is on disk, not a way to inflate RAM use. Publication may need
    # another artifact-sized temporary file, budgeted separately at launch.
    cache_bytes = cache_override if cache_override is not None else int(min(
        ram_total*0.4, hardware['disk_total_bytes']*0.025))
    return dict(policy='adaptive-v3', settings=asdict(settings), hardware=hardware,
                worker_ceiling=workers, worker_cpu_ids=cpus[:available_slots],
                cpu_reserve_satisfied=len(cpus)>settings.reserve_cpus,
                ram_budget_bytes=int(ram_total*settings.ram_target_fraction),
                ram_reserve_bytes=int(ram_total*(1-settings.ram_target_fraction)),
                cache_max_bytes=cache_bytes, backend_by_model=backend, gpu_probe=probe,
                worker_override=worker_override, cache_override=cache_override,
                # Four float32 working matrices per shared float64 host bound,
                # plus the fixed context/histogram allowance at admission.
                gpu_capacity_matrix_factor=2,
                gpu_worker_pool='one persistent process per verified device; all CPU/GPU fits share the total admission ceiling',
                gpu_backend_comparison_rule='Common training-schema matrix bound across every configured pipeline for the same dataset/split/seed/fold/condition/model; static oversize uses CPU for the whole comparison in auto mode',
                memory_limit_type='soft estimated admission; native allocation is not forcibly capped')


def initialize_worker(cpu_ids: list[int] | None = None):
    """Limit worker CPU affinity and exit if its coordinator disappears."""
    if cpu_ids:
        try:
            psutil.Process().cpu_affinity(cpu_ids)
        except (AttributeError, psutil.Error):
            pass
    parent = mp.parent_process()
    if parent is not None:
        def watch():
            while parent.is_alive():
                time.sleep(1)
            os._exit(70)
        threading.Thread(target=watch, daemon=True).start()


def measured_worker(function):
    """Sample per-task resident memory rather than a reused process lifetime peak."""
    @wraps(function)
    def measured(payload):
        process = psutil.Process()
        peak = [process.memory_info().rss]
        stop = threading.Event()
        def sample():
            while not stop.wait(0.1):
                try:
                    peak[0] = max(peak[0], process.memory_info().rss)
                except psutil.Error:
                    return
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            result = function(payload)
            peak[0] = max(peak[0], process.memory_info().rss)
            result['worker_peak_rss_bytes'] = peak[0]
            result['worker_rss_measurement'] = 'task RSS sampled at 100 ms; not allocator-enforced peak'
            return result
        finally:
            stop.set()
            sampler.join()
    return measured


class ResourceAdmission:
    """Admission changes concurrency, never model hyperparameters or data."""
    def __init__(self, plan: dict):
        self.plan = plan
        self.model_peak_bytes = {}
        self.telemetry = dict(admissions=0, ram_waits=0, gpu_waits=0,
                              preparation_waits=0,
                              observed_process_tree_rss_bytes=0,
                              minimum_available_ram_bytes=None, gpu_observations={})
        self._last_sample = None
        self._sample_time = 0.0

    def sample(self, *, fresh=False):
        now=time.monotonic()
        if not fresh and self._last_sample is not None and now-self._sample_time < 0.5:
            return self._last_sample
        vm=psutil.virtual_memory()
        process=psutil.Process()
        resident=0
        child_resident=0
        for child in [process, *process.children(recursive=True)]:
            try:
                amount=child.memory_info().rss
                resident+=amount
                if child.pid!=process.pid:
                    child_resident+=amount
            except psutil.Error:pass
        minimum=self.telemetry['minimum_available_ram_bytes']
        self.telemetry['minimum_available_ram_bytes']=vm.available if minimum is None else min(minimum,vm.available)
        self.telemetry['observed_process_tree_rss_bytes']=max(resident,self.telemetry['observed_process_tree_rss_bytes'])
        self._last_sample=dict(available_bytes=vm.available,resident_bytes=resident,
                              child_resident_bytes=child_resident)
        self._sample_time=now
        return self._last_sample

    def preparation_allowed(self, estimated_bytes: int, pending: list[dict]) -> bool:
        if estimated_bytes > self.plan['ram_budget_bytes']:
            raise MemoryError('Feature preparation estimate exceeds adaptive RAM budget')
        sample = self.sample(fresh=True)
        reserved = max(0, sum(item.get('ram_bytes', 0) for item in pending)
                       - sample.get('child_resident_bytes', 0))
        allowed = (sample['available_bytes']-self.plan['ram_reserve_bytes'] >= estimated_bytes+reserved
                   and sample['resident_bytes']+estimated_bytes+reserved <= self.plan['ram_budget_bytes'])
        if not allowed:
            self.telemetry['preparation_waits'] += 1
        return allowed

    def reservation(self, model: str, matrix_bytes: int, pending: list[dict],
                    *, comparison_matrix_bytes: int | None = None) -> dict | None:
        sample=self.sample(fresh=True)
        factor={'random_forest':8,'extra_trees':8,'linear_svm':8,'knn':4}.get(model,6)
        estimate=max(512*1024**2+matrix_bytes*factor,self.model_peak_bytes.get(model,0)+matrix_bytes)
        if estimate > self.plan['ram_budget_bytes']:
            raise MemoryError('One model estimate exceeds adaptive RAM budget; increase budget on a capable host')
        # Reserve the payloads waiting in the process queue in addition to
        # available RAM. Running estimates also cover still-growing fits.
        reserved=max(0, sum(item.get('ram_bytes',0) for item in pending)
                     - sample.get('child_resident_bytes', 0))
        if (len(pending)>=self.plan['worker_ceiling']
                or sample['available_bytes']-self.plan['ram_reserve_bytes'] < estimate+reserved
                or sample['resident_bytes']+estimate+reserved > self.plan['ram_budget_bytes']):
            self.telemetry['ram_waits']+=1
            return None
        reservation=dict(ram_bytes=estimate,model=model,gpu_device=None,gpu_ram_part=None,
                         backend=self.plan['backend_by_model'].get(model,'cpu'),
                         backend_reason='frozen host capability policy')
        common_bytes = matrix_bytes if comparison_matrix_bytes is None else comparison_matrix_bytes
        if common_bytes < matrix_bytes:
            raise ValueError('Actual matrix exceeds common comparison bound')
        reservation['comparison_matrix_bytes'] = common_bytes
        if self.plan['backend_by_model'].get(model)=='gpu':
            required_vram = 256*1024**2+common_bytes*self.plan['gpu_capacity_matrix_factor']
            verified = self.plan.get('gpu_probe', {}).get(model, {}).get('devices')
            devices = [d for d in self.plan['hardware']['gpu_devices']
                       if verified is None or verified.get(str(d['device_index']), {}).get('supported')]
            if not any(required_vram <= int(d['total_bytes']*self.plan['settings']['vram_target_fraction'])
                       for d in devices):
                if self.plan['settings']['gpu_policy'] == 'require':
                    raise MemoryError('One GPU fit estimate exceeds every device VRAM budget')
                reservation.update(backend='cpu',backend_reason='static matrix estimate exceeds GPU VRAM budget')
                self.telemetry['admissions']+=1
                return reservation
            occupied={p.get('gpu_device') for p in pending if p.get('gpu_device') is not None}
            current={d['uuid']:d for d in gpu_inventory()}
            for device in devices:
                ordinal=device['device_index']
                live=current.get(device['uuid'])
                if ordinal in occupied or live is None:continue
                total=device['total_bytes']
                headroom=int(total*(1-self.plan['settings']['vram_target_fraction']))
                usable=live['free_bytes']-headroom
                self.telemetry['gpu_observations'][device['uuid']]=dict(
                    free_bytes=live['free_bytes'],reserve_bytes=headroom,observed_at_unix=time.time())
                if usable >= required_vram:
                    reservation.update(gpu_device=ordinal,gpu_ram_part=min(
                        self.plan['settings']['vram_target_fraction'],usable/total))
                    break
            else:
                self.telemetry['gpu_waits']+=1
                return None
        self.telemetry['admissions']+=1
        return reservation

    def completed(self, model: str, fit: dict):
        peak=int(fit.get('worker_peak_rss_bytes') or 0)
        self.model_peak_bytes[model]=max(peak,self.model_peak_bytes.get(model,0))
        self.sample(fresh=True)
