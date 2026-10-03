"""Verified batch eviction. Results and scientific evidence are never deleted.

Only selected DataFrame pickles and derived numeric arrays are disposable.
Windows uses exclusive handles, verifies all bytes through those handles, and
marks deletion only after every gate passes. No recursive directory deletion.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import psutil

from src.artifact_integrity import atomic_json, file_sha256
from src.prepared_inputs import verify_unit
from src.task_manifest import _canonical

POLICY_VERSION = 'verified_rolling_matrices_v1'
DISPOSABLE_ROLES = frozenset(('selected_training_features', 'selected_held_out_features',
                             'model_train_numeric_matrix', 'model_test_numeric_matrix'))


class CacheCheckFailed(RuntimeError):
    """Do not delete or advance to the next unit."""


class CacheNotReady(CacheCheckFailed):
    """Writer has not yet confirmed the fsynced ledger export."""


def cache_bytes(root):
    """Disposable matrix budget; retained evidence uses the free-disk guard."""
    return sum(path.stat().st_size for path in Path(root).rglob('*')
               if path.is_file() and (path.name.endswith(('_train.pkl', '_test.pkl', '.model_float32.npy'))))


def capacity_problem(root, config, *, measured_bytes=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < config.min_free_bytes:
        return f'Free disk {free} bytes is below reserve {config.min_free_bytes}'
    if measured_bytes is not None and measured_bytes >= config.cache_max_bytes:
        return f'Cache {measured_bytes} bytes reached budget {config.cache_max_bytes}'
    return None


def checked_cache_path(value, root):
    root = Path(root).resolve()
    candidate = Path(value).absolute()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise CacheCheckFailed('Deletion target outside the declared cache root: ' + str(value))
    for parent in (candidate, *candidate.parents):
        if parent.is_symlink() or (parent.exists() and getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400):
            raise CacheCheckFailed('Symlink or reparse point in deletion path: ' + str(parent))
        if parent == root:
            break
    if resolved.exists() and (not resolved.is_file() or resolved.stat().st_nlink != 1):
        raise CacheCheckFailed('Deletion target is not a regular, singly linked file')
    return resolved


def workers_released(processes):
    for item in processes:
        try:
            process = psutil.Process(item['pid'])
            if process.create_time() == item['create_time'] and process.is_running():
                raise CacheCheckFailed(f"Worker or descendant {item['pid']} still exists")
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            raise CacheCheckFailed('Cannot verify worker exit') from error


def register_worker(root, process, *, run_id, unit_id):
    """Persist ownership before a command can let this process open inputs.

    Registrations survive a coordinator crash and protect other runs/manifest
    databases sharing this cache namespace. PID reuse is fenced by creation time.
    """
    pid = process['pid'] if isinstance(process, dict) else process.pid
    created = process['create_time'] if isinstance(process, dict) else process.create_time()
    item = {'pid': pid, 'create_time': created,
            'run_id': run_id, 'unit_id': unit_id}
    token = hashlib.sha256(_canonical(item).encode()).hexdigest()[:32]
    atomic_json(Path(root) / '.workers' / (token + '.json'), item)
    return item


def verify_registered_workers(root):
    records = []
    for path in (Path(root) / '.workers').glob('*.json'):
        checked_cache_path(path, root)
        records.append(json.loads(path.read_text(encoding='utf-8')))
    workers_released(records)


def completed_unit_rows(store, run_id, unit_id, *, require_exports=True):
    rows = store.unit_snapshot(run_id, unit_id)
    validation_version = store.run_config(run_id).get('numerical_validation_version')
    models = [row for row in rows if row['stage'] == 'model']
    parents = [row for row in rows if row['scientific_task_id'] == unit_id and row['stage'] == 'precompute']
    if len(parents) != 1 or not models:
        raise CacheCheckFailed('No complete, declared preparation/model membership')
    for row in rows:
        if row['state'] != 'completed' or row['active_attempt_id'] is not None or row['result_attempt_state'] != 'completed':
            raise CacheCheckFailed('Pending, running, failed, skipped, or uncommitted task: ' + row['scientific_task_id'])
        if row['result_json'] is None or hashlib.sha256(row['result_json'].encode()).hexdigest() != row['payload_hash']:
            raise CacheCheckFailed('Authoritative result missing or corrupt')
        result, task = json.loads(row['result_json']), json.loads(row['payload_json'])
        if any(result.get(key) != expected for key, expected in
               (('run_id', run_id), ('scientific_task_id', row['scientific_task_id']), ('attempt_id', row['result_attempt_id']))):
            raise CacheCheckFailed('Result ownership mismatch')
        if row['stage'] == 'model':
            if validation_version and (task.get('numerical_validation_version') != validation_version
                    or result.get('numerical_validation_version') != validation_version):
                raise CacheCheckFailed('Numerical validation compatibility unverified')
            if result.get('status') != 'success' or not isinstance(result.get('metric_status'), dict):
                raise CacheCheckFailed('Missing successful metric evidence')
            from src.evaluation import METRIC_RANGES, METRIC_SEMANTICS_VERSION
            if result.get('metric_semantics_version') != METRIC_SEMANTICS_VERSION:
                raise CacheCheckFailed('Metric semantics unverified')
            for name, (low, high) in METRIC_RANGES.items():
                status, value = result['metric_status'].get(name), result.get(name)
                if status == 'complete':
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                        raise CacheCheckFailed('Missing/nonfinite/out-of-range completed metric: ' + name)
                elif not isinstance(status, str) or not status.startswith('undefined_') or value is not None:
                    raise CacheCheckFailed('Metric unavailable without a declared reason: ' + name)
            for key in ('seed', 'fold', 'condition', 'pipeline', 'model', 'split_policy'):
                if result.get(key) != task.get(key):
                    raise CacheCheckFailed('Result scientific identity mismatch: ' + key)
            if result.get('dataset') != task.get('dataset'):
                raise CacheCheckFailed('Result dataset mismatch')
            if result.get('dataset_fingerprint') != (task.get('data_identity') or {}).get('fingerprint'):
                raise CacheCheckFailed('Result dataset bytes/row policy mismatch')
            if require_exports and (row['export_hash'] != row['payload_hash'] or row['byte_offset'] is None):
                raise CacheNotReady('Ledger fsync acknowledgement pending')
    return parents[0], models


def verify_export_bytes(models, ledger):
    resolved = str(Path(ledger).resolve())
    with Path(ledger).open('rb') as source:
        for row in models:
            if row['ledger_path'] != resolved:
                raise CacheCheckFailed('Result export refers to a different ledger')
            source.seek(row['byte_offset'])
            line = source.read(row['byte_length'])
            if not line.endswith(b'\n') or hashlib.sha256(_canonical(json.loads(line)).encode()).hexdigest() != row['payload_hash']:
                raise CacheCheckFailed('Saved result bytes missing or conflicting')


class _ExclusiveFiles:
    """Hold every target until all checks pass, then delete the exact allowlist."""
    def __init__(self, items, root):
        self.items, self.root, self.handles = items, root, []

    def __enter__(self):
        try:
            if os.name == 'nt':
                import ctypes
                from ctypes import wintypes
                self.api = ctypes.WinDLL('kernel32', use_last_error=True)
                self.api.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
                self.api.CreateFileW.restype = wintypes.HANDLE
                self.api.ReadFile.argtypes = (wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                              ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
                self.api.ReadFile.restype = wintypes.BOOL
                self.api.SetFileInformationByHandle.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
                self.api.SetFileInformationByHandle.restype = wintypes.BOOL
                self.api.CloseHandle.argtypes = (wintypes.HANDLE,)
                self.api.CloseHandle.restype = wintypes.BOOL
            for item in self.items:
                path = checked_cache_path(item['path'], self.root)
                if os.name == 'nt':
                    import ctypes
                    from ctypes import wintypes
                    # GENERIC_READ | DELETE, no sharing, existing regular file.
                    handle = self.api.CreateFileW(str(path), 0x80010000, 0, None, 3, 0x00200000, None)
                    if handle == ctypes.c_void_p(-1).value:
                        raise ctypes.WinError(ctypes.get_last_error())
                    self.handles.append((path, handle))
                    buffer = ctypes.create_string_buffer(4 * 1024**2)
                    count = wintypes.DWORD()
                    digest = hashlib.sha256()
                    while True:
                        if not self.api.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None):
                            raise ctypes.WinError(ctypes.get_last_error())
                        if not count.value: break
                        digest.update(buffer.raw[:count.value])
                else:
                    import fcntl
                    handle = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
                    self.handles.append((path, handle))
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    digest = hashlib.sha256()
                    while block := os.read(handle, 4 * 1024**2): digest.update(block)
                if digest.hexdigest() != item['sha256']:
                    raise CacheCheckFailed('Deletion target changed after preparation')
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def delete(self):
        if os.name == 'nt':
            import ctypes
            marked = []
            try:
                for _, handle in self.handles:
                    flag = ctypes.c_ubyte(1)  # FILE_DISPOSITION_INFO.DeleteFile: BOOLEAN
                    if not self.api.SetFileInformationByHandle(handle, 4, ctypes.byref(flag), ctypes.sizeof(flag)):
                        raise ctypes.WinError(ctypes.get_last_error())
                    marked.append(handle)
            except BaseException:
                # Clear successful marks before closing if any later file refuses
                # deletion (including a view whose original file handle closed).
                for handle in marked:
                    flag = ctypes.c_ubyte(0)
                    if not self.api.SetFileInformationByHandle(handle, 4, ctypes.byref(flag), ctypes.sizeof(flag)):
                        raise CacheCheckFailed('Windows could not roll back a deletion mark')
                raise
        else:
            for path, handle in self.handles:
                a, b = path.stat(), os.fstat(handle)
                if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                    raise CacheCheckFailed('Deletion path replaced while locked')
            for path, _ in self.handles: path.unlink()

    def __exit__(self, *_):
        for _, handle in reversed(self.handles):
            if os.name == 'nt': self.api.CloseHandle(handle)
            else: os.close(handle)
        self.handles.clear()


def _receipt_path(store, run_id, unit_id):
    token = hashlib.sha256(run_id.encode()).hexdigest()[:24]
    unit_token = hashlib.sha256(unit_id.encode()).hexdigest()[:24]
    return store.db_path.parent / 'cache_lifecycle' / token / (unit_token + '.json')


def verify_eviction(store, run_id, unit_id, *, allow_intent=False):
    row = store.cache_eviction(run_id, unit_id)
    if row is None or row['state'] not in (('intent', 'completed') if allow_intent else ('completed',)):
        raise CacheCheckFailed('No verified completed eviction receipt')
    receipt = json.loads(row['receipt_json'])
    path = Path(row['receipt_path'])
    if not path.is_file() or file_sha256(path) != row['receipt_sha256'] or json.loads(path.read_text(encoding='utf-8')) != receipt:
        raise CacheCheckFailed('Eviction receipt missing or tampered')
    parent, models = completed_unit_rows(store, run_id, unit_id, require_exports=False)
    actual = {r['scientific_task_id']: r['payload_hash'] for r in [parent, *models]}
    if receipt['schema'] != POLICY_VERSION or receipt['result_hashes'] != actual or receipt['run_id'] != run_id or receipt['unit_id'] != unit_id:
        raise CacheCheckFailed('Eviction receipt no longer matches completed results')
    for item in receipt['retained_artifacts']:
        if file_sha256(item['path']) != item['sha256']:
            raise CacheCheckFailed('Retained scientific evidence missing or changed')
    return receipt


def snapshot_eviction_evidence(snapshot, root, *, hash_file=file_sha256):
    """Read-only historical availability proof, distinct from live input bytes."""
    output = {}
    if not snapshot: return output
    tasks = {row['scientific_task_id']: row for row in snapshot['tasks']}
    results = {row['scientific_task_id']: row for row in snapshot['durable_results']}
    attempts = {row['attempt_id']: row for row in snapshot['attempts']}
    unit_members = {}
    for tid, task in tasks.items():
        if task['stage'] == 'precompute':
            unit_members.setdefault(tid, {})[tid] = task
        for dependency in json.loads(task['payload_json']).get('depends_on', []):
            unit_members.setdefault(dependency, {})[tid] = task
    resolve = lambda value: Path(value) if Path(value).is_absolute() else Path(root) / value
    for row in snapshot.get('cache_evictions', []):
        if row['state'] != 'completed': continue
        receipt = json.loads(row['receipt_json'])
        path = resolve(row['receipt_path'])
        if hash_file(path) != row['receipt_sha256'] or json.loads(path.read_text(encoding='utf-8')) != receipt:
            raise CacheCheckFailed('Eviction receipt integrity failed')
        if receipt.get('schema') != POLICY_VERSION or receipt.get('run_id') != row['run_id'] or receipt.get('unit_id') != row['unit_id']:
            raise CacheCheckFailed('Eviction receipt identity failed')
        members = unit_members.get(row['unit_id'], {})
        if len(members) < 2 or set(receipt['result_hashes']) != set(members):
            raise CacheCheckFailed('Eviction receipt membership failed')
        for tid, task in members.items():
            result = results.get(tid)
            if task['state'] != 'completed' or result is None or result['payload_hash'] != receipt['result_hashes'][tid] or hashlib.sha256(result['payload_json'].encode()).hexdigest() != result['payload_hash']:
                raise CacheCheckFailed('Eviction result ownership/integrity failed')
            if attempts.get(result['attempt_id'], {}).get('state') != 'completed':
                raise CacheCheckFailed('Eviction attempt incomplete')
        expected = {(resolve(item['path']).resolve(), item['sha256'], item['role'])
                    for item in json.loads(results[row['unit_id']]['payload_json']).get('artifacts', [])}
        for item in receipt['retained_artifacts']:
            if hash_file(resolve(item['path'])) != item['sha256']:
                raise CacheCheckFailed('Eviction evidence unavailable')
        for item in receipt['deleted_artifacts']:
            key = (resolve(item['path']).resolve(), item['sha256'], item['role'])
            if item['role'] not in DISPOSABLE_ROLES or key not in expected:
                raise CacheCheckFailed('Eviction allowlist differs from verified preparation')
            output[(key[0], key[1])] = {'receipt_path': str(path), 'receipt_sha256': row['receipt_sha256'],
                                        'verified_at': receipt['verified_at']}
    return output


def evict_unit(store, run_id, unit_id, ledger, root, *, exited_processes=()):
    """The coordinator holds the protocol lease and admits no work during this."""
    workers_released(exited_processes)
    parent, models = completed_unit_rows(store, run_id, unit_id)
    verify_export_bytes(models, ledger)
    result = json.loads(parent['result_json'])
    descriptor = result.get('prepared_descriptor')
    if not descriptor:
        raise CacheCheckFailed('Preparation has no verified descriptor')
    existing = store.cache_eviction(run_id, unit_id)
    if existing and existing['state'] in ('intent', 'completed'):
        receipt = verify_eviction(store, run_id, unit_id, allow_intent=True)
    else:
        verify_unit(descriptor)
        artifacts = result.get('artifacts', [])
        by_path = {}
        for item in artifacts:
            path = checked_cache_path(item['path'], root)
            if str(path) in by_path and by_path[str(path)]['sha256'] != item['sha256']:
                raise CacheCheckFailed('Conflicting artifact identities')
            by_path[str(path)] = {**item, 'path': str(path)}
        deletable = [item for item in by_path.values() if item['role'] in DISPOSABLE_ROLES]
        retained = [item for item in by_path.values() if item['role'] not in DISPOSABLE_ROLES]
        if not deletable:
            raise CacheCheckFailed('No declared disposable matrices')
        expected = {(str(Path(item['path']).resolve()), item['sha256']) for item in artifacts}
        from src.pipeline_runner import GPU_MODELS
        # Common inputs are precisely the descriptor, split, and preprocessing
        # artifacts. Pipeline artifacts carry their owner in preparation.
        common = {(str(Path(item['path']).resolve()), item['sha256']) for item in artifacts
                  if item.get('pipeline') is None}
        for row in models:
            model_result = json.loads(row['result_json'])
            task = json.loads(row['payload_json'])
            executed = model_result.get('artifacts', [])
            required = common | {(str(Path(item['path']).resolve()), item['sha256']) for item in artifacts
                if item.get('pipeline') == task['pipeline'] and
                (task['model'] not in GPU_MODELS or item['role'] not in ('model_train_numeric_matrix', 'model_test_numeric_matrix'))}
            actual = {(str(Path(item['path']).resolve()), item['sha256']) for item in executed}
            if not executed or actual != required or not actual.issubset(expected):
                raise CacheCheckFailed('Model input evidence does not match its prepared unit')
        for item in retained:
            if file_sha256(item['path']) != item['sha256']:
                raise CacheCheckFailed('Retained artifact missing or corrupt')
        receipt = {'schema': POLICY_VERSION, 'run_id': run_id, 'unit_id': unit_id,
            'prepared_descriptor': descriptor, 'cache_root': str(Path(root).resolve()),
            'result_hashes': {r['scientific_task_id']: r['payload_hash'] for r in [parent, *models]},
            'deleted_artifacts': deletable, 'retained_artifacts': retained,
            'workers_released': list(exited_processes), 'verified_at': datetime.now(timezone.utc).isoformat()}
    path = _receipt_path(store, run_id, unit_id)
    # Durable intent makes a crash during deletion explicit and recoverable.
    atomic_json(path, receipt)
    store.record_cache_eviction(run_id, unit_id, 'intent', receipt, path)
    remaining = [item for item in receipt['deleted_artifacts'] if checked_cache_path(item['path'], root).exists()]
    try:
        with _ExclusiveFiles(remaining, root) as owned:
            # Recheck task/attempt ownership and evidence after all handles open.
            verify_eviction(store, run_id, unit_id, allow_intent=True)
            workers_released(exited_processes)
            owned.delete()
        if any(Path(item['path']).exists() for item in receipt['deleted_artifacts']):
            raise CacheCheckFailed('Windows has not released every deleted file')
    except BaseException as error:
        # Keep intent and original hashes. No new unit may start after this error.
        raise CacheCheckFailed('Cache retained or eviction incomplete: ' + str(error)) from error
    store.record_cache_eviction(run_id, unit_id, 'completed', receipt, path)
    return {'unit_id': unit_id, 'deleted_files': len(receipt['deleted_artifacts']),
            'status': 'completed', 'receipt_path': str(path)}
