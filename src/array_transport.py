"""Bounded, regenerable NumPy transport for spawned classifier workers.

The coordinator publishes each large numeric input once. Workers open private
copy-on-write views: mutations cannot reach another fit or the backing files.
These files are execution auxiliaries; durable task/result identities never
depend on a mapping path, and recovery regenerates them from the feature cache.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import psutil

from src.provenance import atomic_write_json, stable_digest

ARRAY_FIELDS = ('xtr', 'xte', 'y_train_enc', 'y_test_enc')


@contextmanager
def task_arrays(payload):
    """Close every mapping before a worker advertises completion."""
    descriptors = payload.get('_mapped_arrays', {})
    result = {k:v for k,v in payload.items() if k != '_mapped_arrays'}
    opened = []
    try:
        for name, entry in descriptors.items():
            if name not in ARRAY_FIELDS:
                raise ValueError('Unexpected mapped array field')
            array = np.load(entry['path'], mmap_mode='c', allow_pickle=False)
            opened.append(array)
            if list(array.shape) != entry['shape'] or array.dtype.str != entry['dtype']:
                raise ValueError('Mapped array shape/dtype changed')
            result[name] = array
        yield result
    finally:
        for array in opened:
            array._mmap.close()


class ArrayTransport:
    def __init__(self, root: Path, *, budget_bytes: int, threshold_bytes: int, prior_stats: dict | None = None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget_bytes, self.threshold_bytes = budget_bytes, threshold_bytes
        self.groups = {}
        self.stats = dict(groups_published=0, mapped_task_submissions=0,
                          serialized_task_submissions=0, peak_disk_bytes=0,
                          cleanup_deferred=0, threshold_bytes=threshold_bytes,
                          budget_bytes=budget_bytes)
        self._reap_stale()
        if prior_stats:
            for field in ('groups_published','mapped_task_submissions','serialized_task_submissions','cleanup_deferred'):
                self.stats[field] += int(prior_stats.get(field, 0))
            self.stats['peak_disk_bytes'] = max(self.stats['peak_disk_bytes'], int(prior_stats.get('peak_disk_bytes', 0)))
        self.epoch = self.root/f'{os.getpid()}-{uuid.uuid4().hex}'
        self.epoch.mkdir()
        atomic_write_json(self.epoch/'owner.json', dict(pid=os.getpid(),
                          create_time=psutil.Process().create_time()))

    def _checked(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError('Transport cleanup escaped its run directory')
        return path

    def _remove_epoch(self, epoch):
        """Delete only known transport files inside a verified absolute root."""
        epoch = self._checked(epoch)
        for group in epoch.iterdir():
            if group.is_dir():
                for path in group.iterdir():
                    path = self._checked(path)
                    if path.is_file() and (path.suffix == '.npy' or path.name == 'manifest.json'):
                        path.unlink()
                self._checked(group).rmdir()
        owner = self._checked(epoch/'owner.json')
        owner.unlink(missing_ok=True)
        epoch.rmdir()

    def _reap_stale(self):
        for epoch in self.root.iterdir():
            if not epoch.is_dir(): continue
            try:
                owner = json.loads((epoch/'owner.json').read_text(encoding='utf-8'))
                try: alive = psutil.Process(owner['pid']).create_time() == owner['create_time']
                except psutil.Error: alive = False
                if not alive: self._remove_epoch(epoch)
            except (OSError, ValueError, KeyError):
                self.stats['cleanup_deferred'] += 1

    def disk_bytes(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def payload(self, payload, key, *, mode='auto'):
        size = sum(payload[n].nbytes for n in ARRAY_FIELDS)
        eligible = all(not payload[n].dtype.hasobject for n in ARRAY_FIELDS)
        if (not eligible or (mode == 'auto' and size < self.threshold_bytes)
                or size > self.budget_bytes):
            self.stats['serialized_task_submissions'] += 1
            return payload, 'pickle'
        if key not in self.groups:
            current = self.disk_bytes()
            # Headers/manifests have a small publication allowance. Never
            # wait on transport space while holding a classifier lease.
            allowance = size + 4096*len(ARRAY_FIELDS)
            if current+allowance > self.budget_bytes or shutil.disk_usage(self.root).free < 2*allowance:
                self.stats['serialized_task_submissions'] += 1
                return payload, 'pickle'
            group = self.epoch/stable_digest(key)
            group.mkdir()
            descriptors = {}
            try:
                for name in ARRAY_FIELDS:
                    path = group/f'{name}.npy'
                    np.save(path, payload[name], allow_pickle=False)
                    descriptors[name] = dict(path=str(path.resolve()),
                                             shape=list(payload[name].shape), dtype=payload[name].dtype.str)
                atomic_write_json(group/'manifest.json', dict(key=key, arrays=descriptors))
            except Exception:
                for path in group.iterdir(): self._checked(path).unlink()
                group.rmdir()
                raise
            self.groups[key] = (group, descriptors)
            self.stats['groups_published'] += 1
            self.stats['peak_disk_bytes'] = max(self.stats['peak_disk_bytes'], self.disk_bytes())
        result = {k:v for k,v in payload.items() if k not in ARRAY_FIELDS}
        result['_mapped_arrays'] = self.groups[key][1]
        self.stats['mapped_task_submissions'] += 1
        return result, 'mapped'

    def release(self, key):
        record = self.groups.get(key)
        if record is None: return
        group, _ = record
        try:
            for path in group.iterdir(): self._checked(path).unlink()
            group.rmdir()
            self.groups.pop(key)
        except OSError:
            self.stats['cleanup_deferred'] += 1

    def close(self):
        for key in list(self.groups): self.release(key)
        try: self._remove_epoch(self.epoch)
        except OSError: self.stats['cleanup_deferred'] += 1
        self._reap_stale()
