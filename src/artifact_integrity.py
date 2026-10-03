"""Content identities and atomic, locked artifact publication."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_SCHEMA_VERSION = "verified_feature_cache_v2"
PREPROCESSING_SEMANTICS_VERSION = "fold_local_mean_scale_exact_onehot_v1"


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path, *, check_cancel=None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if check_cancel is not None:
                check_cancel()
            digest.update(block)
    return digest.hexdigest()


def array_identity(values) -> dict:
    values = np.asarray(values)
    if values.dtype.hasobject:
        content = fingerprint(values.tolist())
    else:
        content = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    return {"dtype": str(values.dtype), "shape": list(values.shape), "sha256": content}


def frame_identity(frame: pd.DataFrame) -> dict:
    if any(isinstance(dtype,pd.SparseDtype) for dtype in frame.dtypes):
        # Hash the exact logical dense row order in bounded blocks.
        digest=hashlib.sha256()
        for start in range(0,len(frame),128):
            digest.update(np.ascontiguousarray(frame.iloc[start:start+128].to_numpy(dtype=np.float64)).tobytes())
        values={'dtype':'float64','shape':list(frame.shape),'sha256':digest.hexdigest()}
    else:
        values=array_identity(frame.to_numpy())
    return {"columns": list(map(str,frame.columns)), "dtypes": list(map(str,frame.dtypes)),
        "index": array_identity(frame.index.to_numpy()), "values": values, "shape": list(frame.shape)}


def dataset_identity(path, *, frame=None) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"availability": "unavailable", "fingerprint": None}
    frame = pd.read_csv(path) if frame is None else frame
    target = "target_label" if "target_label" in frame.columns else "target"
    if target not in frame:
        raise ValueError("Dataset lacks declared target column")
    identity = {"availability": "available", "bytes_sha256": file_sha256(path), "bytes": path.stat().st_size,
        "target": target, "coordinates": list(map(str,frame.columns)), "dtypes": list(map(str,frame.dtypes)), "rows": len(frame),
        "parser": {"engine": "pandas.read_csv_defaults", "pandas_version": pd.__version__}, "row_selection": "stored_csv_row_order"}
    return {**identity, "fingerprint": fingerprint(identity)}


@contextmanager
def artifact_lock(path, *, timeout_seconds=60.0):
    """OS lock is released even if its owning worker exits; no stale lock lease."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0,2)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    deadline = time.monotonic()+timeout_seconds
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except (OSError,BlockingIOError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Artifact lock deadline: {path}")
                time.sleep(.02)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
    finally:
        handle.close()


def atomic_bytes(path, payload: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name = tempfile.mkstemp(prefix=".publish_",dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_bytes(path,json.dumps(value,sort_keys=True,allow_nan=False).encode("utf-8"))


def atomic_pickle(path,value):
    atomic_bytes(path,pickle.dumps(value,protocol=5))


def validate_feature_cache(train_path,test_path,meta_path,dependency):
    """No reader accepts existence, incomplete publication, or corrupted bytes."""
    try:
        metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        if metadata.get("cache_schema") != CACHE_SCHEMA_VERSION or metadata.get("dependency_signature") != dependency:
            return None
        for role,path in (("train",train_path),("test",test_path)):
            if file_sha256(path) != metadata["artifacts"][role]["sha256"]:
                return None
        train,test = pd.read_pickle(train_path),pd.read_pickle(test_path)
        for role,frame in (("train",train),("test",test)):
            if frame_identity(frame) != metadata["artifacts"][role]["identity"]:
                return None
            if list(frame.columns) != metadata["selected_feature_identities"]:
                return None
        if train.shape[1] != test.shape[1] or train.shape[1] != metadata["actual_estimator_input_dimension"]:
            return None
        return train,test,metadata
    except (OSError,ValueError,KeyError,EOFError,pickle.UnpicklingError,TypeError):
        return None
