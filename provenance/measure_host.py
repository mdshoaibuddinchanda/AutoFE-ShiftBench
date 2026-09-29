"""Capture Windows host, storage, GPU, Conda, and package identity.

Run this script on the intended execution PC.  It does not create or modify
an environment and writes only the requested JSON output.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path


def _powershell(script: str) -> str:
    try:
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def _package_importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _disk_io_probe(root: Path, size_mib: int = 32) -> dict[str, object]:
    """Bounded sequential write/read probe on the intended run volume."""
    size = max(1, int(size_mib)) * 1024 * 1024
    root.mkdir(parents=True, exist_ok=True)
    path = root / f".friend_pc_disk_probe_{os.getpid()}.bin"
    payload = b"A" * (1024 * 1024)
    started = time.perf_counter()
    try:
        with path.open("wb") as stream:
            remaining = size
            while remaining:
                chunk = payload[: min(len(payload), remaining)]
                stream.write(chunk)
                remaining -= len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        write_s = time.perf_counter() - started
        started = time.perf_counter()
        with path.open("rb") as stream:
            while stream.read(1024 * 1024):
                pass
        read_s = time.perf_counter() - started
        return {
            "path": str(path.parent), "bytes": size,
            "write_seconds": write_s, "read_seconds": read_s,
            "write_mib_per_s": size / 1024**2 / write_s if write_s else None,
            "read_mib_per_s": size / 1024**2 / read_s if read_s else None,
        }
    except OSError as exc:
        return {"path": str(path.parent), "bytes": size, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        path.unlink(missing_ok=True)


def main(
    output: str | Path = "provenance/host_measurement.json",
    *,
    probe_root: str | Path | None = None,
    probe_mib: int = 32,
) -> dict:
    cpu = _powershell("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,SocketDesignation | ConvertTo-Json -Compress")
    gpu = _powershell("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM,VideoProcessor | ConvertTo-Json -Compress")
    memory = _powershell("Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress")
    disks = _powershell("Get-PhysicalDisk | Select-Object FriendlyName,MediaType,BusType,Size,HealthStatus | ConvertTo-Json -Compress")
    topology = _powershell("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,ThreadCount,MaxClockSpeed,CurrentClockSpeed,SocketDesignation,Manufacturer | ConvertTo-Json -Compress")
    power_plan = _powershell("powercfg /GETACTIVESCHEME")
    try:
        cpu_value = json.loads(cpu)
    except json.JSONDecodeError:
        cpu_value = cpu
    try:
        gpu_value = json.loads(gpu)
    except json.JSONDecodeError:
        gpu_value = gpu
    try:
        memory_value = json.loads(memory)
    except json.JSONDecodeError:
        memory_value = memory
    try:
        disks_value = json.loads(disks)
    except json.JSONDecodeError:
        disks_value = disks
    try:
        topology_value = json.loads(topology)
    except json.JSONDecodeError:
        topology_value = topology
    disk_usage = {}
    for drive in sorted({Path.cwd().anchor, "D:\\"}):
        try:
            usage = shutil.disk_usage(drive)
            disk_usage[drive] = {
                "total_bytes": usage.total, "used_bytes": usage.used,
                "free_bytes": usage.free, "free_gib": usage.free / 1024**3,
            }
        except OSError as exc:
            disk_usage[drive] = f"unavailable:{exc}"
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "featuretools", "woodwork", "xgboost", "lightgbm", "catboost", "psutil"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    payload = {
        "artifact_type": "host_measurement",
        "measured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "computer_name": os.environ.get("COMPUTERNAME"),
        "python": sys.version, "python_executable": sys.executable,
        "sys_prefix": sys.prefix, "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cpu": cpu_value, "cpu_topology": topology_value,
        "power_plan": power_plan,
        "gpu": gpu_value,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "memory": memory_value,
        "physical_disks": disks_value, "disk_usage": disk_usage,
        "disk_io_probe": _disk_io_probe(Path(probe_root) if probe_root else Path.cwd(), probe_mib),
        "packages": packages,
        "gpu_backend_probe": {
            "xgboost_importable": _package_importable("xgboost"),
            "catboost_importable": _package_importable("catboost"),
            "nvidia_smi": _powershell("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits"),
        },
        "note": "Run this probe on the intended friend PC; this file is valid only for the host named above. P/E topology is UNKNOWN when Windows does not expose heterogeneous-core fields.",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="provenance/host_measurement.json")
    parser.add_argument("--probe-root", help="Volume/path for the bounded sequential I/O probe")
    parser.add_argument("--probe-mib", type=int, default=32)
    args = parser.parse_args()
    main(args.output, probe_root=args.probe_root, probe_mib=args.probe_mib)
