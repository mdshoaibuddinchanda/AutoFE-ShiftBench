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


def main(output: str | Path = "provenance/host_measurement.json") -> dict:
    cpu = _powershell("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,SocketDesignation | ConvertTo-Json -Compress")
    gpu = _powershell("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM,VideoProcessor | ConvertTo-Json -Compress")
    memory = _powershell("Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress")
    disks = _powershell("Get-PhysicalDisk | Select-Object FriendlyName,MediaType,BusType,Size,HealthStatus | ConvertTo-Json -Compress")
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
        "python": sys.version, "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "cpu": cpu_value, "gpu": gpu_value, "memory": memory_value,
        "physical_disks": disks_value, "disk_usage": disk_usage,
        "packages": packages,
        "note": "Run this probe on the intended friend PC; this file is valid only for the host named above.",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return payload


if __name__ == "__main__":
    main()
