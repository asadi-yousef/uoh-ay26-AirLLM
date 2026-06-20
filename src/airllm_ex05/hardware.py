"""Hardware environment collection."""

import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

from airllm_ex05.metrics import bytes_to_gb
from airllm_ex05.models import HardwareInfo, write_json_model


def collect_hardware_info() -> HardwareInfo:
    """Collect CPU, memory, GPU, CUDA, OS, Python, and storage details."""
    gpu_model, vram_total_gb, cuda_available = _gpu_info()
    return HardwareInfo(
        cpu_model=_cpu_model(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True) or 0,
        ram_total_gb=bytes_to_gb(psutil.virtual_memory().total) or 0.0,
        gpu_model=gpu_model,
        vram_total_gb=vram_total_gb,
        cuda_available=cuda_available,
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        python_version=sys.version.split()[0],
        storage=_storage_info(),
    )


def save_hardware_info(output_path: Path) -> Path:
    """Collect and save hardware information."""
    return write_json_model(collect_hardware_info(), output_path)


def _cpu_model() -> str:
    processor = platform.processor().strip()
    if processor:
        return processor
    try:
        output = subprocess.run(
            ["wmic", "cpu", "get", "name"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return platform.machine() or "unknown"
    lines = [line.strip() for line in output.splitlines() if line.strip() and line.strip() != "Name"]
    return lines[0] if lines else platform.machine() or "unknown"


def _gpu_info() -> tuple[str | None, float | None, bool]:
    try:
        import torch
    except ImportError:
        return None, None, False
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return None, None, False
    props = torch.cuda.get_device_properties(0)
    return props.name, bytes_to_gb(props.total_memory), True


def _storage_info() -> list[dict[str, Any]]:
    partitions = []
    for partition in psutil.disk_partitions(all=False):
        if not _is_fixed_partition(partition.opts):
            continue
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except OSError:
            continue
        partitions.append(
            {
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "filesystem": partition.fstype,
                "total_gb": bytes_to_gb(usage.total),
                "free_gb": bytes_to_gb(usage.free),
            }
        )
    return partitions


def _is_fixed_partition(options: str) -> bool:
    return re.search(r"cdrom|remote", options, flags=re.IGNORECASE) is None
