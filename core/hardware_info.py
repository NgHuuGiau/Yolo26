from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import List

import psutil

try:
    import GPUtil
except ImportError:  # pragma: no cover
    GPUtil = None

torch = None
TORCH_IMPORT_ERROR = None


@dataclass
class HardwareInfo:
    cpu_name: str
    ram_gb: float
    gpu_name: str
    vram_gb: float
    cuda_available: bool
    os_name: str
    gpu_count: int
    torch_version: str = "Khong co PyTorch"
    torch_cuda_version: str = "CPU-only"
    cuda_runtime_status: str = "Khong"
    cuda_runtime_reason: str = "Chua kiem tra"
    gpu_hardware_available: bool = False

    def pretty_report(self) -> str:
        return (
            "===== KIEM TRA CAU HINH =====\n"
            f"CPU: {self.cpu_name}\n"
            f"RAM: {self.ram_gb:.1f} GB\n"
            f"GPU: {self.gpu_name}\n"
            f"VRAM: {self.vram_gb:.1f} GB\n"
            f"CUDA: {'Co' if self.cuda_available else 'Khong'}\n"
            f"PyTorch: {self.torch_version}\n"
            f"CUDA build: {self.torch_cuda_version}\n"
            f"OS: {self.os_name}"
        )

    def summary(self) -> dict:
        return {
            "cpu_name": self.cpu_name,
            "ram_gb": round(self.ram_gb, 2),
            "gpu_name": self.gpu_name,
            "vram_gb": round(self.vram_gb, 2),
            "cuda_available": self.cuda_available,
            "os_name": self.os_name,
            "gpu_count": self.gpu_count,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cuda_runtime_status": self.cuda_runtime_status,
            "cuda_runtime_reason": self.cuda_runtime_reason,
            "gpu_hardware_available": self.gpu_hardware_available,
        }


def _detect_cpu_name() -> str:
    cpu_name = platform.processor().strip()
    if cpu_name:
        return cpu_name
    return platform.uname().processor or "Unknown CPU"


def _detect_gpu_from_gputil() -> tuple[str, float, int]:
    if GPUtil is None:
        return "Khong phat hien GPU", 0.0, 0
    try:
        gpus: List = GPUtil.getGPUs()
    except Exception:
        return "Khong phat hien GPU", 0.0, 0
    if not gpus:
        return "Khong phat hien GPU", 0.0, 0
    primary = gpus[0]
    return primary.name, round(primary.memoryTotal / 1024, 2), len(gpus)


def _load_torch():
    global torch, TORCH_IMPORT_ERROR
    if torch is not None or TORCH_IMPORT_ERROR is not None:
        return torch
    try:
        import torch as imported_torch

        torch = imported_torch
    except Exception as exc:  # pragma: no cover
        TORCH_IMPORT_ERROR = exc
        torch = None
    return torch


def detect_hardware() -> HardwareInfo:
    torch_module = torch if torch is not None else _load_torch()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    gpu_name, vram_gb, gpu_count = _detect_gpu_from_gputil()
    gpu_hardware_available = gpu_count > 0
    cuda_available = bool(torch_module and torch_module.cuda.is_available())
    torch_version = getattr(torch_module, "__version__", "Khong co PyTorch") if torch_module is not None else "Khong co PyTorch"
    torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None) if torch_module is not None else None
    torch_cuda_label = torch_cuda_version or "CPU-only"
    cuda_runtime_status = "Co" if cuda_available else "Khong"
    cuda_runtime_reason = "PyTorch CUDA runtime san sang"

    if torch_module is None:
        cuda_runtime_reason = (
            f"Khong khoi tao duoc PyTorch: {TORCH_IMPORT_ERROR}"
            if TORCH_IMPORT_ERROR is not None
            else "Chua cai PyTorch trong moi truong hien tai"
        )
    elif gpu_hardware_available and not torch_cuda_version:
        cuda_runtime_reason = "PyTorch hien tai la ban CPU-only, chua ho tro CUDA"
    elif gpu_hardware_available and torch_cuda_version and not cuda_available:
        cuda_runtime_reason = "PyTorch co CUDA build nhung runtime chua khoi tao duoc GPU"
    elif not gpu_hardware_available:
        cuda_runtime_reason = "Khong phat hien GPU tuong thich de chay CUDA"

    if cuda_available and torch_module is not None:
        index = torch_module.cuda.current_device()
        gpu_name = torch_module.cuda.get_device_name(index)
        properties = torch_module.cuda.get_device_properties(index)
        vram_gb = properties.total_memory / (1024**3)
        gpu_count = torch_module.cuda.device_count()
        gpu_hardware_available = gpu_count > 0

    return HardwareInfo(
        cpu_name=_detect_cpu_name(),
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        cuda_available=cuda_available,
        os_name=f"{platform.system()} {platform.release()}",
        gpu_count=gpu_count,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_label,
        cuda_runtime_status=cuda_runtime_status,
        cuda_runtime_reason=cuda_runtime_reason,
        gpu_hardware_available=gpu_hardware_available,
    )
