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
    torch_version: str = "Không có PyTorch"
    torch_cuda_version: str = "CPU-only"
    cuda_runtime_status: str = "Không"
    cuda_runtime_reason: str = "Chưa kiểm tra"
    gpu_hardware_available: bool = False
    cpu_usage_percent: float | None = None
    ram_usage_percent: float | None = None
    gpu_usage_percent: float | None = None
    vram_usage_percent: float | None = None

    def pretty_report(self) -> str:
        cpu_usage = f"{self.cpu_usage_percent:.1f}%" if self.cpu_usage_percent is not None else "Không rõ"
        ram_usage = f"{self.ram_usage_percent:.1f}%" if self.ram_usage_percent is not None else "Không rõ"
        gpu_usage = f"{self.gpu_usage_percent:.1f}%" if self.gpu_usage_percent is not None else "Không rõ"
        vram_usage = f"{self.vram_usage_percent:.1f}%" if self.vram_usage_percent is not None else "Không rõ"
        return (
            "===== KIỂM TRA CẤU HÌNH =====\n"
            f"CPU: {self.cpu_name}\n"
            f"RAM: {self.ram_gb:.1f} GB\n"
            f"GPU: {self.gpu_name}\n"
            f"VRAM: {self.vram_gb:.1f} GB\n"
            f"CPU dùng: {cpu_usage}\n"
            f"RAM dùng: {ram_usage}\n"
            f"GPU dùng: {gpu_usage}\n"
            f"VRAM dùng: {vram_usage}\n"
            f"CUDA: {'Có' if self.cuda_available else 'Không'}\n"
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
            "cpu_usage_percent": round(self.cpu_usage_percent, 2) if self.cpu_usage_percent is not None else None,
            "ram_usage_percent": round(self.ram_usage_percent, 2) if self.ram_usage_percent is not None else None,
            "gpu_usage_percent": round(self.gpu_usage_percent, 2) if self.gpu_usage_percent is not None else None,
            "vram_usage_percent": round(self.vram_usage_percent, 2) if self.vram_usage_percent is not None else None,
        }


def _detect_cpu_name() -> str:
    cpu_name = platform.processor().strip()
    if cpu_name:
        return cpu_name
    return platform.uname().processor or "Không rõ CPU"


def _detect_gpu_from_gputil() -> tuple[str, float, int, float | None, float | None]:
    if GPUtil is None:
        return "Không phát hiện GPU", 0.0, 0, None, None
    try:
        gpus: List = GPUtil.getGPUs()
    except Exception:
        return "Không phát hiện GPU", 0.0, 0, None, None
    if not gpus:
        return "Không phát hiện GPU", 0.0, 0, None, None
    primary = gpus[0]
    gpu_usage_percent = float(primary.load) * 100 if getattr(primary, "load", None) is not None else None
    vram_usage_percent = float(primary.memoryUtil) * 100 if getattr(primary, "memoryUtil", None) is not None else None
    return primary.name, round(primary.memoryTotal / 1024, 2), len(gpus), gpu_usage_percent, vram_usage_percent


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
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024**3)
    ram_usage_percent = float(getattr(memory, "percent", 0.0))
    cpu_usage_percent = float(psutil.cpu_percent(interval=0.15))
    gpu_name, vram_gb, gpu_count, gpu_usage_percent, vram_usage_percent = _detect_gpu_from_gputil()
    gpu_hardware_available = gpu_count > 0
    cuda_available = bool(torch_module and torch_module.cuda.is_available())
    torch_version = getattr(torch_module, "__version__", "Không có PyTorch") if torch_module is not None else "Không có PyTorch"
    torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None) if torch_module is not None else None
    torch_cuda_label = torch_cuda_version or "CPU-only"
    cuda_runtime_status = "Có" if cuda_available else "Không"
    cuda_runtime_reason = "PyTorch CUDA runtime sẵn sàng"

    if torch_module is None:
        cuda_runtime_reason = (
            f"Không khởi tạo được PyTorch: {TORCH_IMPORT_ERROR}"
            if TORCH_IMPORT_ERROR is not None
            else "Chưa cài PyTorch trong môi trường hiện tại"
        )
    elif gpu_hardware_available and not torch_cuda_version:
        cuda_runtime_reason = "PyTorch hiện tại là bản CPU-only, chưa hỗ trợ CUDA"
    elif gpu_hardware_available and torch_cuda_version and not cuda_available:
        cuda_runtime_reason = "PyTorch có CUDA build nhưng runtime chưa khởi tạo được GPU"
    elif not gpu_hardware_available:
        cuda_runtime_reason = "Không phát hiện GPU tương thích để chạy CUDA"

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
        cpu_usage_percent=cpu_usage_percent,
        ram_usage_percent=ram_usage_percent,
        gpu_usage_percent=gpu_usage_percent,
        vram_usage_percent=vram_usage_percent,
    )


def get_live_usage_snapshot() -> dict[str, float | None]:
    try:
        memory = psutil.virtual_memory()
        ram_usage_percent = float(getattr(memory, "percent", 0.0))
    except Exception:
        ram_usage_percent = None

    try:
        cpu_usage_percent = float(psutil.cpu_percent(interval=None))
    except Exception:
        cpu_usage_percent = None

    _gpu_name, _vram_gb, _gpu_count, gpu_usage_percent, vram_usage_percent = _detect_gpu_from_gputil()
    return {
        "cpu_usage_percent": cpu_usage_percent,
        "ram_usage_percent": ram_usage_percent,
        "gpu_usage_percent": gpu_usage_percent,
        "vram_usage_percent": vram_usage_percent,
    }
