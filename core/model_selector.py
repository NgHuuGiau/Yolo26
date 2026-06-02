from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.hardware_detector import HardwareInfo
from utils.file_utils import load_yaml_cached


SETTINGS_PATH = Path("config/settings.yaml")
MODEL_BACKUPS = {
    "yolo26s.pt": ("gpu_model", "yolo11s.pt", "yolov8s.pt"),
    "yolo26n.pt": ("cpu_model", "yolo11n.pt", "yolov8n.pt"),
    "yolo26m.pt": (None, "yolo11m.pt", "yolov8m.pt"),
    "yolo11s.pt": (None, "yolov8s.pt"),
    "yolo11n.pt": (None, "yolov8n.pt"),
}
MODE_DEVICE_HINTS = {"high": "gpu", "medium": "gpu"}
FALLBACK_CHAINS = {
    "high": ("high", "fallback_gpu_1", "fallback_gpu_2", "fallback_cpu", "fallback_cpu_weak"),
    "medium": ("medium", "fallback_gpu_2", "fallback_cpu", "fallback_cpu_weak"),
    "low": ("low", "fallback_cpu", "fallback_cpu_weak"),
    "fallback_cpu": ("fallback_cpu", "fallback_cpu_weak"),
    "fallback_cpu_weak": ("fallback_cpu_weak",),
}


@dataclass
class RuntimeConfig:
    mode: str
    profile_name: str
    requested_profile_name: str
    requested_device: str
    resolved_device: str
    requested_model_name: str
    primary_model_name: str
    candidate_models: list[str]
    requested_imgsz: int
    imgsz: int
    conf: float
    max_det: int
    use_half: bool
    camera_width: int
    camera_height: int
    font_size: int
    box_thickness: int
    label_font_scale: float
    active_model_name: str = ""
    fallback_chain: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "profile_name": self.profile_name,
            "requested_profile_name": self.requested_profile_name,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "requested_model_name": self.requested_model_name,
            "primary_model_name": self.primary_model_name,
            "candidate_models": self.candidate_models,
            "requested_imgsz": self.requested_imgsz,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "max_det": self.max_det,
            "use_half": self.use_half,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
        }

    def pretty_report(self) -> str:
        return (
            "===== CAU HINH DUOC CHON =====\n"
            f"Che do: {self.mode}\n"
            f"Profile: {self.profile_name}\n"
            f"Device: {self.resolved_device}\n"
            f"Model: {self.primary_model_name}\n"
            f"imgsz: {self.imgsz}\n"
            f"Half: {'on' if self.use_half else 'off'}\n"
            f"Camera size: {self.camera_width}x{self.camera_height}"
        )


def _camera_preset(settings: dict) -> dict:
    return settings.get("display_camera", settings["camera_large"])


def build_candidates(model_name: str, settings: dict) -> list[str]:
    backup_key, *fallbacks = MODEL_BACKUPS.get(model_name, (None, "yolov8n.pt"))
    backup_model = settings.get("stable_backup", {}).get(backup_key) if backup_key else None
    return list(dict.fromkeys([model_name, *(item for item in [backup_model, *fallbacks] if item)]))


def _profile_tuple(profile_name: str, settings: dict) -> tuple[str, str, int]:
    profile = settings["models"][profile_name]
    return profile["device"], profile["model"], int(profile["imgsz"])


def _first_matching_profile(metric: float, profiles: tuple[tuple[str, float, str], ...], default: str) -> str:
    for config_key, minimum, fallback in profiles:
        if metric >= minimum:
            return config_key or fallback
    return default


def _default_cpu_profile_name(hardware: HardwareInfo, settings: dict) -> str:
    auto_profiles = settings.get("auto_profiles", {})
    profiles = (
        (auto_profiles.get("cpu_balanced", {}).get("profile"), auto_profiles.get("cpu_balanced", {}).get("min_ram_gb", 16), "low"),
        (auto_profiles.get("cpu_safe", {}).get("profile"), auto_profiles.get("cpu_safe", {}).get("min_ram_gb", 8), "fallback_cpu"),
    )
    return _first_matching_profile(hardware.ram_gb, profiles, auto_profiles.get("cpu_minimum", {}).get("profile", "fallback_cpu_weak"))


def _default_gpu_profile_name(hardware: HardwareInfo, settings: dict) -> str:
    auto_profiles = settings.get("auto_profiles", {})
    profiles = (
        (auto_profiles.get("gpu_high_end", {}).get("profile"), auto_profiles.get("gpu_high_end", {}).get("min_vram_gb", 8), "high"),
        (auto_profiles.get("gpu_balanced", {}).get("profile"), 3.5, "medium"),
    )
    return _first_matching_profile(hardware.vram_gb, profiles, auto_profiles.get("gpu_entry", {}).get("profile", "low"))


def _default_profile_for_auto(hardware: HardwareInfo, settings: dict) -> str:
    min_vram = settings.get("auto_profiles", {}).get("gpu_entry", {}).get("min_vram_gb", 2)
    if hardware.cuda_available and hardware.vram_gb >= min_vram:
        return _default_gpu_profile_name(hardware, settings)
    return _default_cpu_profile_name(hardware, settings)


def _coerce_profile_name(mode: str, hardware: HardwareInfo, settings: dict) -> str:
    if mode == "auto":
        return _default_profile_for_auto(hardware, settings)
    if mode == "low":
        return mode if hardware.cuda_available else _default_cpu_profile_name(hardware, settings)
    if not hardware.cuda_available:
        return _default_cpu_profile_name(hardware, settings)
    if mode == "high":
        return "low" if hardware.vram_gb < 3.5 else "medium" if hardware.vram_gb < 4 else "high"
    if mode == "medium":
        return "low" if hardware.vram_gb < 3 else "medium"
    return mode


def _requested_profile_name(mode: str) -> str:
    return mode if mode in {"high", "medium", "low"} else "auto"


def _mode_profile(mode: str, hardware: HardwareInfo, settings: dict) -> tuple[str, str, int, str]:
    profile_name = _coerce_profile_name(mode, hardware, settings)
    requested_device, model_name, imgsz = _profile_tuple(profile_name, settings)
    if requested_device == "auto":
        requested_device = "gpu" if hardware.cuda_available else "cpu"
    return requested_device, model_name, imgsz, profile_name


def _resolved_device(requested_device: str, hardware: HardwareInfo) -> str:
    return "cuda:0" if requested_device == "gpu" and hardware.cuda_available else "cpu"


def _build_fallback_chain(profile_name: str, settings: dict) -> list[dict]:
    return [settings["models"][name] for name in FALLBACK_CHAINS.get(profile_name, FALLBACK_CHAINS["low"])]


def load_settings() -> dict:
    return load_yaml_cached(str(SETTINGS_PATH))


def select_runtime_config(mode: str, hardware: HardwareInfo) -> RuntimeConfig:
    settings = load_settings()
    requested_profile_name = _requested_profile_name(mode)
    requested_device, model_name, imgsz, profile_name = _mode_profile(mode, hardware, settings)
    requested_profile = settings["models"].get(requested_profile_name, settings["models"]["low"]) if requested_profile_name != "auto" else None
    camera = _camera_preset(settings)
    resolved_device = _resolved_device(requested_device, hardware)
    return RuntimeConfig(
        mode=mode,
        profile_name=profile_name,
        requested_profile_name=requested_profile_name,
        requested_device=MODE_DEVICE_HINTS.get(requested_profile_name, "auto"),
        resolved_device=resolved_device,
        requested_model_name=requested_profile["model"] if requested_profile else "auto",
        primary_model_name=model_name,
        candidate_models=build_candidates(model_name, settings),
        requested_imgsz=int(requested_profile["imgsz"]) if requested_profile else 0,
        imgsz=imgsz,
        conf=float(settings["inference"]["confidence"]),
        max_det=int(settings["inference"].get("max_det", 100)),
        use_half=bool(settings["inference"].get("use_half_for_cuda", False)) and resolved_device.startswith("cuda"),
        camera_width=int(camera["width"]),
        camera_height=int(camera["height"]),
        font_size=int(camera["font_size"]),
        box_thickness=int(camera["box_thickness"]),
        label_font_scale=float(camera["label_font_scale"]),
        fallback_chain=_build_fallback_chain(profile_name, settings),
    )
