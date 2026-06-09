from __future__ import annotations

from dataclasses import replace

from core.hardware_info import detect_hardware
from core.model_selector import RuntimeConfig, build_candidates, load_settings, select_runtime_config


GPU_PROFILE_SPECS = {
    "enthusiast": {
        "high": {"model": "yolo11x.pt", "device": "cuda:0", "imgsz": 960, "max_det": 200},
        "medium": {"model": "yolo11l.pt", "device": "cuda:0", "imgsz": 768, "max_det": 170},
        "low": {"model": "yolo11m.pt", "device": "cuda:0", "imgsz": 640, "max_det": 140},
    },
    "strong": {
        "high": {"model": "yolo11l.pt", "device": "cuda:0", "imgsz": 768, "max_det": 180},
        "medium": {"model": "yolo11m.pt", "device": "cuda:0", "imgsz": 640, "max_det": 150},
        "low": {"model": "yolo11s.pt", "device": "cuda:0", "imgsz": 512, "max_det": 120},
    },
    "entry": {
        "high": {"model": "yolo11s.pt", "device": "cuda:0", "imgsz": 640, "max_det": 150},
        "medium": {"model": "yolo11s.pt", "device": "cuda:0", "imgsz": 512, "max_det": 120},
        "low": {"model": "yolo11n.pt", "device": "cuda:0", "imgsz": 416, "max_det": 100},
    },
    "weak": {
        "high": {"model": "yolo11s.pt", "device": "cuda:0", "imgsz": 416, "max_det": 100},
        "medium": {"model": "yolo11n.pt", "device": "cuda:0", "imgsz": 416, "max_det": 90},
        "low": {"model": "yolo11n.pt", "device": "cuda:0", "imgsz": 320, "max_det": 70},
    },
}
CPU_PROFILE_SPECS = {
    "strong": {
        "high": {"model": "yolo11n.pt", "device": "cpu", "imgsz": 416, "max_det": 80},
        "medium": {"model": "yolo11n.pt", "device": "cpu", "imgsz": 352, "max_det": 60},
        "low": {"model": "yolo11n.pt", "device": "cpu", "imgsz": 320, "max_det": 50},
    },
    "weak": {
        "high": {"model": "yolo11n.pt", "device": "cpu", "imgsz": 320, "max_det": 60},
        "medium": {"model": "yolo11n.pt", "device": "cpu", "imgsz": 320, "max_det": 50},
        "low": {"model": "yolo11n.pt", "device": "cpu", "imgsz": 256, "max_det": 40},
    },
}
MODEL_QUALITY_SCORES = {
    "yolo11x.pt": 100,
    "yolo11l.pt": 92,
    "yolo11m.pt": 84,
    "yolo11s.pt": 74,
    "yolo11n.pt": 58,
}

MODE_ORDER = ("high", "medium", "low")
YOLO11_VARIANTS = ("yolo11x.pt", "yolo11l.pt", "yolo11m.pt", "yolo11s.pt", "yolo11n.pt")
MODE_META = {
    "high": {"label": "mạnh nhất", "title": "MẠNH NHẤT", "meaning": "mức cao nhất có thể ưu tiên"},
    "medium": {"label": "trung bình", "title": "TRUNG BÌNH", "meaning": "mức cân bằng dễ dùng nhất"},
    "low": {"label": "yếu nhất", "title": "YẾU NHẤT", "meaning": "mức nhẹ nhất / dễ chạy nhất"},
}


def mode_label(mode: str) -> str:
    return MODE_META.get(mode, MODE_META["low"])["label"]


def mode_title(mode: str) -> str:
    return MODE_META.get(mode, MODE_META["low"])["title"]


def load_level(hardware) -> str:
    cpu = float(getattr(hardware, "cpu_usage_percent", 0.0) or 0.0)
    ram = float(getattr(hardware, "ram_usage_percent", 0.0) or 0.0)
    gpu = float(getattr(hardware, "gpu_usage_percent", 0.0) or 0.0)
    vram = float(getattr(hardware, "vram_usage_percent", 0.0) or 0.0)
    peak = max(cpu, ram, gpu, vram)
    if peak >= 85:
        return "very_busy"
    if peak >= 70:
        return "busy"
    if peak >= 50:
        return "warm"
    return "cool"


def gpu_tier(hardware) -> str:
    if not getattr(hardware, "cuda_available", False):
        return "cpu_only"

    name = str(getattr(hardware, "gpu_name", "")).lower()
    vram = float(getattr(hardware, "vram_gb", 0.0) or 0.0)
    if "4090" in name or "4080" in name or vram >= 16:
        return "enthusiast"
    if "3090" in name or "3080" in name or "3070" in name or vram >= 8:
        return "strong"
    if "3060" in name or "3050" in name or "1660" in name or vram >= 4:
        return "entry"
    if vram >= 2:
        return "weak"
    return "cpu_only"


def profile_specs_for_hardware(hardware) -> dict[str, dict]:
    tier = gpu_tier(hardware)
    ram_gb = float(getattr(hardware, "ram_gb", 0.0) or 0.0)
    if tier in GPU_PROFILE_SPECS:
        return GPU_PROFILE_SPECS[tier]
    return CPU_PROFILE_SPECS["strong" if ram_gb >= 16 else "weak"]


def default_mode_for_hardware(hardware) -> str:
    tier = gpu_tier(hardware)
    load = load_level(hardware)
    if load == "very_busy":
        return "low"
    if load in {"busy", "warm"}:
        return "medium" if tier in {"enthusiast", "strong", "entry"} else "low"
    if tier in {"enthusiast", "strong"}:
        return "high"
    if tier == "entry":
        return "medium"
    return "low"


def ceiling_mode_for_hardware(hardware) -> str:
    tier = gpu_tier(hardware)
    if tier in {"enthusiast", "strong", "entry"}:
        return "high"
    if tier == "weak":
        return "medium"
    return "low"


def quality_score(runtime: RuntimeConfig) -> int:
    model_score = MODEL_QUALITY_SCORES.get(runtime.primary_model_name, 50)
    imgsz_bonus = min(18, max(0, (int(runtime.imgsz) - 320) // 32))
    det_bonus = min(10, max(0, (int(runtime.max_det) - 60) // 15))
    return min(100, model_score + imgsz_bonus + det_bonus)


def stability_score(mode: str, hardware) -> int:
    base = {"high": 68, "medium": 88, "low": 96}[mode]
    penalty = {"cool": 0, "warm": 6, "busy": 14, "very_busy": 24}[load_level(hardware)]
    return max(40, min(99, base - penalty))


def optimized_runtime(mode: str, hardware) -> RuntimeConfig:
    settings = load_settings()
    base = select_runtime_config(mode=mode, hardware=hardware)
    spec = profile_specs_for_hardware(hardware)[mode]
    resolved_device = spec["device"]
    requested_device = "gpu" if resolved_device.startswith("cuda") else "cpu"
    return replace(
        base,
        mode=mode,
        profile_name=mode,
        requested_profile_name=mode,
        requested_device=requested_device,
        resolved_device=resolved_device,
        requested_model_name=spec["model"],
        primary_model_name=spec["model"],
        candidate_models=build_candidates(spec["model"], settings),
        requested_imgsz=int(spec["imgsz"]),
        imgsz=int(spec["imgsz"]),
        max_det=int(spec["max_det"]),
        use_half=bool(settings["inference"].get("use_half_for_cuda", False)) and resolved_device.startswith("cuda"),
    )


def select_runtime_config_optimized(mode: str, hardware):
    if mode == "auto":
        return select_runtime_config_optimized(default_mode_for_hardware(hardware), hardware)
    return optimized_runtime(mode, hardware)


def build_recommendations(hardware=None) -> dict[str, RuntimeConfig]:
    hardware = hardware or detect_hardware()
    recommendations = {mode: select_runtime_config_optimized(mode, hardware) for mode in MODE_ORDER}
    recommendations["auto"] = select_runtime_config_optimized("auto", hardware)
    return recommendations
