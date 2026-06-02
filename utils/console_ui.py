from __future__ import annotations

import os
import sys
import time
from typing import Any


MODE_CHOICES = {"0": "exit", "1": "auto", "2": "high", "3": "medium", "4": "low"}
MODE_LABELS = {"auto": "Mac dinh tu dong", "high": "Cao nhat", "medium": "Trung binh", "low": "Yeu"}
PROFILE_LABELS = {
    "high": "GPU cuc manh",
    "medium": "GPU can bang",
    "low": "GPU/CPU uu tien muot",
    "fallback_cpu": "CPU an toan",
    "fallback_cpu_weak": "CPU toi thieu",
}
PROMPT_OPTIONS = (
    ("1 | Tu dong", "Can bang an toan | he thong tu quyet dinh.", "\033[92m"),
    ("2 | Cao nhat", "Uu tien chat luong | yolo26s.pt | imgsz 768.", "\033[95m"),
    ("3 | Trung binh", "Uu tien can bang | yolo26s.pt | imgsz 640.", "\033[96m"),
    ("4 | Yeu", "Uu tien muot | yolo26n.pt | imgsz 512.", "\033[93m"),
)
PERFORMANCE_HINTS = (
    (85, "Che do rat manh, he thong dang uu tien chat luong va hieu nang toi da."),
    (65, "Che do can bang manh, hop cho da so may tam trung va manh."),
    (45, "Che do on dinh, uu tien van hanh muot va it loi."),
)

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"
ORANGE = "\033[38;5;208m"
CARD_WIDTH = 96
BOOT_BAR_WIDTH = 24


def _clear_terminal() -> None:
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def _terminal_columns(default: int = 120) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def _line(text: str = "", color: str = "") -> str:
    return f"{color}{text}{RESET}" if color else text


def _pad(text: str, width: int = CARD_WIDTH) -> str:
    return text[:width].ljust(width)


def _rule(char: str = "=") -> str:
    return char * CARD_WIDTH


def _section(title: str, color: str = CYAN) -> str:
    return _line(_pad(f"[ {title} ]"), BOLD + color)


def _row(label: str, value: str = "", color: str = "", *, bounded: bool = True) -> str:
    content = f"{label:<18} {value}".rstrip()
    return _line(_pad(content) if bounded else content, color)


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def profile_label(profile_name: str) -> str:
    return PROFILE_LABELS.get(profile_name, profile_name)


def power_score(runtime: Any, hardware: Any) -> int:
    score = 20
    if str(getattr(runtime, "resolved_device", "")).startswith("cuda"):
        score += 35
    score += min(20, int(float(getattr(hardware, "vram_gb", 0.0)) * 2))
    score += min(10, max(0, (int(getattr(runtime, "imgsz", 0)) - 320) // 64))
    score += 5 if getattr(runtime, "use_half", False) else 0
    score += {"high": 10, "medium": 5}.get(getattr(runtime, "profile_name", ""), 0)
    return max(0, min(100, score))


def progress_bar_colored(score: int, width: int | None = None) -> str:
    normalized = max(0, min(100, score))
    if width is None:
        width = min(BOOT_BAR_WIDTH, max(12, (_terminal_columns() - len("0[]100")) // 2))
    filled = round((normalized / 100) * width)
    yellow_zone = round(width * 0.34)
    orange_zone = round(width * 0.33)
    parts: list[str] = []
    for index in range(width):
        if index >= filled:
            parts.append(f"{DIM}·{RESET}")
            continue
        color = YELLOW if index < yellow_zone else ORANGE if index < yellow_zone + orange_zone else RED
        parts.append(f"{color}█{RESET}")
    return f"0[{' '.join(parts)}]100"


def performance_hint(score: int) -> str:
    for minimum, message in PERFORMANCE_HINTS:
        if score >= minimum:
            return message
    return "Che do an toan, phu hop may yeu hoac dang chay CPU du phong."


def _status_color(ok: bool | None) -> str:
    return GREEN if ok is True else RED if ok is False else YELLOW


def _render_prompt(print_fn=print) -> None:
    _clear_terminal()
    lines = [
        _line(_rule("="), CYAN),
        _line(_pad("YOLO REALTIME CAMERA :: CHON CAU HINH CHAY"), BOLD + CYAN),
        _line(_rule("="), CYAN),
        _row("Goi y", "Mode 3 = Trung binh | can bang muot / chinh xac", YELLOW),
        _row("Phong cach", "Lua chon cau hinh theo muc uu tien va suc manh phan cung", DIM),
        _line(_rule("-"), CYAN),
        _section("CAC LUA CHON", MAGENTA),
    ]
    for label, value, color in PROMPT_OPTIONS:
        lines.append(_row(label, value, color))
    lines.extend(
        [
            _line(_rule("."), DIM),
            _row("0 | Thoat", "Thoat khoi chuong trinh ngay tai day.", RED),
            _line(_rule("-"), CYAN),
        ]
    )
    for line in lines:
        print_fn(line)


def prompt_runtime_mode(input_fn=input, print_fn=print) -> str:
    while True:
        _render_prompt(print_fn=print_fn)
        mode = MODE_CHOICES.get(input_fn(_line("Nhap lua chon cua ban (0/1/2/3/4): ", BOLD)).strip())
        if mode == "exit":
            raise SystemExit(0)
        if mode:
            print_fn("")
            print_fn(_line(f"Da chon: {mode_label(mode)}", GREEN))
            return mode
        print_fn(_line("Lua chon khong hop le. Vui long nhap 0, 1, 2, 3 hoac 4.", RED))
        input_fn(_line("Nhan Enter de chon lai...", DIM))


def mode_to_ui_defaults(mode: str) -> tuple[str, str]:
    return ("auto", "medium") if mode == "auto" else ("manual", mode)


def _dashboard_values(runtime: Any, hardware: Any, camera_index: int) -> dict[str, Any]:
    requested_profile = getattr(runtime, "requested_profile_name", getattr(runtime, "mode", "auto"))
    gpu_count = int(getattr(hardware, "gpu_count", 0) or 0)
    gpu_hardware_available = bool(getattr(hardware, "gpu_hardware_available", gpu_count > 0))
    cuda_available = bool(getattr(hardware, "cuda_available", False))
    resolved_device = getattr(runtime, "resolved_device", "-")
    primary_model_name = getattr(runtime, "primary_model_name", "-")
    requested_model_name = getattr(runtime, "requested_model_name", "-")
    requested_device = getattr(runtime, "requested_device", "-")
    requested_imgsz = getattr(runtime, "requested_imgsz", "-")
    cuda_target = "Tu dong theo phan cung" if requested_profile == "auto" else f"{requested_model_name} / {requested_device} / imgsz {requested_imgsz}"
    if resolved_device.startswith("cuda"):
        cuda_model_status = primary_model_name
    elif gpu_hardware_available:
        cuda_model_status = "GPU co san nhung moi truong PyTorch hien tai chua dung duoc CUDA"
    else:
        cuda_model_status = "Khong co GPU/CUDA de chay model bang CUDA"
    return {
        "chosen_label": mode_label(getattr(runtime, "mode", "auto")),
        "runtime_profile": getattr(runtime, "profile_name", getattr(runtime, "mode", "auto")),
        "requested_profile": requested_profile,
        "score": power_score(runtime, hardware),
        "actual_runtime": f"{primary_model_name} / {resolved_device} / imgsz {getattr(runtime, 'imgsz', '-')}",
        "cuda_target": cuda_target,
        "cuda_model_status": cuda_model_status,
        "cuda_available": cuda_available,
        "gpu_hardware_available": gpu_hardware_available,
        "torch_color": GREEN if getattr(hardware, "torch_cuda_version", "CPU-only") != "CPU-only" else YELLOW if gpu_hardware_available else RED,
        "reason_color": GREEN if cuda_available else YELLOW if gpu_hardware_available else RED,
        "cuda_color": _status_color(True if cuda_available else None if gpu_hardware_available else False),
        "model_color": _status_color(True if resolved_device.startswith("cuda") else None if gpu_hardware_available else False),
        "profile_color": _status_color(None if requested_profile == "auto" else (getattr(runtime, "profile_name", "") == requested_profile)),
        "camera_index": camera_index,
    }


def print_runtime_dashboard(title: str, runtime: Any, hardware: Any, camera_index: int, print_fn=print) -> None:
    values = _dashboard_values(runtime, hardware, camera_index)
    lines = [
        _line(_rule("="), CYAN),
        _line(_pad(title), BOLD + CYAN),
        _line(_rule("="), CYAN),
        _row("Lua chon", values["chosen_label"], MAGENTA),
        _row("Muc tieu", f"{values['requested_profile']} -> {values['cuda_target']}", MAGENTA),
        _row("Thuc te", f"{profile_label(values['runtime_profile'])} ({values['runtime_profile']})", values["profile_color"]),
        _row("Thanh khoi dong", progress_bar_colored(100, width=BOOT_BAR_WIDTH), YELLOW, bounded=False),
        _line(_rule("-"), CYAN),
        _section("PHAN CUNG", GREEN),
        _row("CPU", getattr(hardware, "cpu_name", "Khong ro CPU"), GREEN),
        _row("RAM / OS", f"{float(getattr(hardware, 'ram_gb', 0.0) or 0.0):.1f} GB / {getattr(hardware, 'os_name', 'Khong ro OS')}", GREEN),
        _row("GPU", getattr(hardware, "gpu_name", "Khong ro"), GREEN),
        _row("VRAM / GPU count", f"{float(getattr(hardware, 'vram_gb', 0.0) or 0.0):.1f} GB / {int(getattr(hardware, 'gpu_count', 0) or 0)}", GREEN),
        _row("Torch / build", f"{getattr(hardware, 'torch_version', 'Khong co PyTorch')} / {getattr(hardware, 'torch_cuda_version', 'CPU-only')}", values["torch_color"]),
        _row("CUDA runtime", getattr(hardware, "cuda_runtime_status", "Khong"), values["cuda_color"]),
        _row("Ly do CUDA", getattr(hardware, "cuda_runtime_reason", "Chua kiem tra"), values["reason_color"], bounded=False),
        _line(_rule("-"), CYAN),
        _section("RUNTIME", CYAN),
        _row("Model dang chay", values["actual_runtime"], values["model_color"]),
        _row("Model CUDA", values["cuda_model_status"], values["model_color"], bounded=False),
        _row("imgsz / max_det", f"{getattr(runtime, 'imgsz', '-')} / {getattr(runtime, 'max_det', '-')}", CYAN),
        _row("Camera / Index", f"{getattr(runtime, 'camera_width', '-')}x{getattr(runtime, 'camera_height', '-')} / {camera_index}", CYAN),
        _row("Half precision", "Bat" if getattr(runtime, "use_half", False) else "Tat", CYAN),
        _line(_rule("-"), CYAN),
        _section("SAN SANG", YELLOW),
        _row("Trang thai", performance_hint(values["score"]), YELLOW, bounded=False),
        _line(_rule("-"), CYAN),
    ]
    for line in lines:
        print_fn(line)


class BootProgress:
    def __init__(self, title: str, enabled: bool | None = None) -> None:
        self.enabled = sys.stdout.isatty() and os.getenv("YOLO_DISABLE_PROGRESS") != "1" if enabled is None else enabled
        self.title = title
        self.current = 0
        self.started = False

    def start(self) -> None:
        if not self.enabled or self.started:
            return
        self.started = True
        self._render_line()

    def _clear_line(self) -> None:
        print(f"\r{' ' * _terminal_columns()}\r", end="", flush=True)

    def _render_line(self) -> None:
        self._clear_line()
        print(_line(progress_bar_colored(self.current), YELLOW), end="", flush=True)

    def advance_to(self, target: int, label: str) -> None:
        if not self.enabled:
            self.current = max(self.current, target)
            return
        self.start()
        target = max(self.current, min(100, target))
        while self.current < target:
            self.current += 1
            self._render_line()
            time.sleep(0.01 if self.current < 90 else 0.016)

    def finish(self, label: str = "San sang mo camera") -> None:
        self.advance_to(100, label)
        if self.enabled:
            self._clear_line()
