from __future__ import annotations

import os
import sys
import time
from typing import Any


MODE_CHOICES = {
    "0": "exit",
    "1": "auto",
    "2": "high",
    "3": "medium",
    "4": "low",
}

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"
ORANGE = "\033[38;5;208m"

MODE_LABELS = {
    "auto": "Mac dinh tu dong",
    "high": "Cao nhat",
    "medium": "Trung binh",
    "low": "Yeu",
}

CARD_WIDTH = 96
BOOT_BAR_WIDTH = 24


def _clear_terminal() -> None:
    if not sys.stdout.isatty():
        return
    os.system("cls" if os.name == "nt" else "clear")


def _terminal_columns(default: int = 120) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def _line(text: str = "", color: str = "") -> str:
    return f"{color}{text}{RESET}" if color else text


def _pad(text: str, width: int = CARD_WIDTH) -> str:
    trimmed = text[:width]
    return trimmed + (" " * max(0, width - len(trimmed)))


def _rule(char: str = "=") -> str:
    return char * CARD_WIDTH


def _section(title: str, color: str = CYAN) -> str:
    return _line(_pad(f"[ {title} ]"), BOLD + color)


def _row(label: str, value: str = "", color: str = "") -> str:
    left = f"{label:<18}"
    content = f"{left} {value}".rstrip()
    return _line(_pad(content), color)


def _row_unbounded(label: str, value: str = "", color: str = "") -> str:
    left = f"{label:<18}"
    content = f"{left} {value}".rstrip()
    return _line(content, color)


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def profile_label(profile_name: str) -> str:
    mapping = {
        "high": "GPU cuc manh",
        "medium": "GPU can bang",
        "low": "GPU/CPU uu tien muot",
        "fallback_cpu": "CPU an toan",
        "fallback_cpu_weak": "CPU toi thieu",
    }
    return mapping.get(profile_name, profile_name)


def power_score(runtime: Any, hardware: Any) -> int:
    score = 20
    if str(getattr(runtime, "resolved_device", "")).startswith("cuda"):
        score += 35
    score += min(20, int(float(getattr(hardware, "vram_gb", 0.0)) * 2))
    score += min(10, max(0, (int(getattr(runtime, "imgsz", 0)) - 320) // 64))
    if getattr(runtime, "use_half", False):
        score += 5
    if getattr(runtime, "profile_name", "") == "high":
        score += 10
    elif getattr(runtime, "profile_name", "") == "medium":
        score += 5
    return max(0, min(100, score))


def progress_bar_colored(score: int, width: int | None = None) -> str:
    normalized = max(0, min(100, score))
    if width is None:
        available = max(12, (_terminal_columns() - len("0[]100")) // 2)
        width = min(BOOT_BAR_WIDTH, available)
    filled = round((normalized / 100) * width)
    yellow_zone = round(width * 0.34)
    orange_zone = round(width * 0.33)
    parts: list[str] = []
    for index in range(width):
        if index < filled:
            if index < yellow_zone:
                color = YELLOW
            elif index < yellow_zone + orange_zone:
                color = ORANGE
            else:
                color = RED
            parts.append(f"{color}█{RESET}")
        else:
            parts.append(f"{DIM}·{RESET}")
    return f"0[{' '.join(parts)}]100"


def performance_hint(score: int) -> str:
    if score >= 85:
        return "Che do rat manh, he thong dang uu tien chat luong va hieu nang toi da."
    if score >= 65:
        return "Che do can bang manh, hop cho da so may tam trung va manh."
    if score >= 45:
        return "Che do on dinh, uu tien van hanh muot va it loi."
    return "Che do an toan, phu hop may yeu hoac dang chay CPU du phong."


def _status_color(ok: bool | None) -> str:
    if ok is True:
        return GREEN
    if ok is False:
        return RED
    return YELLOW


def _render_prompt(print_fn=print) -> None:
    _clear_terminal()
    print_fn(_line(_rule("="), CYAN))
    print_fn(_line(_pad("YOLO REALTIME CAMERA :: CHON CAU HINH CHAY"), BOLD + CYAN))
    print_fn(_line(_rule("="), CYAN))
    print_fn(_row("Goi y", "Mode 3 = Trung binh | can bang muot / chinh xac", YELLOW))
    print_fn(_row("Phong cach", "Lua chon cau hinh theo muc uu tien va suc manh phan cung", DIM))
    print_fn(_line(_rule("-"), CYAN))
    print_fn(_section("CAC LUA CHON", MAGENTA))
    print_fn(_row("1 | Tu dong", "Can bang an toan | he thong tu quyet dinh.", GREEN))
    print_fn(_row("2 | Cao nhat", "Uu tien chat luong | yolo26s.pt | imgsz 768.", MAGENTA))
    print_fn(_row("3 | Trung binh", "Uu tien can bang | yolo26s.pt | imgsz 640.", CYAN))
    print_fn(_row("4 | Yeu", "Uu tien muot | yolo26n.pt | imgsz 512.", YELLOW))
    print_fn(_line(_rule("."), DIM))
    print_fn(_row("0 | Thoat", "Thoat khoi chuong trinh ngay tai day.", RED))
    print_fn(_line(_rule("-"), CYAN))


def prompt_runtime_mode(input_fn=input, print_fn=print) -> str:
    while True:
        _render_prompt(print_fn=print_fn)
        choice = input_fn(_line("Nhap lua chon cua ban (0/1/2/3/4): ", BOLD)).strip()
        mode = MODE_CHOICES.get(choice)
        if mode:
            if mode == "exit":
                raise SystemExit(0)
            print_fn("")
            print_fn(_line(f"Da chon: {mode_label(mode)}", GREEN))
            return mode
        print_fn(_line("Lua chon khong hop le. Vui long nhap 0, 1, 2, 3 hoac 4.", RED))
        input_fn(_line("Nhan Enter de chon lai...", DIM))


def mode_to_ui_defaults(mode: str) -> tuple[str, str]:
    if mode == "auto":
        return "auto", "medium"
    return "manual", mode


def print_runtime_dashboard(title: str, runtime: Any, hardware: Any, camera_index: int, print_fn=print) -> None:
    chosen_label = mode_label(getattr(runtime, "mode", "auto"))
    runtime_profile = getattr(runtime, "profile_name", getattr(runtime, "mode", "auto"))
    requested_profile = getattr(runtime, "requested_profile_name", getattr(runtime, "mode", "auto"))
    score = power_score(runtime, hardware)
    requested_model_name = getattr(runtime, "requested_model_name", "-")
    primary_model_name = getattr(runtime, "primary_model_name", "-")
    requested_device = getattr(runtime, "requested_device", "-")
    resolved_device = getattr(runtime, "resolved_device", "-")
    requested_imgsz = getattr(runtime, "requested_imgsz", "-")
    imgsz = getattr(runtime, "imgsz", "-")
    max_det = getattr(runtime, "max_det", "-")
    camera_width = getattr(runtime, "camera_width", "-")
    camera_height = getattr(runtime, "camera_height", "-")
    cpu_name = getattr(hardware, "cpu_name", "Khong ro CPU")
    gpu_name = getattr(hardware, "gpu_name", "Khong ro")
    vram_gb = float(getattr(hardware, "vram_gb", 0.0) or 0.0)
    ram_gb = float(getattr(hardware, "ram_gb", 0.0) or 0.0)
    cuda_available = bool(getattr(hardware, "cuda_available", False))
    os_name = getattr(hardware, "os_name", "Khong ro OS")
    gpu_count = int(getattr(hardware, "gpu_count", 0) or 0)
    torch_version = getattr(hardware, "torch_version", "Khong co PyTorch")
    torch_cuda_version = getattr(hardware, "torch_cuda_version", "CPU-only")
    cuda_runtime_status = getattr(hardware, "cuda_runtime_status", "Khong")
    cuda_runtime_reason = getattr(hardware, "cuda_runtime_reason", "Chua kiem tra")
    gpu_hardware_available = bool(getattr(hardware, "gpu_hardware_available", gpu_count > 0))
    cuda_target = f"{requested_model_name} / {requested_device} / imgsz {requested_imgsz}"
    if requested_profile == "auto":
        cuda_target = "Tu dong theo phan cung"
    actual_runtime = f"{primary_model_name} / {resolved_device} / imgsz {imgsz}"
    cuda_model_status = primary_model_name if resolved_device.startswith("cuda") else "Khong co model nao dang chay bang CUDA"
    if not cuda_available and gpu_hardware_available:
        cuda_model_status = "GPU co san nhung moi truong PyTorch hien tai chua dung duoc CUDA"
    elif not gpu_hardware_available:
        cuda_model_status = "Khong co GPU/CUDA de chay model bang CUDA"
    target_met = runtime_profile == requested_profile or requested_profile == "auto"
    if requested_profile == "auto":
        target_met = None
    profile_color = _status_color(True if target_met else None if gpu_hardware_available else False)
    cuda_color = _status_color(True if cuda_available else None if gpu_hardware_available else False)
    model_color = _status_color(True if resolved_device.startswith("cuda") else None if gpu_hardware_available else False)
    reason_color = GREEN if cuda_available else YELLOW if gpu_hardware_available else RED
    torch_color = GREEN if torch_cuda_version != "CPU-only" else YELLOW if gpu_hardware_available else RED

    print_fn(_line(_rule("="), CYAN))
    print_fn(_line(_pad(title), BOLD + CYAN))
    print_fn(_line(_rule("="), CYAN))
    print_fn(_row("Lua chon", chosen_label, MAGENTA))
    print_fn(_row("Muc tieu", f"{requested_profile} -> {cuda_target}", MAGENTA))
    print_fn(_row("Thuc te", f"{profile_label(runtime_profile)} ({runtime_profile})", profile_color))
    print_fn(_row_unbounded("Thanh khoi dong", progress_bar_colored(100, width=BOOT_BAR_WIDTH), YELLOW))
    print_fn(_line(_rule("-"), CYAN))
    print_fn(_section("PHAN CUNG", GREEN))
    print_fn(_row("CPU", cpu_name, GREEN))
    print_fn(_row("RAM / OS", f"{ram_gb:.1f} GB / {os_name}", GREEN))
    print_fn(_row("GPU", gpu_name, GREEN))
    print_fn(_row("VRAM / GPU count", f"{vram_gb:.1f} GB / {gpu_count}", GREEN))
    print_fn(_row("Torch / build", f"{torch_version} / {torch_cuda_version}", torch_color))
    print_fn(_row("CUDA runtime", cuda_runtime_status, cuda_color))
    print_fn(_row_unbounded("Ly do CUDA", cuda_runtime_reason, reason_color))
    print_fn(_line(_rule("-"), CYAN))
    print_fn(_section("RUNTIME", CYAN))
    print_fn(_row("Model dang chay", actual_runtime, model_color))
    print_fn(_row_unbounded("Model CUDA", cuda_model_status, model_color))
    print_fn(_row("imgsz / max_det", f"{imgsz} / {max_det}", CYAN))
    print_fn(_row("Camera / Index", f"{camera_width}x{camera_height} / {camera_index}", CYAN))
    print_fn(_row("Half precision", "Bat" if getattr(runtime, "use_half", False) else "Tat", CYAN))
    print_fn(_line(_rule("-"), CYAN))
    print_fn(_section("SAN SANG", YELLOW))
    print_fn(_row_unbounded("Trang thai", performance_hint(score), YELLOW))
    print_fn(_line(_rule("-"), CYAN))


class BootProgress:
    def __init__(self, title: str, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = sys.stdout.isatty() and os.getenv("YOLO_DISABLE_PROGRESS") != "1"
        self.enabled = enabled
        self.title = title
        self.current = 0
        self.started = False

    def start(self) -> None:
        if not self.enabled or self.started:
            return
        self.started = True
        self._render_line()

    def _clear_line(self) -> None:
        clear = " " * _terminal_columns()
        print(f"\r{clear}\r", end="", flush=True)

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
