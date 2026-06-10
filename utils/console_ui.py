from __future__ import annotations

import os
import sys
import time
import unicodedata
from typing import Any


MODE_CHOICES = {"0": "exit", "1": "high", "2": "medium", "3": "low"}
START_TARGET_CHOICES = {"0": "exit", "1": "ui", "2": "camera"}
MODE_LABELS = {
    "auto": "Tự động theo máy",
    "high": "Mạnh nhất",
    "medium": "Trung bình",
    "low": "Yếu nhất",
}
START_TARGET_LABELS = {
    "ui": "UX/UI desktop (run_chat.py)",
    "camera": "Camera realtime",
}
PROFILE_LABELS = {
    "high": "Mạnh nhất",
    "medium": "Trung bình",
    "low": "Yếu nhất",
    "fallback_cpu": "CPU an toàn",
    "fallback_cpu_weak": "CPU tối thiểu",
}
PROMPT_OPTIONS = (
    ("1 | MẠNH NHẤT", "Mức cao nhất máy còn gánh được.", "\033[92m"),
    ("2 | TRUNG BÌNH", "Mức cân bằng để dùng thường xuyên.", "\033[93m"),
    ("3 | YẾU NHẤT", "Mức nhẹ nhất để ưu tiên độ mượt.", "\033[93m"),
)
PERFORMANCE_HINTS = (
    (85, "Runtime rất khỏe, ưu tiên chất lượng và tốc độ tối đa."),
    (65, "Runtime cân bằng tốt, hợp cho đa số tình huống sử dụng."),
    (40, "Runtime tạm ổn, ưu tiên ổn định hơn hiệu năng."),
)

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
ORANGE = "\033[38;5;208m"
BLUE = "\033[38;5;81m"
DIM = "\033[2m"
CARD_WIDTH = 96
BOOT_BAR_WIDTH = 24


def _ensure_utf8_console() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        return


_ensure_utf8_console()


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
    glyph = {"=": "═", "-": "─", ".": "·"}.get(char, char)
    return glyph * CARD_WIDTH


def _section(title: str, color: str = CYAN) -> str:
    return _line(_pad(f"◆ {title}"), BOLD + color)


def _row(label: str, value: str = "", color: str = "", *, bounded: bool = True) -> str:
    content = f"│ {label:<16} {value}".rstrip()
    return _line(_pad(content) if bounded else content, color)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return without_marks.replace("đ", "d").replace("Đ", "D").replace("Ä‘", "d").replace("Ä", "D").lower()


def line(text: str = "", color: str = "") -> str:
    return _line(text, color)


def pad(text: str, width: int = CARD_WIDTH) -> str:
    return _pad(text, width)


def rule(char: str = "=") -> str:
    return _rule(char)


def section(title: str, color: str = CYAN) -> str:
    return _section(title, color)


def row(label: str, value: str = "", color: str = "", *, bounded: bool = True) -> str:
    return _row(label, value, color, bounded=bounded)


def header(title: str, color: str = CYAN) -> list[str]:
    border = _rule("=")[:-2]
    return [
        _line(f"╔{border}╗", color),
        _line(f"║ {_pad(title, CARD_WIDTH - 4)} ║", BOLD + color),
        _line(f"╚{border}╝", color),
    ]


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def profile_label(profile_name: str) -> str:
    return PROFILE_LABELS.get(profile_name, profile_name)


def power_score(runtime: Any, hardware: Any) -> int:
    score = 20
    if str(getattr(runtime, "resolved_device", "")).startswith("cuda"):
        score += 35
    score += min(20, int(float(getattr(hardware, "vram_gb", 0.0) or 0.0) * 2))
    score += min(10, max(0, (int(getattr(runtime, "imgsz", 0)) - 320) // 64))
    score += 5 if getattr(runtime, "use_half", False) else 0
    score += {"high": 10, "medium": 5}.get(getattr(runtime, "profile_name", ""), 0)
    return max(0, min(100, score))


def _score_color(score: int) -> str:
    return GREEN if score >= 70 else YELLOW if score >= 40 else RED


def _usage_color(percent: float | None) -> str:
    if percent is None:
        return YELLOW
    if percent < 60:
        return GREEN
    if percent < 85:
        return YELLOW
    return RED


def progress_bar_colored(score: int, width: int | None = None) -> str:
    normalized = max(0, min(100, score))
    if width is None:
        width = min(BOOT_BAR_WIDTH, max(12, (_terminal_columns() - len("0[]100")) // 2))
    filled = round((normalized / 100) * width)
    parts: list[str] = []
    for index in range(width):
        if index >= filled:
            parts.append(" ")
        else:
            color = YELLOW if index < width // 3 else ORANGE if index < (width * 2) // 3 else RED
            parts.append(f"{color}\u2588{RESET}")
    return f"0[{' '.join(parts)}]100"


def _usage_row(label: str, percent: float | None) -> str:
    if percent is None:
        return _row(label, "Không rõ", YELLOW, bounded=False)
    return _row(
        label,
        f"{percent:5.1f}% {progress_bar_colored(round(percent), width=12)}",
        _usage_color(percent),
        bounded=False,
    )


def dashboard_boot_bar(_score: int, width: int = BOOT_BAR_WIDTH) -> str:
    return progress_bar_colored(100, width)


def performance_hint(score: int) -> str:
    for minimum, message in PERFORMANCE_HINTS:
        if score >= minimum:
            return message
    return "Runtime đang bị giới hạn. Nên kiểm tra CUDA, model hoặc giảm mức chạy."


def _runtime_line(mode: str, runtime: Any) -> str:
    return (
        f"{mode_label(mode)} -> "
        f"{getattr(runtime, 'primary_model_name', '-')} | "
        f"{getattr(runtime, 'resolved_device', '-')} | "
        f"imgsz {getattr(runtime, 'imgsz', '-')} | "
        f"max_det {getattr(runtime, 'max_det', '-')}"
    )


def _render_prompt(hardware: Any | None = None, recommendations: dict[str, Any] | None = None, print_fn=print) -> None:
    _clear_terminal()
    suggested_runtime = recommendations.get("auto") if recommendations else None
    suggested_mode = getattr(suggested_runtime, "mode", "medium") if suggested_runtime else "medium"
    hardware_summary = (
        f"CPU: {getattr(hardware, 'cpu_name', 'Không rõ CPU')} | "
        f"RAM: {float(getattr(hardware, 'ram_gb', 0.0) or 0.0):.1f} GB | "
        f"GPU: {getattr(hardware, 'gpu_name', 'Không rõ GPU')} | "
        f"VRAM: {float(getattr(hardware, 'vram_gb', 0.0) or 0.0):.1f} GB"
        if hardware is not None
        else "Chưa có dữ liệu phần cứng."
    )
    lines = [
        _line(_rule("="), CYAN),
        _line(_pad("YOLO REALTIME CAMERA : CHỌN CẤU HÌNH CHẠY"), BOLD + CYAN),
        _line(_rule("="), CYAN),
        _row("Phần cứng", hardware_summary, GREEN, bounded=False),
        _row("Đề xuất", f"{mode_label(suggested_mode)} | hệ thống đã thăm dò máy trước khi chạy.", YELLOW, bounded=False),
        _row("Ý nghĩa", "Chọn 1 trong 3 mức mạnh nhất / trung bình / yếu nhất.", DIM, bounded=False),
        _line(_rule("-"), CYAN),
        _section("3 LỰA CHỌN", MAGENTA),
    ]
    for label, value, color in PROMPT_OPTIONS:
        lines.append(_row(label, value, color))
        if recommendations:
            option_mode = MODE_CHOICES[label.split("|", 1)[0].strip()]
            runtime = recommendations.get(option_mode)
            if runtime is not None:
                lines.append(_row("  Máy này", _runtime_line(option_mode, runtime), DIM, bounded=False))
    lines.extend(
        [
            _line(_rule("."), DIM),
            _row("0 | THOÁT", "Đóng chương trình ngay tại đây.", RED),
            _line(_rule("-"), CYAN),
        ]
    )
    for item in lines:
        print_fn(item)


def prompt_runtime_mode(hardware: Any | None = None, recommendations: dict[str, Any] | None = None, input_fn=input, print_fn=print) -> str:
    while True:
        _render_prompt(hardware=hardware, recommendations=recommendations, print_fn=print_fn)
        mode = MODE_CHOICES.get(input_fn(_line("Nhập lựa chọn của bạn (0/1/2/3): ", BOLD)).strip())
        if mode == "exit":
            raise SystemExit(0)
        if mode:
            print_fn("")
            print_fn(_line(f"Đã chọn: {mode_label(mode)}", GREEN))
            return mode
        print_fn(_line("Lựa chọn không hợp lệ. Vui lòng nhập 0, 1, 2 hoặc 3.", RED))
        input_fn(_line("Nhấn Enter để chọn lại...", DIM))


def mode_to_ui_defaults(mode: str) -> tuple[str, str]:
    return ("auto", "medium") if mode == "auto" else ("manual", mode)


def launch_target_label(target: str) -> str:
    return START_TARGET_LABELS.get(target, target)


def prompt_launch_target(
    *,
    selected_mode: str,
    selected_model: str,
    preferred_target: str = "ui",
    input_fn=input,
    print_fn=print,
) -> str:
    while True:
        _clear_terminal()
        lines = [
            _line(_rule("="), CYAN),
            _line(_pad("YOLO REALTIME CAMERA :: CHỌN KIỂU KHỞI ĐỘNG"), BOLD + CYAN),
            _line(_rule("="), CYAN),
_row("Cấu hình", f"{mode_label(selected_mode)} | {selected_model}", GREEN, bounded=False),
             _row("Đề xuất", launch_target_label(preferred_target), YELLOW, bounded=False),
            _line(_rule("-"), CYAN),
_section("2 LỰA CHỌN", MAGENTA),
             _row("1 | UI DESKTOP", "Mở giao diện desktop / chat.", GREEN),
             _row("2 | CAMERA", "Chạy detect realtime chỉ với camera.", YELLOW),
             _line(_rule("."), DIM),
             _row("0 | THOÁT", "Đóng chương trình ngay tại đây.", RED),
            _line(_rule("-"), CYAN),
        ]
        for item in lines:
            print_fn(item)
        target = START_TARGET_CHOICES.get(input_fn(_line("Nhập lựa chọn của bạn (0/1/2): ", BOLD)).strip())
        if target == "exit":
            raise SystemExit(0)
        if target:
            print_fn("")
            print_fn(_line(f"Đã chọn kiểu chạy: {launch_target_label(target)}", GREEN))
            return target
        print_fn(_line("Lựa chọn không hợp lệ. Vui lòng nhập 0, 1, 2 hoặc 3.", RED))
        input_fn(_line("Nhấn Enter để chọn lại...", DIM))


def _dashboard_values(runtime: Any, hardware: Any, camera_index: int, launch_target: str | None = None) -> dict[str, Any]:
    requested_profile = getattr(runtime, "requested_profile_name", getattr(runtime, "mode", "auto"))
    gpu_count = int(getattr(hardware, "gpu_count", 0) or 0)
    gpu_hardware_available = bool(getattr(hardware, "gpu_hardware_available", gpu_count > 0))
    cuda_available = bool(getattr(hardware, "cuda_available", False))
    resolved_device = getattr(runtime, "resolved_device", "-")
    primary_model_name = getattr(runtime, "primary_model_name", "-")
    requested_model_name = getattr(runtime, "requested_model_name", "-")
    requested_device = getattr(runtime, "requested_device", "-")
    requested_imgsz = getattr(runtime, "requested_imgsz", "-")
    score = power_score(runtime, hardware)
    requested_gpu_mode = requested_profile in {"high", "medium"} or requested_device == "gpu"
    runtime_on_gpu = resolved_device.startswith("cuda")
    profile_match = getattr(runtime, "profile_name", "") == requested_profile if requested_profile != "auto" else True
    cuda_target = "Tự động theo phần cứng" if requested_profile == "auto" else f"{requested_model_name} / {requested_device} / imgsz {requested_imgsz}"

    if runtime_on_gpu:
        cuda_model_status = primary_model_name
    elif gpu_hardware_available:
        cuda_model_status = "GPU có sẵn nhưng PyTorch hiện tại chưa dùng được CUDA"
    else:
        cuda_model_status = "Không có GPU/CUDA để chạy model bằng CUDA"

    cuda_color = GREEN if cuda_available else RED if requested_gpu_mode or gpu_hardware_available else YELLOW
    reason_color = cuda_color
    model_color = GREEN if runtime_on_gpu else RED if requested_gpu_mode else YELLOW
    if requested_profile == "auto":
        profile_color = GREEN if runtime_on_gpu else YELLOW
    elif profile_match:
        profile_color = GREEN
    elif runtime_on_gpu:
        profile_color = YELLOW
    else:
        profile_color = RED

    return {
        "chosen_label": mode_label(getattr(runtime, "mode", "auto")),
        "runtime_profile": getattr(runtime, "profile_name", getattr(runtime, "mode", "auto")),
        "requested_profile": requested_profile,
        "score": score,
        "actual_runtime": f"{primary_model_name} / {resolved_device} / imgsz {getattr(runtime, 'imgsz', '-')} / max_det {getattr(runtime, 'max_det', '-')}",
        "cuda_target": cuda_target,
        "cuda_model_status": cuda_model_status,
        "torch_color": GREEN if getattr(hardware, "torch_cuda_version", "CPU-only") != "CPU-only" else YELLOW if gpu_hardware_available else RED,
        "cuda_color": cuda_color,
        "reason_color": reason_color,
        "model_color": model_color,
        "profile_color": profile_color,
        "hardware_section_color": GREEN if cuda_available else RED if gpu_hardware_available else YELLOW,
        "runtime_section_color": model_color,
        "ready_section_color": _score_color(score),
        "ready_text_color": _score_color(score),
        "camera_index": camera_index,
        "launch_target": launch_target_label(launch_target) if launch_target else None,
    }


def print_runtime_dashboard(title: str, runtime: Any, hardware: Any, camera_index: int, launch_target: str | None = None, print_fn=print) -> None:
    values = _dashboard_values(runtime, hardware, camera_index, launch_target)
    lines = [
        _line(_rule("="), CYAN),
        _line(_pad(title), BOLD + CYAN),
        _line(_rule("="), CYAN),
        _row("Kiểu chạy", values["launch_target"] or "-", CYAN),
        _row("Lựa chọn", values["chosen_label"], GREEN),
        _row("Mục tiêu", f"{values['requested_profile']} -> {values['cuda_target']}", MAGENTA),
        _row("Thực tế", f"{profile_label(values['runtime_profile'])} ({values['runtime_profile']})", values["profile_color"]),
        _row("Mức sẵn sàng", dashboard_boot_bar(values["score"], BOOT_BAR_WIDTH), values["ready_text_color"], bounded=False),
        _line(_rule("-"), CYAN),
        _section("PHẦN CỨNG", values["hardware_section_color"]),
        _row("CPU", getattr(hardware, "cpu_name", "Không rõ CPU"), GREEN),
        _row("RAM / OS", f"{float(getattr(hardware, 'ram_gb', 0.0) or 0.0):.1f} GB / {getattr(hardware, 'os_name', 'Không rõ OS')}", GREEN),
        _row("GPU", getattr(hardware, "gpu_name", "Không rõ"), values["hardware_section_color"]),
        _row("VRAM / GPU", f"{float(getattr(hardware, 'vram_gb', 0.0) or 0.0):.1f} GB / {int(getattr(hardware, 'gpu_count', 0) or 0)}", values["hardware_section_color"]),
        _row("Torch / build", f"{getattr(hardware, 'torch_version', 'Không có PyTorch')} / {getattr(hardware, 'torch_cuda_version', 'CPU-only')}", values["torch_color"]),
        _row("CUDA runtime", getattr(hardware, "cuda_runtime_status", "Không"), values["cuda_color"]),
        _row("Lý do CUDA", getattr(hardware, "cuda_runtime_reason", "Chưa kiểm tra"), values["reason_color"], bounded=False),
        _line(_rule("-"), CYAN),
        _section("RUNTIME", values["runtime_section_color"]),
        _row("Model đang chạy", values["actual_runtime"], values["model_color"]),
        _row("Model CUDA", values["cuda_model_status"], values["model_color"], bounded=False),
        _row("Camera / Index", f"{getattr(runtime, 'camera_width', '-')}x{getattr(runtime, 'camera_height', '-')} / {camera_index}", CYAN),
        _row("Half precision", "Bật" if getattr(runtime, "use_half", False) else "Tắt", GREEN if getattr(runtime, "use_half", False) else YELLOW),
        _line(_rule("-"), CYAN),
        _section("SẴN SÀNG", values["ready_section_color"]),
        _row("Trạng thái", performance_hint(values["score"]), values["ready_text_color"], bounded=False),
        _line(_rule("-"), CYAN),
    ]
    for item in lines:
        print_fn(item)


def explain_runtime_failure(error: Exception) -> tuple[str, list[str], list[str]]:
    message = str(error)
    lower_message = _normalize_text(message)
    if "khong mo duoc camera" in lower_message or "camera lien tuc khong tra ve frame" in lower_message:
        return (
            "Không mở được webcam hoặc webcam không trả về frame.",
            [
                "Kiểm tra webcam đang được cắm và không bị app khác chiếm.",
                "Thử đổi camera index sang 1 hoặc 2.",
                "Nếu đang mở app camera khác, hãy tắt trước khi chạy lại.",
            ],
            [
                r".\.venv\Scripts\python run_app.py --camera-index 1",
                r".\.venv\Scripts\python run_detect.py --camera-index 1",
            ],
        )
    if "khong khoi tao duoc ultralytics" in lower_message or "khong khoi tao duoc detector" in lower_message:
        return (
            "Không nạp được YOLO / ultralytics để bắt đầu nhận diện.",
            [
                "Kiểm tra môi trường .venv và gói ultralytics, torch, torchvision.",
                "Kiểm tra model local trong models/pretrained hoặc models/trained.",
                "Nếu máy yếu hoặc lỗi CUDA, thử mode low.",
            ],
            [
                r".\.venv\Scripts\python run_app.py --mode low",
                r".\.venv\Scripts\python run_detect.py --mode low",
            ],
        )
    if "cuda" in lower_message or "pytorch" in lower_message or "torch" in lower_message:
        return (
            "Môi trường PyTorch / CUDA đang có vấn đề hoặc chưa sẵn sàng.",
            [
                "Kiểm tra torch.cuda.is_available() trong .venv.",
                "Nếu không cần GPU, thử mode low để chạy nhẹ hơn.",
                "Nếu cần GPU, cài lại bản torch đúng với CUDA của máy.",
            ],
            [
                r".\.venv\Scripts\python -c \"import torch; print(torch.cuda.is_available())\"",
                r".\.venv\Scripts\python run_app.py --mode low",
            ],
        )
    return (
        message or "Không xác định được lỗi cụ thể.",
        [
            "Kiểm tra webcam, model local, PyTorch / CUDA và quyền truy cập camera.",
            "Thử chạy lại bằng mode low để giảm tải runtime.",
        ],
        [
            r".\.venv\Scripts\python run_app.py --mode low",
            r".\.venv\Scripts\python run_detect.py --mode low",
        ],
    )


def print_runtime_failure(title: str, error: Exception, *, print_fn=print) -> None:
    reason, suggestions, commands = explain_runtime_failure(error)
    lines = [
        _line(_rule("="), CYAN),
        _line(_pad(title), BOLD + RED),
        _line(_rule("="), CYAN),
        _section("LÝ DO", RED),
        _row("Lý do", reason, RED, bounded=False),
        _row("Chi tiết", str(error), YELLOW, bounded=False),
        _line(_rule("-"), CYAN),
        _section("GỢI Ý", GREEN),
    ]
    for index, suggestion in enumerate(suggestions, start=1):
        lines.append(_row(f"Bước {index}", suggestion, GREEN if index == len(suggestions) else YELLOW, bounded=False))
    lines.extend([_line(_rule("-"), CYAN), _section("LỆNH THỬ", CYAN)])
    for index, command in enumerate(commands, start=1):
        lines.append(_row(f"Lệnh {index}", command, CYAN, bounded=False))
    lines.append(_line(_rule("="), CYAN))
    for item in lines:
        print_fn(item)


class BootProgress:
    def __init__(self, title: str, enabled: bool | None = None) -> None:
        self.enabled = sys.stdout.isatty() and os.getenv("YOLO_DISABLE_PROGRESS") != "1" if enabled is None else enabled
        self.title = title
        self.current = 0
        self.current_label = "Đang chuẩn bị khởi động"
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
        title = " ".join(str(self.title).split())
        label = " ".join(str(self.current_label).split())
        content = f"{title} | {label} | {progress_bar_colored(self.current)}"
        print(_line(content, _score_color(self.current)), end="", flush=True)

    def advance_to(self, target: int, label: str) -> None:
        self.current_label = label
        if not self.enabled:
            self.current = max(self.current, target)
            return
        self.start()
        target = max(self.current, min(100, target))
        while self.current < target:
            self.current += 1
            self._render_line()
            time.sleep(0.01 if self.current < 90 else 0.016)

    def finish(self, label: str = "Sẵn sàng mở camera") -> None:
        self.advance_to(100, label)
        if self.enabled:
            self._clear_line()
