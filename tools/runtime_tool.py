from __future__ import annotations

from pathlib import Path

from core.hardware_info import detect_hardware
from core.runtime_advisor import (
    MODE_ORDER,
    YOLO11_VARIANTS,
    build_recommendations,
    ceiling_mode_for_hardware,
    gpu_tier,
    load_level,
    mode_label,
    mode_meaning,
    mode_title,
    quality_score,
    stability_score,
)
from core.model_selector import load_settings
from utils.console_ui import (
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    YELLOW,
    header,
    line,
    row,
    rule,
    section,
)
from utils.console_ui import prompt_runtime_mode as render_runtime_prompt
from utils.file_utils import load_yaml_cached


MODEL_CONFIG_PATH = "config/model_config.yaml"


def _usage_text(value) -> str:
    if value is None:
        return "không rõ"
    return f"{float(value):.1f}%"


def _usage_color(value) -> str:
    if value is None:
        return YELLOW
    if float(value) < 60:
        return GREEN
    if float(value) < 85:
        return YELLOW
    return RED


def _available_models() -> tuple[list[str], list[str]]:
    available = []
    missing = []
    for name in YOLO11_VARIANTS:
        if Path("models/pretrained", name).exists() or Path(name).exists():
            available.append(name)
        else:
            missing.append(name)
    if Path("models/trained/best.pt").exists():
        available.insert(0, "models/trained/best.pt")
    return available, missing


def _summary_line(mode: str, runtime) -> str:
    return (
        f"{mode_title(mode)} ({mode}) -> "
        f"model={runtime.primary_model_name}, "
        f"thiết bị={runtime.resolved_device}, "
        f"imgsz={runtime.imgsz}, "
        f"max_det={runtime.max_det}, "
        f"fallback={', '.join(runtime.candidate_models)}"
    )


def _mode_reason(mode: str, runtime, hardware) -> str:
    quality = quality_score(runtime)
    stability = stability_score(mode, hardware)
    if mode == "high":
        return f"Trần cao nhất máy còn gánh được. chất lượng={quality}/100, ổn định={stability}/100."
    if mode == "medium":
        return f"Mức cân bằng đẹp nhất để dùng thường xuyên. chất lượng={quality}/100, ổn định={stability}/100."
    return f"Mức an toàn nhất khi ưu tiên độ mượt. chất lượng={quality}/100, ổn định={stability}/100."


def _best_solution_text(auto_runtime, hardware) -> str:
    load_text = {
        "cool": "tải hiện tại nhẹ",
        "warm": "tải hiện tại trung bình",
        "busy": "tải hiện tại khá cao",
        "very_busy": "tải hiện tại rất cao",
    }[load_level(hardware)]
    return (
        f"Đề xuất nên chạy ngay là {mode_label(auto_runtime.mode)} vì {load_text}, "
        f"với {auto_runtime.primary_model_name} / {auto_runtime.resolved_device} / "
        f"imgsz {auto_runtime.imgsz} / max_det {auto_runtime.max_det}."
    )


def _wow_conclusion(hardware, recommendations, auto_runtime) -> list[str]:
    ceiling_mode = ceiling_mode_for_hardware(hardware)
    ceiling_runtime = recommendations[ceiling_mode]
    return [
        (
            f"- trần tối đa máy đang gánh được: {mode_label(ceiling_mode)} ({ceiling_mode}) / "
            f"{ceiling_runtime.primary_model_name} / {ceiling_runtime.resolved_device} / "
            f"imgsz {ceiling_runtime.imgsz} / max_det {ceiling_runtime.max_det}"
        ),
        (
            f"- mức đẹp nhất để chạy thường xuyên: trung bình / "
            f"{recommendations['medium'].primary_model_name} / {recommendations['medium'].resolved_device} / "
            f"imgsz {recommendations['medium'].imgsz} / max_det {recommendations['medium'].max_det}"
        ),
        (
            f"- mức an toàn nhất khi muốn mượt: yếu nhất / "
            f"{recommendations['low'].primary_model_name} / {recommendations['low'].resolved_device} / "
            f"imgsz {recommendations['low'].imgsz} / max_det {recommendations['low'].max_det}"
        ),
        (
            f"- đề xuất chạy ngay lúc này: {mode_label(auto_runtime.mode)} ({auto_runtime.mode}) / "
            f"{auto_runtime.primary_model_name} / {auto_runtime.resolved_device} / "
            f"imgsz {auto_runtime.imgsz} / max_det {auto_runtime.max_det}"
        ),
    ]


def _menu_line(mode: str, runtime) -> str:
    return (
        f"{mode_title(mode):<11} | "
        f"{mode_meaning(mode):<30} | "
        f"{runtime.primary_model_name:<10} | "
        f"{runtime.resolved_device:<6} | "
        f"imgsz {runtime.imgsz:<4} | "
        f"max_det {runtime.max_det}"
    )


def prompt_runtime_mode(hardware=None, recommendations=None, input_fn=input, print_fn=print) -> str:
    hardware = hardware or detect_hardware()
    recommendations = recommendations or build_recommendations(hardware)
    return render_runtime_prompt(
        hardware=hardware,
        recommendations=recommendations,
        input_fn=input_fn,
        print_fn=print_fn,
    )


def _print_lines(lines: list[str]) -> None:
    for item in lines:
        print(item)


def _model_local_text(models: list[str]) -> str:
    return ", ".join(models) if models else "không có model local nào"


def _mode_color(mode: str) -> str:
    return {"high": GREEN, "medium": YELLOW, "low": MAGENTA}.get(mode, CYAN)


def _project_model_text(settings: dict, mode: str) -> str:
    profile = settings["models"][mode]
    device = "gpu" if mode in {"high", "medium"} else "auto"
    return f"{profile['model']} / {device} / imgsz {profile['imgsz']}"


def main() -> None:
    hardware = detect_hardware()
    settings = load_settings()
    model_config = load_yaml_cached(MODEL_CONFIG_PATH)
    recommendations = build_recommendations(hardware)
    auto_runtime = recommendations["auto"]
    available_models, missing_models = _available_models()

    _print_lines(header("BỘ TƯ VẤN RUNTIME YOLO :: THĂM DÒ MÁY VÀ ĐỀ XUẤT 3 MỨC TỐI ƯU"))
    _print_lines(
        [
            section("TỔNG QUAN MÁY", GREEN),
            row("CPU", hardware.cpu_name, GREEN, bounded=False),
            row("RAM / OS", f"{hardware.ram_gb:.1f} GB / {hardware.os_name}", GREEN, bounded=False),
            row("GPU", hardware.gpu_name, GREEN if hardware.cuda_available else YELLOW, bounded=False),
            row("VRAM", f"{hardware.vram_gb:.1f} GB", GREEN if hardware.vram_gb else YELLOW, bounded=False),
            row("CUDA", "có" if hardware.cuda_available else "không", GREEN if hardware.cuda_available else RED, bounded=False),
            row("PyTorch", hardware.torch_version, CYAN, bounded=False),
            row("CUDA build", hardware.torch_cuda_version, CYAN, bounded=False),
            row("Phân hạng GPU", gpu_tier(hardware), YELLOW, bounded=False),
            row("Tải CPU", _usage_text(hardware.cpu_usage_percent), _usage_color(hardware.cpu_usage_percent), bounded=False),
            row("Tải GPU", _usage_text(hardware.gpu_usage_percent), _usage_color(hardware.gpu_usage_percent), bounded=False),
            row("Tải VRAM", _usage_text(hardware.vram_usage_percent), _usage_color(hardware.vram_usage_percent), bounded=False),
            row("Trạng thái tải", load_level(hardware), MAGENTA, bounded=False),
            line(rule("-"), CYAN),
            section("YOLO11 VÀ MODEL LOCAL", MAGENTA),
            row("5 phiên bản", ", ".join(YOLO11_VARIANTS), CYAN, bounded=False),
            row("Model sẵn sàng", _model_local_text(available_models), GREEN, bounded=False),
            row("Model còn thiếu", _model_local_text(missing_models) if missing_models else "đã có đủ các model chính", YELLOW, bounded=False),
            line(rule("-"), CYAN),
            section("ĐỊNH NGHĨA 3 MỨC", YELLOW),
            row("Mạnh nhất", "mức cao nhất máy còn gánh được", GREEN, bounded=False),
            row("Trung bình", "mức cân bằng đẹp nhất để chạy thường xuyên", YELLOW, bounded=False),
            row("Yếu nhất", "mức nhẹ nhất để ưu tiên độ mượt và an toàn", MAGENTA, bounded=False),
            line(rule("-"), CYAN),
            section("3 MỨC TỐI ƯU TRÊN MÁY NÀY", CYAN),
        ]
    )
    for mode in MODE_ORDER:
        runtime = recommendations[mode]
        color = _mode_color(mode)
        _print_lines(
            [
                row(mode_title(mode), _summary_line(mode, runtime), color, bounded=False),
                row("  Đánh giá", f"chất lượng {quality_score(runtime)}/100 | ổn định {stability_score(mode, hardware)}/100", DIM, bounded=False),
                row("  Giải thích", _mode_reason(mode, runtime, hardware), DIM, bounded=False),
                line(rule("."), DIM),
            ]
        )

    preferred = model_config.get("preferred_models", {})
    priority_order = model_config.get("priority_order", [])
    _print_lines(
        [
            line(rule("-"), CYAN),
            section("CẤU HÌNH DỰ ÁN", GREEN),
            row("Thiết lập high", _project_model_text(settings, "high"), GREEN, bounded=False),
            row("Thiết lập medium", _project_model_text(settings, "medium"), YELLOW, bounded=False),
            row("Thiết lập low", _project_model_text(settings, "low"), MAGENTA, bounded=False),
            row(
                "Model ưu tiên",
                (
                    f"primary {preferred.get('primary_gpu', 'không rõ')} | "
                    f"backup GPU {preferred.get('stable_backup_gpu', 'không rõ')} | "
                    f"backup CPU {preferred.get('stable_backup_cpu', 'không rõ')}"
                ),
                CYAN,
                bounded=False,
            ),
        ]
    )
    if priority_order:
        print(row("Thứ tự load", ", ".join(priority_order), DIM, bounded=False))

    _print_lines(
        [
            line(rule("-"), CYAN),
            section("KẾT LUẬN WOW", MAGENTA),
        ]
    )
    for item in _wow_conclusion(hardware, recommendations, auto_runtime):
        print(row("Giải pháp", item.removeprefix("- ").strip(), GREEN, bounded=False))
    print(row("Đề xuất tốt nhất", _best_solution_text(auto_runtime, hardware), YELLOW, bounded=False))
    print(line(rule("="), CYAN))
