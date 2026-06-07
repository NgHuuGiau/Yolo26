from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from utils.console_ui import BootProgress, print_runtime_dashboard, print_runtime_failure
from utils.file_utils import ensure_project_directories


def build_runtime_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        default=None,
        choices=["auto", "high", "medium", "low"],
        help="Chế độ runtime.",
    )
    parser.add_argument(
        "--camera-index",
        default=0,
        type=int,
        help="Chỉ số camera OpenCV.",
    )
    return parser


def run_camera_entrypoint(
    *,
    args: argparse.Namespace,
    boot_title: str,
    dashboard_title: str,
    boot_finish_message: str,
    error_message: str,
    logger: Any,
    detect_hardware_fn: Callable[[], Any],
    select_runtime_config_fn: Callable[..., Any],
    run_camera_session_fn: Callable[..., None],
    prompt_runtime_mode_fn: Callable[..., str],
) -> int:
    ensure_project_directories()
    try:
        hardware = detect_hardware_fn()
        if hasattr(hardware, "pretty_report") and callable(getattr(hardware, "pretty_report")):
            print(hardware.pretty_report())
            print()

        recommendations = None
        if args.mode is None:
            recommendations = {
                mode: select_runtime_config_fn(mode=mode, hardware=hardware)
                for mode in ("auto", "high", "medium", "low")
            }

        selected_mode = args.mode or prompt_runtime_mode_fn(hardware=hardware, recommendations=recommendations)
        progress = BootProgress(boot_title)
        progress.advance_to(12, "Đã nhận cấu hình từ người dùng")
        progress.advance_to(48, "Đã kiểm tra CPU / GPU / CUDA")

        runtime = recommendations.get(selected_mode) if recommendations else None
        if runtime is None:
            runtime = select_runtime_config_fn(mode=selected_mode, hardware=hardware)

        progress.advance_to(82, "Đã chọn runtime và model phù hợp")
        progress.finish(boot_finish_message)
        print_runtime_dashboard(
            title=dashboard_title,
            runtime=runtime,
            hardware=hardware,
            camera_index=args.camera_index,
        )
        run_camera_session_fn(runtime=runtime, camera_index=args.camera_index)
    except Exception as exc:
        logger.error(error_message, exc)
        print_runtime_failure(f"{dashboard_title} :: KHÔNG THỂ KHỞI CHẠY", exc)
        raise
    return 0
