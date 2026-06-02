from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from utils.file_utils import ensure_project_directories
from utils.console_ui import BootProgress, print_runtime_dashboard


def build_runtime_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        default=None,
        choices=["auto", "high", "medium", "low"],
        help="Runtime mode.",
    )
    parser.add_argument(
        "--camera-index",
        default=0,
        type=int,
        help="OpenCV camera index.",
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
    prompt_runtime_mode_fn: Callable[[], str],
) -> int:
    ensure_project_directories()
    selected_mode = args.mode or prompt_runtime_mode_fn()
    progress = BootProgress(boot_title)
    progress.advance_to(12, "Da nhan cau hinh tu nguoi dung")
    hardware = detect_hardware_fn()
    progress.advance_to(48, "Da kiem tra CPU / GPU / CUDA")
    runtime = select_runtime_config_fn(mode=selected_mode, hardware=hardware)
    progress.advance_to(82, "Da chon runtime va model phu hop")
    progress.finish(boot_finish_message)
    print_runtime_dashboard(
        title=dashboard_title,
        runtime=runtime,
        hardware=hardware,
        camera_index=args.camera_index,
    )
    try:
        run_camera_session_fn(runtime=runtime, camera_index=args.camera_index)
    except Exception as exc:
        logger.error(error_message, exc)
        raise
    return 0
