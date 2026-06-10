from __future__ import annotations

import argparse

from app.chat_ai_app import build_chat_arg_parser
from utils.console_ui import BootProgress


def build_targeted_parser(description: str) -> argparse.ArgumentParser:
    parser = build_chat_arg_parser(description)
    parser.add_argument(
        "--target",
        default=None,
        choices=["ui", "camera"],
        help="Kieu khoi dong: ui desktop hoac camera realtime.",
    )
    return parser


def run_targeted_entrypoint(
    *,
    args,
    preferred_target: str,
    ui_title: str,
    dashboard_title: str,
    resolve_start_bundle_fn,
    launch_chat_ai_app_fn,
    print_runtime_dashboard_fn,
    run_camera_session_fn,
) -> int:
    start_options = resolve_start_bundle_fn(
        requested_mode=args.mode,
        requested_model=args.model,
        requested_target=args.target,
        preferred_target=preferred_target,
    )
    if start_options.launch_target == "ui":
        return launch_chat_ai_app_fn(
            window_title=ui_title,
            camera_index=args.camera_index,
            app_mode=start_options.selected_mode,
            selected_model=start_options.selected_model,
        )
    progress = BootProgress(dashboard_title)
    progress.advance_to(16, "Đang nhận cấu hình khởi động")
    progress.advance_to(42, "Đang kiểm tra CPU / GPU / CUDA")
    progress.advance_to(68, "Đang chọn model và runtime phù hợp")
    progress.advance_to(88, "Đang chuẩn bị mở camera")
    progress.finish("Sẵn sàng mở camera")
    print_runtime_dashboard_fn(
        title=dashboard_title,
        runtime=start_options.runtime,
        hardware=start_options.hardware,
        camera_index=args.camera_index,
        launch_target=start_options.launch_target,
    )
    run_camera_session_fn(runtime=start_options.runtime, camera_index=args.camera_index)
    return 0
