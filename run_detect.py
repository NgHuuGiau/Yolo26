import argparse

from app.chat_bootstrap import resolve_start_bundle
from app.chat_ai_app import build_chat_arg_parser, launch_chat_ai_app
from core.camera_runner import run_camera_session
from utils.console_ui import print_runtime_dashboard


def parse_args() -> argparse.Namespace:
    parser = build_chat_arg_parser("Chạy YOLO realtime camera ở chế độ nhận diện CLI.")
    parser.add_argument(
        "--target",
        default=None,
        choices=["ui", "camera"],
        help="Kieu khoi dong: ui desktop hoac camera realtime.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_options = resolve_start_bundle(
        requested_mode=args.mode,
        requested_model=args.model,
        requested_target=args.target,
        preferred_target="camera",
    )
    if start_options.launch_target == "ui":
        return launch_chat_ai_app(
            window_title="Chat AI Detect",
            camera_index=args.camera_index,
            app_mode=start_options.selected_mode,
            selected_model=start_options.selected_model,
        )

    print_runtime_dashboard(
        title="YOLO Detect Camera",
        runtime=start_options.runtime,
        hardware=start_options.hardware,
        camera_index=args.camera_index,
        launch_target=start_options.launch_target,
    )
    run_camera_session(runtime=start_options.runtime, camera_index=args.camera_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
