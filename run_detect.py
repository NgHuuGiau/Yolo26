import argparse

from app.chat_bootstrap import resolve_start_options
from app.chat_ai_app import build_chat_arg_parser, launch_chat_ai_app


def parse_args() -> argparse.Namespace:
    return build_chat_arg_parser("Chạy YOLO realtime camera ở chế độ nhận diện CLI.").parse_args()


def main() -> int:
    args = parse_args()
    selected_mode, selected_model = resolve_start_options(
        requested_mode=args.mode,
        requested_model=args.model,
    )
    return launch_chat_ai_app(
        window_title="Chat AI Detect",
        camera_index=args.camera_index,
        app_mode=selected_mode,
        selected_model=selected_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
