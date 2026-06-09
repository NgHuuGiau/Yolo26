import argparse

from app.chat_bootstrap import resolve_start_bundle
from app.chat_ai_app import launch_chat_ai_app
from app.runtime_entry import build_targeted_parser, run_targeted_entrypoint
from core.camera_runner import run_camera_session
from utils.console_ui import print_runtime_dashboard


def parse_args() -> argparse.Namespace:
    return build_targeted_parser("Chay YOLO realtime camera o che do desktop.").parse_args()


def main() -> int:
    return run_targeted_entrypoint(
        args=parse_args(),
        preferred_target="ui",
        ui_title="Chat AI",
        dashboard_title="YOLO Camera Realtime",
        resolve_start_bundle_fn=resolve_start_bundle,
        launch_chat_ai_app_fn=launch_chat_ai_app,
        print_runtime_dashboard_fn=print_runtime_dashboard,
        run_camera_session_fn=run_camera_session,
    )


if __name__ == "__main__":
    raise SystemExit(main())
