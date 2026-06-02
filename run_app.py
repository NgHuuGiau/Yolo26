import argparse

from app.camera_app import build_runtime_arg_parser, run_camera_entrypoint
from core.camera_runner import run_camera_session
from core.hardware_info import detect_hardware
from core.model_selector import select_runtime_config
from utils.logger import get_logger
from utils.console_ui import prompt_runtime_mode


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    return build_runtime_arg_parser("Run YOLO realtime camera desktop mode.").parse_args()


def main() -> int:
    return run_camera_entrypoint(
        args=parse_args(),
        boot_title="YOLO REALTIME CAMERA - DANG CHUAN BI CHE DO DESKTOP",
        dashboard_title="YOLO REALTIME CAMERA - CHE DO DESKTOP PYTHON",
        boot_finish_message="San sang mo cua so camera Python / OpenCV",
        error_message="Desktop camera session failed: %s. Goi y: kiem tra webcam, model local, CUDA hoac thu mode low.",
        logger=logger,
        detect_hardware_fn=detect_hardware,
        select_runtime_config_fn=select_runtime_config,
        run_camera_session_fn=run_camera_session,
        prompt_runtime_mode_fn=prompt_runtime_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
