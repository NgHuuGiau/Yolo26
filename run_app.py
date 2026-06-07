import argparse

from app.camera_app import build_runtime_arg_parser, run_camera_entrypoint
from core.camera_runner import run_camera_session
from core.hardware_info import detect_hardware
from core.runtime_advisor import select_runtime_config_optimized
from tools.runtime_tool import prompt_runtime_mode
from utils.logger import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    return build_runtime_arg_parser("Chạy YOLO realtime camera ở chế độ desktop.").parse_args()


def main() -> int:
    return run_camera_entrypoint(
        args=parse_args(),
        boot_title="YOLO REALTIME CAMERA - ĐANG CHUẨN BỊ CHẾ ĐỘ DESKTOP",
        dashboard_title="YOLO REALTIME CAMERA - CHẾ ĐỘ DESKTOP PYTHON",
        boot_finish_message="Sẵn sàng mở cửa sổ camera Python / OpenCV",
        error_message="Desktop camera session failed: %s. Gợi ý: kiểm tra webcam, model local, CUDA hoặc thử mode low.",
        logger=logger,
        detect_hardware_fn=detect_hardware,
        select_runtime_config_fn=select_runtime_config_optimized,
        run_camera_session_fn=run_camera_session,
        prompt_runtime_mode_fn=prompt_runtime_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
