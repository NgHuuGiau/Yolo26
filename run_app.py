import argparse

from app.runtime_entry import build_runtime_arg_parser, run_camera_entrypoint
from core.camera_detector import run_camera_session
from core.hardware_detector import detect_hardware
from core.model_selector import select_runtime_config
from utils.file_utils import ensure_project_directories
from utils.logger import get_logger
from utils.runtime_prompt import prompt_runtime_mode

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    return build_runtime_arg_parser("Run YOLO realtime camera desktop mode.").parse_args()


def main() -> int:
    args = parse_args()
    selected_mode = args.mode or prompt_runtime_mode()
    progress = BootProgress("YOLO REALTIME CAMERA - DANG CHUAN BI CHE DO DESKTOP")
    progress.advance_to(12, "Da nhan cau hinh tu nguoi dung")
    hardware = detect_hardware()
    progress.advance_to(48, "Da kiem tra CPU / GPU / CUDA")
    runtime = select_runtime_config(mode=selected_mode, hardware=hardware)
    progress.advance_to(82, "Da chon runtime va model phu hop")
    progress.finish("San sang mo cua so camera Python / OpenCV")
    print_runtime_dashboard(
        title="YOLO REALTIME CAMERA - CHE DO DESKTOP PYTHON",
        runtime=runtime,
        hardware=hardware,
        camera_index=args.camera_index,
    )
    try:
        run_camera_session(runtime=runtime, camera_index=args.camera_index)
    except Exception as exc:
        logger.error(
            "Desktop camera session failed: %s. Goi y: kiem tra webcam, model local, CUDA hoac thu mode low.",
            exc,
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
