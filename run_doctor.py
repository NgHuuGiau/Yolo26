from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from core.hardware_info import detect_hardware
from core.model_selector import select_runtime_config
from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
from utils.file_utils import ensure_project_directories


YOLO11_MODELS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt")
PRETRAINED_DIR = Path("models/pretrained")
RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")
PROCESSED_TRAIN_DIR = Path("dataset/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/processed/images/val")


@dataclass
class CameraProbeResult:
    level: str
    summary: str
    detail: str

    @property
    def color(self) -> str:
        return {"PASS": GREEN, "WARN": YELLOW, "ERROR": RED}.get(self.level, CYAN)


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def _present_and_missing_models(model_dir: Path = PRETRAINED_DIR) -> tuple[list[str], list[str]]:
    present = [name for name in YOLO11_MODELS if (model_dir / name).exists()]
    missing = [name for name in YOLO11_MODELS if name not in present]
    return present, missing


def _open_camera_capture(index: int):
    import cv2

    return cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)


def _probe_camera(index: int = 0) -> CameraProbeResult:
    try:
        capture = _open_camera_capture(index)
    except Exception as exc:
        return CameraProbeResult(
            level="ERROR",
            summary=f"Camera thật       ERROR | Không tạo được camera index {index}",
            detail=f"Lý do không chạy   {exc}",
        )

    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        return CameraProbeResult(
            level="WARN",
            summary=f"Camera thật       WARN  | Không mở được camera index {index}",
            detail="Lý do không chạy   Webcam không sẵn sàng, đang bị app khác chiếm hoặc chưa cắm.",
        )

    width = 0
    height = 0
    ok = False
    try:
        for _ in range(3):
            success, frame = capture.read()
            if success and frame is not None:
                height, width = frame.shape[:2]
                ok = True
                break
    finally:
        capture.release()

    if not ok:
        return CameraProbeResult(
            level="WARN",
            summary=f"Camera thật       WARN  | Mở được camera index {index} nhưng không đọc được frame",
            detail="Lý do không chạy   Webcam mở được nhưng không trả về khung hình hợp lệ.",
        )

    return CameraProbeResult(
        level="PASS",
        summary=f"Camera thật       PASS  | Đọc frame thành công tại index {index}",
        detail=f"Chi tiết          {width}x{height}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra sức khỏe toàn hệ thống cho dự án YOLO.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index để kiểm tra webcam thật.")
    parser.add_argument(
        "--skip-camera-check",
        action="store_true",
        help="Bỏ qua bước kiểm tra camera thật.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_directories()
    hardware = detect_hardware()
    present_models, missing_models = _present_and_missing_models()
    raw_images = _count_files(RAW_IMAGES_DIR)
    raw_labels = _count_files(RAW_LABELS_DIR)
    train_images = _count_files(PROCESSED_TRAIN_DIR)
    val_images = _count_files(PROCESSED_VAL_DIR)
    camera_probe = None if args.skip_camera_check else _probe_camera(args.camera_index)

    recommendations = {
        "Cao nhất": select_runtime_config("high", hardware),
        "Trung bình": select_runtime_config("medium", hardware),
        "Yếu": select_runtime_config("low", hardware),
    }

    for item in header("YOLO DOCTOR :: KIỂM TRA TOÀN HỆ THỐNG"):
        print(item)

    print(section("PHẦN CỨNG", GREEN if hardware.cuda_available else YELLOW))
    print(row("CPU", hardware.cpu_name, GREEN))
    print(row("RAM / OS", f"{hardware.ram_gb:.1f} GB / {hardware.os_name}", GREEN))
    print(row("GPU", hardware.gpu_name, GREEN if hardware.gpu_hardware_available else YELLOW))
    print(row("VRAM / GPU", f"{hardware.vram_gb:.1f} GB / {hardware.gpu_count}", GREEN if hardware.gpu_hardware_available else YELLOW))
    print(row("PyTorch", hardware.torch_version, GREEN if hardware.torch_version != "Không có PyTorch" else RED, bounded=False))
    print(row("CUDA", hardware.cuda_runtime_reason, GREEN if hardware.cuda_available else YELLOW, bounded=False))

    if camera_probe is not None:
        print(line(rule("-"), CYAN))
        print(section("CAMERA THẬT", camera_probe.color))
        print(row("Trạng thái", camera_probe.summary.split("|", 1)[-1].strip(), camera_probe.color, bounded=False))
        print(row("Chi tiết", camera_probe.detail.replace("Chi tiết          ", "").replace("Lý do không chạy   ", ""), camera_probe.color, bounded=False))

    print(line(rule("-"), CYAN))
    print(section("MODEL YOLO11", GREEN if not missing_models else YELLOW))
    print(row("Đã có", ", ".join(present_models) if present_models else "Chưa có model nào", GREEN if present_models else RED, bounded=False))
    if missing_models:
        print(row("Thiếu", ", ".join(missing_models), RED, bounded=False))
    else:
        print(row("Trạng thái", "Đã có đủ 5 model YOLO11.", GREEN, bounded=False))

    print(line(rule("-"), CYAN))
    print(section("GỢI Ý CHẠY THEO MÁY", GREEN))
    for label, runtime in recommendations.items():
        value = f"{runtime.primary_model_name} / {runtime.resolved_device} / imgsz {runtime.imgsz}"
        color = GREEN if runtime.primary_model_name not in {"yolo11n.pt"} else YELLOW
        print(row(label, value, color, bounded=False))

    print(line(rule("-"), CYAN))
    dataset_ok = raw_images > 0 and raw_labels > 0
    split_ok = train_images > 0 and val_images > 0
    print(section("DỮ LIỆU", GREEN if dataset_ok else YELLOW))
    print(row("Raw images", f"{RAW_IMAGES_DIR} ({raw_images} file)", GREEN if raw_images else RED, bounded=False))
    print(row("Raw labels", f"{RAW_LABELS_DIR} ({raw_labels} file)", GREEN if raw_labels else RED, bounded=False))
    print(row("Train split", f"{PROCESSED_TRAIN_DIR} ({train_images} file)", GREEN if train_images else YELLOW, bounded=False))
    print(row("Val split", f"{PROCESSED_VAL_DIR} ({val_images} file)", GREEN if val_images else YELLOW, bounded=False))

    print(line(rule("-"), CYAN))
    ready = bool(present_models) and dataset_ok
    print(section("KẾT LUẬN", GREEN if ready else YELLOW))
    if not present_models:
        print(row("Lý do", "Chưa có model local trong models/pretrained.", RED, bounded=False))
    elif missing_models:
        print(row("Lý do", "Máy vẫn chạy được, nhưng chưa có đủ 5 model để chọn hết mọi mức.", YELLOW, bounded=False))
    else:
        print(row("Model", "Đã sẵn sàng để chạy đủ các mức YOLO11.", GREEN, bounded=False))

    if camera_probe is not None and camera_probe.level != "PASS":
        print(row("Camera", camera_probe.detail.replace("Lý do không chạy   ", ""), YELLOW, bounded=False))

    if not dataset_ok:
        print(row("Dataset", "Chưa có dữ liệu raw để train.", YELLOW, bounded=False))
    elif not split_ok:
        print(row("Dataset", "Đã có raw nhưng chưa split train/val.", YELLOW, bounded=False))
    else:
        print(row("Dataset", "Dữ liệu train/val đã sẵn sàng.", GREEN, bounded=False))

    print(line(rule("-"), CYAN))
    print(section("LỆNH NÊN CHẠY", CYAN))
    command_index = 1
    if missing_models:
        print(command_row(command_index, r".\.venv\Scripts\python training\download_models.py"))
        command_index += 1
    if not dataset_ok:
        print(command_row(command_index, r".\.venv\Scripts\python training\prepare_dataset.py"))
        command_index += 1
    elif not split_ok:
        print(command_row(command_index, r".\.venv\Scripts\python training\validate_dataset.py"))
        command_index += 1
        print(command_row(command_index, r".\.venv\Scripts\python training\split_dataset.py"))
        command_index += 1
    else:
        print(command_row(command_index, r".\.venv\Scripts\python run_app.py"))
        command_index += 1
        print(command_row(command_index, r".\.venv\Scripts\python run_train.py"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
