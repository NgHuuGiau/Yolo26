from __future__ import annotations

from pathlib import Path

from core.hardware_info import detect_hardware
from core.model_selector import select_runtime_config
from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
from utils.file_utils import ensure_project_directories


YOLO26_MODELS = ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt")
PRETRAINED_DIR = Path("models/pretrained")
RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")
PROCESSED_TRAIN_DIR = Path("dataset/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/processed/images/val")


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def _present_and_missing_models(model_dir: Path = PRETRAINED_DIR) -> tuple[list[str], list[str]]:
    present = [name for name in YOLO26_MODELS if (model_dir / name).exists()]
    missing = [name for name in YOLO26_MODELS if name not in present]
    return present, missing


def main() -> None:
    ensure_project_directories()
    hardware = detect_hardware()
    present_models, missing_models = _present_and_missing_models()
    raw_images = _count_files(RAW_IMAGES_DIR)
    raw_labels = _count_files(RAW_LABELS_DIR)
    train_images = _count_files(PROCESSED_TRAIN_DIR)
    val_images = _count_files(PROCESSED_VAL_DIR)

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

    print(line(rule("-"), CYAN))
    print(section("MODEL YOLO26", GREEN if not missing_models else YELLOW))
    print(row("Đã có", ", ".join(present_models) if present_models else "Chưa có model nào", GREEN if present_models else RED, bounded=False))
    if missing_models:
        print(row("Thiếu", ", ".join(missing_models), RED, bounded=False))
    else:
        print(row("Trạng thái", "Đã có đủ 5 model YOLO26.", GREEN, bounded=False))

    print(line(rule("-"), CYAN))
    print(section("GỢI Ý CHẠY THEO MÁY", GREEN))
    for label, runtime in recommendations.items():
        value = f"{runtime.primary_model_name} / {runtime.resolved_device} / imgsz {runtime.imgsz}"
        print(row(label, value, GREEN if runtime.primary_model_name != "yolo26n.pt" else YELLOW, bounded=False))

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
        print(row("Model", "Đã sẵn sàng để chạy đủ các mức YOLO26.", GREEN, bounded=False))
    if not dataset_ok:
        print(row("Dataset", "Chưa có dữ liệu raw để train.", YELLOW, bounded=False))
    elif not split_ok:
        print(row("Dataset", "Đã có raw nhưng chưa split train/val.", YELLOW, bounded=False))
    else:
        print(row("Dataset", "Dữ liệu train/val đã sẵn sàng.", GREEN, bounded=False))

    print(line(rule("-"), CYAN))
    print(section("LỆNH NÊN CHẠY", CYAN))
    if missing_models:
        print(command_row(1, r".\.venv\Scripts\python training\download_models.py"))
    if not dataset_ok:
        print(command_row(2, r".\.venv\Scripts\python training\prepare_dataset.py"))
    elif not split_ok:
        print(command_row(2, r".\.venv\Scripts\python training\validate_dataset.py"))
        print(command_row(3, r".\.venv\Scripts\python training\split_dataset.py"))
    else:
        print(command_row(2, r".\.venv\Scripts\python run_app.py"))
        print(command_row(3, r".\.venv\Scripts\python run_train.py"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
