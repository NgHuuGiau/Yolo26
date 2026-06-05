from __future__ import annotations

import importlib
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.model_paths import resolve_data_config_path, resolve_trained_model_path, resolve_model_source
except ModuleNotFoundError:
    from model_paths import resolve_data_config_path, resolve_trained_model_path, resolve_model_source

try:
    from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
except ModuleNotFoundError:
    from terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section

YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
PROCESSED_VAL_DIR = Path("dataset/processed/images/val")


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    if YOLO is None and ULTRALYTICS_IMPORT_ERROR is None:
        try:
            YOLO = importlib.import_module("ultralytics").YOLO
        except Exception as exc:  # pragma: no cover
            ULTRALYTICS_IMPORT_ERROR = exc
    if YOLO is None:
        raise RuntimeError(f"Không khởi tạo được ultralytics/YOLO: {ULTRALYTICS_IMPORT_ERROR}")
    return YOLO


def resolve_validation_model_path():
    model_path = resolve_trained_model_path(required=False, fallback="yolo11n.pt")
    if model_path == Path(model_path.name):
        model_path = Path("models/pretrained") / model_path.name
    return resolve_model_source(model_path)


def _ensure_validation_dataset_ready() -> None:
    if not PROCESSED_VAL_DIR.exists() or not any(PROCESSED_VAL_DIR.iterdir()):
        raise FileNotFoundError(
            "Chưa có ảnh trong dataset/processed/images/val. "
            "Hãy bỏ dữ liệu vào dataset/raw và chạy training/split_dataset.py trước."
        )


def _print_validation_ready_help(error: FileNotFoundError) -> None:
    val_count = len(list(PROCESSED_VAL_DIR.glob("*"))) if PROCESSED_VAL_DIR.exists() else 0
    for item in header("YOLO VALIDATION :: DỮ LIỆU CHƯA SẴN SÀNG", color=RED):
        print(item)
    print(section("LÝ DO", RED))
    print(row("Lý do không chạy", str(error), RED, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("KIỂM TRA NHANH", YELLOW))
    print(row("Val images", f"{PROCESSED_VAL_DIR} ({val_count} file)", GREEN if val_count > 0 else RED, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("CÁC BƯỚC CẦN LÀM", GREEN))
    print(row("Bước 1", "Bỏ ảnh và label vào dataset/raw", YELLOW, bounded=False))
    print(row("Bước 2", "Chạy training/validate_dataset.py", YELLOW))
    print(row("Bước 3", "Chạy training/split_dataset.py", YELLOW))
    print(row("Bước 4", "Chạy lại training/validate_model.py", GREEN))
    print(line(rule("-"), CYAN))
    print(section("Ý NGHĨA LỆNH", CYAN))
    print(row("Lệnh này", "Dùng tập val trong dataset/processed để đo kết quả model.", YELLOW, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("LỆNH NHANH", CYAN))
    print(command_row(1, r".\.venv\Scripts\python training\validate_dataset.py"))
    print(command_row(2, r".\.venv\Scripts\python training\split_dataset.py"))
    print(command_row(3, r".\.venv\Scripts\python training\validate_model.py"))
    print(line(rule("="), CYAN))


def main() -> None:
    try:
        _ensure_validation_dataset_ready()
    except FileNotFoundError as exc:
        _print_validation_ready_help(exc)
        raise SystemExit(1)
    model_path = resolve_validation_model_path()
    model = _require_yolo()(str(model_path))
    metrics = model.val(
        data=str(resolve_data_config_path()),
        project=str(Path("runs/val").resolve()),
        name="validation",
    )
    print(metrics)


if __name__ == "__main__":
    main()
