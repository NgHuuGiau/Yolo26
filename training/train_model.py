from __future__ import annotations

import importlib
import shutil
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source
except ModuleNotFoundError:
    from model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source

try:
    from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
except ModuleNotFoundError:
    from terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section

try:
    from training.auto_label_raw import auto_label_raw_images
    from training.split_dataset import _copy_split, _reset_processed_dirs, _split_items, audit_raw_dataset
except ModuleNotFoundError:
    from auto_label_raw import auto_label_raw_images
    from split_dataset import _copy_split, _reset_processed_dirs, _split_items, audit_raw_dataset

YOLO = None
ULTRALYTICS_IMPORT_ERROR = None

from utils.file_utils import ensure_project_directories, load_yaml
from utils.logger import get_logger


logger = get_logger(__name__)
PROCESSED_TRAIN_DIR = Path("dataset/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/processed/images/val")
RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")


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


def _training_kwargs(config: dict) -> dict:
    kwargs = {key: value for key, value in config.items() if key != "fallback_model"}
    if "project" in kwargs:
        kwargs["project"] = str(Path(kwargs["project"]).resolve())
    if "data" in kwargs:
        kwargs["data"] = str(resolve_data_config_path())
    return kwargs


def _ensure_training_dataset_ready() -> None:
    if not PROCESSED_TRAIN_DIR.exists() or not any(PROCESSED_TRAIN_DIR.iterdir()):
        raise FileNotFoundError(
            "Chưa có ảnh trong dataset/processed/images/train. "
            "Hãy bỏ dữ liệu vào dataset/raw và chạy training/split_dataset.py trước."
        )
    if not PROCESSED_VAL_DIR.exists() or not any(PROCESSED_VAL_DIR.iterdir()):
        raise FileNotFoundError(
            "Chưa có ảnh trong dataset/processed/images/val. "
            "Hãy bỏ dữ liệu vào dataset/raw và chạy training/split_dataset.py trước."
        )


def _auto_prepare_training_dataset() -> dict[str, object]:
    ensure_project_directories()
    report = {
        "raw_images": 0,
        "auto_labeled": 0,
        "eligible": 0,
        "no_detection": [],
    }
    audit = audit_raw_dataset()
    report["raw_images"] = audit.raw_image_count
    if audit.raw_image_count == 0:
        return report

    needs_auto_label = bool(audit.missing_labels)
    if needs_auto_label:
        auto_report = auto_label_raw_images(overwrite=False, conf=0.25, device="cpu")
        report["auto_labeled"] = int(auto_report.get("generated", 0))
        report["no_detection"] = list(auto_report.get("no_detection", []))
        audit = audit_raw_dataset()

    eligible = audit.eligible
    report["eligible"] = len(eligible)
    if not eligible:
        return report

    _reset_processed_dirs()
    split_map = _split_items(eligible)
    for split_name, items in split_map.items():
        _copy_split(split_name, items)
    return report


def _copy_best_weight(run_dir: Path) -> None:
    best_weight = run_dir / "weights" / "best.pt"
    target = TRAINED_BEST_MODEL_PATH
    if best_weight.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weight, target)
        logger.info("Copied best.pt to %s", target)


def _print_dataset_ready_help(error: FileNotFoundError) -> None:
    raw_images_count = len(list(RAW_IMAGES_DIR.glob("*"))) if RAW_IMAGES_DIR.exists() else 0
    raw_labels_count = len(list(RAW_LABELS_DIR.glob("*"))) if RAW_LABELS_DIR.exists() else 0
    raw_images_color = GREEN if raw_images_count > 0 else RED
    raw_labels_color = GREEN if raw_labels_count > 0 else RED
    for item in header("YOLO TRAINING :: DỮ LIỆU CHƯA SẴN SÀNG", color=RED):
        print(item)
    print(section("LÝ DO", RED))
    print(row("Lý do không chạy", str(error), RED, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("KIỂM TRA NHANH", YELLOW))
    print(row("Raw images", f"{RAW_IMAGES_DIR} ({raw_images_count} file)", raw_images_color, bounded=False))
    print(row("Raw labels", f"{RAW_LABELS_DIR} ({raw_labels_count} file)", raw_labels_color, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("CÁC BƯỚC CẦN LÀM", GREEN))
    print(row("Bước 1", f"Bỏ ảnh vào {RAW_IMAGES_DIR}" if raw_images_count == 0 else f"Đã có {raw_images_count} ảnh trong {RAW_IMAGES_DIR}", raw_images_color, bounded=False))
    print(row("Bước 2", f"Bỏ label vào {RAW_LABELS_DIR}" if raw_labels_count == 0 else f"Đã có {raw_labels_count} label trong {RAW_LABELS_DIR}", raw_labels_color, bounded=False))
    print(row("Bước 3", "Chạy training/validate_dataset.py", YELLOW))
    print(row("Bước 4", "Chạy training/split_dataset.py", YELLOW))
    print(row("Bước 5", "Chạy lại run_train.py", GREEN))
    print(line(rule("-"), CYAN))
    print(section("Ý NGHĨA LỆNH", CYAN))
    print(row("Lệnh này", "Đọc dataset đã split trong dataset/processed và bắt đầu huấn luyện.", YELLOW, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("LỆNH NHANH", CYAN))
    print(command_row(1, r".\.venv\Scripts\python training\validate_dataset.py"))
    print(command_row(2, r".\.venv\Scripts\python training\split_dataset.py"))
    print(command_row(3, r".\.venv\Scripts\python run_train.py"))
    print(line(rule("="), CYAN))


def main() -> None:
    ensure_project_directories()
    _auto_prepare_training_dataset()
    try:
        _ensure_training_dataset_ready()
    except FileNotFoundError as exc:
        _print_dataset_ready_help(exc)
        raise SystemExit(1)
    config = load_yaml("training/train_config.yaml")
    model_name = config["model"]
    yolo_cls = _require_yolo()
    try:
        model = yolo_cls(str(resolve_model_source(model_name)))
        results = model.train(**_training_kwargs(config))
    except Exception as exc:
        logger.warning("Primary training config failed: %s", exc)
        fallback_model = config.get("fallback_model", "yolo26n.pt")
        config["model"] = fallback_model
        config["imgsz"] = min(int(config["imgsz"]), 416)
        config["batch"] = min(int(config["batch"]), 4)
        model = yolo_cls(str(resolve_model_source(fallback_model)))
        results = model.train(**_training_kwargs(config))

    save_dir = Path(getattr(results, "save_dir", config["project"]))
    _copy_best_weight(save_dir)
    logger.info("Training completed. Output: %s", save_dir)


if __name__ == "__main__":
    main()
