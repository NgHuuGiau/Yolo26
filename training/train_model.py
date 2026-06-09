from __future__ import annotations

import shutil
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.auto_label_raw import auto_label_raw_images
    from training.common import GREEN, RED, YELLOW, count_files, print_help_screen, require_yolo
    from training.model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source
    from training.split_dataset import _copy_split, _reset_processed_dirs, _split_items, audit_raw_dataset
except ModuleNotFoundError:
    from auto_label_raw import auto_label_raw_images
    from common import GREEN, RED, YELLOW, count_files, print_help_screen, require_yolo
    from model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source
    from split_dataset import _copy_split, _reset_processed_dirs, _split_items, audit_raw_dataset

from utils.file_utils import ensure_project_directories, load_yaml
from utils.logger import get_logger


logger = get_logger(__name__)
YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
PROCESSED_TRAIN_DIR = Path("dataset/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/processed/images/val")
RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    YOLO, ULTRALYTICS_IMPORT_ERROR = require_yolo(YOLO, ULTRALYTICS_IMPORT_ERROR)
    return YOLO


def _training_kwargs(config: dict) -> dict:
    kwargs = {key: value for key, value in config.items() if key != "fallback_model"}
    if "project" in kwargs:
        kwargs["project"] = str(Path(kwargs["project"]).resolve())
    if "data" in kwargs:
        kwargs["data"] = str(resolve_data_config_path())
    return kwargs


def _ensure_training_dataset_ready() -> None:
    required_dirs = (
        (PROCESSED_TRAIN_DIR, "dataset/processed/images/train"),
        (PROCESSED_VAL_DIR, "dataset/processed/images/val"),
    )
    for directory, label in required_dirs:
        if not directory.exists() or not any(directory.iterdir()):
            raise FileNotFoundError(
                f"Chua co anh trong {label}. Hay bo du lieu vao dataset/raw va chay training/split_dataset.py truoc."
            )


def _auto_prepare_training_dataset() -> dict[str, object]:
    ensure_project_directories()
    report = {"raw_images": 0, "auto_labeled": 0, "eligible": 0, "no_detection": []}
    audit = audit_raw_dataset()
    report["raw_images"] = audit.raw_image_count
    if not audit.raw_image_count:
        return report
    if audit.missing_labels:
        auto_report = auto_label_raw_images(overwrite=False, conf=0.25, device="cpu")
        report["auto_labeled"] = int(auto_report.get("generated", 0))
        report["no_detection"] = list(auto_report.get("no_detection", []))
        audit = audit_raw_dataset()
    eligible = audit.eligible
    report["eligible"] = len(eligible)
    if not eligible:
        return report
    _reset_processed_dirs()
    for split_name, items in _split_items(eligible).items():
        _copy_split(split_name, items)
    return report


def _copy_best_weight(run_dir: Path) -> None:
    best_weight = run_dir / "weights" / "best.pt"
    if best_weight.exists():
        TRAINED_BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weight, TRAINED_BEST_MODEL_PATH)
        logger.info("Copied best.pt to %s", TRAINED_BEST_MODEL_PATH)


def _print_dataset_ready_help(error: FileNotFoundError) -> None:
    raw_images_count = count_files(RAW_IMAGES_DIR)
    raw_labels_count = count_files(RAW_LABELS_DIR)
    raw_images_color = GREEN if raw_images_count > 0 else RED
    raw_labels_color = GREEN if raw_labels_count > 0 else RED
    print_help_screen(
        title="YOLO TRAINING :: DU LIEU CHUA SAN SANG",
        reason=str(error),
        checks=[
            ("Raw images", f"{RAW_IMAGES_DIR} ({raw_images_count} file)", raw_images_color),
            ("Raw labels", f"{RAW_LABELS_DIR} ({raw_labels_count} file)", raw_labels_color),
        ],
        steps=[
            ("Buoc 1", f"Bo anh vao {RAW_IMAGES_DIR}" if raw_images_count == 0 else f"Da co {raw_images_count} anh trong {RAW_IMAGES_DIR}", raw_images_color),
            ("Buoc 2", f"Bo label vao {RAW_LABELS_DIR}" if raw_labels_count == 0 else f"Da co {raw_labels_count} label trong {RAW_LABELS_DIR}", raw_labels_color),
            ("Buoc 3", "Chay training/validate_dataset.py", YELLOW),
            ("Buoc 4", "Chay training/split_dataset.py", YELLOW),
            ("Buoc 5", "Chay lai run_train.py", GREEN),
        ],
        meaning="Doc dataset da split trong dataset/processed va bat dau huan luyen.",
        commands=[
            r".\.venv\Scripts\python training\validate_dataset.py",
            r".\.venv\Scripts\python training\split_dataset.py",
            r".\.venv\Scripts\python run_train.py",
        ],
    )


def main() -> None:
    ensure_project_directories()
    _auto_prepare_training_dataset()
    try:
        _ensure_training_dataset_ready()
    except FileNotFoundError as exc:
        _print_dataset_ready_help(exc)
        raise SystemExit(1)
    config = load_yaml("training/train_config.yaml")
    yolo_cls = _require_yolo()
    model_name = config["model"]
    try:
        results = yolo_cls(str(resolve_model_source(model_name))).train(**_training_kwargs(config))
    except Exception as exc:
        logger.warning("Primary training config failed: %s", exc)
        fallback_model = config.get("fallback_model", "yolo11n.pt")
        config.update(model=fallback_model, imgsz=min(int(config["imgsz"]), 416), batch=min(int(config["batch"]), 4))
        results = yolo_cls(str(resolve_model_source(fallback_model))).train(**_training_kwargs(config))
    save_dir = Path(getattr(results, "save_dir", config["project"]))
    _copy_best_weight(save_dir)
    logger.info("Training completed. Output: %s", save_dir)


if __name__ == "__main__":
    main()
