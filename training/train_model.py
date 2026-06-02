from __future__ import annotations

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
    from ultralytics import YOLO
    ULTRALYTICS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc

from utils.file_utils import ensure_project_directories, load_yaml
from utils.logger import get_logger


logger = get_logger(__name__)
PROCESSED_TRAIN_DIR = Path("dataset/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/processed/images/val")


def _require_yolo():
    if YOLO is None:
        raise RuntimeError(f"Khong khoi tao duoc ultralytics/YOLO: {ULTRALYTICS_IMPORT_ERROR}")
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
            "Chua co anh trong dataset/processed/images/train. "
            "Hay bo du lieu vao dataset/raw va chay training/split_dataset.py truoc."
        )
    if not PROCESSED_VAL_DIR.exists() or not any(PROCESSED_VAL_DIR.iterdir()):
        raise FileNotFoundError(
            "Chua co anh trong dataset/processed/images/val. "
            "Hay bo du lieu vao dataset/raw va chay training/split_dataset.py truoc."
        )


def _copy_best_weight(run_dir: Path) -> None:
    best_weight = run_dir / "weights" / "best.pt"
    target = TRAINED_BEST_MODEL_PATH
    if best_weight.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weight, target)
        logger.info("Copied best.pt to %s", target)


def main() -> None:
    ensure_project_directories()
    _ensure_training_dataset_ready()
    config = load_yaml("training/train_config.yaml")
    model_name = config["model"]
    yolo_cls = _require_yolo()
    try:
        model = yolo_cls(str(resolve_model_source(model_name)))
        results = model.train(**_training_kwargs(config))
    except Exception as exc:
        logger.warning("Primary training config failed: %s", exc)
        fallback_model = config.get("fallback_model", "yolo11n.pt")
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
