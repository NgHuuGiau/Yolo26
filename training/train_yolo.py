from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

from utils.file_utils import load_yaml
from utils.logger import get_logger


logger = get_logger(__name__)


def _copy_best_weight(run_dir: Path) -> None:
    best_weight = run_dir / "weights" / "best.pt"
    target = Path("models/trained/best.pt")
    if best_weight.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weight, target)
        logger.info("Copied best.pt to %s", target)


def main() -> None:
    config = load_yaml("training/train_config.yaml")
    model_name = config["model"]
    try:
        model = YOLO(model_name)
        results = model.train(**config)
    except Exception as exc:
        logger.warning("Primary training config failed: %s", exc)
        fallback_model = config.get("fallback_model", "yolo11n.pt")
        config["model"] = fallback_model
        config["imgsz"] = min(int(config["imgsz"]), 416)
        config["batch"] = min(int(config["batch"]), 4)
        model = YOLO(fallback_model)
        results = model.train(**config)

    save_dir = Path(getattr(results, "save_dir", config["project"]))
    _copy_best_weight(save_dir)
    logger.info("Training completed. Output: %s", save_dir)


if __name__ == "__main__":
    main()
