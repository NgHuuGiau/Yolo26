from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def resolve_validation_model_path() -> Path:
    trained_model_path = Path("models/trained/best.pt")
    if trained_model_path.exists():
        return trained_model_path
    return Path("yolo11n.pt")


def main() -> None:
    model_path = resolve_validation_model_path()
    model = YOLO(str(model_path))
    metrics = model.val(data="training/data.yaml", project="runs/val", name="validation")
    print(metrics)


if __name__ == "__main__":
    main()
