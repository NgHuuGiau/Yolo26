from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def resolve_export_model_path() -> Path:
    model_path = Path("models/trained/best.pt")
    if not model_path.exists():
        raise FileNotFoundError("Khong tim thay models/trained/best.pt")
    return model_path


def main() -> None:
    model_path = resolve_export_model_path()
    model = YOLO(str(model_path))
    model.export(format="onnx")
    print("Export model xong.")


if __name__ == "__main__":
    main()
