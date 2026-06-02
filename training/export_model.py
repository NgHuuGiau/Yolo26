from __future__ import annotations

from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.model_paths import resolve_trained_model_path
except ModuleNotFoundError:
    from model_paths import resolve_trained_model_path

try:
    from ultralytics import YOLO
    ULTRALYTICS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc


def _require_yolo():
    if YOLO is None:
        raise RuntimeError(f"Khong khoi tao duoc ultralytics/YOLO: {ULTRALYTICS_IMPORT_ERROR}")
    return YOLO


def resolve_export_model_path():
    return resolve_trained_model_path(required=True)


def main() -> None:
    model_path = resolve_export_model_path()
    model = _require_yolo()(str(model_path))
    model.export(format="onnx")
    print("Export model xong.")


if __name__ == "__main__":
    main()
