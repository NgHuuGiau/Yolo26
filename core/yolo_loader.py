from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from ultralytics import YOLO

from core.model_selector import RuntimeConfig
from utils.file_utils import load_yaml_cached
from utils.logger import get_logger


logger = get_logger(__name__)
MODEL_CONFIG_PATH = Path("config/model_config.yaml")


@dataclass
class LoadedModel:
    model: YOLO
    model_name: str
    source_path: str


def _candidate_paths(model_name: str) -> list[str]:
    trained_path = Path("models/trained/best.pt")
    pretrained_path = Path("models/pretrained") / model_name
    local_root_path = Path(model_name)
    configured_priority = load_yaml_cached(str(MODEL_CONFIG_PATH)).get("priority_order", [])
    candidates: list[str] = []

    for configured_item in configured_priority:
        configured_str = str(configured_item)
        if configured_str == "models/trained/best.pt" and trained_path.exists():
            candidates.append(str(trained_path))
        elif configured_str == f"models/pretrained/{model_name}" and pretrained_path.exists():
            candidates.append(str(pretrained_path))
        elif configured_str == model_name and local_root_path.exists():
            candidates.append(str(local_root_path))

    if trained_path.exists():
        candidates.append(str(trained_path))
    if pretrained_path.exists():
        candidates.append(str(pretrained_path))
    if local_root_path.exists():
        candidates.append(str(local_root_path))
    return list(dict.fromkeys(candidates))


def load_yolo_model(runtime: RuntimeConfig) -> Tuple[LoadedModel, str]:
    errors = []
    for model_name in runtime.candidate_models:
        candidate_paths = _candidate_paths(model_name)
        if not candidate_paths:
            errors.append(f"{model_name}: khong tim thay file model local")
            logger.warning("No local file found for model %s", model_name)
            continue
        for candidate in candidate_paths:
            try:
                logger.info("Trying model candidate: %s", candidate)
                model = YOLO(candidate)
                runtime.active_model_name = model_name
                return LoadedModel(model=model, model_name=model_name, source_path=candidate), runtime.resolved_device
            except Exception as exc:  # pragma: no cover
                errors.append(f"{candidate}: {exc}")
                logger.warning("Failed to load %s: %s", candidate, exc)
    raise RuntimeError(
        "Khong the load bat ky model local nao. "
        "Hay kiem tra models/pretrained hoac models/trained.\n" + "\n".join(errors)
    )
