from __future__ import annotations

from pathlib import Path

from utils.file_utils import load_yaml, save_yaml


TRAINED_BEST_MODEL_PATH = Path("models/trained/best.pt")
PRETRAINED_MODELS_DIR = Path("models/pretrained")
TRAINING_DATA_CONFIG_PATH = Path("training/data.yaml")
GENERATED_DATA_CONFIG_PATH = Path("training/.generated_data.yaml")


def resolve_model_source(model_name: str | Path) -> Path:
    model_path = Path(model_name)
    if model_path.exists():
        return model_path
    pretrained_path = PRETRAINED_MODELS_DIR / model_path.name
    if pretrained_path.exists():
        return pretrained_path
    return model_path


def resolve_trained_model_path(*, required: bool, fallback: str | None = None) -> Path:
    if TRAINED_BEST_MODEL_PATH.exists():
        return TRAINED_BEST_MODEL_PATH
    if fallback is not None:
        return resolve_model_source(fallback)
    if required:
        raise FileNotFoundError(f"Khong tim thay {TRAINED_BEST_MODEL_PATH}")
    return TRAINED_BEST_MODEL_PATH


def resolve_data_config_path() -> Path:
    config = load_yaml(TRAINING_DATA_CONFIG_PATH)
    dataset_root = (TRAINING_DATA_CONFIG_PATH.parent / config["path"]).resolve()
    normalized = dict(config)
    normalized["path"] = str(dataset_root)
    save_yaml(GENERATED_DATA_CONFIG_PATH, normalized)
    return GENERATED_DATA_CONFIG_PATH
