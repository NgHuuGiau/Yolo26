from __future__ import annotations

import random
import shutil
from pathlib import Path

from utils.file_utils import ensure_project_directories


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DATASET_RANDOM_SEED = 42
RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")
PROCESSED_IMAGES_DIR = Path("dataset/processed/images")
PROCESSED_LABELS_DIR = Path("dataset/processed/labels")


def _copy_pair(image_path: Path, split: str) -> None:
    label_path = RAW_LABELS_DIR / f"{image_path.stem}.txt"
    dst_image = PROCESSED_IMAGES_DIR / split / image_path.name
    dst_label = PROCESSED_LABELS_DIR / split / label_path.name
    shutil.copy2(image_path, dst_image)
    if label_path.exists():
        shutil.copy2(label_path, dst_label)


def _load_raw_images() -> list[Path]:
    images = [path for path in RAW_IMAGES_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort(key=lambda path: path.name.lower())
    random.Random(DATASET_RANDOM_SEED).shuffle(images)
    return images


def main() -> None:
    ensure_project_directories()
    raw_images = _load_raw_images()
    if not raw_images:
        print("Khong co anh trong dataset/raw/images")
        return

    total = len(raw_images)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    splits = {
        "train": raw_images[:train_end],
        "val": raw_images[train_end:val_end],
        "test": raw_images[val_end:],
    }
    for split, files in splits.items():
        for file_path in files:
            _copy_pair(file_path, split)
    print(f"Da chia dataset: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()
