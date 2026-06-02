from __future__ import annotations

from dataclasses import dataclass
import random
import shutil
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from utils.file_utils import ensure_project_directories


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DATASET_RANDOM_SEED = 42
RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")
PROCESSED_IMAGES_DIR = Path("dataset/processed/images")
PROCESSED_LABELS_DIR = Path("dataset/processed/labels")
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class DatasetAudit:
    raw_image_count: int
    eligible_images: list[Path]
    missing_labels: list[Path]
    invalid_labels: list[tuple[Path, str]]
    orphan_labels: list[Path]
    empty_labels: list[Path]


def print_audit_summary(audit: DatasetAudit, *, detail_limit: int = 5) -> None:
    print(f"Tong anh raw: {audit.raw_image_count}")
    print(f"Anh hop le de train: {len(audit.eligible_images)}")
    print(f"Anh thieu label: {len(audit.missing_labels)}")
    print(f"Label rong: {len(audit.empty_labels)}")
    print(f"Label loi: {len(audit.invalid_labels)}")
    print(f"Label mo coi: {len(audit.orphan_labels)}")

    for image_path in audit.missing_labels[:detail_limit]:
        print(f"Bo qua anh thieu label: {image_path.name}")
    for label_path, issue in audit.invalid_labels[:detail_limit]:
        print(f"Bo qua label loi: {label_path.name} -> {issue}")
    for label_path in audit.orphan_labels[:detail_limit]:
        print(f"Label mo coi: {label_path.name}")


def _label_path_for(image_path: Path) -> Path:
    return RAW_LABELS_DIR / f"{image_path.stem}.txt"


def _copy_pair(image_path: Path, split: str) -> None:
    label_path = _label_path_for(image_path)
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


def _load_raw_labels() -> list[Path]:
    labels = [path for path in RAW_LABELS_DIR.iterdir() if path.suffix.lower() == ".txt"]
    labels.sort(key=lambda path: path.name.lower())
    return labels


def _validate_label_file(label_path: Path) -> tuple[bool, str]:
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return True, "empty"

    for line_number, line in enumerate(content.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            return False, f"line {line_number}: expected 5 values, got {len(parts)}"
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            return False, f"line {line_number}: non-numeric value"
        if class_id < 0:
            return False, f"line {line_number}: class_id must be >= 0"
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            return False, f"line {line_number}: coordinates must be in [0, 1]"
    return True, "ok"


def audit_raw_dataset() -> DatasetAudit:
    images = _load_raw_images()
    labels = _load_raw_labels()
    label_map = {label.stem: label for label in labels}
    missing_labels: list[Path] = []
    invalid_labels: list[tuple[Path, str]] = []
    empty_labels: list[Path] = []
    eligible_images: list[Path] = []

    for image_path in images:
        label_path = label_map.get(image_path.stem)
        if label_path is None:
            missing_labels.append(image_path)
            continue
        is_valid, status = _validate_label_file(label_path)
        if not is_valid:
            invalid_labels.append((label_path, status))
            continue
        if status == "empty":
            empty_labels.append(label_path)
        eligible_images.append(image_path)

    image_stems = {image_path.stem for image_path in images}
    orphan_labels = [label_path for label_path in labels if label_path.stem not in image_stems]
    return DatasetAudit(
        raw_image_count=len(images),
        eligible_images=eligible_images,
        missing_labels=missing_labels,
        invalid_labels=invalid_labels,
        orphan_labels=orphan_labels,
        empty_labels=empty_labels,
    )


def _reset_processed_directories() -> None:
    for split in SPLIT_NAMES:
        image_dir = PROCESSED_IMAGES_DIR / split
        label_dir = PROCESSED_LABELS_DIR / split
        if image_dir.exists():
            shutil.rmtree(image_dir)
        if label_dir.exists():
            shutil.rmtree(label_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)


def _build_splits(images: list[Path]) -> dict[str, list[Path]]:
    total = len(images)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def main() -> None:
    ensure_project_directories()
    audit = audit_raw_dataset()
    if audit.raw_image_count == 0:
        print("Khong co anh trong dataset/raw/images")
        return

    print_audit_summary(audit)

    if not audit.eligible_images:
        print("Khong co anh hop le nao de chia dataset")
        return

    _reset_processed_directories()
    splits = _build_splits(audit.eligible_images)
    for split, files in splits.items():
        for file_path in files:
            _copy_pair(file_path, split)
    print(f"Da chia dataset: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()
