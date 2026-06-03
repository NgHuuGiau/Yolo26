from __future__ import annotations

import shutil
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
from utils.file_utils import ensure_project_directories


SAMPLE_IMAGES_DIR = Path("dataset/sample/images")
SAMPLE_LABELS_DIR = Path("dataset/sample/labels")
RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def promote_samples(*, overwrite: bool = False) -> dict[str, int]:
    ensure_project_directories()
    moved = 0
    skipped = 0
    missing_labels = 0
    images = sorted(path for path in SAMPLE_IMAGES_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS) if SAMPLE_IMAGES_DIR.exists() else []
    for image_path in images:
        label_path = SAMPLE_LABELS_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        raw_image = RAW_IMAGES_DIR / image_path.name
        raw_label = RAW_LABELS_DIR / label_path.name
        if (raw_image.exists() or raw_label.exists()) and not overwrite:
            skipped += 1
            continue
        shutil.copy2(image_path, raw_image)
        shutil.copy2(label_path, raw_label)
        moved += 1
    return {
        "sample_images": len(images),
        "moved": moved,
        "skipped": skipped,
        "missing_labels": missing_labels,
    }


def main() -> None:
    report = promote_samples()
    for item in header("YOLO SAMPLE :: CHUYỂN MẪU SANG RAW"):
        print(item)
    print(section("KẾT QUẢ", GREEN if report["moved"] else YELLOW))
    print(row("Ảnh sample", str(report["sample_images"]), GREEN if report["sample_images"] else YELLOW))
    print(row("Đã chuyển", str(report["moved"]), GREEN if report["moved"] else YELLOW))
    print(row("Bỏ qua", str(report["skipped"]), YELLOW if report["skipped"] else GREEN))
    print(row("Thiếu label", str(report["missing_labels"]), RED if report["missing_labels"] else GREEN))
    print(line(rule("-"), CYAN))
    print(section("Ý NGHĨA", CYAN))
    print(row("Lệnh này", "Copy ảnh + label từ dataset/sample sang dataset/raw để chuẩn bị train.", YELLOW, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("LỆNH TIẾP", CYAN))
    print(command_row(1, r".\.venv\Scripts\python training\validate_dataset.py"))
    print(command_row(2, r".\.venv\Scripts\python training\split_dataset.py"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
