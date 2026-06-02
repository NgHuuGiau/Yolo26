from __future__ import annotations

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from utils.file_utils import ensure_project_directories


DATASET_READY_LINES = (
    "Dataset folders are ready.",
    "Raw images  : dataset/raw/images",
    "Raw labels  : dataset/raw/labels",
    "Split output: dataset/processed/images and dataset/processed/labels",
    "Sample data : dataset/sample/images and dataset/sample/labels",
)


def main() -> None:
    ensure_project_directories()
    for line in DATASET_READY_LINES:
        print(line)


if __name__ == "__main__":
    main()
