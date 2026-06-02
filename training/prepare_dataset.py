from pathlib import Path


def main() -> None:
    ensure_project_directories()
    print("Dataset folders are ready.")
    print("Raw images  : dataset/raw/images")
    print("Raw labels  : dataset/raw/labels")
    print("Split output: dataset/processed/images and dataset/processed/labels")
    print("Sample data : dataset/sample/images and dataset/sample/labels")


if __name__ == "__main__":
    main()
