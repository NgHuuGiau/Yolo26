from pathlib import Path


def main() -> None:
    for relative_path in [
        "dataset/raw/images",
        "dataset/raw/labels",
        "dataset/processed/images/train",
        "dataset/processed/images/val",
        "dataset/processed/images/test",
        "dataset/processed/labels/train",
        "dataset/processed/labels/val",
        "dataset/processed/labels/test",
    ]:
        Path(relative_path).mkdir(parents=True, exist_ok=True)
    print("Dataset folders are ready.")


if __name__ == "__main__":
    main()
