from utils.file_utils import ensure_project_directories


def main() -> None:
    ensure_project_directories()
    print("Dataset folders are ready.")


if __name__ == "__main__":
    main()
