from __future__ import annotations

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from training.terminal_ui import CYAN, GREEN, YELLOW, header, line, row, rule, section
from utils.file_utils import ensure_project_directories


def main() -> None:
    ensure_project_directories()
    for item in header("YOLO DATASET :: CHUẨN BỊ THƯ MỤC"):
        print(item)
    print(section("TRẠNG THÁI", GREEN))
    print(row("Kết quả", "Đã tạo sẵn các thư mục dataset.", GREEN))
    print(line(rule("-"), CYAN))
    print(section("ĐƯỜNG DẪN", YELLOW))
    print(row("Raw images", "dataset/raw/images"))
    print(row("Raw labels", "dataset/raw/labels"))
    print(row("Split output", "dataset/processed/images and dataset/processed/labels", bounded=False))
    print(line(rule("-"), CYAN))
    print(section("Ý NGHĨA LỆNH", CYAN))
    print(row("Lệnh này", "Chỉ tạo sẵn thư mục, chưa train và chưa split.", YELLOW, bounded=False))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
