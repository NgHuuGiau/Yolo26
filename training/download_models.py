from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
from utils.file_utils import ensure_project_directories


BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
MODEL_URLS = {
    "yolo26n.pt": f"{BASE_URL}/yolo26n.pt",
    "yolo26s.pt": f"{BASE_URL}/yolo26s.pt",
    "yolo26m.pt": f"{BASE_URL}/yolo26m.pt",
    "yolo26l.pt": f"{BASE_URL}/yolo26l.pt",
    "yolo26x.pt": f"{BASE_URL}/yolo26x.pt",
}
PRETRAINED_DIR = Path("models/pretrained")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tải model YOLO26 vào models/pretrained")
    parser.add_argument("--models", nargs="*", choices=sorted(MODEL_URLS), help="Chỉ tải các model được chỉ định.")
    parser.add_argument("--force", action="store_true", help="Tải lại ngay cả khi file đã tồn tại.")
    return parser.parse_args()


def download_models(model_names: list[str] | None = None, *, force: bool = False) -> tuple[list[str], list[str]]:
    ensure_project_directories()
    requested = model_names or list(MODEL_URLS)
    downloaded: list[str] = []
    skipped: list[str] = []
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    for model_name in requested:
        target = PRETRAINED_DIR / model_name
        if target.exists() and not force:
            skipped.append(model_name)
            continue
        urlretrieve(MODEL_URLS[model_name], target)
        downloaded.append(model_name)
    return downloaded, skipped


def main() -> None:
    args = _parse_args()
    requested = args.models or list(MODEL_URLS)
    for item in header("YOLO MODEL :: TẢI MODEL YOLO26"):
        print(item)
    print(section("MỤC TIÊU", GREEN))
    print(row("Thư mục", str(PRETRAINED_DIR), GREEN))
    print(row("Model yêu cầu", ", ".join(requested), YELLOW, bounded=False))
    print(line(rule("-"), CYAN))
    try:
        downloaded, skipped = download_models(requested, force=args.force)
    except Exception as exc:
        print(section("LỖI", RED))
        print(row("Lý do không chạy", str(exc), RED, bounded=False))
        print(line(rule("-"), CYAN))
        print(section("LỆNH THỬ", CYAN))
        print(command_row(1, r".\.venv\Scripts\python training\download_models.py --force"))
        print(line(rule("="), CYAN))
        raise

    print(section("KẾT QUẢ", GREEN if downloaded else YELLOW))
    print(row("Đã tải", ", ".join(downloaded) if downloaded else "Không có model nào được tải mới", GREEN if downloaded else YELLOW, bounded=False))
    print(row("Bỏ qua", ", ".join(skipped) if skipped else "Không có", YELLOW if skipped else GREEN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("LỆNH TIẾP", CYAN))
    print(command_row(1, r".\.venv\Scripts\python run_doctor.py"))
    print(command_row(2, r".\.venv\Scripts\python run_app.py"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
