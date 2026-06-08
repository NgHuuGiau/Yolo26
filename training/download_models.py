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
    "yolo11n.pt": f"{BASE_URL}/yolo11n.pt",
    "yolo11s.pt": f"{BASE_URL}/yolo11s.pt",
    "yolo11m.pt": f"{BASE_URL}/yolo11m.pt",
    "yolo11l.pt": f"{BASE_URL}/yolo11l.pt",
    "yolo11x.pt": f"{BASE_URL}/yolo11x.pt",
}
PRETRAINED_DIR = Path("models/pretrained")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YOLO11 models to models/pretrained")
    parser.add_argument("--models", nargs="*", choices=sorted(MODEL_URLS), help="Download only specified models.")
    parser.add_argument("--force", action="store_true", help="Redownload even if file exists.")
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
    for item in header("YOLO MODEL :: DOWNLOAD YOLO11"):
        print(item)
    print(section("TARGET", GREEN))
    print(row("Directory", str(PRETRAINED_DIR), GREEN))
    print(row("Models requested", ", ".join(requested), YELLOW, bounded=False))
    print(line(rule("-"), CYAN))
    try:
        downloaded, skipped = download_models(requested, force=args.force)
    except Exception as exc:
        print(section("ERROR", RED))
        print(row("Reason cannot run", str(exc), RED, bounded=False))
        print(line(rule("-"), CYAN))
        print(section("TRY COMMANDS", CYAN))
        print(command_row(1, r".\.venv\Scripts\python training\download_models.py --force"))
        print(line(rule("="), CYAN))
        raise

    print(section("RESULT", GREEN if downloaded else YELLOW))
    print(row("Downloaded", ", ".join(downloaded) if downloaded else "No new models downloaded", GREEN if downloaded else YELLOW, bounded=False))
    print(row("Skipped", ", ".join(skipped) if skipped else "None", YELLOW if skipped else GREEN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("NEXT COMMANDS", CYAN))
    print(command_row(1, r".\.venv\Scripts\python run_doctor.py"))
    print(command_row(2, r".\.venv\Scripts\python run_app.py"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
