from __future__ import annotations

import importlib
from pathlib import Path

try:
    from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
except ModuleNotFoundError:
    from terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section


def require_yolo(current, current_error):
    if current is None and current_error is None:
        try:
            current = importlib.import_module("ultralytics").YOLO
        except Exception as exc:  # pragma: no cover
            current_error = exc
    if current is None:
        raise RuntimeError(f"Không khởi tạo được ultralytics/YOLO: {current_error}")
    return current, current_error


def count_files(path: Path) -> int:
    return len(list(path.glob("*"))) if path.exists() else 0


def print_help_screen(
    *,
    title: str,
    reason: str,
    checks: list[tuple[str, str, str]],
    steps: list[tuple[str, str, str]],
    meaning: str,
    commands: list[str],
) -> None:
    for item in header(title, color=RED):
        print(item)
    print(section("LÝ DO", RED))
    print(row("Lý do không chạy", reason, RED, bounded=False))
    print(line(rule("-"), CYAN))
    if checks:
        print(section("KIỂM TRA NHANH", YELLOW))
        for label, value, color in checks:
            print(row(label, value, color, bounded=False))
        print(line(rule("-"), CYAN))
    print(section("CÁC BƯỚC CẦN LÀM", GREEN))
    for label, value, color in steps:
        print(row(label, value, color, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("Ý NGHĨA LỆNH", CYAN))
    print(row("Lệnh này", meaning, YELLOW, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("LỆNH NHANH", CYAN))
    for index, command in enumerate(commands, start=1):
        print(command_row(index, command))
    print(line(rule("="), CYAN))
