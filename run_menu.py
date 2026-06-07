from __future__ import annotations

import os
import subprocess
import sys

from training.terminal_ui import CYAN, GREEN, RED, YELLOW, header, line, row, rule, section


MENU_OPTIONS = {
    "1": ("run_app.py", "Mở camera app chính"),
    "2": ("run_detect.py", "Mở detect camera"),
    "3": ("run_tools.py", "Xem cấu hình máy và 3 mức tối ưu"),
    "4": ("run_train.py", "Chạy huấn luyện"),
    "5": ("run_tests.py", "Chạy toàn bộ test"),
    "6": ("run_doctor.py", "Kiểm tra toàn hệ thống"),
    "0": ("", "Thoát"),
}


def _render_menu(print_fn=print) -> None:
    for item in header("YOLO HUB :: ĐIỀU HƯỚNG TOÀN BỘ DỰ ÁN"):
        print_fn(item)
    print_fn(section("LỐI VÀO CHÍNH", GREEN))
    for key in ("1", "2", "3", "4", "5", "6"):
        script_name, description = MENU_OPTIONS[key]
        color = GREEN if key in {"1", "2", "3"} else YELLOW
        print_fn(row(f"{key} | {script_name}", description, color, bounded=False))
    print_fn(line(rule("-"), CYAN))
    print_fn(section("THOÁT", YELLOW))
    print_fn(row("0 | Thoát", "Đóng menu ngay tại đây.", YELLOW, bounded=False))
    print_fn(line(rule("="), CYAN))


def _clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _run_script(script_name: str) -> int:
    return subprocess.call([sys.executable, script_name])


def main(input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> int:
    while True:
        _render_menu(print_fn=print_fn)
        choice = input_fn("Nhập lựa chọn của bạn (0/1/2/3/4/5/6): ").strip()
        if choice == "0":
            print_fn(line("Đã thoát menu.", YELLOW))
            return 0
        option = MENU_OPTIONS.get(choice)
        if option is None:
            print_fn(line("Lựa chọn không hợp lệ. Hãy nhập lại.", RED))
            continue
        script_name, description = option
        clear_terminal_fn()
        print_fn(line(f"Đang chạy: {script_name} - {description}", CYAN))
        exit_code = run_script_fn(script_name)
        if exit_code == 0:
            print_fn(line(f"Đã chạy xong {script_name}. Quay lại menu.", GREEN))
        else:
            print_fn(line(f"{script_name} kết thúc với mã {exit_code}. Quay lại menu.", YELLOW))


if __name__ == "__main__":
    raise SystemExit(main())
