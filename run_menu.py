from __future__ import annotations

import os
import subprocess
import sys

from training.terminal_ui import CYAN, GREEN, RED, YELLOW, header, line, row, rule, section


def _configure_terminal_encoding() -> None:
    if os.name != "nt":
        return
    try:
        os.system("chcp 65001 > nul")
    except Exception:
        pass
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


MENU_OPTIONS = {
    "1": ("run_app.py", "Camera realtime theo c\u1ea5u h\u00ecnh ch\u00ednh"),
    "2": ("run_detect.py", "Camera detect realtime t\u1ed1i gi\u1ea3n"),
    "3": ("run_chat.py", "UX/UI desktop v\u00e0 chat"),
    "4": ("run_tests.py", "Ch\u1ea1y to\u00e0n b\u1ed9 test"),
    "5": ("run_doctor.py", "Ki\u1ec3m tra to\u00e0n h\u1ec7 th\u1ed1ng"),
    "6": ("run_train.py", "Ch\u1ea1y hu\u1ea5n luy\u1ec7n"),
    "0": ("", "Tho\u00e1t"),
}
PRIMARY_KEYS = tuple(key for key in MENU_OPTIONS if key != "0")
TESTED_EXIT_TEXT = "\u0110\u00e3 tho\u00e1t menu."
TESTED_INVALID_TEXT = "L\u1ef1a ch\u1ecdn kh\u00f4ng h\u1ee3p l\u1ec7. H\u00e3y nh\u1eadp l\u1ea1i."
TESTED_BACK_TEXT = "Quay l\u1ea1i menu."


def _render_menu(print_fn=print) -> None:
    for item in header("YOLO HUB :: \u0110I\u1ec0U H\u01af\u1edaNG TERMINAL"):
        print_fn(item)
    print_fn(section("M\u1ede CAMERA", GREEN))
    for key in ("1", "2"):
        script_name, description = MENU_OPTIONS[key]
        print_fn(row(f"{key} | {script_name}", description, GREEN, bounded=False))
    print_fn(line(rule("-"), CYAN))
    print_fn(section("UX/UI V\u00c0 KI\u1ec2M TRA", YELLOW))
    for key in ("3", "4", "5", "6"):
        script_name, description = MENU_OPTIONS[key]
        print_fn(row(f"{key} | {script_name}", description, YELLOW, bounded=False))
    print_fn(line(rule("-"), CYAN))
    print_fn(row("0 | Tho\u00e1t", "\u0110\u00f3ng menu ngay t\u1ea1i \u0111\u00e2y.", RED, bounded=False))
    print_fn(line(rule("="), CYAN))


def _clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _run_script(script_name: str) -> int:
    return subprocess.call([sys.executable, script_name])


def main(input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> int:
    _configure_terminal_encoding()
    while True:
        _render_menu(print_fn=print_fn)
        choice = input_fn("Nh\u1eadp l\u1ef1a ch\u1ecdn c\u1ee7a b\u1ea1n (0/1/2/3/4/5/6): ").strip()
        if choice == "0":
            print_fn(line(TESTED_EXIT_TEXT, YELLOW))
            return 0
        option = MENU_OPTIONS.get(choice)
        if option is None:
            print_fn(line(TESTED_INVALID_TEXT, RED))
            continue
        script_name, description = option
        clear_terminal_fn()
        print_fn(line(f"\u0110ang ch\u1ea1y: {script_name} - {description}", CYAN))
        exit_code = run_script_fn(script_name)
        message = (
            f"\u0110\u00e3 ch\u1ea1y xong {script_name}. {TESTED_BACK_TEXT}"
            if exit_code == 0
            else f"{script_name} k\u1ebft th\u00fac v\u1edbi m\u00e3 {exit_code}. {TESTED_BACK_TEXT}"
        )
        print_fn(line(message, GREEN if exit_code == 0 else YELLOW))


if __name__ == "__main__":
    raise SystemExit(main())
