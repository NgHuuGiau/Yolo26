from __future__ import annotations

import os
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[38;5;81m"
DIM = "\033[2m"
CARD_WIDTH = 88


def _ensure_utf8_console() -> None:
    try:
        if os.name == "nt":
            os.system("chcp 65001 > nul")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        return


_ensure_utf8_console()


def line(text: str = "", color: str = "") -> str:
    return f"{color}{text}{RESET}" if color else text


def pad(text: str, width: int = CARD_WIDTH) -> str:
    return text[:width].ljust(width)


def rule(char: str = "=") -> str:
    glyph = {
        "=": "\u2550",
        "-": "\u2500",
        ".": "\u00b7",
    }.get(char, char)
    return glyph * CARD_WIDTH


def section(title: str, color: str = CYAN) -> str:
    return line(pad(f"\u25c6 {title}"), BOLD + color)


def row(label: str, value: str = "", color: str = "", *, bounded: bool = True) -> str:
    content = f"\u2502 {label:<16} {value}".rstrip()
    return line(pad(content) if bounded else content, color)


def header(title: str, *, color: str = CYAN) -> list[str]:
    border = rule("=")[:-2]
    return [
        line(f"\u2554{border}\u2557", color),
        line(f"\u2551 {pad(title, CARD_WIDTH - 4)} \u2551", BOLD + color),
        line(f"\u255a{border}\u255d", color),
    ]


def command_row(index: int, command: str) -> str:
    return row(f"L\u1ec7nh {index}", command, BLUE if index == 1 else CYAN, bounded=False)
