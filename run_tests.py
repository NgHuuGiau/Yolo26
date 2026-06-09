from __future__ import annotations

import argparse
import contextlib
import io
import logging
import math
import sys
import time
import unittest
from dataclasses import dataclass

from utils.file_utils import ensure_project_directories


def _ensure_utf8_console() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        return


_ensure_utf8_console()


RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
ORANGE = "\033[38;5;208m"
CARD_WIDTH = 88


def color(text: str, style: str = "") -> str:
    return f"{style}{text}{RESET}" if style else text


def pad(text: str, width: int = CARD_WIDTH) -> str:
    trimmed = text[:width]
    return trimmed + (" " * max(0, width - len(trimmed)))


def rule(char: str = "=") -> str:
    return char * CARD_WIDTH


def section(title: str, style: str = CYAN) -> str:
    return color(pad(f"[ {title} ]"), BOLD + style)


def meter(current: int, total: int, width: int = 18, filled_char: str = "█", empty_char: str = "·") -> str:
    if total <= 0:
        return " ".join([empty_char] * width)
    filled = min(width, math.ceil((current / total) * width))
    cells = [filled_char] * filled + [empty_char] * (width - filled)
    return " ".join(cells)


def status_meter(current: int, total: int, ok_style: str, hot_style: str | None = None) -> tuple[str, str]:
    hot_style = hot_style or ok_style
    text = meter(current, total)
    style = hot_style if current else ok_style
    return text, style


@dataclass
class CameraCheckResult:
    level: str
    summary: str
    detail: str

    @property
    def style(self) -> str:
        return {"PASS": GREEN, "WARN": YELLOW, "ERROR": RED}.get(self.level, CYAN)

    @property
    def ok(self) -> bool:
        return self.level == "PASS"


def _open_camera_capture(index: int):
    import cv2
    # Sử dụng CAP_DSHOW để khởi động camera nhanh và ổn định hơn trên Windows
    return cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)


def check_camera(index: int = 0, attempts: int = 3) -> CameraCheckResult:
    try:
        capture = _open_camera_capture(index)
    except Exception as exc:
        return CameraCheckResult(
            level="ERROR",
            summary=f"Camera thật       ERROR | Không tạo được camera index {index}",
            detail=f"Lý do không chạy   {exc}",
        )

    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        return CameraCheckResult(
            level="WARN",
            summary=f"Camera thật       WARN  | Không mở được camera index {index}",
            detail="Lý do không chạy   Camera không sẵn sàng, đang bị app khác chiếm hoặc không có webcam.",
        )

    frame_width = 0
    frame_height = 0
    got_frame = False
    try:
        for _ in range(max(1, attempts)):
            success, frame = capture.read()
            if success and frame is not None:
                frame_height, frame_width = frame.shape[:2]
                got_frame = True
                break
    finally:
        capture.release()

    if not got_frame:
        return CameraCheckResult(
            level="WARN",
            summary=f"Camera thật       WARN  | Mở được camera index {index} nhưng không đọc được frame",
            detail="Lý do không chạy   Webcam mở được nhưng không trả về khung hình hợp lệ.",
        )

    return CameraCheckResult(
        level="PASS",
        summary=f"Camera thật       PASS  | Đọc frame thành công tại index {index}",
        detail=f"Chi tiết          {frame_width}x{frame_height}",
    )


class PrettyTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity, total_tests: int = 0):
        super().__init__(stream, descriptions, verbosity)
        self.test_start_time = 0.0
        self.test_outputs: dict[unittest.case.TestCase, str] = {}
        self._stdout_capture: io.StringIO | None = None
        self._stderr_capture: io.StringIO | None = None
        self._stdout_redirect = None
        self._stderr_redirect = None
        self.total_tests = total_tests
        self.current_test = 0
        self.passed_count = 0
        self.failed_count = 0
        self.error_count = 0
        self.skipped_count = 0
        self._last_module = ""

    def _progress_label(self) -> str:
        width = max(2, len(str(self.total_tests or 0)))
        return f"{self.current_test:0{width}d}/{self.total_tests:0{width}d}"

    def _write_module_header(self, test) -> None:
        module_name = test.__class__.__module__.split(".")[-1]
        if module_name == self._last_module:
            return
        self._last_module = module_name
        self.stream.write("\n")
        self.stream.write(section(f"MODULE :: {module_name}", YELLOW) + "\n")
        self.stream.flush()

    def startTest(self, test):
        super().startTest(test)
        self.current_test += 1
        self.test_start_time = time.perf_counter()
        self._stdout_capture = io.StringIO()
        self._stderr_capture = io.StringIO()
        self._stdout_redirect = contextlib.redirect_stdout(self._stdout_capture)
        self._stderr_redirect = contextlib.redirect_stderr(self._stderr_capture)
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()
        self._write_module_header(test)
        self.stream.write(color(pad(f"[ TEST {self._progress_label()} ] {self.getDescription(test)}"), CYAN) + "\n")
        self.stream.flush()

    def stopTest(self, test):
        if self._stdout_redirect is not None:
            self._stdout_redirect.__exit__(None, None, None)
        if self._stderr_redirect is not None:
            self._stderr_redirect.__exit__(None, None, None)
        stdout_value = self._stdout_capture.getvalue() if self._stdout_capture else ""
        stderr_value = self._stderr_capture.getvalue() if self._stderr_capture else ""
        self.test_outputs[test] = stdout_value + stderr_value
        super().stopTest(test)

    def addSuccess(self, test):
        elapsed = time.perf_counter() - self.test_start_time
        self.passed_count += 1
        self.stream.write(color(pad(f"    PASS   {elapsed:.3f}s   | đã pass {self.passed_count}/{self.total_tests}"), GREEN) + "\n")
        self.stream.flush()
        super().addSuccess(test)

    def addFailure(self, test, err):
        elapsed = time.perf_counter() - self.test_start_time
        self.failed_count += 1
        self.stream.write(color(pad(f"    FAIL   {elapsed:.3f}s   | lỗi {self.failed_count}"), RED) + "\n")
        self.stream.flush()
        super().addFailure(test, err)

    def addError(self, test, err):
        elapsed = time.perf_counter() - self.test_start_time
        self.error_count += 1
        self.stream.write(color(pad(f"    ERROR  {elapsed:.3f}s   | error {self.error_count}"), MAGENTA) + "\n")
        self.stream.flush()
        super().addError(test, err)

    def addSkip(self, test, reason):
        elapsed = time.perf_counter() - self.test_start_time
        self.skipped_count += 1
        self.stream.write(color(pad(f"    SKIP   {elapsed:.3f}s   | {reason}"), YELLOW) + "\n")
        self.stream.flush()
        super().addSkip(test, reason)

    def printErrors(self):
        return


class SilentStream:
    def __init__(self, real_stream):
        self.real_stream = real_stream

    def write(self, text):
        stripped = text.strip()
        if not text:
            return
        if stripped == "-" * 70:
            return
        if text.startswith(("Ran ", "OK", "FAILED", "FAILED ", "OK ", "\nOK", "\nFAILED")):
            return
        if stripped in {"OK", "FAILED"}:
            return
        if stripped.startswith("Ran "):
            return
        if stripped.startswith("FAILED"):
            return
        if stripped.startswith("OK"):
            return
        if stripped.startswith("(failures="):
            return
        self.real_stream.write(text)

    def writeln(self, text=None):
        self.write("" if text is None else f"{text}\n")

    def flush(self):
        self.real_stream.flush()


class PrettyTestRunner(unittest.TextTestRunner):
    resultclass = PrettyTestResult

    def __init__(
        self,
        *args,
        total_tests: int = 0,
        camera_result: CameraCheckResult | None = None,
        strict_camera: bool = False,
        **kwargs,
    ):
        stream = kwargs.pop("stream", sys.stderr)
        super().__init__(*args, stream=SilentStream(stream), **kwargs)
        self.output_stream = stream
        self.total_tests = total_tests
        self.camera_result = camera_result
        self.strict_camera = strict_camera

    def _check_dependencies(self) -> list[str]:
        missing = []
        deps = {
            "PySide6": "Giao diện (UI)",
            "faster_whisper": "Voice Recognition",
            "pyaudio": "Audio Input",
            "pygments": "Syntax Highlighting"
        }
        import importlib.util
        for lib, desc in deps.items():
            if importlib.util.find_spec(lib.split('.')[0]) is None:
                missing.append(f"{lib} ({desc})")
        return missing

    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity, total_tests=self.total_tests)

    def _write_camera_section(self) -> None:
        if self.camera_result is not None:
            self.output_stream.write(section("KIỂM TRA THIẾT BỊ", YELLOW) + "\n")
            self.output_stream.write(color(pad(self.camera_result.summary), self.camera_result.style) + "\n")
            self.output_stream.write(color(pad(self.camera_result.detail), self.camera_result.style) + "\n")
            self.output_stream.write(color(rule("-"), CYAN) + "\n")

        missing_deps = self._check_dependencies()
        if missing_deps:
            self.output_stream.write(section("CẢNH BÁO THƯ VIỆN", RED) + "\n")
            for dep in missing_deps:
                self.output_stream.write(color(pad(f"Thiếu: {dep}"), YELLOW) + "\n")
            self.output_stream.write(color(rule("-"), CYAN) + "\n")

    def run(self, test):
        self.output_stream.write(color(rule("="), CYAN) + "\n")
        self.output_stream.write(color(pad("YOLO PROJECT :: SYSTEM TEST DASHBOARD"), BOLD + CYAN) + "\n")
        self.output_stream.write(color(rule("="), CYAN) + "\n")
        self.output_stream.write(section("TỔNG QUAN", GREEN) + "\n")
        self.output_stream.write(color(pad(f"Tổng số test      {self.total_tests}"), GREEN) + "\n")
        self.output_stream.write(color(pad("Chế độ            Chạy từng test, hiện kết quả theo thời gian thực"), GREEN) + "\n")
        self.output_stream.write(color(pad(f"Tiến độ khởi động  [{meter(0, self.total_tests)}] 0/{self.total_tests}"), YELLOW) + "\n")
        self.output_stream.write(color(pad("Trạng thái        Đang quét toàn bộ hệ thống..."), YELLOW) + "\n")
        self.output_stream.write(color(rule("-"), CYAN) + "\n")
        self._write_camera_section()
        self.output_stream.flush()

        start = time.perf_counter()
        result: PrettyTestResult = super().run(test)
        duration = time.perf_counter() - start

        pass_bar, pass_style = status_meter(result.passed_count, self.total_tests, GREEN)
        fail_bar, fail_style = status_meter(result.failed_count, self.total_tests, GREEN, RED)
        error_bar, error_style = status_meter(result.error_count, self.total_tests, GREEN, MAGENTA)
        skip_bar, skip_style = status_meter(result.skipped_count, self.total_tests, GREEN, YELLOW)

        self.output_stream.write("\n" + color(rule("="), CYAN) + "\n")
        self.output_stream.write(color(pad("YOLO PROJECT :: TỔNG KẾT KIỂM THỬ"), BOLD + CYAN) + "\n")
        self.output_stream.write(color(rule("="), CYAN) + "\n")
        self.output_stream.write(section("TỔNG KẾT", CYAN) + "\n")
        self.output_stream.write(color(pad(f"Tiến độ tổng      [{meter(result.current_test, self.total_tests)}] {result.current_test}/{self.total_tests}"), CYAN) + "\n")
        self.output_stream.write(color(pad(f"PASS              [{pass_bar}] {result.passed_count}/{self.total_tests}"), pass_style) + "\n")
        self.output_stream.write(color(pad(f"FAIL              [{fail_bar}] {result.failed_count}"), fail_style) + "\n")
        self.output_stream.write(color(pad(f"ERROR             [{error_bar}] {result.error_count}"), error_style) + "\n")
        self.output_stream.write(color(pad(f"SKIP              [{skip_bar}] {result.skipped_count}"), skip_style) + "\n")
        self.output_stream.write(color(pad(f"Thời gian tổng    {duration:.3f}s"), CYAN) + "\n")
        if self.camera_result is not None:
            self.output_stream.write(color(pad(self.camera_result.summary), self.camera_result.style) + "\n")

        if result.failures or result.errors:
            self.output_stream.write("\n" + color(rule("-"), RED) + "\n")
            self.output_stream.write(section("CHI TIẾT TEST LỖI", RED) + "\n")
            self.output_stream.write(color(rule("-"), RED) + "\n")
            for test, traceback_text in result.failures + result.errors:
                self.output_stream.write(color(f"{self.resultclass.getDescription(result, test)}\n", RED))
                captured_output = result.test_outputs.get(test, "").strip()
                if captured_output:
                    self.output_stream.write(color("Output:\n", ORANGE))
                    self.output_stream.write(captured_output + "\n")
                self.output_stream.write(traceback_text + "\n")

        self.output_stream.write(color(rule("-"), CYAN) + "\n")
        suite_ok = result.wasSuccessful()
        camera_ok = self.camera_result is None or self.camera_result.ok or not self.strict_camera
        if suite_ok and camera_ok:
            self.output_stream.write(color(pad("KẾT QUẢ CUỐI      TOÀN BỘ TEST PASS"), BOLD + GREEN) + "\n")
        elif suite_ok and not camera_ok:
            self.output_stream.write(color(pad("KẾT QUẢ CUỐI      UNIT TEST PASS, NHƯNG CAMERA THẬT KHÔNG ĐẠT"), BOLD + RED) + "\n")
        else:
            self.output_stream.write(color(pad("KẾT QUẢ CUỐI      CÓ TEST LỖI, CẦN KIỂM TRA LẠI"), BOLD + RED) + "\n")
        self.output_stream.flush()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy test toàn hệ thống cho dự án YOLO.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index để kiểm tra camera thật.")
    parser.add_argument(
        "--skip-camera-check",
        action="store_true",
        help="Bỏ qua bước kiểm tra camera thật trước khi chạy unit test.",
    )
    parser.add_argument(
        "--strict-camera",
        action="store_true",
        help="Nếu camera thật không đạt thì coi như bài test tổng thể fail.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_directories()
    suite = unittest.defaultTestLoader.discover("tests")
    camera_result = None if args.skip_camera_check else check_camera(index=args.camera_index)
    runner = PrettyTestRunner(
        verbosity=0,
        total_tests=suite.countTestCases(),
        stream=sys.stdout,
        camera_result=camera_result,
        strict_camera=args.strict_camera,
    )
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        result = runner.run(suite)
    finally:
        logging.disable(previous_disable_level)

    suite_ok = result.wasSuccessful()
    camera_ok = camera_result is None or camera_result.ok or not args.strict_camera
    return 0 if suite_ok and camera_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
