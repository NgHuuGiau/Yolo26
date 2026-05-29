from __future__ import annotations

import contextlib
import io
import logging
import math
import sys
import time
import unittest

from utils.file_utils import ensure_project_directories


RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
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
        self.stream.write(color(pad(f"    PASS   {elapsed:.3f}s   | da pass {self.passed_count}/{self.total_tests}"), GREEN) + "\n")
        self.stream.flush()
        super().addSuccess(test)

    def addFailure(self, test, err):
        elapsed = time.perf_counter() - self.test_start_time
        self.failed_count += 1
        self.stream.write(color(pad(f"    FAIL   {elapsed:.3f}s   | loi {self.failed_count}"), RED) + "\n")
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

    def __init__(self, *args, total_tests: int = 0, **kwargs):
        stream = kwargs.pop("stream", sys.stderr)
        super().__init__(*args, stream=SilentStream(stream), **kwargs)
        self.output_stream = stream
        self.total_tests = total_tests

    def _render_progress(self, current: int, total: int, style: str = CYAN) -> str:
        return color(pad(f"[{meter(current, total)}] {current}/{total}"), style)

    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity, total_tests=self.total_tests)

    def run(self, test):
        self.output_stream.write(color(rule("="), CYAN) + "\n")
        self.output_stream.write(color(pad("YOLO PROJECT :: SYSTEM TEST DASHBOARD"), BOLD + CYAN) + "\n")
        self.output_stream.write(color(rule("="), CYAN) + "\n")
        self.output_stream.write(section("TONG QUAN", GREEN) + "\n")
        self.output_stream.write(color(pad(f"Tong so test      {self.total_tests}"), GREEN) + "\n")
        self.output_stream.write(color(pad("Che do            Chay tung test, hien ket qua theo thoi gian that"), GREEN) + "\n")
        self.output_stream.write(color(pad(f"Tien do khoi dong  [{meter(0, self.total_tests)}] 0/{self.total_tests}"), YELLOW) + "\n")
        self.output_stream.write(color(pad("Trang thai        Dang quet toan bo he thong..."), YELLOW) + "\n")
        self.output_stream.write(color(rule("-"), CYAN) + "\n")
        self.output_stream.flush()

        start = time.perf_counter()
        result: PrettyTestResult = super().run(test)
        duration = time.perf_counter() - start

        pass_bar, pass_style = status_meter(result.passed_count, self.total_tests, GREEN)
        fail_bar, fail_style = status_meter(result.failed_count, self.total_tests, GREEN, RED)
        error_bar, error_style = status_meter(result.error_count, self.total_tests, GREEN, MAGENTA)
        skip_bar, skip_style = status_meter(result.skipped_count, self.total_tests, GREEN, YELLOW)

        self.output_stream.write("\n" + color(rule("="), CYAN) + "\n")
        self.output_stream.write(color(pad("YOLO PROJECT :: TONG KET KIEM THU"), BOLD + CYAN) + "\n")
        self.output_stream.write(color(rule("="), CYAN) + "\n")
        self.output_stream.write(section("TONG KET", CYAN) + "\n")
        self.output_stream.write(color(pad(f"Tien do tong      [{meter(result.current_test, self.total_tests)}] {result.current_test}/{self.total_tests}"), CYAN) + "\n")
        self.output_stream.write(color(pad(f"PASS              [{pass_bar}] {result.passed_count}/{self.total_tests}"), pass_style) + "\n")
        self.output_stream.write(color(pad(f"FAIL              [{fail_bar}] {result.failed_count}"), fail_style) + "\n")
        self.output_stream.write(color(pad(f"ERROR             [{error_bar}] {result.error_count}"), error_style) + "\n")
        self.output_stream.write(color(pad(f"SKIP              [{skip_bar}] {result.skipped_count}"), skip_style) + "\n")
        self.output_stream.write(color(pad(f"Thoi gian tong    {duration:.3f}s"), CYAN) + "\n")

        if result.failures or result.errors:
            self.output_stream.write("\n" + color(rule("-"), RED) + "\n")
            self.output_stream.write(section("CHI TIET TEST LOI", RED) + "\n")
            self.output_stream.write(color(rule("-"), RED) + "\n")
            for test, traceback_text in result.failures + result.errors:
                self.output_stream.write(color(f"{self.resultclass.getDescription(result, test)}\n", RED))
                captured_output = result.test_outputs.get(test, "").strip()
                if captured_output:
                    self.output_stream.write(color("Output:\n", ORANGE))
                    self.output_stream.write(captured_output + "\n")
                self.output_stream.write(traceback_text + "\n")

        self.output_stream.write(color(rule("-"), CYAN) + "\n")
        if result.wasSuccessful():
            self.output_stream.write(color(pad("KET QUA CUOI      TOAN BO TEST PASS"), BOLD + GREEN) + "\n")
        else:
            self.output_stream.write(color(pad("KET QUA CUOI      CO TEST LOI, CAN KIEM TRA LAI"), BOLD + RED) + "\n")
        self.output_stream.flush()
        return result


def main() -> int:
    ensure_project_directories()
    suite = unittest.defaultTestLoader.discover("tests")
    runner = PrettyTestRunner(verbosity=0, total_tests=suite.countTestCases(), stream=sys.stdout)
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        result = runner.run(suite)
    finally:
        logging.disable(previous_disable_level)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
