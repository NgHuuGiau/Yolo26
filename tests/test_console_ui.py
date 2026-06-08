from __future__ import annotations

import unittest

from utils.console_ui import _usage_row, explain_runtime_failure, progress_bar_colored


class ConsoleUiTests(unittest.TestCase):
    def test_progress_bar_uses_unicode_bar_and_dot(self) -> None:
        bar = progress_bar_colored(50, width=6)
        self.assertIn("\u2588", bar)
        self.assertIn("\u00b7", bar)

    def test_explain_runtime_failure_for_camera_error(self) -> None:
        reason, suggestions, commands = explain_runtime_failure(RuntimeError("Không mở được camera."))
        self.assertTrue("webcam" in reason.lower() or "camera" in reason.lower())
        self.assertTrue(any("camera index" in suggestion.lower() or "webcam" in suggestion.lower() for suggestion in suggestions))
        self.assertTrue(any("--camera-index 1" in command for command in commands))

    def test_usage_row_renders_percent_and_bar(self) -> None:
        row = _usage_row("CPU", 42.0)
        self.assertIn("42.0%", row)
        self.assertIn("\u2588", row)

    def test_usage_row_handles_unknown_value(self) -> None:
        row = _usage_row("GPU", None)
        self.assertIn("Không rõ", row)
