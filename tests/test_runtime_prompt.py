from __future__ import annotations

import unittest

from utils.console_ui import mode_to_ui_defaults, prompt_launch_target, prompt_runtime_mode


class RuntimePromptTests(unittest.TestCase):
    def test_prompt_runtime_mode_exits_on_zero(self) -> None:
        answers = iter(["0"])
        printed: list[str] = []
        with self.assertRaises(SystemExit) as ctx:
            prompt_runtime_mode(input_fn=lambda _: next(answers), print_fn=printed.append)
        self.assertEqual(ctx.exception.code, 0)

    def test_prompt_runtime_mode_accepts_valid_choice(self) -> None:
        answers = iter(["2"])
        printed: list[str] = []
        mode = prompt_runtime_mode(input_fn=lambda _: next(answers), print_fn=printed.append)
        self.assertEqual(mode, "medium")
        self.assertTrue(any("YOLO REALTIME CAMERA" in line for line in printed))

    def test_prompt_runtime_mode_retries_on_invalid_choice(self) -> None:
        answers = iter(["9", "", "3"])
        printed: list[str] = []
        mode = prompt_runtime_mode(input_fn=lambda _: next(answers), print_fn=printed.append)
        self.assertEqual(mode, "low")
        self.assertGreater(len(printed), 1)

    def test_prompt_launch_target_accepts_camera(self) -> None:
        answers = iter(["2"])
        printed: list[str] = []
        target = prompt_launch_target(
            selected_mode="medium",
            selected_model="yolo11s.pt",
            preferred_target="camera",
            input_fn=lambda _: next(answers),
            print_fn=printed.append,
        )
        self.assertEqual(target, "camera")
        self.assertTrue(any("CHỌN KIỂU KHỞI ĐỘNG" in line for line in printed))

    def test_mode_to_ui_defaults_maps_values(self) -> None:
        self.assertEqual(mode_to_ui_defaults("auto"), ("auto", "medium"))
        self.assertEqual(mode_to_ui_defaults("high"), ("manual", "high"))
