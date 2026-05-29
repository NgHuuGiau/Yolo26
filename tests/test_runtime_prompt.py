from __future__ import annotations

import unittest

from utils.runtime_prompt import mode_to_ui_defaults, prompt_runtime_mode


class RuntimePromptTests(unittest.TestCase):
    def test_prompt_runtime_mode_exits_on_zero(self) -> None:
        answers = iter(["0"])
        printed: list[str] = []

        with self.assertRaises(SystemExit) as ctx:
            prompt_runtime_mode(
                input_fn=lambda _: next(answers),
                print_fn=printed.append,
            )

        self.assertEqual(ctx.exception.code, 0)

    def test_prompt_runtime_mode_accepts_valid_choice(self) -> None:
        answers = iter(["3"])
        printed: list[str] = []

        mode = prompt_runtime_mode(
            input_fn=lambda _: next(answers),
            print_fn=printed.append,
        )

        self.assertEqual(mode, "medium")
        self.assertTrue(any("CHON CAU HINH CHAY" in line for line in printed))

    def test_prompt_runtime_mode_retries_on_invalid_choice(self) -> None:
        answers = iter(["9", "", "4"])
        printed: list[str] = []

        mode = prompt_runtime_mode(
            input_fn=lambda _: next(answers),
            print_fn=printed.append,
        )

        self.assertEqual(mode, "low")
        self.assertTrue(any("Lua chon khong hop le" in line for line in printed))

    def test_mode_to_ui_defaults_maps_values(self) -> None:
        self.assertEqual(mode_to_ui_defaults("auto"), ("auto", "medium"))
        self.assertEqual(mode_to_ui_defaults("high"), ("manual", "high"))
