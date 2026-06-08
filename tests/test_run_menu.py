from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import run_menu


class RunMenuTests(unittest.TestCase):
    def test_main_exits_on_zero(self) -> None:
        outputs: list[str] = []
        clear_terminal = MagicMock()
        result = run_menu.main(input_fn=lambda _: "0", print_fn=outputs.append, run_script_fn=MagicMock(), clear_terminal_fn=clear_terminal)
        self.assertEqual(result, 0)
        self.assertTrue(any("Da thoat menu" in line for line in outputs))
        clear_terminal.assert_not_called()

    def test_main_runs_selected_script(self) -> None:
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        clear_terminal = MagicMock()
        answers = iter(["6", "0"])
        result = run_menu.main(input_fn=lambda _: next(answers), print_fn=outputs.append, run_script_fn=run_script, clear_terminal_fn=clear_terminal)
        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_doctor.py")
        self.assertTrue(any("Quay lai menu" in line or "Quay lại menu" in line for line in outputs))
        clear_terminal.assert_called_once()

    def test_main_retries_on_invalid_choice(self) -> None:
        outputs: list[str] = []
        answers = iter(["9", "5", "0"])
        run_script = MagicMock(return_value=0)
        clear_terminal = MagicMock()
        result = run_menu.main(input_fn=lambda _: next(answers), print_fn=outputs.append, run_script_fn=run_script, clear_terminal_fn=clear_terminal)
        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_tests.py")
        self.assertTrue(any("Lua chon khong hop le" in line or "Khong hop le" in line or "Không hợp lệ" in line for line in outputs))
        clear_terminal.assert_called_once()

    def test_main_returns_to_menu_after_nonzero_exit(self) -> None:
        outputs: list[str] = []
        answers = iter(["3", "0"])
        run_script = MagicMock(return_value=1)
        clear_terminal = MagicMock()
        result = run_menu.main(input_fn=lambda _: next(answers), print_fn=outputs.append, run_script_fn=run_script, clear_terminal_fn=clear_terminal)
        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_tools.py")
        self.assertTrue(any("ket thuc voi ma" in line or "kết thúc với mã" in line for line in outputs))
        clear_terminal.assert_called_once()


if __name__ == "__main__":
    unittest.main()