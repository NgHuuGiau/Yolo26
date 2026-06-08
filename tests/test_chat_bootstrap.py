from __future__ import annotations

import unittest
from unittest.mock import patch

from app.chat_bootstrap import resolve_start_options


class ChatBootstrapTests(unittest.TestCase):
    @patch("app.chat_bootstrap.prompt_runtime_mode", return_value="high")
    @patch("app.chat_bootstrap.select_runtime_config_optimized")
    @patch("app.chat_bootstrap.detect_hardware")
    def test_resolve_start_options_uses_selected_runtime_model_when_mode_is_prompted(
        self,
        detect_hardware_mock,
        select_runtime_mock,
        prompt_mode_mock,
    ) -> None:
        hardware = object()
        detect_hardware_mock.return_value = hardware

        def runtime_for(mode: str):
            return type("Runtime", (), {"primary_model_name": f"{mode}.pt"})()

        select_runtime_mock.side_effect = lambda *, mode, hardware: runtime_for(mode)

        selected_mode, selected_model = resolve_start_options(requested_mode=None, requested_model=None)

        self.assertEqual(selected_mode, "high")
        self.assertEqual(selected_model, "high.pt")
        prompt_mode_mock.assert_called_once()
        self.assertEqual(select_runtime_mock.call_count, 4)

