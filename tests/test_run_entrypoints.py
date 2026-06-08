from __future__ import annotations

import unittest
from unittest.mock import patch

import run_app
import run_detect
import run_train
from utils.console_ui import mode_to_ui_defaults


class RunEntrypointsTests(unittest.TestCase):
    @patch("run_detect.resolve_start_options")
    @patch("run_detect.launch_chat_ai_app")
    @patch("run_detect.parse_args")
    def test_run_detect_main_opens_chat_ui(self, parse_args_mock, launch_chat_mock, resolve_start_options_mock) -> None:
        args = type("Args", (), {"mode": "auto", "model": None, "camera_index": 1})()
        parse_args_mock.return_value = args
        resolve_start_options_mock.return_value = ("auto", "yolo11s.pt")
        launch_chat_mock.return_value = 0
        run_detect.main()
        resolve_start_options_mock.assert_called_once_with(requested_mode="auto", requested_model=None)
        launch_chat_mock.assert_called_once_with(
            window_title="Chat AI Detect",
            camera_index=1,
            app_mode="auto",
            selected_model="yolo11s.pt",
        )

    @patch("run_app.resolve_start_options")
    @patch("run_app.launch_chat_ai_app")
    @patch("run_app.parse_args")
    def test_run_app_main_prompts_for_mode_before_opening_chat_ui(
        self,
        parse_args_mock,
        launch_chat_mock,
        resolve_start_options_mock,
    ) -> None:
        args = type("Args", (), {"mode": None, "model": None, "camera_index": 2})()
        parse_args_mock.return_value = args
        resolve_start_options_mock.return_value = ("high", "models/trained/best.pt")
        launch_chat_mock.return_value = 0
        exit_code = run_app.main()
        self.assertEqual(exit_code, 0)
        resolve_start_options_mock.assert_called_once_with(requested_mode=None, requested_model=None)
        launch_chat_mock.assert_called_once_with(
            window_title="Chat AI",
            camera_index=2,
            app_mode="high",
            selected_model="models/trained/best.pt",
        )

    @patch("run_train.main")
    def test_run_train_module_exposes_training_main(self, train_main_mock) -> None:
        run_train.main()
        train_main_mock.assert_called_once()

    def test_mode_to_ui_defaults_maps_modes(self) -> None:
        self.assertEqual(mode_to_ui_defaults("auto"), ("auto", "medium"))
        self.assertEqual(mode_to_ui_defaults("high"), ("manual", "high"))
