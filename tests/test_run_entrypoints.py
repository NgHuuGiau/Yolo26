from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_app
import run_detect
import run_train
from utils.console_ui import mode_to_ui_defaults


class RunEntrypointsTests(unittest.TestCase):
    @patch("run_detect.run_camera_session")
    @patch("run_detect.print_runtime_dashboard")
    @patch("run_detect.resolve_start_bundle")
    @patch("run_detect.parse_args")
    def test_run_detect_main_runs_camera_when_camera_target_is_selected(
        self,
        parse_args_mock,
        resolve_start_bundle_mock,
        print_dashboard_mock,
        run_camera_session_mock,
    ) -> None:
        args = type("Args", (), {"mode": "auto", "model": None, "camera_index": 1, "target": None})()
        parse_args_mock.return_value = args
        start_options = SimpleNamespace(
            selected_mode="auto",
            selected_model="yolo11s.pt",
            launch_target="camera",
            runtime=object(),
            hardware=object(),
        )
        resolve_start_bundle_mock.return_value = start_options

        exit_code = run_detect.main()

        self.assertEqual(exit_code, 0)
        resolve_start_bundle_mock.assert_called_once_with(
            requested_mode="auto",
            requested_model=None,
            requested_target=None,
            preferred_target="camera",
        )
        print_dashboard_mock.assert_called_once_with(
            title="YOLO Detect Camera",
            runtime=start_options.runtime,
            hardware=start_options.hardware,
            camera_index=1,
            launch_target="camera",
        )
        run_camera_session_mock.assert_called_once_with(runtime=start_options.runtime, camera_index=1)

    @patch("run_app.resolve_start_bundle")
    @patch("run_app.launch_chat_ai_app")
    @patch("run_app.parse_args")
    def test_run_app_main_opens_chat_ui_when_ui_target_is_selected(
        self,
        parse_args_mock,
        launch_chat_mock,
        resolve_start_bundle_mock,
    ) -> None:
        args = type("Args", (), {"mode": None, "model": None, "camera_index": 2, "target": None})()
        parse_args_mock.return_value = args
        resolve_start_bundle_mock.return_value = SimpleNamespace(
            selected_mode="high",
            selected_model="models/trained/best.pt",
            launch_target="ui",
            runtime=object(),
            hardware=object(),
        )
        launch_chat_mock.return_value = 0

        exit_code = run_app.main()

        self.assertEqual(exit_code, 0)
        resolve_start_bundle_mock.assert_called_once_with(
            requested_mode=None,
            requested_model=None,
            requested_target=None,
            preferred_target="ui",
        )
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
