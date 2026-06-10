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

    @patch("run_app.run_camera_session")
    @patch("run_app.print_runtime_dashboard")
    @patch("run_app.resolve_start_bundle")
    @patch("run_app.parse_args")
    def test_run_app_main_runs_camera_only(
        self,
        parse_args_mock,
        resolve_start_bundle_mock,
        print_dashboard_mock,
        run_camera_session_mock,
    ) -> None:
        args = type("Args", (), {"mode": None, "model": None, "camera_index": 2, "target": "ui"})()
        parse_args_mock.return_value = args
        runtime = object()
        hardware = object()
        resolve_start_bundle_mock.return_value = SimpleNamespace(
            selected_mode="high",
            selected_model="models/trained/best.pt",
            launch_target="camera",
            runtime=runtime,
            hardware=hardware,
        )

        exit_code = run_app.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(args.target, "camera")
        resolve_start_bundle_mock.assert_called_once_with(
            requested_mode=None,
            requested_model=None,
            requested_target="camera",
            preferred_target="camera",
        )
        print_dashboard_mock.assert_called_once_with(
            title="YOLO Camera Realtime",
            runtime=runtime,
            hardware=hardware,
            camera_index=2,
            launch_target="camera",
        )
        run_camera_session_mock.assert_called_once_with(runtime=runtime, camera_index=2)

    @patch("run_train.main")
    def test_run_train_module_exposes_training_main(self, train_main_mock) -> None:
        run_train.main()
        train_main_mock.assert_called_once()

    def test_mode_to_ui_defaults_maps_modes(self) -> None:
        self.assertEqual(mode_to_ui_defaults("auto"), ("auto", "medium"))
        self.assertEqual(mode_to_ui_defaults("high"), ("manual", "high"))
