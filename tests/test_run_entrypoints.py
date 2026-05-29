from __future__ import annotations

import unittest
from unittest.mock import patch

import run_app
import run_detect
import run_train
from utils.runtime_prompt import mode_to_ui_defaults


class RunEntrypointsTests(unittest.TestCase):
    @patch("run_detect.run_camera_session")
    @patch("run_detect.select_runtime_config")
    @patch("run_detect.detect_hardware")
    @patch("run_detect.parse_args")
    def test_run_detect_main_wires_runtime_to_camera(
        self,
        parse_args_mock,
        detect_hardware_mock,
        select_runtime_mock,
        run_camera_mock,
    ) -> None:
        args = type("Args", (), {"mode": "auto", "camera_index": 1})()
        parse_args_mock.return_value = args
        detect_hardware_mock.return_value = object()
        runtime = type("Runtime", (), {"summary": lambda self: {"mode": "auto"}})()
        select_runtime_mock.return_value = runtime

        run_detect.main()

        select_runtime_mock.assert_called_once_with(mode="auto", hardware=detect_hardware_mock.return_value)
        run_camera_mock.assert_called_once_with(runtime=runtime, camera_index=1)

    @patch("run_detect.prompt_runtime_mode", return_value="medium")
    @patch("run_detect.run_camera_session")
    @patch("run_detect.select_runtime_config")
    @patch("run_detect.detect_hardware")
    @patch("run_detect.parse_args")
    def test_run_detect_prompts_for_mode_when_not_provided(
        self,
        parse_args_mock,
        detect_hardware_mock,
        select_runtime_mock,
        run_camera_mock,
        prompt_mode_mock,
    ) -> None:
        args = type("Args", (), {"mode": None, "camera_index": 0})()
        parse_args_mock.return_value = args
        detect_hardware_mock.return_value = object()
        runtime = type("Runtime", (), {"summary": lambda self: {"mode": "medium"}})()
        select_runtime_mock.return_value = runtime

        run_detect.main()

        prompt_mode_mock.assert_called_once()
        select_runtime_mock.assert_called_once_with(mode="medium", hardware=detect_hardware_mock.return_value)
        run_camera_mock.assert_called_once_with(runtime=runtime, camera_index=0)

    @patch("run_detect.logger")
    @patch("run_detect.run_camera_session", side_effect=RuntimeError("camera failed"))
    @patch("run_detect.select_runtime_config")
    @patch("run_detect.detect_hardware")
    @patch("run_detect.parse_args")
    def test_run_detect_logs_helpful_error_when_runtime_fails(
        self,
        parse_args_mock,
        detect_hardware_mock,
        select_runtime_mock,
        _run_camera_mock,
        logger_mock,
    ) -> None:
        args = type("Args", (), {"mode": "low", "camera_index": 0})()
        parse_args_mock.return_value = args
        detect_hardware_mock.return_value = object()
        runtime = type("Runtime", (), {"summary": lambda self: {"mode": "low"}})()
        select_runtime_mock.return_value = runtime

        with self.assertRaises(RuntimeError):
            run_detect.main()

        logger_mock.error.assert_called_once()

    @patch("run_train.main")
    def test_run_train_module_exposes_training_main(self, train_main_mock) -> None:
        run_train.main()
        train_main_mock.assert_called_once()

    @patch("run_app.run_camera_session")
    @patch("run_app.select_runtime_config")
    @patch("run_app.detect_hardware")
    @patch("run_app.prompt_runtime_mode", return_value="high")
    @patch("run_app.parse_args")
    def test_run_app_main_invokes_desktop_camera_flow(
        self,
        parse_args_mock,
        prompt_mode_mock,
        detect_hardware_mock,
        select_runtime_mock,
        run_camera_mock,
    ) -> None:
        parse_args_mock.return_value = type("Args", (), {"mode": None, "camera_index": 2})()
        detect_hardware_mock.return_value = object()
        runtime = type(
            "Runtime",
            (),
            {
                "mode": "high",
                "profile_name": "high",
                "primary_model_name": "yolo26s.pt",
                "resolved_device": "cuda:0",
                "imgsz": 640,
                "camera_width": 1000,
                "camera_height": 650,
            },
        )()
        select_runtime_mock.return_value = runtime
        exit_code = run_app.main()
        self.assertEqual(exit_code, 0)
        prompt_mode_mock.assert_called_once()
        select_runtime_mock.assert_called_once_with(mode="high", hardware=detect_hardware_mock.return_value)
        run_camera_mock.assert_called_once_with(runtime=runtime, camera_index=2)

    def test_mode_to_ui_defaults_maps_modes(self) -> None:
        self.assertEqual(mode_to_ui_defaults("auto"), ("auto", "medium"))
        self.assertEqual(mode_to_ui_defaults("high"), ("manual", "high"))
