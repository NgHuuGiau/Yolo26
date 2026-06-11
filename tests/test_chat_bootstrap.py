from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from app.chat_bootstrap import resolve_start_bundle, resolve_start_options


class ChatBootstrapTests(unittest.TestCase):
    @patch("app.chat_bootstrap.prompt_runtime_mode")
    @patch("app.chat_bootstrap.select_runtime_config_optimized")
    @patch("app.chat_bootstrap.detect_hardware")
    def test_resolve_start_bundle_prompts_for_camera_mode_when_mode_is_missing(
        self,
        detect_hardware_mock,
        select_runtime_mock,
        prompt_runtime_mode_mock,
    ) -> None:
        hardware = object()
        detect_hardware_mock.return_value = hardware
        prompt_runtime_mode_mock.return_value = "medium"

        @dataclass
        class FakeRuntime:
            primary_model_name: str
            requested_model_name: str
            candidate_models: list[str]

        def runtime_for(mode: str):
            return FakeRuntime(
                primary_model_name=f"{mode}.pt",
                requested_model_name=f"{mode}.pt",
                candidate_models=[f"{mode}.pt"],
            )

        select_runtime_mock.side_effect = lambda *, mode, hardware: runtime_for(mode)

        start_options = resolve_start_bundle(
            requested_mode=None,
            requested_model=None,
            requested_target=None,
            preferred_target="camera",
        )

        self.assertEqual(start_options.selected_mode, "medium")
        self.assertEqual(start_options.selected_model, "medium.pt")
        self.assertEqual(start_options.launch_target, "camera")
        prompt_runtime_mode_mock.assert_called_once_with(
            hardware=hardware,
            recommendations={
                "auto": runtime_for("auto"),
                "high": runtime_for("high"),
                "medium": runtime_for("medium"),
                "low": runtime_for("low"),
            },
        )
        self.assertEqual(select_runtime_mock.call_count, 4)

    @patch("app.chat_bootstrap.prompt_launch_target")
    @patch("app.chat_bootstrap.select_runtime_config_optimized")
    @patch("app.chat_bootstrap.detect_hardware")
    def test_resolve_start_options_keeps_legacy_tuple_contract(
        self,
        detect_hardware_mock,
        select_runtime_mock,
        prompt_target_mock,
    ) -> None:
        hardware = object()
        detect_hardware_mock.return_value = hardware
        
        @dataclass
        class FakeRuntime:
            primary_model_name: str
            requested_model_name: str
            candidate_models: list[str]

        runtime = FakeRuntime(
            primary_model_name="medium.pt",
            requested_model_name="medium.pt",
            candidate_models=["medium.pt"],
        )
        select_runtime_mock.return_value = runtime

        selected_mode, selected_model = resolve_start_options(requested_mode="medium", requested_model=None)

        self.assertEqual((selected_mode, selected_model), ("medium", "medium.pt"))
        prompt_target_mock.assert_not_called()

    @patch("app.chat_bootstrap.select_runtime_config_optimized")
    @patch("app.chat_bootstrap.detect_hardware")
    def test_resolve_start_bundle_skips_prompt_model_when_requested_model_is_provided(
        self,
        detect_hardware_mock,
        select_runtime_mock,
    ) -> None:
        hardware = object()
        detect_hardware_mock.return_value = hardware

        @dataclass
        class FakeRuntime:
            primary_model_name: str
            requested_model_name: str
            candidate_models: list[str]

        runtime = FakeRuntime(
            primary_model_name="medium.pt",
            requested_model_name="medium.pt",
            candidate_models=["medium.pt"],
        )
        select_runtime_mock.side_effect = lambda *, mode, hardware: runtime

        start_options = resolve_start_bundle(
            requested_mode="medium",
            requested_model="custom.pt",
            requested_target="camera",
            preferred_target="camera",
        )

        self.assertEqual(start_options.selected_model, "custom.pt")

    @patch("app.chat_bootstrap.select_runtime_config_optimized")
    @patch("app.chat_bootstrap.detect_hardware")
    def test_resolve_start_options_defaults_ui_to_medium_mode(
        self,
        detect_hardware_mock,
        select_runtime_mock,
    ) -> None:
        hardware = object()
        detect_hardware_mock.return_value = hardware

        @dataclass
        class FakeRuntime:
            primary_model_name: str
            requested_model_name: str
            candidate_models: list[str]

        select_runtime_mock.side_effect = lambda *, mode, hardware: FakeRuntime(
            primary_model_name=f"{mode}.pt",
            requested_model_name=f"{mode}.pt",
            candidate_models=[f"{mode}.pt"],
        )

        selected_mode, selected_model = resolve_start_options(requested_mode=None, requested_model=None)

        self.assertEqual((selected_mode, selected_model), ("medium", "medium.pt"))

    @patch("app.chat_bootstrap.prompt_runtime_mode")
    @patch("app.chat_bootstrap.select_runtime_config_optimized")
    @patch("app.chat_bootstrap.detect_hardware")
    def test_resolve_start_bundle_keeps_ui_default_without_prompt(
        self,
        detect_hardware_mock,
        select_runtime_mock,
        prompt_runtime_mode_mock,
    ) -> None:
        hardware = object()
        detect_hardware_mock.return_value = hardware

        @dataclass
        class FakeRuntime:
            primary_model_name: str
            requested_model_name: str
            candidate_models: list[str]

        select_runtime_mock.side_effect = lambda *, mode, hardware: FakeRuntime(
            primary_model_name=f"{mode}.pt",
            requested_model_name=f"{mode}.pt",
            candidate_models=[f"{mode}.pt"],
        )

        start_options = resolve_start_bundle(
            requested_mode=None,
            requested_model=None,
            requested_target="ui",
            preferred_target="camera",
        )

        self.assertEqual(start_options.selected_mode, "medium")
        self.assertEqual(start_options.launch_target, "ui")
        prompt_runtime_mode_mock.assert_not_called()
