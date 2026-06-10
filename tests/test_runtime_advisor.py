from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.model_selector import RuntimeConfig
from core.runtime_advisor import (
    build_recommendations,
    ceiling_mode_for_hardware,
    default_mode_for_hardware,
    gpu_tier,
    load_level,
    optimized_runtime,
    profile_specs_for_hardware,
    quality_score,
    select_runtime_config_optimized,
    stability_score,
)


def _hardware(**overrides):
    base = dict(
        cpu_name="Intel",
        ram_gb=16.0,
        gpu_name="NVIDIA GeForce RTX 3080",
        vram_gb=10.0,
        cuda_available=True,
        os_name="Windows 11",
        gpu_count=1,
        cpu_usage_percent=20.0,
        ram_usage_percent=25.0,
        gpu_usage_percent=30.0,
        vram_usage_percent=35.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _runtime(**overrides) -> RuntimeConfig:
    base = dict(
        mode="medium",
        profile_name="medium",
        requested_profile_name="medium",
        requested_device="gpu",
        resolved_device="cuda:0",
        requested_model_name="yolo11s.pt",
        primary_model_name="yolo11s.pt",
        candidate_models=["yolo11s.pt"],
        requested_imgsz=640,
        imgsz=640,
        conf=0.25,
        max_det=150,
        use_half=True,
        camera_width=800,
        camera_height=600,
        font_size=16,
        box_thickness=2,
        label_font_scale=0.8,
        active_model_name="",
        hardware_tier="strong GPU",
        fallback_chain=[],
    )
    base.update(overrides)
    return RuntimeConfig(**base)


class RuntimeAdvisorTests(unittest.TestCase):
    def test_load_level_uses_peak_usage(self) -> None:
        self.assertEqual(load_level(_hardware(cpu_usage_percent=86.0)), "very_busy")
        self.assertEqual(load_level(_hardware(ram_usage_percent=72.0)), "busy")
        self.assertEqual(load_level(_hardware(gpu_usage_percent=50.0)), "warm")
        self.assertEqual(load_level(_hardware(vram_usage_percent=15.0)), "cool")

    def test_gpu_tier_and_profile_specs_follow_hardware_strength(self) -> None:
        enthusiast = _hardware(gpu_name="RTX 4090", vram_gb=24.0)
        cpu_only = _hardware(cuda_available=False, gpu_name="None", vram_gb=0.0, ram_gb=8.0)

        self.assertEqual(gpu_tier(enthusiast), "enthusiast")
        self.assertEqual(profile_specs_for_hardware(enthusiast)["high"]["model"], "yolo11x.pt")
        self.assertEqual(gpu_tier(cpu_only), "cpu_only")
        self.assertEqual(profile_specs_for_hardware(cpu_only)["low"]["device"], "cpu")

    def test_default_and_ceiling_modes_reflect_load_and_gpu_capability(self) -> None:
        strong_gpu = _hardware(cpu_usage_percent=10.0, gpu_name="RTX 3080", vram_gb=10.0)
        busy_entry = _hardware(cpu_usage_percent=78.0, gpu_name="RTX 3050", vram_gb=4.0)
        cpu_only = _hardware(cuda_available=False, gpu_name="None", vram_gb=0.0, ram_gb=8.0)

        self.assertEqual(default_mode_for_hardware(strong_gpu), "high")
        self.assertEqual(default_mode_for_hardware(busy_entry), "medium")
        self.assertEqual(default_mode_for_hardware(cpu_only), "low")
        self.assertEqual(ceiling_mode_for_hardware(strong_gpu), "high")
        self.assertEqual(ceiling_mode_for_hardware(_hardware(gpu_name="GTX 1650", vram_gb=2.0)), "medium")
        self.assertEqual(ceiling_mode_for_hardware(cpu_only), "low")

    def test_quality_and_stability_scores_use_runtime_shape_and_load(self) -> None:
        runtime = _runtime(primary_model_name="yolo11x.pt", imgsz=960, max_det=200)

        self.assertGreaterEqual(quality_score(runtime), 100)
        self.assertEqual(stability_score("high", _hardware(cpu_usage_percent=90.0)), 44)
        self.assertEqual(stability_score("low", _hardware(cpu_usage_percent=10.0)), 96)

    @patch("core.runtime_advisor.build_candidates")
    @patch("core.runtime_advisor.load_settings")
    @patch("core.runtime_advisor.select_runtime_config")
    def test_optimized_runtime_overrides_base_runtime_with_profile_specs(
        self,
        select_runtime_config_mock,
        load_settings_mock,
        build_candidates_mock,
    ) -> None:
        select_runtime_config_mock.return_value = _runtime(
            primary_model_name="base.pt",
            candidate_models=["base.pt"],
            imgsz=320,
            max_det=60,
            use_half=False,
        )
        load_settings_mock.return_value = {"inference": {"use_half_for_cuda": True}}
        build_candidates_mock.return_value = ["yolo11l.pt", "yolo11m.pt"]

        runtime = optimized_runtime("high", _hardware(gpu_name="RTX 3080", vram_gb=10.0))

        self.assertEqual(runtime.profile_name, "high")
        self.assertEqual(runtime.primary_model_name, "yolo11l.pt")
        self.assertEqual(runtime.requested_device, "gpu")
        self.assertEqual(runtime.resolved_device, "cuda:0")
        self.assertEqual(runtime.imgsz, 768)
        self.assertEqual(runtime.max_det, 180)
        self.assertEqual(runtime.candidate_models, ["yolo11l.pt", "yolo11m.pt"])
        self.assertTrue(runtime.use_half)

    @patch("core.runtime_advisor.optimized_runtime")
    def test_select_runtime_config_optimized_auto_uses_default_mode(self, optimized_runtime_mock) -> None:
        expected = _runtime(mode="medium", profile_name="medium")
        optimized_runtime_mock.return_value = expected

        runtime = select_runtime_config_optimized("auto", _hardware(gpu_name="RTX 3050", vram_gb=4.0, cpu_usage_percent=75.0))

        self.assertIs(runtime, expected)
        optimized_runtime_mock.assert_called_once()
        self.assertEqual(optimized_runtime_mock.call_args.args[0], "medium")

    @patch("core.runtime_advisor.select_runtime_config_optimized")
    @patch("core.runtime_advisor.detect_hardware")
    def test_build_recommendations_includes_auto_and_all_modes(
        self,
        detect_hardware_mock,
        select_runtime_config_optimized_mock,
    ) -> None:
        hardware = _hardware()
        detect_hardware_mock.return_value = hardware
        select_runtime_config_optimized_mock.side_effect = lambda mode, _hardware: f"runtime:{mode}"

        recommendations = build_recommendations()

        self.assertEqual(
            recommendations,
            {
                "high": "runtime:high",
                "medium": "runtime:medium",
                "low": "runtime:low",
                "auto": "runtime:auto",
            },
        )


if __name__ == "__main__":
    unittest.main()
