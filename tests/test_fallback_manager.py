from __future__ import annotations

import unittest
from unittest.mock import patch

from core.fallback_manager import _resolve_fallback_device, iter_fallback_configs
from core.model_selector import RuntimeConfig


def _runtime(**overrides) -> RuntimeConfig:
    base = dict(
        mode="high",
        profile_name="high",
        requested_profile_name="high",
        requested_device="gpu",
        resolved_device="cuda:0",
        requested_model_name="yolo11x.pt",
        primary_model_name="yolo11x.pt",
        candidate_models=["yolo11x.pt"],
        requested_imgsz=960,
        imgsz=960,
        conf=0.25,
        max_det=200,
        use_half=True,
        camera_width=800,
        camera_height=600,
        font_size=16,
        box_thickness=2,
        label_font_scale=0.8,
        active_model_name="",
        hardware_tier="strong GPU",
        fallback_chain=[
            {"device": "gpu", "model": "yolo11l.pt", "imgsz": 768},
            {"device": "gpu", "model": "yolo11l.pt", "imgsz": 768},
            {"device": "cpu", "model": "yolo11n.pt", "imgsz": 320},
        ],
    )
    base.update(overrides)
    return RuntimeConfig(**base)


class FallbackManagerTests(unittest.TestCase):
    def test_resolve_fallback_device_honors_requested_device(self) -> None:
        runtime = _runtime(resolved_device="cuda:0")

        self.assertEqual(_resolve_fallback_device(runtime, "gpu"), "cuda:0")
        self.assertEqual(_resolve_fallback_device(runtime, "auto"), "cuda:0")
        self.assertEqual(_resolve_fallback_device(runtime, "cpu"), "cpu")

    def test_resolve_fallback_device_auto_uses_cpu_when_runtime_is_cpu(self) -> None:
        runtime = _runtime(resolved_device="cpu", use_half=False)

        self.assertEqual(_resolve_fallback_device(runtime, "auto"), "cpu")

    @patch("core.fallback_manager.build_candidates")
    @patch("core.fallback_manager.load_settings")
    def test_iter_fallback_configs_skips_duplicates_and_disables_half_on_cpu(
        self,
        load_settings_mock,
        build_candidates_mock,
    ) -> None:
        load_settings_mock.return_value = {"inference": {"use_half_for_cuda": True}}
        build_candidates_mock.side_effect = lambda model, _settings: [model, "backup.pt"]

        configs = list(iter_fallback_configs(_runtime()))

        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].primary_model_name, "yolo11l.pt")
        self.assertEqual(configs[0].resolved_device, "cuda:0")
        self.assertTrue(configs[0].use_half)
        self.assertEqual(configs[0].candidate_models, ["yolo11l.pt", "backup.pt"])
        self.assertEqual(configs[1].primary_model_name, "yolo11n.pt")
        self.assertEqual(configs[1].resolved_device, "cpu")
        self.assertFalse(configs[1].use_half)


if __name__ == "__main__":
    unittest.main()
