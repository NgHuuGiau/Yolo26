from __future__ import annotations

import unittest

from core.fallback_manager import iter_fallback_configs
from core.hardware_info import HardwareInfo
from core.model_selector import build_candidates, select_runtime_config
from utils.file_utils import load_yaml


class ModelSelectorTests(unittest.TestCase):
    def test_auto_selects_high_for_strong_gpu(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core Ultra 9",
            ram_gb=32.0,
            gpu_name="NVIDIA GeForce RTX 4080 Laptop GPU",
            vram_gb=12.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        runtime = select_runtime_config("auto", hardware)
        self.assertEqual(runtime.profile_name, "high")
        self.assertEqual(runtime.primary_model_name, "yolo11x.pt")
        self.assertEqual(runtime.imgsz, 960)
        self.assertEqual(runtime.resolved_device, "cuda:0")
        self.assertEqual(runtime.hardware_tier, "strong GPU")
        self.assertTrue(runtime.use_half)

    def test_auto_selects_low_for_rtx_3050_ti_4gb(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            vram_gb=3.95,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        runtime = select_runtime_config("auto", hardware)
        self.assertEqual(runtime.profile_name, "low")
        self.assertEqual(runtime.primary_model_name, "yolo11n.pt")
        self.assertEqual(runtime.imgsz, 320)
        self.assertEqual(runtime.resolved_device, "cuda:0")
        self.assertTrue(runtime.use_half)

    def test_runtime_config_includes_hardware_tier(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i5-9300H",
            ram_gb=8.0,
            gpu_name="NVIDIA GeForce MX450",
            vram_gb=2.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        runtime = select_runtime_config("auto", hardware)
        self.assertEqual(runtime.hardware_tier, "entry GPU")
        self.assertEqual(runtime.profile_name, "low")
        self.assertEqual(runtime.primary_model_name, "yolo11n.pt")
        self.assertEqual(runtime.resolved_device, "cpu")

    def test_auto_selects_low_for_entry_gpu(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i5-9300H",
            ram_gb=8.0,
            gpu_name="NVIDIA GeForce MX450",
            vram_gb=2.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        runtime = select_runtime_config("auto", hardware)
        self.assertEqual(runtime.profile_name, "low")
        self.assertEqual(runtime.primary_model_name, "yolo11n.pt")
        self.assertEqual(runtime.imgsz, 320)
        self.assertEqual(runtime.resolved_device, "cpu")
        self.assertFalse(runtime.use_half)

    def test_auto_selects_cpu_balanced_profile_without_cuda(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="Không phát hiện GPU",
            vram_gb=0.0,
            cuda_available=False,
            os_name="Windows 11",
            gpu_count=0,
        )
        runtime = select_runtime_config("auto", hardware)
        self.assertEqual(runtime.profile_name, "low")
        self.assertEqual(runtime.primary_model_name, "yolo11n.pt")
        self.assertEqual(runtime.imgsz, 320)
        self.assertEqual(runtime.resolved_device, "cpu")
        self.assertFalse(runtime.use_half)

    def test_auto_selects_cpu_minimum_profile_for_weak_machine(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i3",
            ram_gb=4.0,
            gpu_name="Không phát hiện GPU",
            vram_gb=0.0,
            cuda_available=False,
            os_name="Windows 10",
            gpu_count=0,
        )
        runtime = select_runtime_config("auto", hardware)
        self.assertEqual(runtime.profile_name, "fallback_cpu_weak")
        self.assertEqual(runtime.primary_model_name, "yolo11n.pt")
        self.assertEqual(runtime.imgsz, 320)
        self.assertEqual(runtime.resolved_device, "cpu")
        self.assertFalse(runtime.use_half)

    def test_manual_high_downgrades_on_cpu_only_machine(self) -> None:
        hardware = HardwareInfo(
            cpu_name="AMD Ryzen 5",
            ram_gb=8.0,
            gpu_name="Không phát hiện GPU",
            vram_gb=0.0,
            cuda_available=False,
            os_name="Windows 11",
            gpu_count=0,
        )
        runtime = select_runtime_config("high", hardware)
        self.assertEqual(runtime.profile_name, "fallback_cpu")
        self.assertEqual(runtime.primary_model_name, "yolo11n.pt")
        self.assertEqual(runtime.imgsz, 320)
        self.assertEqual(runtime.resolved_device, "cpu")
        self.assertFalse(runtime.use_half)

    def test_manual_high_downgrades_on_weak_gpu(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i5",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce GTX 1050",
            vram_gb=3.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        runtime = select_runtime_config("high", hardware)
        self.assertEqual(runtime.profile_name, "low")
        self.assertEqual(runtime.primary_model_name, "yolo11n.pt")
        self.assertEqual(runtime.imgsz, 320)
        self.assertEqual(runtime.resolved_device, "cuda:0")
        self.assertTrue(runtime.use_half)

    def test_manual_low_uses_lightest_gpu_model(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        runtime = select_runtime_config("low", hardware)
        self.assertEqual(runtime.profile_name, "low")
        self.assertEqual(runtime.primary_model_name, "yolo11n.pt")
        self.assertEqual(runtime.imgsz, 320)
        self.assertEqual(runtime.resolved_device, "cuda:0")

    def test_high_mode_uses_large_camera_preset(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        runtime = select_runtime_config("high", hardware)
        self.assertEqual(runtime.camera_width, 800)
        self.assertEqual(runtime.camera_height, 600)
        self.assertEqual(runtime.imgsz, 640)
        self.assertEqual(runtime.max_det, 150)

    def test_all_modes_keep_same_display_camera_preset(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        sizes = {
            mode: (
                select_runtime_config(mode, hardware).camera_width,
                select_runtime_config(mode, hardware).camera_height,
            )
            for mode in ["auto", "high", "medium", "low"]
        }
        self.assertEqual(
            sizes,
            {
                "auto": (800, 600),
                "high": (800, 600),
                "medium": (800, 600),
                "low": (800, 600),
            },
        )

    def test_build_candidates_uses_backup_order(self) -> None:
        settings = load_yaml("config/settings.yaml")
        candidates = build_candidates("yolo11x.pt", settings)
        self.assertEqual(candidates, ["yolo11x.pt", "yolo11l.pt", "yolo11m.pt", "yolo11s.pt", "yolo11n.pt"])

    def test_fallback_configs_are_unique_and_degrade_to_cpu(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        runtime = select_runtime_config("high", hardware)
        fallbacks = list(iter_fallback_configs(runtime))
        self.assertGreaterEqual(len(fallbacks), 4)
        self.assertEqual(fallbacks[-1].resolved_device, "cpu")
        self.assertEqual(fallbacks[-1].primary_model_name, "yolo11n.pt")
