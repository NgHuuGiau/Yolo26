from __future__ import annotations

import unittest

from core.fallback_manager import iter_fallback_configs
from core.hardware_detector import HardwareInfo
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
        self.assertEqual(runtime.primary_model_name, "yolo26s.pt")
        self.assertEqual(runtime.imgsz, 768)
        self.assertEqual(runtime.resolved_device, "cuda:0")
        self.assertTrue(runtime.use_half)

    def test_auto_selects_medium_for_rtx_3050_ti_4gb(self) -> None:
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
        self.assertEqual(runtime.profile_name, "medium")
        self.assertEqual(runtime.primary_model_name, "yolo26s.pt")
        self.assertEqual(runtime.imgsz, 640)
        self.assertEqual(runtime.resolved_device, "cuda:0")
        self.assertTrue(runtime.use_half)

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
        self.assertEqual(runtime.primary_model_name, "yolo26n.pt")
        self.assertEqual(runtime.imgsz, 512)
        self.assertEqual(runtime.resolved_device, "cuda:0")
        self.assertTrue(runtime.use_half)

    def test_auto_selects_cpu_balanced_profile_without_cuda(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="Khong phat hien GPU",
            vram_gb=0.0,
            cuda_available=False,
            os_name="Windows 11",
            gpu_count=0,
        )
        runtime = select_runtime_config("auto", hardware)
        self.assertEqual(runtime.profile_name, "low")
        self.assertEqual(runtime.primary_model_name, "yolo26n.pt")
        self.assertEqual(runtime.imgsz, 512)
        self.assertEqual(runtime.resolved_device, "cpu")
        self.assertFalse(runtime.use_half)

    def test_auto_selects_cpu_minimum_profile_for_weak_machine(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i3",
            ram_gb=4.0,
            gpu_name="Khong phat hien GPU",
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
            gpu_name="Khong phat hien GPU",
            vram_gb=0.0,
            cuda_available=False,
            os_name="Windows 11",
            gpu_count=0,
        )
        runtime = select_runtime_config("high", hardware)
        self.assertEqual(runtime.profile_name, "fallback_cpu")
        self.assertEqual(runtime.primary_model_name, "yolo26n.pt")
        self.assertEqual(runtime.imgsz, 416)
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
        self.assertEqual(runtime.primary_model_name, "yolo26n.pt")
        self.assertEqual(runtime.imgsz, 512)
        self.assertEqual(runtime.resolved_device, "cuda:0")
        self.assertTrue(runtime.use_half)

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
        self.assertEqual(runtime.camera_width, 1200)
        self.assertEqual(runtime.camera_height, 750)
        self.assertEqual(runtime.imgsz, 768)
        self.assertEqual(runtime.max_det, 100)

    def test_all_modes_keep_same_large_display_size(self) -> None:
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
                "auto": (1200, 750),
                "high": (1200, 750),
                "medium": (1200, 750),
                "low": (1200, 750),
            },
        )

    def test_build_candidates_uses_backup_order(self) -> None:
        settings = load_yaml("config/settings.yaml")
        candidates = build_candidates("yolo26s.pt", settings)
        self.assertEqual(candidates, ["yolo26s.pt", "yolo11s.pt", "yolov8s.pt"])

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
