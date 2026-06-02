from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.hardware_info import HardwareInfo, detect_hardware


class HardwareDetectorTests(unittest.TestCase):
    def test_pretty_report_contains_expected_fields(self) -> None:
        info = HardwareInfo(
            cpu_name="Intel Core i7",
            ram_gb=16.0,
            gpu_name="RTX 3050 Ti",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        report = info.pretty_report()
        self.assertIn("CPU: Intel Core i7", report)
        self.assertIn("GPU: RTX 3050 Ti", report)
        self.assertIn("CUDA: Co", report)

    @patch("core.hardware_info.platform.release", return_value="11")
    @patch("core.hardware_info.platform.system", return_value="Windows")
    @patch("core.hardware_info._detect_cpu_name", return_value="Intel Core i7-11800H")
    @patch("core.hardware_info.psutil.virtual_memory")
    @patch("core.hardware_info._detect_gpu_from_gputil")
    @patch("core.hardware_info.torch")
    def test_detect_hardware_prefers_torch_cuda_details(
        self,
        torch_mock,
        gpu_detect_mock,
        virtual_memory_mock,
        _cpu_mock,
        _system_mock,
        _release_mock,
    ) -> None:
        virtual_memory_mock.return_value = SimpleNamespace(total=16 * (1024**3))
        gpu_detect_mock.return_value = ("Fallback GPU", 2.0, 1)
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.current_device.return_value = 0
        torch_mock.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3050 Ti Laptop GPU"
        torch_mock.cuda.get_device_properties.return_value = SimpleNamespace(total_memory=4 * (1024**3))
        torch_mock.cuda.device_count.return_value = 1

        info = detect_hardware()

        self.assertEqual(info.cpu_name, "Intel Core i7-11800H")
        self.assertEqual(info.gpu_name, "NVIDIA GeForce RTX 3050 Ti Laptop GPU")
        self.assertAlmostEqual(info.vram_gb, 4.0, places=1)
        self.assertTrue(info.cuda_available)
        self.assertEqual(info.gpu_count, 1)
        self.assertEqual(info.os_name, "Windows 11")
        self.assertEqual(info.summary()["gpu_count"], 1)
