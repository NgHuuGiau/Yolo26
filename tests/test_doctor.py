from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import run_doctor


class DoctorTests(unittest.TestCase):
    def test_present_and_missing_models_reports_expected_values(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "yolo11n.pt").write_text("n", encoding="utf-8")
            (model_dir / "yolo11s.pt").write_text("s", encoding="utf-8")
            present, missing = run_doctor._present_and_missing_models(model_dir)
        self.assertEqual(present, ["yolo11n.pt", "yolo11s.pt"])
        self.assertEqual(missing, ["yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])

    def test_count_files_ignores_missing_directory(self) -> None:
        self.assertEqual(run_doctor._count_files(Path("missing-dir-for-doctor-test")), 0)

    @patch("run_doctor._open_camera_capture")
    def test_probe_camera_reports_warn_when_camera_cannot_open(self, open_camera_mock) -> None:
        capture = Mock()
        capture.isOpened.return_value = False
        open_camera_mock.return_value = capture

        result = run_doctor._probe_camera(0)

        self.assertEqual(result.level, "WARN")
        self.assertIn("camera index 0", result.summary)

    @patch("run_doctor._open_camera_capture")
    def test_probe_camera_reports_pass_when_frame_is_read(self, open_camera_mock) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, Mock(shape=(720, 1280, 3)))
        open_camera_mock.return_value = capture

        result = run_doctor._probe_camera(1)

        self.assertEqual(result.level, "PASS")
        self.assertIn("index 1", result.summary)
        self.assertIn("1280x720", result.detail)

    @patch("builtins.print")
    @patch("run_doctor.select_runtime_config")
    @patch("run_doctor.detect_hardware")
    @patch("run_doctor.ensure_project_directories")
    @patch("run_doctor.parse_args")
    def test_main_reports_missing_models_and_raw_dataset_guidance(
        self,
        parse_args_mock,
        _ensure_dirs_mock,
        detect_hardware_mock,
        select_runtime_config_mock,
        print_mock,
    ) -> None:
        parse_args_mock.return_value = Mock(camera_index=0, skip_camera_check=True, fix=False)
        detect_hardware_mock.return_value = Mock(
            cpu_name="Intel Core i7",
            ram_gb=16.0,
            os_name="Windows 11",
            gpu_name="KhĂ´ng phĂ¡t hiá»‡n GPU",
            gpu_count=0,
            vram_gb=0.0,
            torch_version="2.0",
            cuda_runtime_reason="CPU-only",
            cuda_available=False,
            gpu_hardware_available=False,
        )
        select_runtime_config_mock.side_effect = [
            Mock(primary_model_name="yolo11n.pt", resolved_device="cpu", imgsz=320),
            Mock(primary_model_name="yolo11n.pt", resolved_device="cpu", imgsz=320),
            Mock(primary_model_name="yolo11n.pt", resolved_device="cpu", imgsz=320),
        ]

        with patch("run_doctor._present_and_missing_models", return_value=([], ["yolo11n.pt"])), patch(
            "run_doctor._count_files",
            side_effect=[0, 0, 0, 0, 0],
        ):
            run_doctor.main()

        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("download_models.py", output)
        self.assertIn("prepare_dataset.py", output)
        self.assertIn("model local", output)


if __name__ == "__main__":
    unittest.main()
