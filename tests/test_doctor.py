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
        self.assertIn("Không mở được camera index 0", result.summary)


if __name__ == "__main__":
    unittest.main()
