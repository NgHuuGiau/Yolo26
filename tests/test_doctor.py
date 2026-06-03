from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run_doctor


class DoctorTests(unittest.TestCase):
    def test_present_and_missing_models_reports_expected_values(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "yolo26n.pt").write_text("n", encoding="utf-8")
            (model_dir / "yolo26s.pt").write_text("s", encoding="utf-8")
            present, missing = run_doctor._present_and_missing_models(model_dir)
        self.assertEqual(present, ["yolo26n.pt", "yolo26s.pt"])
        self.assertEqual(missing, ["yolo26m.pt", "yolo26l.pt", "yolo26x.pt"])

    def test_count_files_ignores_missing_directory(self) -> None:
        self.assertEqual(run_doctor._count_files(Path("missing-dir-for-doctor-test")), 0)


if __name__ == "__main__":
    unittest.main()
