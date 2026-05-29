from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.file_utils import ensure_project_directories, load_yaml, save_yaml


class FileUtilsTests(unittest.TestCase):
    def test_save_yaml_and_load_yaml_roundtrip(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            yaml_path = Path(temp_dir) / "sample.yaml"
            payload = {
                "system": {"auto_detect_hardware": True},
                "camera": {"width": 1000, "height": 650},
            }
            save_yaml(yaml_path, payload)
            loaded = load_yaml(yaml_path)
            self.assertEqual(loaded, payload)

    def test_ensure_project_directories_creates_expected_structure(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                import os

                os.chdir(temp_dir)
                ensure_project_directories()
                self.assertTrue(Path("dataset/raw/images").is_dir())
                self.assertTrue(Path("dataset/processed/labels/test").is_dir())
                self.assertTrue(Path("output/logs").is_dir())
                self.assertTrue(Path("runs/detect").is_dir())
            finally:
                os.chdir(previous_cwd)
