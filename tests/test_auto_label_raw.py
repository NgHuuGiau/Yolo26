from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from training import auto_label_raw


class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def tolist(self):
        return list(self._value)


class _FakeBox:
    def __init__(self, cls_id: int, xyxy: list[float]):
        self.cls = [_FakeTensor(cls_id)]
        self.xyxy = [_FakeTensor(xyxy)]


class AutoLabelRawTests(unittest.TestCase):
    @patch("training.auto_label_raw._resolve_auto_label_model_path", return_value=Path("models/pretrained/yolo11s.pt"))
    @patch("training.auto_label_raw._require_yolo")
    def test_auto_label_raw_images_generates_txt_files(self, require_yolo_mock, _model_path_mock) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "dataset/raw/images"
            label_dir = root / "dataset/raw/labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 80), color="white").save(image_dir / "sample.jpg")

            fake_model = SimpleNamespace(
                predict=lambda **kwargs: [SimpleNamespace(boxes=[_FakeBox(0, [10.0, 20.0, 60.0, 70.0])])]
            )
            require_yolo_mock.return_value = lambda model_path: fake_model

            current = Path.cwd()
            try:
                import os

                os.chdir(root)
                report = auto_label_raw.auto_label_raw_images()
            finally:
                os.chdir(current)

            label_path = label_dir / "sample.txt"
            self.assertTrue(label_path.exists())
            content = label_path.read_text(encoding="utf-8").strip()
            self.assertTrue(content.startswith("0 "))
            self.assertEqual(report["generated"], 1)
            self.assertEqual(report["no_detection"], [])

    @patch("training.auto_label_raw._resolve_auto_label_model_path", return_value=Path("models/pretrained/yolo11s.pt"))
    @patch("training.auto_label_raw._require_yolo")
    def test_auto_label_raw_images_reports_images_with_no_detection(self, require_yolo_mock, _model_path_mock) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "dataset/raw/images"
            label_dir = root / "dataset/raw/labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color="white").save(image_dir / "empty.jpg")

            fake_model = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(boxes=[])])
            require_yolo_mock.return_value = lambda model_path: fake_model

            current = Path.cwd()
            try:
                import os

                os.chdir(root)
                report = auto_label_raw.auto_label_raw_images()
            finally:
                os.chdir(current)

            self.assertFalse((label_dir / "empty.txt").exists())
            self.assertEqual(report["generated"], 0)
            self.assertEqual(report["no_detection"], ["empty.jpg"])


if __name__ == "__main__":
    unittest.main()
