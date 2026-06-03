from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training import promote_samples


class PromoteSamplesTests(unittest.TestCase):
    def test_promote_samples_copies_matching_pairs(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            root = Path(temp_dir)
            sample_images = root / "sample_images"
            sample_labels = root / "sample_labels"
            raw_images = root / "raw_images"
            raw_labels = root / "raw_labels"
            for path in (sample_images, sample_labels, raw_images, raw_labels):
                path.mkdir(parents=True, exist_ok=True)
            (sample_images / "a.jpg").write_text("img", encoding="utf-8")
            (sample_labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
            with (
                patch.object(promote_samples, "SAMPLE_IMAGES_DIR", sample_images),
                patch.object(promote_samples, "SAMPLE_LABELS_DIR", sample_labels),
                patch.object(promote_samples, "RAW_IMAGES_DIR", raw_images),
                patch.object(promote_samples, "RAW_LABELS_DIR", raw_labels),
            ):
                report = promote_samples.promote_samples()
                self.assertEqual(report["moved"], 1)
                self.assertTrue((raw_images / "a.jpg").exists())
                self.assertTrue((raw_labels / "a.txt").exists())

    def test_promote_samples_reports_missing_labels(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            root = Path(temp_dir)
            sample_images = root / "sample_images"
            sample_labels = root / "sample_labels"
            raw_images = root / "raw_images"
            raw_labels = root / "raw_labels"
            for path in (sample_images, sample_labels, raw_images, raw_labels):
                path.mkdir(parents=True, exist_ok=True)
            (sample_images / "a.jpg").write_text("img", encoding="utf-8")
            with (
                patch.object(promote_samples, "SAMPLE_IMAGES_DIR", sample_images),
                patch.object(promote_samples, "SAMPLE_LABELS_DIR", sample_labels),
                patch.object(promote_samples, "RAW_IMAGES_DIR", raw_images),
                patch.object(promote_samples, "RAW_LABELS_DIR", raw_labels),
            ):
                report = promote_samples.promote_samples()
        self.assertEqual(report["moved"], 0)
        self.assertEqual(report["missing_labels"], 1)


if __name__ == "__main__":
    unittest.main()
