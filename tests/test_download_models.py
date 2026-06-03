from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training import download_models


class DownloadModelsTests(unittest.TestCase):
    @patch("training.download_models.urlretrieve")
    def test_download_models_downloads_requested_files(self, urlretrieve_mock) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            with patch.object(download_models, "PRETRAINED_DIR", Path(temp_dir)):
                downloaded, skipped = download_models.download_models(["yolo26n.pt", "yolo26s.pt"])
        self.assertEqual(downloaded, ["yolo26n.pt", "yolo26s.pt"])
        self.assertEqual(skipped, [])
        self.assertEqual(urlretrieve_mock.call_count, 2)

    @patch("training.download_models.urlretrieve")
    def test_download_models_skips_existing_when_not_forced(self, urlretrieve_mock) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            target_dir = Path(temp_dir)
            (target_dir / "yolo26n.pt").write_text("existing", encoding="utf-8")
            with patch.object(download_models, "PRETRAINED_DIR", target_dir):
                downloaded, skipped = download_models.download_models(["yolo26n.pt"])
        self.assertEqual(downloaded, [])
        self.assertEqual(skipped, ["yolo26n.pt"])
        urlretrieve_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
