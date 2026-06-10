from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from utils.console_ui import BootProgress


class BootProgressTests(unittest.TestCase):
    @patch("utils.console_ui.time.sleep", return_value=None)
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_progress_line_includes_title_and_status_label(self, fake_stdout, _sleep_mock) -> None:
        progress = BootProgress("YOLO Camera Realtime", enabled=True)

        progress.advance_to(42, "Đang kiểm tra CUDA")

        rendered = fake_stdout.getvalue()
        self.assertIn("YOLO Camera Realtime", rendered)
        self.assertIn("Đang kiểm tra CUDA", rendered)
        self.assertIn("Mức sẵn sàng", rendered)
        self.assertIn("0 [", rendered)

    def test_disabled_progress_still_tracks_latest_label_and_percent(self) -> None:
        progress = BootProgress("YOLO", enabled=False)

        progress.advance_to(42, "Đang chọn model")

        self.assertEqual(progress.current, 42)
        self.assertEqual(progress.current_label, "Đang chọn model")


if __name__ == "__main__":
    unittest.main()
