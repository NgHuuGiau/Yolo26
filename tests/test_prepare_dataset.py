from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from training import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def test_main_creates_dataset_directories_and_prints_help(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                output = io.StringIO()
                with redirect_stdout(output):
                    prepare_dataset.main()
                self.assertTrue(Path("dataset/raw/images").exists())
                self.assertTrue(Path("dataset/raw/labels").exists())
                self.assertIn("Dataset folders are ready.", output.getvalue())
            finally:
                os.chdir(previous_cwd)
