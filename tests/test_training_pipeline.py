from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from training import export_model, train_yolo, validate_model


class TrainingPipelineTests(unittest.TestCase):
    @patch("training.train_yolo._copy_best_weight")
    @patch("training.train_yolo.YOLO")
    @patch("training.train_yolo.load_yaml")
    def test_train_main_falls_back_to_lighter_model(self, load_yaml_mock, yolo_mock, copy_best_mock) -> None:
        load_yaml_mock.return_value = {
            "model": "yolo26s.pt",
            "fallback_model": "yolo26n.pt",
            "data": "training/data.yaml",
            "epochs": 2,
            "imgsz": 512,
            "batch": 8,
            "device": 0,
            "workers": 2,
            "cache": False,
            "amp": True,
            "patience": 20,
            "project": "runs/train",
            "name": "test-run",
        }

        primary_model = MagicMock()
        primary_model.train.side_effect = RuntimeError("oom")
        fallback_model = MagicMock()
        fallback_model.train.return_value = SimpleNamespace(save_dir="runs/train/test-run")
        yolo_mock.side_effect = [primary_model, fallback_model]

        train_yolo.main()

        fallback_model.train.assert_called_once()
        kwargs = fallback_model.train.call_args.kwargs
        self.assertEqual(kwargs["model"], "yolo26n.pt")
        self.assertEqual(kwargs["imgsz"], 416)
        self.assertEqual(kwargs["batch"], 4)
        copy_best_mock.assert_called_once()

    def test_copy_best_weight_copies_when_present(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            run_dir = Path(temp_dir)
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            best_path = weights_dir / "best.pt"
            best_path.write_text("fake-weight", encoding="utf-8")

            target = Path("models/trained/best.pt")
            original = target.read_bytes() if target.exists() else None
            try:
                train_yolo._copy_best_weight(run_dir)
                self.assertTrue(target.exists())
                self.assertEqual(target.read_text(encoding="utf-8"), "fake-weight")
            finally:
                if original is None and target.exists():
                    target.unlink()
                elif original is not None:
                    target.write_bytes(original)

    @patch("training.validate_model.YOLO")
    def test_validate_model_falls_back_to_yolo11n_when_best_missing(self, yolo_mock) -> None:
        model = MagicMock()
        yolo_mock.return_value = model
        with patch("training.validate_model.Path.exists", return_value=False):
            validate_model.main()
        yolo_mock.assert_called_once_with("yolo11n.pt")
        model.val.assert_called_once()

    @patch("training.export_model.YOLO")
    def test_export_model_uses_best_weight(self, yolo_mock) -> None:
        model = MagicMock()
        yolo_mock.return_value = model
        with patch("training.export_model.Path.exists", return_value=True):
            export_model.main()
        yolo_mock.assert_called_once_with(str(Path("models/trained/best.pt")))
        model.export.assert_called_once_with(format="onnx")
