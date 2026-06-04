from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from training import export_model, train_model, validate_model


class TrainingPipelineTests(unittest.TestCase):
    @patch("training.train_model._auto_prepare_training_dataset")
    @patch("training.train_model._ensure_training_dataset_ready")
    @patch("training.train_model._copy_best_weight")
    @patch("training.train_model.YOLO")
    @patch("training.train_model.load_yaml")
    def test_train_main_falls_back_to_lighter_model(
        self, load_yaml_mock, yolo_mock, copy_best_mock, ensure_dataset_ready_mock, auto_prepare_mock
    ) -> None:
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
        ensure_dataset_ready_mock.return_value = None
        auto_prepare_mock.return_value = {"raw_images": 0, "auto_labeled": 0, "eligible": 0, "no_detection": []}

        train_model.main()

        fallback_model.train.assert_called_once()
        kwargs = fallback_model.train.call_args.kwargs
        self.assertEqual(kwargs["model"], "yolo26n.pt")
        self.assertEqual(kwargs["imgsz"], 416)
        self.assertEqual(kwargs["batch"], 4)
        copy_best_mock.assert_called_once()
        auto_prepare_mock.assert_called_once()

    @patch("training.train_model._copy_split")
    @patch("training.train_model._split_items", return_value={"train": [("img", "lbl")], "val": [], "test": []})
    @patch("training.train_model._reset_processed_dirs")
    @patch("training.train_model.auto_label_raw_images", return_value={"generated": 2, "no_detection": ["c.jpg"]})
    @patch("training.train_model.audit_raw_dataset")
    def test_auto_prepare_training_dataset_auto_labels_and_splits(
        self,
        audit_mock,
        auto_label_mock,
        reset_mock,
        split_items_mock,
        copy_split_mock,
    ) -> None:
        first_audit = SimpleNamespace(raw_image_count=2, missing_labels=[Path("a.jpg")], eligible=[])
        second_audit = SimpleNamespace(raw_image_count=2, missing_labels=[], eligible=[("img", "lbl")])
        audit_mock.side_effect = [first_audit, second_audit]

        report = train_model._auto_prepare_training_dataset()

        auto_label_mock.assert_called_once_with(overwrite=False, conf=0.25, device="cpu")
        reset_mock.assert_called_once()
        split_items_mock.assert_called_once_with([("img", "lbl")])
        copy_split_mock.assert_any_call("train", [("img", "lbl")])
        self.assertEqual(copy_split_mock.call_count, 3)
        self.assertEqual(report["raw_images"], 2)
        self.assertEqual(report["auto_labeled"], 2)
        self.assertEqual(report["eligible"], 1)
        self.assertEqual(report["no_detection"], ["c.jpg"])

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
                train_model._copy_best_weight(run_dir)
                self.assertTrue(target.exists())
                self.assertEqual(target.read_text(encoding="utf-8"), "fake-weight")
            finally:
                if original is None and target.exists():
                    target.unlink()
                elif original is not None:
                    target.write_bytes(original)

    @patch("training.validate_model._ensure_validation_dataset_ready")
    @patch("training.validate_model.YOLO")
    def test_validate_model_falls_back_to_yolo26n_when_best_missing(self, yolo_mock, ensure_validation_ready_mock) -> None:
        model = MagicMock()
        yolo_mock.return_value = model
        ensure_validation_ready_mock.return_value = None
        with patch("training.validate_model.resolve_trained_model_path", return_value=Path("yolo26n.pt")):
            validate_model.main()
        yolo_mock.assert_called_once_with(str(Path("models/pretrained/yolo26n.pt")))
        model.val.assert_called_once()

    @patch("training.export_model.YOLO")
    def test_export_model_uses_best_weight(self, yolo_mock) -> None:
        model = MagicMock()
        yolo_mock.return_value = model
        with patch("training.export_model.resolve_trained_model_path", return_value=Path("models/trained/best.pt")):
            export_model.main()
        yolo_mock.assert_called_once_with(str(Path("models/trained/best.pt")))
        model.export.assert_called_once_with(format="onnx")
