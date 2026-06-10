from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training import model_paths


class ModelPathsTests(unittest.TestCase):
    def test_resolve_model_source_prefers_existing_explicit_path(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            model_file = Path(temp_dir) / "custom.pt"
            model_file.write_text("model", encoding="utf-8")

            resolved = model_paths.resolve_model_source(model_file)

            self.assertEqual(resolved, model_file)

    def test_resolve_model_source_falls_back_to_pretrained_directory(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            pretrained_dir = Path(temp_dir)
            pretrained_model = pretrained_dir / "yolo11n.pt"
            pretrained_model.write_text("model", encoding="utf-8")

            with patch.object(model_paths, "PRETRAINED_MODELS_DIR", pretrained_dir):
                resolved = model_paths.resolve_model_source("yolo11n.pt")

            self.assertEqual(resolved, pretrained_model)

    def test_resolve_trained_model_path_uses_best_or_fallback(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            temp_root = Path(temp_dir)
            trained_best = temp_root / "best.pt"
            pretrained_dir = temp_root / "pretrained"
            pretrained_dir.mkdir(parents=True, exist_ok=True)
            fallback_model = pretrained_dir / "yolo11n.pt"
            fallback_model.write_text("fallback", encoding="utf-8")

            with patch.object(model_paths, "TRAINED_BEST_MODEL_PATH", trained_best), patch.object(
                model_paths, "PRETRAINED_MODELS_DIR", pretrained_dir
            ):
                self.assertEqual(
                    model_paths.resolve_trained_model_path(required=False, fallback="yolo11n.pt"),
                    fallback_model,
                )
                trained_best.write_text("best", encoding="utf-8")
                self.assertEqual(model_paths.resolve_trained_model_path(required=True), trained_best)

    def test_resolve_trained_model_path_raises_when_required_and_missing(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            trained_best = Path(temp_dir) / "best.pt"

            with patch.object(model_paths, "TRAINED_BEST_MODEL_PATH", trained_best):
                with self.assertRaises(FileNotFoundError):
                    model_paths.resolve_trained_model_path(required=True)

    @patch("training.model_paths.save_yaml")
    @patch("training.model_paths.load_yaml")
    def test_resolve_data_config_path_normalizes_dataset_root(self, load_yaml_mock, save_yaml_mock) -> None:
        load_yaml_mock.return_value = {"path": "../dataset/processed", "train": "images/train", "val": "images/val"}

        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            config_path = Path(temp_dir) / "data.yaml"
            generated_path = Path(temp_dir) / ".generated_data.yaml"
            with patch.object(model_paths, "TRAINING_DATA_CONFIG_PATH", config_path), patch.object(
                model_paths, "GENERATED_DATA_CONFIG_PATH", generated_path
            ):
                resolved = model_paths.resolve_data_config_path()

        self.assertEqual(resolved, generated_path)
        saved_payload = save_yaml_mock.call_args.args[1]
        self.assertEqual(Path(saved_payload["path"]), config_path.parent.joinpath("../dataset/processed").resolve())


if __name__ == "__main__":
    unittest.main()
