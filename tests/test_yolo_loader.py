from __future__ import annotations

import unittest
from unittest.mock import patch

from core.hardware_detector import HardwareInfo
from core.model_selector import select_runtime_config
from core.yolo_loader import _candidate_paths, load_yolo_model


class YoloLoaderTests(unittest.TestCase):
    def _runtime(self):
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        return select_runtime_config("medium", hardware)

    @patch("core.yolo_loader._candidate_paths")
    @patch("core.yolo_loader.YOLO")
    def test_load_yolo_model_tries_candidates_until_success(self, yolo_mock, candidate_paths_mock) -> None:
        runtime = self._runtime()
        runtime.candidate_models = ["yolo26s.pt", "yolo11s.pt"]
        candidate_paths_mock.side_effect = [["missing-a.pt", "missing-b.pt"], ["working.pt"]]
        yolo_mock.side_effect = [RuntimeError("missing-a"), RuntimeError("missing-b"), object()]

        loaded_model, resolved_device = load_yolo_model(runtime)

        self.assertEqual(loaded_model.model_name, "yolo11s.pt")
        self.assertEqual(loaded_model.source_path, "working.pt")
        self.assertEqual(runtime.active_model_name, "yolo11s.pt")
        self.assertEqual(resolved_device, "cuda:0")

    @patch("core.yolo_loader._candidate_paths", return_value=["missing.pt"])
    @patch("core.yolo_loader.YOLO", side_effect=RuntimeError("boom"))
    def test_load_yolo_model_raises_aggregated_error(self, _yolo_mock, _candidate_paths_mock) -> None:
        runtime = self._runtime()
        runtime.candidate_models = ["yolo26s.pt"]

        with self.assertRaises(RuntimeError) as context:
            load_yolo_model(runtime)

        self.assertIn("Khong the load bat ky model local nao", str(context.exception))
        self.assertIn("missing.pt: boom", str(context.exception))

    @patch("core.yolo_loader.load_yaml_cached")
    @patch("pathlib.Path.exists", autospec=True)
    def test_candidate_paths_prioritize_trained_then_pretrained_then_name(
        self,
        exists_mock,
        load_yaml_mock,
    ) -> None:
        load_yaml_mock.return_value = {
            "priority_order": [
                "models/trained/best.pt",
                "models/pretrained/yolo26s.pt",
                "yolo26s.pt",
            ]
        }

        def fake_exists(path_obj):
            return str(path_obj).replace("\\", "/") in {
                "models/trained/best.pt",
                "models/pretrained/yolo26s.pt",
            }

        exists_mock.side_effect = fake_exists

        candidates = _candidate_paths("yolo26s.pt")

        self.assertEqual(
            [candidate.replace("\\", "/") for candidate in candidates],
            ["models/trained/best.pt", "models/pretrained/yolo26s.pt"],
        )

    @patch("core.yolo_loader._candidate_paths", return_value=[])
    def test_load_yolo_model_reports_missing_local_model(self, _candidate_paths_mock) -> None:
        runtime = self._runtime()
        runtime.candidate_models = ["yolo26s.pt"]

        with self.assertRaises(RuntimeError) as context:
            load_yolo_model(runtime)

        self.assertIn("khong tim thay file model local", str(context.exception))
