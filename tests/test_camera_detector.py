from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.camera_detector import CameraDetector, DetectionRecord
from core.hardware_detector import HardwareInfo
from core.model_selector import select_runtime_config
from core.yolo_loader import LoadedModel


class _FakeValue:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeTensorRow:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [_FakeValue(cls_id)]
        self.conf = [_FakeValue(conf)]
        self.xyxy = [_FakeTensorRow(xyxy)]


class CameraDetectorTests(unittest.TestCase):
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

    @patch("core.camera_detector.cv2.VideoCapture")
    @patch("core.camera_detector.iter_fallback_configs")
    @patch("core.camera_detector.load_yolo_model")
    def test_initialize_uses_fallback_runtime_when_primary_fails(
        self,
        load_model_mock,
        fallback_mock,
        video_capture_mock,
    ) -> None:
        primary_runtime = self._runtime()
        fallback_runtime = self._runtime()
        fallback_runtime.primary_model_name = "yolo26n.pt"
        fallback_runtime.imgsz = 416

        model = MagicMock()
        loaded = LoadedModel(model=model, model_name="yolo26n.pt", source_path="working.pt")
        load_model_mock.side_effect = [RuntimeError("cuda oom"), (loaded, "cuda:0")]
        fallback_mock.return_value = [fallback_runtime]

        capture = MagicMock()
        capture.isOpened.return_value = True
        video_capture_mock.return_value = capture

        detector = CameraDetector(runtime=primary_runtime)
        detector.initialize()

        self.assertEqual(detector.runtime.primary_model_name, "yolo26n.pt")
        self.assertIs(detector.capture, capture)

    @patch("core.camera_detector.draw_detection_results", side_effect=lambda image, detections, box_thickness, label_font_scale: image)
    @patch("core.camera_detector.cv2.VideoCapture")
    @patch("core.camera_detector.load_yolo_model")
    def test_read_and_detect_returns_parsed_detections(
        self,
        load_model_mock,
        video_capture_mock,
        _draw_mock,
    ) -> None:
        runtime = self._runtime()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = SimpleNamespace(
            names={0: "person"},
            boxes=[_FakeBox(0, 0.91, [10, 20, 100, 150])],
        )
        fake_model = MagicMock()
        fake_model.predict.return_value = [result]
        load_model_mock.return_value = (
            LoadedModel(model=fake_model, model_name="yolo26s.pt", source_path="yolo26s.pt"),
            "cuda:0",
        )

        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, frame.copy())
        video_capture_mock.return_value = capture

        detector = CameraDetector(runtime=runtime)
        detector.initialize()
        ok, processed_frame, detections, fps = detector.read_and_detect()

        self.assertTrue(ok)
        self.assertEqual(processed_frame.shape, frame.shape)
        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections[0], DetectionRecord)
        self.assertEqual(detections[0].label, "person")
        self.assertGreater(detections[0].confidence, 0.9)
        self.assertGreater(fps, 0.0)

    def test_parse_results_extracts_box_fields(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        results = [SimpleNamespace(names={1: "car"}, boxes=[_FakeBox(1, 0.5, [1, 2, 3, 4])])]
        detections = detector._parse_results(results)
        self.assertEqual(detections[0].bbox, (1, 2, 3, 4))
        self.assertEqual(detections[0].label, "car")

    @patch("core.camera_detector.cv2.VideoCapture")
    @patch("core.camera_detector.load_yolo_model")
    def test_read_and_detect_recovers_after_inference_failure(
        self,
        load_model_mock,
        video_capture_mock,
    ) -> None:
        runtime = self._runtime()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        fake_model = MagicMock()
        fake_model.predict.side_effect = RuntimeError("cuda error")
        load_model_mock.return_value = (
            LoadedModel(model=fake_model, model_name="yolo26s.pt", source_path="yolo26s.pt"),
            "cuda:0",
        )

        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, frame.copy())
        video_capture_mock.return_value = capture

        detector = CameraDetector(runtime=runtime)
        detector.initialize()

        with patch.object(detector, "initialize") as reinitialize_mock:
            ok, processed_frame, detections, fps = detector.read_and_detect()

        self.assertFalse(ok)
        self.assertIsNone(processed_frame)
        self.assertEqual(detections, [])
        self.assertEqual(fps, 0.0)
        self.assertEqual(detector.recovery_count, 1)
        reinitialize_mock.assert_called_once()

    @patch("core.camera_detector.cv2.VideoCapture")
    @patch("core.camera_detector.load_yolo_model")
    def test_read_and_detect_raises_after_many_camera_read_failures(
        self,
        load_model_mock,
        video_capture_mock,
    ) -> None:
        runtime = self._runtime()
        fake_model = MagicMock()
        load_model_mock.return_value = (
            LoadedModel(model=fake_model, model_name="yolo26s.pt", source_path="yolo26s.pt"),
            "cuda:0",
        )
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        video_capture_mock.return_value = capture

        detector = CameraDetector(runtime=runtime)
        detector.initialize()

        for _ in range(detector.max_consecutive_read_failures - 1):
            ok, frame, detections, fps = detector.read_and_detect()
            self.assertFalse(ok)
            self.assertIsNone(frame)
            self.assertEqual(detections, [])
            self.assertEqual(fps, 0.0)

        with self.assertRaises(RuntimeError):
            detector.read_and_detect()
