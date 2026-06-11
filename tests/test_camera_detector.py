from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.camera_runner import (
    CameraDetector,
    DetectionRecord,
    _bbox_iou,
    _dedupe_display_detections,
    _filter_person_detections,
    _fps_tolerance_for_profile,
    _target_fps_for_profile,
    _should_force_camera_only_preview,
)
from core.hardware_info import HardwareInfo
from core.model_selector import select_runtime_config
from core.model_loader import LoadedModel


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

    @patch("core.camera_runner.cv2.VideoCapture")
    @patch("core.camera_runner.iter_fallback_configs")
    @patch("core.camera_runner.load_yolo_model")
    def test_initialize_uses_fallback_runtime_when_primary_fails(
        self,
        load_model_mock,
        fallback_mock,
        video_capture_mock,
    ) -> None:
        primary_runtime = self._runtime()
        fallback_runtime = self._runtime()
        fallback_runtime.primary_model_name = "yolo11n.pt"
        fallback_runtime.imgsz = 416

        model = MagicMock()
        loaded = LoadedModel(model=model, model_name="yolo11n.pt", source_path="working.pt")
        load_model_mock.side_effect = [RuntimeError("cuda oom"), (loaded, "cuda:0")]
        fallback_mock.return_value = [fallback_runtime]

        capture = MagicMock()
        capture.isOpened.return_value = True
        video_capture_mock.return_value = capture

        detector = CameraDetector(runtime=primary_runtime)
        detector.initialize()

        self.assertEqual(detector.runtime.primary_model_name, "yolo11n.pt")
        self.assertIs(detector.capture, capture)

    @patch(
        "core.camera_runner.draw_detection_results",
        side_effect=lambda image, detections, box_thickness, label_font_scale, motion_trails=None: image,
    )
    @patch("core.camera_runner.cv2.VideoCapture")
    @patch("core.camera_runner.load_yolo_model")
    def test_read_and_detect_returns_parsed_detections(
        self,
        load_model_mock,
        video_capture_mock,
        _draw_mock,
    ) -> None:
        runtime = self._runtime()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = SimpleNamespace(
            names={1: "car"},
            boxes=[_FakeBox(1, 0.91, [10, 20, 100, 150])],
        )
        fake_model = MagicMock()
        fake_model.predict.return_value = [result]
        load_model_mock.return_value = (
            LoadedModel(model=fake_model, model_name="yolo11s.pt", source_path="yolo11s.pt"),
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
        self.assertEqual(detections[0].label, "car")
        self.assertGreater(detections[0].confidence, 0.9)
        self.assertGreater(fps, 0.0)

    def test_parse_results_extracts_box_fields(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        results = [SimpleNamespace(names={1: "car"}, boxes=[_FakeBox(1, 0.5, [1, 2, 3, 4])])]
        detections = detector._parse_results(results)
        self.assertEqual(detections[0].bbox, (1, 2, 3, 4))
        self.assertEqual(detections[0].label, "car")
        self.assertEqual(detections[0].class_id, 1)

    def test_parse_results_keeps_person_box_for_person_class(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        results = [SimpleNamespace(names={0: "person"}, boxes=[_FakeBox(0, 0.91, [40, 20, 120, 180])])]

        detections = detector._parse_results(results)

        self.assertEqual(detections[0].label, "person")
        self.assertEqual(detections[0].bbox, (40, 20, 120, 180))

    def test_parse_results_normalizes_cell_phone_to_phone(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        results = [SimpleNamespace(names={1: "cell phone"}, boxes=[_FakeBox(1, 0.88, [50, 30, 90, 110])])]

        detections = detector._parse_results(results)

        self.assertEqual(detections[0].label, "phone")
        self.assertEqual(detections[0].bbox, (50, 30, 90, 110))

    @patch("core.camera_runner.cv2.VideoCapture")
    @patch("core.camera_runner.load_yolo_model")
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
            LoadedModel(model=fake_model, model_name="yolo11s.pt", source_path="yolo11s.pt"),
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

    @patch("core.camera_runner.cv2.VideoCapture")
    @patch("core.camera_runner.load_yolo_model")
    def test_read_and_detect_raises_after_many_camera_read_failures(
        self,
        load_model_mock,
        video_capture_mock,
    ) -> None:
        runtime = self._runtime()
        fake_model = MagicMock()
        load_model_mock.return_value = (
            LoadedModel(model=fake_model, model_name="yolo11s.pt", source_path="yolo11s.pt"),
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

    def test_adjust_detect_interval_increases_when_fps_is_below_target(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        detector = CameraDetector(runtime=select_runtime_config("low", hardware))
        detector.base_detect_interval = 5
        detector.detect_interval = 5
        detector.max_detect_interval = 8
        detector.frame_index = 24
        detector.last_detect_adjust_frame = 0

        detector._adjust_detect_interval(26.0)

        self.assertEqual(detector.detect_interval, 6)

    def test_adjust_detect_interval_decreases_when_fps_is_above_target(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        detector = CameraDetector(runtime=select_runtime_config("low", hardware))
        detector.base_detect_interval = 5
        detector.detect_interval = 7
        detector.max_detect_interval = 8
        detector.frame_index = 24
        detector.last_detect_adjust_frame = 0

        detector._adjust_detect_interval(34.5)

        self.assertEqual(detector.detect_interval, 6)

    def test_low_profile_uses_more_aggressive_inference_limits(self) -> None:
        hardware = HardwareInfo(
            cpu_name="Intel Core i7-11800H",
            ram_gb=16.0,
            gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            gpu_count=1,
        )
        detector = CameraDetector(runtime=select_runtime_config("low", hardware))

        self.assertEqual(detector._effective_inference_imgsz(), 320)
        self.assertEqual(detector._effective_max_det(), 10)
        self.assertEqual(detector._effective_confidence(), 0.35)
        self.assertEqual(detector._effective_box_thickness(), 2)
        self.assertGreaterEqual(detector._effective_label_font_scale(), 0.62)

    def test_fps_policy_targets_match_requested_profiles(self) -> None:
        self.assertEqual(_target_fps_for_profile("high"), 15)
        self.assertEqual(_target_fps_for_profile("medium"), 18)
        self.assertEqual(_target_fps_for_profile("low"), 30)
        self.assertEqual(_fps_tolerance_for_profile("high"), 1.0)
        self.assertEqual(_fps_tolerance_for_profile("medium"), 2.5)
        self.assertEqual(_fps_tolerance_for_profile("low"), 1.5)
        self.assertTrue(_should_force_camera_only_preview("high"))
        self.assertTrue(_should_force_camera_only_preview("medium"))
        self.assertTrue(_should_force_camera_only_preview("low"))

    def test_read_and_detect_suppresses_detections_on_high_motion(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.camera_stream = MagicMock()
        moved_frame = np.full((80, 80, 3), 200, dtype=np.uint8)
        detector.camera_stream.read_latest_frame.return_value = moved_frame
        detector.camera_stream.last_error_message = ""
        detector.camera_stream.last_status_message = ""
        detector.camera_stream.consecutive_read_failures = 0
        detector.loaded_model = LoadedModel(model=MagicMock(), model_name="yolo11s.pt", source_path="yolo11s.pt")
        detector.runtime.resolved_device = "cpu"
        stable_gray = np.zeros((40, 40), dtype=np.uint8)
        detector.previous_gray = stable_gray
        detector.last_raw_frame = np.zeros((80, 80, 3), dtype=np.uint8)
        detector.loaded_model.model.predict.return_value = [
            SimpleNamespace(names={0: "person"}, boxes=[_FakeBox(0, 0.9, [10, 10, 50, 50])])
        ]
        detector.frame_index = 0

        ok, processed_frame, detections, fps = detector.read_and_detect()

        self.assertTrue(ok)
        self.assertEqual(processed_frame.shape, moved_frame.shape)
        self.assertEqual(len(detections), 1)
        self.assertIn("Đang theo dõi vật thể chuyển động", detector.last_status_message)

    def test_read_and_detect_shows_detections_on_stable_frame(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.camera_stream = MagicMock()
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        detector.camera_stream.read_latest_frame.return_value = frame
        detector.camera_stream.last_error_message = ""
        detector.camera_stream.last_status_message = ""
        detector.camera_stream.consecutive_read_failures = 0
        detector.loaded_model = LoadedModel(model=MagicMock(), model_name="yolo11s.pt", source_path="yolo11s.pt")
        detector.runtime.resolved_device = "cpu"
        detector.previous_gray = None
        detector.last_raw_frame = frame.copy()
        detector.loaded_model.model.predict.return_value = [
            SimpleNamespace(names={1: "car"}, boxes=[_FakeBox(1, 0.9, [10, 10, 50, 50])])
        ]
        detector.frame_index = 0

        ok, processed_frame, detections, fps = detector.read_and_detect()

        self.assertTrue(ok)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "car")

    def test_read_and_detect_keeps_person_detections_as_person(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.camera_stream = MagicMock()
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        detector.camera_stream.read_latest_frame.return_value = frame
        detector.camera_stream.last_error_message = ""
        detector.camera_stream.last_status_message = ""
        detector.camera_stream.consecutive_read_failures = 0
        detector.loaded_model = LoadedModel(model=MagicMock(), model_name="yolo11s.pt", source_path="yolo11s.pt")
        detector.runtime.resolved_device = "cpu"
        detector.previous_gray = None
        detector.last_raw_frame = frame.copy()
        detector.loaded_model.model.predict.return_value = [
            SimpleNamespace(names={0: "person"}, boxes=[_FakeBox(0, 0.9, [10, 10, 50, 50])])
        ]
        detector.frame_index = 0

        ok, processed_frame, detections, fps = detector.read_and_detect()

        self.assertTrue(ok)
        self.assertEqual(processed_frame.shape, frame.shape)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")

    def test_read_and_detect_normalizes_phone_alias_to_phone(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.camera_stream = MagicMock()
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        detector.camera_stream.read_latest_frame.return_value = frame
        detector.camera_stream.last_error_message = ""
        detector.camera_stream.last_status_message = ""
        detector.camera_stream.consecutive_read_failures = 0
        detector.loaded_model = LoadedModel(model=MagicMock(), model_name="yolo11s.pt", source_path="yolo11s.pt")
        detector.runtime.resolved_device = "cpu"
        detector.previous_gray = None
        detector.last_raw_frame = frame.copy()
        detector.loaded_model.model.predict.return_value = [
            SimpleNamespace(names={1: "cell phone"}, boxes=[_FakeBox(1, 0.9, [10, 10, 30, 40])])
        ]
        detector.frame_index = 0

        ok, processed_frame, detections, fps = detector.read_and_detect()

        self.assertTrue(ok)
        self.assertEqual(processed_frame.shape, frame.shape)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "phone")

    def test_read_and_detect_does_not_keep_stale_boxes_when_current_frame_has_no_detection(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.camera_stream = MagicMock()
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        detector.camera_stream.read_latest_frame.return_value = frame
        detector.camera_stream.last_error_message = ""
        detector.camera_stream.last_status_message = ""
        detector.camera_stream.consecutive_read_failures = 0
        detector.loaded_model = LoadedModel(model=MagicMock(), model_name="yolo11s.pt", source_path="yolo11s.pt")
        detector.runtime.resolved_device = "cpu"
        detector.previous_gray = None
        detector.last_raw_frame = frame.copy()
        detector.last_detections = [DetectionRecord(class_id=1, label="car", confidence=0.9, bbox=(10, 10, 50, 50))]
        detector.loaded_model.model.predict.return_value = [SimpleNamespace(names={1: "car"}, boxes=[])]
        detector.frame_index = 0

        ok, processed_frame, detections, fps = detector.read_and_detect()

        self.assertTrue(ok)
        self.assertEqual(detections, [])
        self.assertEqual(detector.last_detections, [])
        self.assertEqual(detector.previous_display_detections, [])

    def test_smooth_display_detections_tracks_same_person_without_keeping_fixed_box(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.previous_display_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.91, bbox=(20, 20, 80, 120))
        ]
        detector.previous_observed_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.91, bbox=(20, 20, 80, 120))
]

        result = detector._smooth_display_detections(
            [DetectionRecord(class_id=0, label="person", confidence=0.95, bbox=(30, 30, 90, 130))]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "person")
        self.assertEqual(result[0].bbox, (21, 21, 81, 121))
        self.assertEqual(detector.previous_display_detections[0].bbox, (21, 21, 81, 121))
        self.assertEqual(detector.previous_observed_detections[0].bbox, (30, 30, 90, 130))

    def test_smooth_display_detections_clears_previous_boxes_when_current_frame_is_empty(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.previous_display_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.91, bbox=(20, 20, 80, 120))
        ]
        detector.previous_observed_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.91, bbox=(20, 20, 80, 120))
        ]

        result = detector._smooth_display_detections([])

        self.assertEqual(result, [])
        self.assertEqual(detector.previous_display_detections, [])
        self.assertEqual(detector.previous_observed_detections, [])

    def test_smooth_display_detections_matches_fast_motion_using_center_distance(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.previous_display_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.88, bbox=(27, 27, 87, 127))
        ]
        detector.previous_observed_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.91, bbox=(30, 30, 90, 130))
        ]

        result = detector._smooth_display_detections(
            [DetectionRecord(class_id=0, label="person", confidence=0.94, bbox=(55, 55, 115, 155))]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox, (54, 54, 114, 154))

    def test_smooth_display_detections_reduces_small_jitter_between_frames(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.previous_display_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.90, bbox=(40, 40, 100, 160))
        ]
        detector.previous_observed_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.90, bbox=(40, 40, 100, 160))
        ]

        result = detector._smooth_display_detections(
            [DetectionRecord(class_id=0, label="person", confidence=0.92, bbox=(42, 41, 102, 161))]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox, (40, 40, 100, 160))

    def test_smooth_display_detections_keeps_face_box_stable_for_tiny_motion(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.previous_display_detections = [
            DetectionRecord(class_id=0, label="face", confidence=0.90, bbox=(40, 40, 100, 100))
        ]
        detector.previous_observed_detections = [
            DetectionRecord(class_id=0, label="face", confidence=0.90, bbox=(40, 40, 100, 100))
        ]

        result = detector._smooth_display_detections(
            [DetectionRecord(class_id=0, label="face", confidence=0.92, bbox=(41, 41, 101, 101))]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox, (40, 40, 100, 100))

    def test_smooth_display_detections_allows_face_box_to_follow_real_motion(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.previous_display_detections = [
            DetectionRecord(class_id=0, label="face", confidence=0.90, bbox=(40, 40, 100, 100))
        ]
        detector.previous_observed_detections = [
            DetectionRecord(class_id=0, label="face", confidence=0.90, bbox=(40, 40, 100, 100))
        ]

        result = detector._smooth_display_detections(
            [DetectionRecord(class_id=0, label="face", confidence=0.93, bbox=(58, 52, 118, 112))]
        )

        self.assertEqual(len(result), 1)
        self.assertNotEqual(result[0].bbox, (40, 40, 100, 100))

    def test_smooth_display_detections_keeps_track_id_and_updates_motion_trail(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.previous_display_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.90, bbox=(40, 40, 100, 160), track_id=7)
        ]
        detector.previous_observed_detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.90, bbox=(40, 40, 100, 160), track_id=7)
        ]
        detector.display_trails = {7: [(70, 100)]}

        result = detector._smooth_display_detections(
            [DetectionRecord(class_id=0, label="person", confidence=0.93, bbox=(60, 55, 120, 175))]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].track_id, 7)
        self.assertIn(7, detector.display_trails)
        self.assertGreaterEqual(len(detector.display_trails[7]), 2)

    def test_smooth_display_detections_assigns_track_id_for_new_person(self) -> None:
        detector = CameraDetector(runtime=self._runtime())

        result = detector._smooth_display_detections(
            [DetectionRecord(class_id=0, label="person", confidence=0.95, bbox=(30, 30, 90, 150))]
        )

        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0].track_id, 1)
        self.assertIn(result[0].track_id, detector.display_trails)

    def test_effective_display_detections_keeps_multiple_people_when_not_overlapping(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.72, bbox=(10, 10, 50, 50)),
            DetectionRecord(class_id=0, label="person", confidence=0.95, bbox=(80, 12, 130, 62)),
            DetectionRecord(class_id=0, label="person", confidence=0.48, bbox=(140, 60, 180, 100)),
        ]

        result = detector._effective_display_detections(detections)

        self.assertEqual(len(result), 2)
        self.assertTrue(all(item.label == "person" for item in result))
        self.assertEqual([round(item.confidence, 2) for item in result], [0.95, 0.72])

    def test_effective_display_detections_removes_overlapping_people_duplicates(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.72, bbox=(10, 10, 50, 50)),
            DetectionRecord(class_id=0, label="person", confidence=0.95, bbox=(12, 12, 52, 52)),
            DetectionRecord(class_id=0, label="person", confidence=0.81, bbox=(100, 20, 150, 80)),
        ]

        result = detector._effective_display_detections(detections)

        self.assertEqual(len(result), 2)
        self.assertEqual([round(item.confidence, 2) for item in result], [0.95, 0.81])

    def test_effective_display_detections_filters_low_confidence_noise(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detections = [
            DetectionRecord(class_id=1, label="car", confidence=0.44, bbox=(10, 10, 50, 50)),
            DetectionRecord(class_id=1, label="car", confidence=0.81, bbox=(12, 12, 52, 52)),
        ]

        result = detector._effective_display_detections(detections)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "car")
        self.assertAlmostEqual(result[0].confidence, 0.81)

    def test_filter_person_detections_rejects_weak_person_confidence(self) -> None:
        detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.59, bbox=(40, 20, 120, 170)),
            DetectionRecord(class_id=0, label="person", confidence=0.78, bbox=(45, 25, 125, 175)),
        ]

        result = _filter_person_detections(detections, (200, 200, 3))

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.78)

    def test_effective_inference_imgsz_uses_higher_medium_resolution_for_accuracy(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.runtime.profile_name = "medium"
        detector.runtime.imgsz = 768

        self.assertEqual(detector._effective_inference_imgsz(), 640)

    def test_dedupe_display_detections_removes_overlapping_same_label_boxes(self) -> None:
        detections = [
            DetectionRecord(class_id=1, label="car", confidence=0.88, bbox=(10, 10, 60, 60)),
            DetectionRecord(class_id=1, label="car", confidence=0.76, bbox=(12, 12, 58, 58)),
            DetectionRecord(class_id=2, label="bottle", confidence=0.91, bbox=(100, 20, 130, 90)),
        ]

        result = _dedupe_display_detections(detections)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "bottle")
        self.assertEqual(result[1].label, "car")

    def test_bbox_iou_returns_high_value_for_near_identical_boxes(self) -> None:
        score = _bbox_iou((10, 10, 60, 60), (12, 12, 58, 58))
        self.assertGreater(score, 0.8)

    def test_filter_person_detections_rejects_large_edge_false_positive(self) -> None:
        detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.74, bbox=(0, 0, 190, 150)),
            DetectionRecord(class_id=0, label="person", confidence=0.80, bbox=(60, 40, 150, 150)),
        ]

        result = _filter_person_detections(detections, (180, 200, 3))

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.80)

    def test_filter_person_detections_keeps_multiple_valid_person_boxes(self) -> None:
        detections = [
            DetectionRecord(class_id=0, label="person", confidence=0.78, bbox=(20, 10, 80, 160)),
            DetectionRecord(class_id=0, label="person", confidence=0.76, bbox=(60, 20, 140, 170)),
        ]

        result = _filter_person_detections(detections, (180, 200, 3))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].bbox, (20, 10, 80, 160))
        self.assertEqual(result[1].bbox, (60, 20, 140, 170))

    def test_filter_person_detections_keeps_phone_with_sufficient_confidence(self) -> None:
        detections = [
            DetectionRecord(class_id=1, label="phone", confidence=0.54, bbox=(20, 20, 40, 60)),
            DetectionRecord(class_id=1, label="phone", confidence=0.78, bbox=(60, 25, 90, 80)),
        ]

        result = _filter_person_detections(detections, (180, 200, 3))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "phone")
        self.assertEqual(result[0].bbox, (60, 25, 90, 80))

    def test_predict_frame_requests_all_classes(self) -> None:
        detector = CameraDetector(runtime=self._runtime())
        fake_model = MagicMock()
        fake_model.predict.return_value = [SimpleNamespace(names={0: "person", 2: "car"}, boxes=[])]
        detector.loaded_model = LoadedModel(model=fake_model, model_name="yolo11s.pt", source_path="yolo11s.pt")
        detector.runtime.resolved_device = "cpu"
        detector.runtime.use_half = False
        frame = np.zeros((80, 80, 3), dtype=np.uint8)

        detector._predict_frame(frame)

        fake_model.predict.assert_called_once()
        self.assertNotIn("classes", fake_model.predict.call_args.kwargs)

    @patch("core.camera_runner.get_live_usage_snapshot", return_value={"cpu_usage_percent": 11.0})
    def test_runtime_health_includes_runtime_and_fallback_observability(self, _usage_mock) -> None:
        detector = CameraDetector(runtime=self._runtime())
        detector.runtime.active_model_name = "yolo11n.pt"
        detector.runtime.resolved_device = "cpu"
        detector.runtime.use_half = False
        detector.active_runtime_summary = "yolo11n.pt | cpu | imgsz 320"
        detector.fallback_chain_tried = [{"model_name": "yolo11s.pt", "resolved_device": "cuda:0", "use_half": True}]
        detector.runtime_step_errors = [{"model_name": "yolo11s.pt", "error": "CUDA boom"}]

        health = detector.runtime_health()

        self.assertEqual(health["active_model_name"], "yolo11n.pt")
        self.assertEqual(health["resolved_device"], "cpu")
        self.assertFalse(health["use_half"])
        self.assertEqual(health["fallback_chain_tried"][0]["model_name"], "yolo11s.pt")
        self.assertEqual(health["step_errors"][0]["error"], "CUDA boom")
        self.assertEqual(health["live_usage_snapshot"]["cpu_usage_percent"], 11.0)
