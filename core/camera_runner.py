from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.fallback_manager import iter_fallback_configs
from core.hardware_info import detect_hardware, get_live_usage_snapshot
from core.model_selector import RuntimeConfig
from core.model_loader import LoadedModel, load_yolo_model
from utils.file_utils import ensure_project_directories, load_yaml, save_yaml
from utils.logger import get_logger
from utils.draw_utils import draw_detection_results


logger = get_logger(__name__)
WINDOW_NAME = "YOLO Realtime Camera"
SAMPLE_IMAGE_DIR = Path("dataset/sample/images")
SAMPLE_LABEL_DIR = Path("dataset/sample/labels")
TRAINING_DATA_YAML = Path("training/data.yaml")
CAPTURE_STABILITY_SECONDS = 5.0
MOTION_STABLE_THRESHOLD = 2.8
MOTION_RESET_THRESHOLD = 7.0
STABLE_FRAMES_REQUIRED = 6
ALLOWED_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
CAMERA_BOTTOM_PADDING = 150
CAPTURE_PANEL_TITLE = "YOLO Capture"
WINDOW_MARGIN = 16
LAYOUT_GAP = 16
CAMERA_PREVIEW_SCALE = 0.86
CAMERA_ONLY_MIN_MARGIN = 24
CAMERA_ONLY_WIDTH = 800
CAMERA_ONLY_HEIGHT = 600
CURRENT_LAYOUT_SIZE = (1, 1)
CURRENT_DISPLAY_SIZE = (1, 1)
LIGHT_BG = (24, 24, 24)
LIGHT_BG_SOFT = (37, 37, 38)
LIGHT_PANEL_FILL = (30, 30, 30)
LIGHT_PANEL_OUTLINE = (58, 58, 61)
LIGHT_PANEL_OUTLINE_SOFT = (77, 77, 81)
LIGHT_TEXT_PRIMARY = (204, 204, 204)
LIGHT_TEXT_SECONDARY = (156, 163, 175)
LIGHT_TEXT_TERTIARY = (122, 128, 136)
LIGHT_ACCENT = (0, 122, 204)
LIGHT_SUCCESS = (78, 201, 176)
LIGHT_WARNING = (220, 220, 170)
DISPLAY_MIN_CONFIDENCE = 0.50
PERSON_MIN_CONFIDENCE = 0.60
DISPLAY_NMS_IOU = 0.45
PERSON_MAX_AREA_RATIO = 0.6
PERSON_MAX_WIDTH_HEIGHT_RATIO = 1.35
PERSON_EDGE_TOUCH_RATIO = 0.02
PERSON_CENTER_BIAS_WEIGHT = 0.35
FACE_LABEL = "face"
PERSON_LABEL = "person"
FACE_REFINEMENT_MIN_HEIGHT_RATIO = 1.15
FACE_BOX_WIDTH_RATIO = 0.58
FACE_BOX_TOP_RATIO = 0.08
FACE_BOX_BOTTOM_RATIO = 0.38
FACE_TRACKING_STICKY_ALPHA = 0.96
FACE_TRACKING_STABLE_MOTION_RATIO = 0.16
FACE_JITTER_FREEZE_RATIO = 0.05
FACE_JITTER_MAX_SIZE_CHANGE_RATIO = 0.10
TRACKING_MATCH_IOU = 0.30
TRACKING_SMOOTHING_ALPHA = 0.75
TRACKING_FAST_SMOOTHING_ALPHA = 0.50
TRACKING_STABLE_SMOOTHING_ALPHA = 0.92
TRACKING_MATCH_CENTER_RATIO = 0.50
TRACKING_PREDICTION_MOTION_RATIO = 0.30
TRACKING_STABLE_MOTION_RATIO = 0.10
TRACK_TRAIL_MAX_POINTS = 10
TRACK_TRAIL_MIN_MOVEMENT_PX = 4.0

@dataclass
class DetectionRecord:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: int = -1


@dataclass
class CapturePreparationState:
    stable_since: float
    previous_gray: np.ndarray | None = None
    motion_score: float = 0.0
    stable_frame_count: int = 0
    status: str = "Giữ camera ổn định trong 5 giây."


def _to_yolo_bbox_line(class_id: int, bbox: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> str:
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    x_center = x1 + (box_width / 2.0)
    y_center = y1 + (box_height / 2.0)
    return (
        f"{class_id} "
        f"{x_center / image_width:.6f} "
        f"{y_center / image_height:.6f} "
        f"{box_width / image_width:.6f} "
        f"{box_height / image_height:.6f}"
    )


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _smooth_bbox(
    previous_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    alpha: float = TRACKING_SMOOTHING_ALPHA,
) -> tuple[int, int, int, int]:
    alpha = max(0.0, min(1.0, float(alpha)))
    return tuple(
        int(round((previous_value * alpha) + (current_value * (1.0 - alpha))))
        for previous_value, current_value in zip(previous_bbox, current_bbox)
    )


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_center_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    center_ax, center_ay = _bbox_center(box_a)
    center_bx, center_by = _bbox_center(box_b)
    return float(((center_ax - center_bx) ** 2 + (center_ay - center_by) ** 2) ** 0.5)


def _bbox_reference_size(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return float(max(width, height))


def _estimate_motion_bbox(
    previous_display_bbox: tuple[int, int, int, int],
    previous_observed_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    deltas = [current_value - previous_value for previous_value, current_value in zip(previous_observed_bbox, current_bbox)]
    predicted = tuple(previous_display_value + delta for previous_display_value, delta in zip(previous_display_bbox, deltas))
    return predicted


def _adaptive_tracking_alpha(
    previous_observed_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> float:
    movement_ratio = _bbox_movement_ratio(previous_observed_bbox, current_bbox)
    if movement_ratio <= TRACKING_STABLE_MOTION_RATIO:
        return TRACKING_STABLE_SMOOTHING_ALPHA
    if movement_ratio >= 0.35:
        return TRACKING_FAST_SMOOTHING_ALPHA
    blend = max(0.0, min(1.0, (movement_ratio - TRACKING_STABLE_MOTION_RATIO) / (0.35 - TRACKING_STABLE_MOTION_RATIO)))
    return TRACKING_STABLE_SMOOTHING_ALPHA + ((TRACKING_FAST_SMOOTHING_ALPHA - TRACKING_STABLE_SMOOTHING_ALPHA) * blend)


def _bbox_movement_ratio(
    previous_observed_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> float:
    movement_distance = _bbox_center_distance(previous_observed_bbox, current_bbox)
    reference_size = max(_bbox_reference_size(previous_observed_bbox), _bbox_reference_size(current_bbox), 1.0)
    return movement_distance / reference_size


def _bbox_size_change_ratio(
    previous_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> float:
    previous_width = max(1, previous_bbox[2] - previous_bbox[0])
    previous_height = max(1, previous_bbox[3] - previous_bbox[1])
    current_width = max(1, current_bbox[2] - current_bbox[0])
    current_height = max(1, current_bbox[3] - current_bbox[1])
    width_ratio = abs(current_width - previous_width) / max(previous_width, current_width, 1)
    height_ratio = abs(current_height - previous_height) / max(previous_height, current_height, 1)
    return max(width_ratio, height_ratio)


def _can_match_detection(
    previous_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    iou_threshold: float = TRACKING_MATCH_IOU,
    center_ratio_threshold: float = TRACKING_MATCH_CENTER_RATIO,
) -> bool:
    overlap = _bbox_iou(previous_bbox, current_bbox)
    if overlap >= iou_threshold:
        return True
    center_distance = _bbox_center_distance(previous_bbox, current_bbox)
    reference_size = max(_bbox_reference_size(previous_bbox), _bbox_reference_size(current_bbox), 1.0)
    return center_distance <= (reference_size * center_ratio_threshold)


def _tracking_match_score(
    previous_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> float:
    overlap = _bbox_iou(previous_bbox, current_bbox)
    center_distance = _bbox_center_distance(previous_bbox, current_bbox)
    reference_size = max(_bbox_reference_size(previous_bbox), _bbox_reference_size(current_bbox), 1.0)
    normalized_distance = min(1.0, center_distance / reference_size)
    return overlap + (1.0 - normalized_distance)


def _match_and_smooth_detections(
    current_detections: list[DetectionRecord],
    previous_detections: list[DetectionRecord],
    previous_observed_detections: list[DetectionRecord] | None = None,
    iou_threshold: float = TRACKING_MATCH_IOU,
) -> list[DetectionRecord]:
    if not current_detections:
        return []
    if not previous_detections:
        return list(current_detections)

    smoothed: list[DetectionRecord] = []
    matched_previous_indices: set[int] = set()
    for current in current_detections:
        best_match_index = -1
        best_score = -1.0
        for index, previous in enumerate(previous_detections):
            if index in matched_previous_indices:
                continue
            if previous.class_id != current.class_id or previous.label != current.label:
                continue
            if not _can_match_detection(previous.bbox, current.bbox, iou_threshold=iou_threshold):
                continue
            score = _tracking_match_score(previous.bbox, current.bbox)
            if score > best_score:
                best_score = score
                best_match_index = index
        if best_match_index >= 0:
            matched_previous_indices.add(best_match_index)
            previous = previous_detections[best_match_index]
            previous_observed = (
                previous_observed_detections[best_match_index]
                if previous_observed_detections is not None and best_match_index < len(previous_observed_detections)
                else previous
            )
            adaptive_alpha = _adaptive_tracking_alpha(previous_observed.bbox, current.bbox)
            movement_ratio = _bbox_movement_ratio(previous_observed.bbox, current.bbox)
            if movement_ratio >= TRACKING_PREDICTION_MOTION_RATIO:
                smoothing_source_bbox = _estimate_motion_bbox(previous.bbox, previous_observed.bbox, current.bbox)
            else:
                smoothing_source_bbox = previous.bbox
            smoothed_bbox = (
                _stabilize_face_bbox(
                    previous_display_bbox=previous.bbox,
                    previous_observed_bbox=previous_observed.bbox,
                    current_bbox=current.bbox,
                    adaptive_alpha=adaptive_alpha,
                )
                if _is_refined_face_label(current.label)
                else _smooth_bbox(smoothing_source_bbox, current.bbox, alpha=adaptive_alpha)
            )
            smoothed.append(
                DetectionRecord(
                    class_id=current.class_id,
                    label=current.label,
                    confidence=current.confidence,
                    bbox=smoothed_bbox,
                    track_id=previous.track_id,
                )
            )
            continue
        smoothed.append(current)
    return smoothed


def _dedupe_display_detections(detections: list[DetectionRecord], iou_threshold: float = DISPLAY_NMS_IOU) -> list[DetectionRecord]:
    selected: list[DetectionRecord] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(
            detection.label == existing.label and _bbox_iou(detection.bbox, existing.bbox) >= iou_threshold
            for existing in selected
        ):
            continue
        selected.append(detection)
    return selected


def _box_area_ratio(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> float:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return (max(0, x2 - x1) * max(0, y2 - y1)) / max(1, frame_w * frame_h)


def _touches_frame_edge(box: tuple[int, int, int, int], frame_shape: tuple[int, ...], margin_ratio: float = PERSON_EDGE_TOUCH_RATIO) -> bool:
    frame_h, frame_w = frame_shape[:2]
    margin_x = max(1, int(frame_w * margin_ratio))
    margin_y = max(1, int(frame_h * margin_ratio))
    x1, y1, x2, y2 = box
    return x1 <= margin_x or y1 <= margin_y or x2 >= frame_w - margin_x or y2 >= frame_h - margin_y


def _person_shape_is_plausible(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return (width / height) <= PERSON_MAX_WIDTH_HEIGHT_RATIO


def _person_priority_score(box: tuple[int, int, int, int], confidence: float, frame_shape: tuple[int, ...]) -> float:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0
    frame_center_x = frame_w / 2.0
    center_distance = abs(center_x - frame_center_x) / max(1.0, frame_center_x)
    center_bonus = max(0.0, 1.0 - center_distance)
    return float(confidence) + (center_bonus * PERSON_CENTER_BIAS_WEIGHT)


def _is_face_detection_label(label: str) -> bool:
    return str(label).lower() in {PERSON_LABEL, FACE_LABEL}


def _refine_person_bbox_to_face(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    if (height / width) < FACE_REFINEMENT_MIN_HEIGHT_RATIO:
        return bbox

    refined_width = max(1, int(round(width * FACE_BOX_WIDTH_RATIO)))
    horizontal_margin = max(0, int(round((width - refined_width) / 2.0)))
    refined_x1 = x1 + horizontal_margin
    refined_x2 = refined_x1 + refined_width
    refined_y1 = y1 + int(round(height * FACE_BOX_TOP_RATIO))
    refined_y2 = y1 + int(round(height * FACE_BOX_BOTTOM_RATIO))
    refined_y2 = max(refined_y1 + 1, refined_y2)
    return (refined_x1, refined_y1, refined_x2, refined_y2)


def _normalize_detection_label_and_bbox(
    label: str,
    bbox: tuple[int, int, int, int],
) -> tuple[str, tuple[int, int, int, int]]:
    normalized_label = str(label).strip().lower()
    return normalized_label, bbox


def _is_refined_face_label(label: str) -> bool:
    return str(label).lower() == FACE_LABEL


def _stabilize_face_bbox(
    previous_display_bbox: tuple[int, int, int, int],
    previous_observed_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    adaptive_alpha: float,
) -> tuple[int, int, int, int]:
    movement_ratio = _bbox_movement_ratio(previous_observed_bbox, current_bbox)
    size_change_ratio = _bbox_size_change_ratio(previous_observed_bbox, current_bbox)
    if movement_ratio <= FACE_JITTER_FREEZE_RATIO and size_change_ratio <= FACE_JITTER_MAX_SIZE_CHANGE_RATIO:
        return previous_display_bbox
    face_alpha = adaptive_alpha
    if movement_ratio <= FACE_TRACKING_STABLE_MOTION_RATIO:
        face_alpha = max(face_alpha, FACE_TRACKING_STICKY_ALPHA)
    return _smooth_bbox(previous_display_bbox, current_bbox, alpha=face_alpha)


def _filter_person_detections(
    detections: list[DetectionRecord],
    frame_shape: tuple[int, ...],
) -> list[DetectionRecord]:
    filtered: list[DetectionRecord] = []

    for item in detections:
        label = str(item.label).lower()

        if label == PERSON_LABEL:
            if item.confidence >= 0.55:
                filtered.append(item)

        elif label == FACE_LABEL:
            if item.confidence >= 0.45:
                filtered.append(item)

        else:
            if item.confidence >= DISPLAY_MIN_CONFIDENCE:
                filtered.append(item)

    return filtered


def _sanitize_sample_name(value: str) -> str:
    cleaned = "".join(char if char in ALLOWED_NAME_CHARS else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_-")


def _next_sample_sequence_name() -> str:
    max_index = 0
    for directory in (SAMPLE_IMAGE_DIR, SAMPLE_LABEL_DIR):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            stem = path.stem.strip()
            if stem.isdigit():
                max_index = max(max_index, int(stem))
    return str(max_index + 1)


def _update_training_data_name(sample_name: str) -> None:
    if not sample_name or sample_name.isdigit():
        return
    if not TRAINING_DATA_YAML.exists():
        return
    config = load_yaml(TRAINING_DATA_YAML) or {}
    names = config.get("names") or {}
    if isinstance(names, list):
        if len(names) == 1:
            names[0] = sample_name
        elif not names:
            names = [sample_name]
        else:
            return
    elif isinstance(names, dict):
        if len(names) == 1 and 0 in names:
            names[0] = sample_name
        elif not names:
            names[0] = sample_name
        else:
            return
    else:
        names = {0: sample_name}
    config["names"] = names
    save_yaml(TRAINING_DATA_YAML, config)

class CameraStream:
    def __init__(self, camera_index: int, max_consecutive_read_failures: int = 5) -> None:
        self.camera_index = camera_index
        self.max_consecutive_read_failures = max_consecutive_read_failures
        self.consecutive_read_failures = 0
        self.last_status_message = "Sẵn sàng khởi tạo camera."
        self.last_error_message = ""
        self.capture: cv2.VideoCapture | None = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop_event = threading.Event()
        self.capture_ready_event = threading.Event()
        self.capture_lock = threading.Lock()
        self.latest_captured_frame: np.ndarray | None = None

    def open(self, width: int, height: int) -> None:
        self.release()
        self.capture = self._open_capture()
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError("Không mở được camera.")
        self.consecutive_read_failures = 0
        self.last_error_message = ""
        self.last_status_message = "Đã mở camera thành công."
        self.latest_captured_frame = None
        self.capture_ready_event.clear()
        self._start_capture_worker()

    def read_latest_frame(self, wait_seconds: float = 0.05) -> np.ndarray | None:
        frame = self._take_latest_frame()
        if frame is not None:
            return frame
        self.capture_ready_event.wait(wait_seconds)
        frame = self._take_latest_frame()
        if frame is not None:
            return frame
        ok, frame = self._read_capture_frame(track_failures=True)
        if ok and frame is not None:
            with self.capture_lock:
                self.latest_captured_frame = frame.copy()
            self.capture_ready_event.set()
            return frame
        return None

    def release(self) -> None:
        self._stop_capture_worker()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.last_status_message = "Camera đã dừng."

    def _start_capture_worker(self) -> None:
        self.capture_stop_event.clear()
        self.capture_ready_event.clear()
        self.capture_thread = threading.Thread(
            target=self._capture_worker_loop,
            name="camera-capture-worker",
            daemon=True,
        )
        self.capture_thread.start()

    def _stop_capture_worker(self) -> None:
        if self.capture_thread is None:
            return
        self.capture_stop_event.set()
        self.capture_thread.join(timeout=1.0)
        self.capture_thread = None
        self.capture_stop_event.clear()
        self.capture_ready_event.clear()
        with self.capture_lock:
            self.latest_captured_frame = None

    def _capture_worker_loop(self) -> None:
        while not self.capture_stop_event.is_set():
            ok, frame = self._read_capture_frame(track_failures=False)
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self.capture_lock:
                self.latest_captured_frame = frame
            self.capture_ready_event.set()

    def _read_capture_frame(self, track_failures: bool) -> tuple[bool, np.ndarray | None]:
        if self.capture is None:
            return False, None
        try:
            result = self.capture.read()
        except Exception:
            result = (False, None)
        if not isinstance(result, tuple) or len(result) != 2:
            result = (False, None)
        ok, frame = result
        if ok and frame is not None:
            if track_failures:
                self.consecutive_read_failures = 0
            return True, frame
        if track_failures:
            self.consecutive_read_failures += 1
            self.last_error_message = "Không đọc được frame từ camera."
            self.last_status_message = f"Mat frame camera ({self.consecutive_read_failures}/{self.max_consecutive_read_failures})."
        return False, None

    def _take_latest_frame(self) -> np.ndarray | None:
        with self.capture_lock:
            if self.latest_captured_frame is None:
                return None
            return self.latest_captured_frame.copy()

    def _open_capture(self) -> cv2.VideoCapture:
        if hasattr(cv2, "CAP_DSHOW"):
            return cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(self.camera_index)


class CameraDetector:
    def __init__(self, runtime: RuntimeConfig, camera_index: int = 0) -> None:
        self.runtime = runtime
        self.camera_index = camera_index
        self.camera_stream: CameraStream | None = None
        self.loaded_model: LoadedModel | None = None
        self.last_frame_ts = time.perf_counter()
        self.smoothed_fps = 0.0
        self.recovery_count = 0
        self.last_status_message = "Sẵn sàng khởi tạo camera."
        self.last_error_message = ""
        self.active_runtime_summary = ""
        self.fallback_chain_tried: list[dict[str, Any]] = []
        self.runtime_step_errors: list[dict[str, Any]] = []
        self.last_raw_frame: np.ndarray | None = None
        self.last_detections: list[DetectionRecord] = []
        self.previous_display_detections: list[DetectionRecord] = []
        self.previous_observed_detections: list[DetectionRecord] = []
        self.display_trails: dict[int, list[tuple[int, int]]] = {}
        self.next_track_id = 1
        self.frame_index = 0
        self.detect_interval = 1
        self.base_detect_interval = 1
        self.max_detect_interval = 1
        self.last_detect_adjust_frame = 0
        self.inference_thread: threading.Thread | None = None
        self.inference_stop_event = threading.Event()
        self.inference_ready_event = threading.Event()
        self.inference_lock = threading.Lock()
        self.pending_inference_frame: np.ndarray | None = None
        self.pending_inference_error: Exception | None = None
        self.previous_gray: np.ndarray | None = None
        self.last_motion_score: float = 0.0

    @property
    def capture(self) -> cv2.VideoCapture | None:
        return self.camera_stream.capture if self.camera_stream is not None else None

    @property
    def max_consecutive_read_failures(self) -> int:
        return self.camera_stream.max_consecutive_read_failures if self.camera_stream is not None else 5

    @property
    def consecutive_read_failures(self) -> int:
        return self.camera_stream.consecutive_read_failures if self.camera_stream is not None else 0

    def initialize(self) -> None:
        last_error: Exception | None = None
        self.fallback_chain_tried = []
        self.runtime_step_errors = []
        for runtime in [self.runtime, *list(iter_fallback_configs(self.runtime))]:
            attempt = {
                "profile_name": runtime.profile_name,
                "model_name": runtime.primary_model_name,
                "resolved_device": runtime.resolved_device,
                "imgsz": int(runtime.imgsz),
                "use_half": bool(runtime.use_half),
            }
            self.fallback_chain_tried.append(attempt)
            try:
                self._stop_inference_worker()
                self.runtime = runtime
                self.loaded_model = None
                self.release()
                self.camera_stream = CameraStream(camera_index=self.camera_index)
                self.camera_stream.open(self.runtime.camera_width, self.runtime.camera_height)
                self.last_frame_ts = time.perf_counter()
                self.frame_index = 0
                self.base_detect_interval = 1
                self.max_detect_interval = 1
                self.detect_interval = 1
                self.last_detect_adjust_frame = 0
                self.last_detections = []
                self.previous_display_detections = []
                self.previous_observed_detections = []
                self.display_trails = {}
                self.next_track_id = 1
                self.pending_inference_error = None
                self.pending_inference_frame = None
                self.previous_gray = None
                self.last_motion_score = 0.0
                self.last_error_message = ""
                self.active_runtime_summary = (
                    f"camera {self.runtime.camera_width}x{self.runtime.camera_height} | "
                    f"profile {self.runtime.profile_name}"
                )
                self.last_status_message = f"Đã khởi tạo camera thành công. Đang chạy với {self.active_runtime_summary}."
                logger.info("Detector initialized with %s", self.runtime.summary())
                return
            except Exception as exc:
                last_error = exc
                self.last_error_message = str(exc)
                self.last_status_message = "Khởi tạo runtime thất bại, đang thử fallback."
                self.runtime_step_errors.append({**attempt, "error": str(exc)})
                logger.warning("Runtime failed, trying fallback: %s", exc)
                self.release()
        raise RuntimeError(f"Không khởi tạo được detector. Lỗi cuối: {last_error}")

    def read_and_detect(self) -> tuple[bool, Any, list[DetectionRecord], float]:
        if self.camera_stream is None:
            raise RuntimeError("Detector chưa được khởi tạo.")

        frame = self.camera_stream.read_latest_frame(wait_seconds=0.05)
        if frame is None:
            self.last_error_message = self.camera_stream.last_error_message
            self.last_status_message = self.camera_stream.last_status_message
            if self.camera_stream.consecutive_read_failures >= self.camera_stream.max_consecutive_read_failures:
                raise RuntimeError("Camera liên tục không trả về frame.")
            return False, None, [], 0.0

        self.last_raw_frame = frame.copy()
        self.frame_index += 1
        self.last_detections = []
        self.previous_display_detections = []
        self.previous_observed_detections = []
        self.display_trails = {}
        self.previous_gray = None
        self.last_motion_score = 0.0
        self.last_status_message = "Camera đang hoạt động."
        fps = self._update_fps()
        return True, frame, [], fps
        if self.camera_stream is None or self.loaded_model is None:
            raise RuntimeError("Detector chưa được khởi tạo.")
        if self.pending_inference_error is not None:
            exc = self.pending_inference_error
            self.pending_inference_error = None
            logger.warning("Inference failed on %s: %s", self.runtime.primary_model_name, exc)
            self.recovery_count += 1
            self.last_error_message = str(exc)
            self.last_status_message = "Suy luận bị lỗi, hệ thống đang tự phục hồi và thử cấu hình an toàn hơn."
            self.initialize()
            return False, None, [], 0.0

        frame = self.camera_stream.read_latest_frame(wait_seconds=0.05)
        if frame is None:
            self.last_error_message = self.camera_stream.last_error_message
            self.last_status_message = self.camera_stream.last_status_message
            if self.camera_stream.consecutive_read_failures >= self.camera_stream.max_consecutive_read_failures:
                raise RuntimeError("Camera liên tục không trả về frame.")
            return False, None, [], 0.0
        self.last_raw_frame = frame.copy()
        self.frame_index += 1
        try:
            current_detections = self._predict_frame(frame)
            self.last_detections = list(current_detections)
        except Exception as exc:
            logger.warning("Inference failed on %s: %s", self.runtime.primary_model_name, exc)
            self.recovery_count += 1
            self.last_error_message = str(exc)
            self.last_status_message = "Suy luận bị lỗi, hệ thống đang tự phục hồi và thử cấu hình an toàn hơn."
            self.initialize()
            return False, None, [], 0.0

        motion_score, current_gray = _compute_motion_score(frame, self.previous_gray)
        self.last_motion_score = motion_score
        self.previous_gray = current_gray
        detections = self._smooth_display_detections(self._effective_display_detections(current_detections))
        if motion_score > MOTION_STABLE_THRESHOLD and detections:
            self.last_status_message = f"Đang theo dõi vật thể chuyển động với {len(detections)} đối tượng."
        else:
            if len(detections) > 0:
                self.last_status_message = f"Đang nhận diện ổn định với {len(detections)} đối tượng."
            else:
                self.last_status_message = "Chưa có vật thể nào được nhận diện."
        processed_frame = draw_detection_results(
            image=frame,
            detections=detections,
            box_thickness=self._effective_box_thickness(),
            label_font_scale=self._effective_label_font_scale(),
            motion_trails=self.display_trails,
        )
        fps = self._update_fps()
        return True, processed_frame, detections, fps

    def _queue_inference(self, frame: np.ndarray) -> None:
        should_submit = self.frame_index == 1 or (self.frame_index % self.detect_interval) == 1
        if not should_submit:
            return
        with self.inference_lock:
            self.pending_inference_frame = frame.copy()
        self.inference_ready_event.set()

    def _adjust_detect_interval(self, fps: float) -> None:
        target_fps = _target_fps_for_profile(self.runtime.profile_name)
        tolerance = _fps_tolerance_for_profile(self.runtime.profile_name)
        if fps < target_fps - tolerance and self.detect_interval < self.max_detect_interval:
            self.detect_interval += 1
        elif fps > target_fps + tolerance and self.detect_interval > self.base_detect_interval:
            self.detect_interval -= 1
        self.last_detect_adjust_frame = self.frame_index

    def _start_inference_worker(self) -> None:
        self.inference_stop_event.clear()
        self.inference_ready_event.clear()
        self.inference_thread = threading.Thread(
            target=self._inference_worker_loop,
            name="yolo-inference-worker",
            daemon=True,
        )
        self.inference_thread.start()

    def _stop_inference_worker(self) -> None:
        if self.inference_thread is None:
            return
        self.inference_stop_event.set()
        self.inference_ready_event.set()
        self.inference_thread.join(timeout=1.0)
        self.inference_thread = None
        self.inference_stop_event.clear()
        self.inference_ready_event.clear()
        self.pending_inference_frame = None

    def _inference_worker_loop(self) -> None:
        while not self.inference_stop_event.is_set():
            self.inference_ready_event.wait(0.05)
            if self.inference_stop_event.is_set():
                return
            with self.inference_lock:
                frame = self.pending_inference_frame
                self.pending_inference_frame = None
                if frame is None:
                    self.inference_ready_event.clear()
                    continue
            try:
                if self.loaded_model is None:
                    continue
                self.last_detections = self._predict_frame(frame)
            except Exception as exc:
                self.pending_inference_error = exc
            finally:
                with self.inference_lock:
                    if self.pending_inference_frame is None:
                        self.inference_ready_event.clear()

    def _predict_frame(self, frame: np.ndarray) -> list[DetectionRecord]:
        if self.loaded_model is None:
            return []
        results = self.loaded_model.model.predict(
            source=frame,
            imgsz=self._effective_inference_imgsz(),
            conf=self._effective_confidence(),
            device=self.runtime.resolved_device,
            half=self.runtime.use_half,
            max_det=self._effective_max_det(),
            verbose=False,
            stream=False,
        )
        parsed = self._parse_results(results)
        return _filter_person_detections(parsed, frame.shape)

    def _effective_inference_imgsz(self) -> int:
        if self.runtime.profile_name == "low":
            return self.runtime.imgsz
        if self.runtime.profile_name == "medium":
            return min(self.runtime.imgsz, 640)
        return self.runtime.imgsz

    def _effective_max_det(self) -> int:
        if self.runtime.profile_name == "low":
            return min(self.runtime.max_det, 10)
        if self.runtime.profile_name == "medium":
            return min(self.runtime.max_det, 30)
        return self.runtime.max_det

    def _effective_confidence(self) -> float:
        if self.runtime.profile_name == "low":
            return max(self.runtime.conf, 0.35)
        if self.runtime.profile_name == "medium":
            return max(self.runtime.conf, 0.30)
        return self.runtime.conf

    def _effective_display_detections(
        self,
        detections: list[DetectionRecord],
    ) -> list[DetectionRecord]:
        filtered: list[DetectionRecord] = []

        for item in detections:
            label = str(item.label).lower()

            if label == PERSON_LABEL:
                min_confidence = 0.55
            elif label == FACE_LABEL:
                min_confidence = 0.45
            else:
                min_confidence = DISPLAY_MIN_CONFIDENCE

            if item.confidence >= min_confidence:
                filtered.append(item)

        cleaned = _dedupe_display_detections(filtered)

        if self.runtime.profile_name == "low":
            return cleaned[:5]
        if self.runtime.profile_name == "medium":
            return cleaned[:10]
        return cleaned[:20]

    def _smooth_display_detections(self, detections: list[DetectionRecord]) -> list[DetectionRecord]:
        smoothed = _match_and_smooth_detections(
            current_detections=detections,
            previous_detections=self.previous_display_detections,
            previous_observed_detections=self.previous_observed_detections,
        )
        self._assign_track_ids(smoothed)
        self._update_display_trails(smoothed)
        self.previous_display_detections = list(smoothed)
        self.previous_observed_detections = list(detections)
        return smoothed

    def _assign_track_ids(self, detections: list[DetectionRecord]) -> None:
        for detection in detections:
            if detection.track_id >= 0:
                continue
            detection.track_id = self.next_track_id
            self.next_track_id += 1

    def _update_display_trails(self, detections: list[DetectionRecord]) -> None:
        next_trails: dict[int, list[tuple[int, int]]] = {}
        for detection in detections:
            center_x, center_y = _bbox_center(detection.bbox)
            center = (int(round(center_x)), int(round(center_y)))
            trail = list(self.display_trails.get(detection.track_id, []))
            if not trail:
                trail.append(center)
            else:
                last_x, last_y = trail[-1]
                movement = float((((center[0] - last_x) ** 2) + ((center[1] - last_y) ** 2)) ** 0.5)
                if movement >= TRACK_TRAIL_MIN_MOVEMENT_PX:
                    trail.append(center)
                else:
                    trail[-1] = center
            next_trails[detection.track_id] = trail[-TRACK_TRAIL_MAX_POINTS:]
        self.display_trails = next_trails

    def _effective_box_thickness(self) -> int:
        if self.runtime.profile_name == "low":
            return max(1, self.runtime.box_thickness - 2)
        if self.runtime.profile_name == "medium":
            return max(1, self.runtime.box_thickness - 1)
        return self.runtime.box_thickness

    def _effective_label_font_scale(self) -> float:
        if self.runtime.profile_name == "low":
            return max(0.62, self.runtime.label_font_scale * 0.72)
        if self.runtime.profile_name == "medium":
            return max(0.72, self.runtime.label_font_scale * 0.86)
        return self.runtime.label_font_scale

    def _update_fps(self) -> float:
        now = time.perf_counter()
        current_fps = 1.0 / max(now - self.last_frame_ts, 1e-6)
        self.last_frame_ts = now
        if self.smoothed_fps == 0.0:
            self.smoothed_fps = current_fps
        else:
            self.smoothed_fps = (self.smoothed_fps * 0.85) + (current_fps * 0.15)
        return self.smoothed_fps

    def _parse_results(self, results: list) -> list[DetectionRecord]:
        parsed: list[DetectionRecord] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                label, bbox = _normalize_detection_label_and_bbox(
                    names.get(cls_id, str(cls_id)),
                    (x1, y1, x2, y2),
                )
                parsed.append(
                    DetectionRecord(
                        class_id=cls_id,
                        label=label,
                        confidence=float(box.conf[0].item()),
                        bbox=bbox,
                    )
                )
        return parsed

    def release(self) -> None:
        self._stop_inference_worker()
        self.previous_display_detections = []
        self.previous_observed_detections = []
        self.display_trails = {}
        if self.camera_stream is not None:
            self.camera_stream.release()
        self.camera_stream = None
        self.last_status_message = "Camera đã dừng."

    def save_training_sample(
        self,
        *,
        frame: np.ndarray,
        detections: list[DetectionRecord],
        sample_name: str | None = None,
    ) -> tuple[Path, Path]:
        ensure_project_directories()
        name_prefix = _sanitize_sample_name(sample_name or "")
        base_name = name_prefix or _next_sample_sequence_name()
        image_path = SAMPLE_IMAGE_DIR / f"{base_name}.jpg"
        label_path = SAMPLE_LABEL_DIR / f"{base_name}.txt"
        suffix = 2
        while image_path.exists() or label_path.exists():
            fallback_base = f"{base_name}_{suffix}"
            image_path = SAMPLE_IMAGE_DIR / f"{fallback_base}.jpg"
            label_path = SAMPLE_LABEL_DIR / f"{fallback_base}.txt"
            suffix += 1
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Không lưu được ảnh: {image_path}")
        label_path.write_text(
            "\n".join(_to_yolo_bbox_line(item.class_id, item.bbox, frame.shape) for item in detections),
            encoding="utf-8",
        )
        _update_training_data_name(name_prefix)
        self.last_status_message = f"Đã lưu mẫu train: {image_path.name} và {label_path.name} ({len(detections)} nhãn)."
        logger.info("Saved training sample: %s | %s", image_path, label_path)
        return image_path, label_path

    def save_current_training_sample(self, sample_name: str | None = None) -> tuple[Path, Path]:
        if self.last_raw_frame is None:
            raise RuntimeError("Chưa có frame nào để lưu.")
        return self.save_training_sample(
            frame=self.last_raw_frame,
            detections=self.last_detections,
            sample_name=sample_name,
        )

    def runtime_health(self) -> dict:
        return {
            "status": self.last_status_message,
            "last_error": self.last_error_message,
            "recovery_count": self.recovery_count,
            "runtime": self.active_runtime_summary,
            "detect_interval": self.detect_interval,
            "active_model_name": self.runtime.active_model_name or self.runtime.primary_model_name,
            "resolved_device": self.runtime.resolved_device,
            "use_half": bool(self.runtime.use_half),
            "fallback_chain_tried": list(self.fallback_chain_tried),
            "step_errors": list(self.runtime_step_errors),
            "live_usage_snapshot": get_live_usage_snapshot(),
        }



def _compute_motion_score(current_frame: np.ndarray, previous_gray: np.ndarray | None) -> tuple[float, np.ndarray]:
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (7, 7), 0)
    current_gray = cv2.resize(current_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if previous_gray is None:
        return 0.0, current_gray
    if previous_gray.shape != current_gray.shape:
        previous_gray = cv2.resize(previous_gray, (current_gray.shape[1], current_gray.shape[0]), interpolation=cv2.INTER_AREA)
    return float(cv2.absdiff(current_gray, previous_gray).mean()), current_gray


def _update_capture_preparation(
    state: CapturePreparationState,
    frame: np.ndarray,
    now: float,
) -> tuple[CapturePreparationState, bool, float]:
    if state.previous_gray is None:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (7, 7), 0)
        current_gray = cv2.resize(current_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return (
            CapturePreparationState(
                stable_since=now,
                previous_gray=current_gray,
                motion_score=0.0,
                stable_frame_count=0,
                status="Đang kiểm tra độ ổn định ban đầu. Giữ yên để chuẩn bị đếm 5 giây.",
            ),
            False,
            CAPTURE_STABILITY_SECONDS,
        )

    motion_score, current_gray = _compute_motion_score(frame, state.previous_gray)
    stable_since = state.stable_since
    stable_frame_count = state.stable_frame_count
    if motion_score > MOTION_RESET_THRESHOLD:
        stable_since = now
        stable_frame_count = 0
        status = "Phát hiện rung/lắc mạnh. Bộ đếm đã quay lại 5 giây."
        remaining = CAPTURE_STABILITY_SECONDS
    elif motion_score > MOTION_STABLE_THRESHOLD:
        stable_since = now
        stable_frame_count = 0
        status = "Vật thể hoặc camera chưa đứng yên. Hãy giữ cố định để chuẩn bị đếm 5 giây."
        remaining = CAPTURE_STABILITY_SECONDS
    else:
        stable_frame_count += 1
        if stable_frame_count < STABLE_FRAMES_REQUIRED:
            stable_since = now
            status = (
                f"Khung hình gần ổn định. "
                f"Cần giữ thêm {STABLE_FRAMES_REQUIRED - stable_frame_count} frame liên tục."
            )
            remaining = CAPTURE_STABILITY_SECONDS
        else:
            status = "Khung hình đã ổn định. Tiếp tục giữ yên để bộ đếm chạy xuống."
            remaining = max(0.0, CAPTURE_STABILITY_SECONDS - (now - stable_since))
    return (
        CapturePreparationState(
            stable_since=stable_since,
            previous_gray=current_gray,
            motion_score=motion_score,
            stable_frame_count=stable_frame_count,
            status=status,
        ),
        remaining <= 0.0,
        remaining,
    )


def _handle_name_input(current_name: str, key: int) -> tuple[str, bool, bool]:
    if key in (13, 10):
        return current_name, True, False
    if key == 27:
        return current_name, False, True
    if key in (8, 127):
        return current_name[:-1], False, False
    if 32 <= key <= 126:
        char = chr(key)
        if char in ALLOWED_NAME_CHARS and len(current_name) < 48:
            return current_name + char, False, False
    return current_name, False, False


def _reset_capture_flow() -> tuple[bool, np.ndarray | None, list[DetectionRecord], str]:
    return False, None, [], ""


def _start_capture_preparation() -> CapturePreparationState:
    return CapturePreparationState(stable_since=time.time())


def _compose_camera_only_layout(frame: np.ndarray) -> np.ndarray:
    global CURRENT_LAYOUT_SIZE
    CURRENT_LAYOUT_SIZE = (CAMERA_ONLY_WIDTH, CAMERA_ONLY_HEIGHT)
    interpolation = cv2.INTER_AREA if frame.shape[1] > CAMERA_ONLY_WIDTH or frame.shape[0] > CAMERA_ONLY_HEIGHT else cv2.INTER_LINEAR
    return cv2.resize(frame, (CAMERA_ONLY_WIDTH, CAMERA_ONLY_HEIGHT), interpolation=interpolation)



def _center_window(window_name: str, width: int, height: int) -> None:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return

        monitor = user32.MonitorFromWindow(hwnd, 2)

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", RECT), ("rcWork", RECT), ("dwFlags", ctypes.c_ulong)]

        window_rect = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(window_rect))
        window_width = int(window_rect.right - window_rect.left)
        window_height = int(window_rect.bottom - window_rect.top)

        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(monitor, ctypes.byref(info))

        work_width = int(info.rcWork.right - info.rcWork.left)
        work_height = int(info.rcWork.bottom - info.rcWork.top)
        x = int(info.rcWork.left + max(WINDOW_MARGIN, (work_width - window_width) // 2))
        y = int(info.rcWork.top + max(WINDOW_MARGIN, (work_height - window_height) // 2))
        cv2.moveWindow(window_name, x, y)
    except Exception:
        return




def _target_fps_for_profile(profile_name: str) -> int:
    return {"high": 15, "medium": 18, "low": 30}.get(profile_name, 18)


def _fps_tolerance_for_profile(profile_name: str) -> float:
    return {"high": 1.0, "medium": 2.5, "low": 1.5}.get(profile_name, 2.0)


def _should_force_camera_only_preview(profile_name: str) -> bool:
    return profile_name in {"high", "medium", "low"}


def _fps_panel_line(fps: float) -> str:
    if fps <= 0:
        return "unknown"
    return f"{fps:.1f}"


def _poll_window_key(wait_ms: int, poll_slice_ms: int = 8) -> int:
    remaining = max(1, wait_ms)
    while remaining > 0:
        step = max(1, min(poll_slice_ms, remaining))
        key = cv2.waitKey(step) & 0xFF
        if key != 255:
            return key
        remaining -= step
    return 255


def run_camera_session(runtime: RuntimeConfig, camera_index: int = 0) -> None:
    global CURRENT_DISPLAY_SIZE
    detector = CameraDetector(runtime=runtime, camera_index=camera_index)
    detector.initialize()
    window_positioned = False
    fps_text = _fps_panel_line(0.0)

    def _handle_runtime_key(key: int) -> bool:
        if key == 255:
            return False
        if key == 27:
            return True
        return False

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            loop_started_at = time.perf_counter()
            ok, frame, _detections, _fps = detector.read_and_detect()
            if not ok:
                key = _poll_window_key(1)
                if _handle_runtime_key(key):
                    break
                continue

            display_frame = frame

            fps_text = _fps_panel_line(_fps)
            composed = _compose_camera_only_layout(display_frame)
            if fps_text:
                cv2.putText(composed, fps_text, (composed.shape[1] - 80, composed.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)

            CURRENT_DISPLAY_SIZE = (composed.shape[1], composed.shape[0])
            cv2.imshow(WINDOW_NAME, composed)
            if not window_positioned:
                cv2.resizeWindow(WINDOW_NAME, composed.shape[1], composed.shape[0])
                _center_window(WINDOW_NAME, composed.shape[1], composed.shape[0])
                window_positioned = True
            key = _poll_window_key(1)
            if _handle_runtime_key(key):
                break
    finally:
        detector.release()
        cv2.destroyAllWindows()
