from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.fallback_manager import iter_fallback_configs
from core.model_selector import RuntimeConfig
from core.model_loader import LoadedModel, load_yolo_model
from utils.file_utils import ensure_project_directories
from utils.logger import get_logger
from utils.draw_utils import draw_detection_results


logger = get_logger(__name__)
WINDOW_NAME = "YOLO Realtime Camera"
ASSISTANT_WINDOW_NAME = "YOLO Capture Assistant"
SAMPLE_IMAGE_DIR = Path("dataset/sample/images")
SAMPLE_LABEL_DIR = Path("dataset/sample/labels")
CAPTURE_STABILITY_SECONDS = 5.0
MOTION_RESET_THRESHOLD = 3.5
ALLOWED_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
ASSISTANT_WINDOW_SIZE = (700, 220)
IDLE_ASSISTANT_TITLE = "Trợ lý chụp mẫu train"
PREPARE_ASSISTANT_TITLE = "Chuẩn bị chụp mẫu"
NAME_PROMPT_TITLE = "Đặt tên mẫu"
IDLE_ASSISTANT_LINES = (
    "Bấm T để chụp dữ liệu huấn luyện.",
    "Hệ thống đếm 5 giây ổn định trước khi lưu.",
    "Cửa sổ này tách riêng khỏi camera để dễ thao tác.",
)


@dataclass
class DetectionRecord:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class CapturePreparationState:
    stable_since: float
    previous_gray: np.ndarray | None = None
    motion_score: float = 0.0
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


def _sanitize_sample_name(value: str) -> str:
    cleaned = "".join(char if char in ALLOWED_NAME_CHARS else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_-")


@lru_cache(maxsize=8)
def _load_unicode_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if not candidate.exists():
            continue
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_assistant_window(title: str, lines: list[str]) -> np.ndarray:
    width, height = ASSISTANT_WINDOW_SIZE
    image = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(8, 8), (width - 8, height - 8)], outline=(0, 220, 255), width=3)
    draw.text((22, 18), title, fill=(0, 220, 255), font=_load_unicode_font(28))
    body_font = _load_unicode_font(18)
    y = 62
    for line in lines:
        draw.text((22, y), line, fill=(240, 240, 240), font=body_font)
        y += 30
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _assistant_lines_for_preparation(remaining_seconds: float, motion_score: float, status: str) -> list[str]:
    return [
        f"Đếm ngược: {int(np.ceil(remaining_seconds))} giây",
        f"Rung/lắc: {motion_score:.2f} | Ngưỡng reset: {MOTION_RESET_THRESHOLD:.2f}",
        status,
        "Giữ yên camera và vật thể. Nếu rung, bộ đếm sẽ bắt đầu lại.",
    ]


def _assistant_lines_for_name_prompt(sample_name: str, detection_count: int) -> list[str]:
    return [
        f"Số nhãn sẽ lưu: {detection_count}",
        f"Tên hiện tại: {sample_name or '(để trống sẽ dùng tên mặc định)'}",
        "Gõ tên rồi nhấn Enter để lưu.",
        "Backspace để xóa, Esc để hủy.",
    ]


def _render_capture_preparation_overlay(*, remaining_seconds: float, motion_score: float, status: str) -> np.ndarray:
    return _render_assistant_window(
        title=PREPARE_ASSISTANT_TITLE,
        lines=_assistant_lines_for_preparation(remaining_seconds, motion_score, status),
    )


def _render_name_prompt(sample_name: str, detection_count: int) -> np.ndarray:
    return _render_assistant_window(
        title=NAME_PROMPT_TITLE,
        lines=_assistant_lines_for_name_prompt(sample_name, detection_count),
    )


@lru_cache(maxsize=1)
def _render_idle_assistant() -> np.ndarray:
    return _render_assistant_window(title=IDLE_ASSISTANT_TITLE, lines=list(IDLE_ASSISTANT_LINES))


class CameraDetector:
    def __init__(self, runtime: RuntimeConfig, camera_index: int = 0) -> None:
        self.runtime = runtime
        self.camera_index = camera_index
        self.capture: cv2.VideoCapture | None = None
        self.loaded_model: LoadedModel | None = None
        self.last_frame_ts = time.perf_counter()
        self.smoothed_fps = 0.0
        self.consecutive_read_failures = 0
        self.max_consecutive_read_failures = 5
        self.recovery_count = 0
        self.last_status_message = "San sang khoi tao camera."
        self.last_error_message = ""
        self.active_runtime_summary = ""
        self.last_raw_frame: np.ndarray | None = None
        self.last_detections: list[DetectionRecord] = []

    def initialize(self) -> None:
        last_error: Exception | None = None
        for runtime in [self.runtime, *list(iter_fallback_configs(self.runtime))]:
            try:
                self.runtime = runtime
                self.loaded_model, self.runtime.resolved_device = load_yolo_model(runtime)
                self.release()
                self.capture = self._open_capture()
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.runtime.camera_width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.runtime.camera_height)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.capture.isOpened():
                    raise RuntimeError("Khong mo duoc camera.")
                self.consecutive_read_failures = 0
                self.last_frame_ts = time.perf_counter()
                self.last_error_message = ""
                self.active_runtime_summary = (
                    f"{self.runtime.active_model_name or self.runtime.primary_model_name} | "
                    f"{self.runtime.resolved_device} | imgsz {self.runtime.imgsz}"
                )
                self.last_status_message = f"Da khoi tao camera thanh cong. Dang chay voi {self.active_runtime_summary}."
                logger.info("Detector initialized with %s", self.runtime.summary())
                return
            except Exception as exc:
                last_error = exc
                self.last_error_message = str(exc)
                self.last_status_message = "Khoi tao runtime that bai, dang thu fallback."
                logger.warning("Runtime failed, trying fallback: %s", exc)
                self.release()
        raise RuntimeError(f"Khong khoi tao duoc detector. Loi cuoi: {last_error}")

    def read_and_detect(self) -> tuple[bool, Any, list[DetectionRecord], float]:
        if self.capture is None or self.loaded_model is None:
            raise RuntimeError("Detector chua duoc khoi tao.")

        ok, frame = self.capture.read()
        if not ok:
            self.consecutive_read_failures += 1
            self.last_error_message = "Khong doc duoc frame tu camera."
            self.last_status_message = f"Mat frame camera ({self.consecutive_read_failures}/{self.max_consecutive_read_failures})."
            if self.consecutive_read_failures >= self.max_consecutive_read_failures:
                raise RuntimeError("Camera lien tuc khong tra ve frame.")
            return False, None, [], 0.0
        self.consecutive_read_failures = 0

        try:
            results = self.loaded_model.model.predict(
                source=frame,
                imgsz=self.runtime.imgsz,
                conf=self.runtime.conf,
                device=self.runtime.resolved_device,
                half=self.runtime.use_half,
                max_det=self.runtime.max_det,
                verbose=False,
                stream=False,
            )
        except Exception as exc:
            logger.warning("Inference failed on %s: %s", self.runtime.primary_model_name, exc)
            self.recovery_count += 1
            self.last_error_message = str(exc)
            self.last_status_message = "Suy luan bi loi, he thong dang tu phuc hoi va thu cau hinh an toan hon."
            self.initialize()
            return False, None, [], 0.0

        detections = self._parse_results(results)
        self.last_raw_frame = frame.copy()
        self.last_detections = detections
        processed_frame = draw_detection_results(
            image=frame,
            detections=detections,
            box_thickness=self.runtime.box_thickness,
            label_font_scale=self.runtime.label_font_scale,
        )
        self.last_status_message = f"Dang nhan dien on dinh voi {len(detections)} doi tuong."
        return True, processed_frame, detections, self._update_fps()

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
                parsed.append(
                    DetectionRecord(
                        class_id=cls_id,
                        label=names.get(cls_id, str(cls_id)),
                        confidence=float(box.conf[0].item()),
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return parsed

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.last_status_message = "Camera da dung."

    def save_training_sample(
        self,
        *,
        frame: np.ndarray,
        detections: list[DetectionRecord],
        sample_name: str | None = None,
    ) -> tuple[Path, Path]:
        ensure_project_directories()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name_prefix = _sanitize_sample_name(sample_name or "")
        base_name = f"{name_prefix}_{timestamp}" if name_prefix else f"capture_{timestamp}_{int(time.time() * 1000) % 1000:03d}"
        image_path = SAMPLE_IMAGE_DIR / f"{base_name}.jpg"
        label_path = SAMPLE_LABEL_DIR / f"{base_name}.txt"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Khong luu duoc anh: {image_path}")
        label_path.write_text(
            "\n".join(_to_yolo_bbox_line(item.class_id, item.bbox, frame.shape) for item in detections),
            encoding="utf-8",
        )
        self.last_status_message = f"Da luu mau train: {image_path.name} va {label_path.name} ({len(detections)} nhan)."
        logger.info("Saved training sample: %s | %s", image_path, label_path)
        return image_path, label_path

    def save_current_training_sample(self, sample_name: str | None = None) -> tuple[Path, Path]:
        if self.last_raw_frame is None:
            raise RuntimeError("Chua co frame nao de luu.")
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
        }

    def _open_capture(self) -> cv2.VideoCapture:
        if hasattr(cv2, "CAP_DSHOW"):
            return cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(self.camera_index)


def _compute_motion_score(current_frame: np.ndarray, previous_gray: np.ndarray | None) -> tuple[float, np.ndarray]:
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    if previous_gray is None:
        return 0.0, current_gray
    return float(cv2.absdiff(current_gray, previous_gray).mean()), current_gray


def _update_capture_preparation(
    state: CapturePreparationState,
    frame: np.ndarray,
    now: float,
) -> tuple[CapturePreparationState, bool, float]:
    motion_score, current_gray = _compute_motion_score(frame, state.previous_gray)
    stable_since = now if motion_score > MOTION_RESET_THRESHOLD else state.stable_since
    status = (
        "Phát hiện rung/lắc. Giữ yên lại để đếm ngược từ đầu."
        if motion_score > MOTION_RESET_THRESHOLD
        else "Khung hình đang ổn định. Tiếp tục giữ yên."
    )
    remaining = max(0.0, CAPTURE_STABILITY_SECONDS - (now - stable_since))
    return (
        CapturePreparationState(
            stable_since=stable_since,
            previous_gray=current_gray,
            motion_score=motion_score,
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


def _show_assistant(panel: np.ndarray) -> None:
    cv2.namedWindow(ASSISTANT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ASSISTANT_WINDOW_NAME, *ASSISTANT_WINDOW_SIZE)
    cv2.imshow(ASSISTANT_WINDOW_NAME, panel)


def _close_assistant() -> None:
    try:
        cv2.destroyWindow(ASSISTANT_WINDOW_NAME)
    except cv2.error:
        pass


def run_camera_session(runtime: RuntimeConfig, camera_index: int = 0) -> None:
    detector = CameraDetector(runtime=runtime, camera_index=camera_index)
    detector.initialize()
    capture_prep: CapturePreparationState | None = None
    naming_mode = False
    typed_name = ""
    frozen_frame: np.ndarray | None = None
    frozen_detections: list[DetectionRecord] = []
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, runtime.camera_width, runtime.camera_height)

        while True:
            ok, frame, _detections, _fps = detector.read_and_detect()
            if not ok:
                continue

            display_frame = frame
            if capture_prep is not None:
                capture_prep, ready, remaining = _update_capture_preparation(
                    capture_prep,
                    detector.last_raw_frame,
                    time.perf_counter(),
                )
                _show_assistant(
                    _render_capture_preparation_overlay(
                        remaining_seconds=remaining,
                        motion_score=capture_prep.motion_score,
                        status=capture_prep.status,
                    ),
                )
                if ready and detector.last_raw_frame is not None:
                    capture_prep = None
                    naming_mode = True
                    typed_name = ""
                    frozen_frame = detector.last_raw_frame.copy()
                    frozen_detections = list(detector.last_detections)
                    detector.last_status_message = "Khung hinh da on dinh. Hay dat ten de luu."
                    _show_assistant(_render_name_prompt(typed_name, len(frozen_detections)))
            elif naming_mode and frozen_frame is not None:
                display_frame = draw_detection_results(
                    image=frozen_frame.copy(),
                    detections=frozen_detections,
                    box_thickness=runtime.box_thickness,
                    label_font_scale=runtime.label_font_scale,
                )
                _show_assistant(_render_name_prompt(typed_name, len(frozen_detections)))

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if naming_mode:
                typed_name, should_save, should_cancel = _handle_name_input(typed_name, key)
                if should_cancel:
                    naming_mode, frozen_frame, frozen_detections, typed_name = _reset_capture_flow()
                    detector.last_status_message = "Da huy luu mau train."
                    _close_assistant()
                    continue
                if should_save and frozen_frame is not None:
                    image_path, label_path = detector.save_training_sample(
                        frame=frozen_frame,
                        detections=frozen_detections,
                        sample_name=typed_name,
                    )
                    logger.info("Da luu %s va %s", image_path.name, label_path.name)
                    naming_mode, frozen_frame, frozen_detections, typed_name = _reset_capture_flow()
                    _close_assistant()
                continue

            if capture_prep is not None:
                if key == 27:
                    capture_prep = None
                    detector.last_status_message = "Da huy che do chup mau train."
                    _close_assistant()
                continue

            if key in (ord("t"), ord("T")):
                capture_prep = CapturePreparationState(stable_since=time.perf_counter())
                detector.last_status_message = "Bat dau dem nguoc 5 giay de chup mau train."
                continue
            if key == 27:
                break
    finally:
        detector.release()
        cv2.destroyAllWindows()
