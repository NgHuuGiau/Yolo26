from __future__ import annotations

import ctypes
import shutil
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tkinter import Tk, filedialog

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
LIVE_USAGE_REFRESH_SECONDS = 0.35
SIDEBAR_REFRESH_SECONDS = 1.0
RESOURCE_PANEL_REFRESH_SECONDS = 0.25
CHAT_PANEL_REFRESH_SECONDS = 0.2
WINDOW_MARGIN = 16
SIDEBAR_WIDTH = 320
LAYOUT_GAP = 16
SIDEBAR_OUTER_PADDING = 18
RIGHT_PANEL_WIDTH = 340
BOTTOM_CHAT_HEIGHT = 128
CAMERA_PREVIEW_SCALE = 0.86
UI_SETTINGS_PATH = Path("config/ui_preferences.yaml")
CAMERA_ONLY_MIN_MARGIN = 24
GWL_STYLE = -16
GCLP_HCURSOR = -12
IDC_ARROW = 32512
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_THICKFRAME = 0x00040000
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020
SIDEBAR_BUTTON_RECTS: dict[str, tuple[int, int, int, int]] = {}
INTERACTIVE_RECTS: dict[str, tuple[int, int, int, int]] = {}
SIDEBAR_SELECTED_VIEW = "add_image"
CURRENT_LANGUAGE = "vi"
CURRENT_THEME = "dark"
CURRENT_LAYOUT_SIZE = (1, 1)
CURRENT_DISPLAY_SIZE = (1, 1)
PANEL_CACHE: dict[str, dict[str, object]] = {
    "sidebar": {"image": None, "key": None, "ts": 0.0},
    "resource": {"image": None, "key": None, "ts": 0.0},
    "chat": {"image": None, "key": None, "ts": 0.0},
    "settings": {"image": None, "key": None, "ts": 0.0},
    "live_background": {"image": None, "key": None, "ts": 0.0},
}
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

TEXTS = {
    "vi": {
        "app_name": "Yolo",
        "app_tagline": "AI Blood & Urine Analyzer",
        "add_image": "Thêm ảnh",
        "settings": "Cài đặt",
        "current_data": "Dữ liệu hiện có",
        "images_count": "Ảnh",
        "image_history": "Lịch sử ảnh",
        "all_history": "Lịch sử toàn bộ",
        "no_recent_images": "Chưa có ảnh gần đây",
        "no_recent_activity": "Chưa có hoạt động gần đây",
        "statistics": "Thống kê",
        "resource_stats": "Thống kê tài nguyên",
        "status": "Trạng thái",
        "ask_anything": "Hỏi bất cứ điều gì...",
        "analysis": "Phân tích",
        "press_t_hint": "Nhấn T để bắt đầu chụp mẫu.",
        "keep_camera_stable": "Giữ camera ổn định khi chụp.",
        "waiting_analysis": "Đang chờ phân tích...",
        "analysis_placeholder": "Kết quả chụp ảnh hoặc AI sẽ hiển thị tại đây.",
        "toggle_performance_hint": "Nhấn P để bật/tắt chế độ mượt.",
        "performance_mode_on": "Chế độ hiệu năng đang bật.",
        "performance_mode_off": "Chế độ hiệu năng đã tắt.",
        "choose_image_title": "Chọn ảnh từ máy",
        "unknown": "Không rõ",
        "settings_title": "Cài đặt hệ thống",
        "settings_subtitle": "Tùy chỉnh giao diện và ngôn ngữ theo phong cách tối kiểu VS Code.",
        "general_info": "Thông tin chung",
        "general_intro": "Màn hình này dùng để quản lý cấu hình giao diện của ứng dụng realtime YOLO.",
        "current_view": "Màn hình hiện tại",
        "language_label": "Ngôn ngữ",
        "theme_label": "Giao diện",
        "latest_image": "Ảnh gần nhất",
        "latest_label": "Nhãn gần nhất",
        "guide_title": "Hướng dẫn nhanh",
        "guide_line_1": "Nhấn vào các tuỳ chọn bên phải để đổi ngôn ngữ hoặc chủ đề.",
        "guide_line_2": "Chế độ tối đang được ưu tiên để giống giao diện VS Code.",
        "guide_line_3": "Quay lại màn hình camera bằng nút Thêm ảnh ở sidebar.",
        "appearance_title": "Tuỳ chỉnh",
        "language_system": "Ngôn ngữ hệ thống",
        "theme_system": "Hệ thống giao diện",
        "language_vi": "Tiếng Việt",
        "language_en": "English",
        "theme_dark": "Tối",
        "theme_light": "Sáng",
        "theme_auto": "Theo hệ thống",
        "theme_active_note": "Giao diện tối đang là mặc định để giữ cảm giác VS Code.",
        "camera_view_name": "Camera realtime",
        "back_to_camera": "Quay lại camera",
    },
    "en": {
        "app_name": "Yolo",
        "app_tagline": "AI Blood & Urine Analyzer",
        "add_image": "Add image",
        "settings": "Settings",
        "current_data": "Current data",
        "images_count": "Images",
        "image_history": "Image history",
        "all_history": "Recent activity",
        "no_recent_images": "No recent images",
        "no_recent_activity": "No recent activity",
        "statistics": "Statistics",
        "resource_stats": "Resource statistics",
        "status": "Status",
        "ask_anything": "Ask anything...",
        "analysis": "Analysis",
        "press_t_hint": "Press T to start sample capture.",
        "keep_camera_stable": "Keep the camera stable while capturing.",
        "waiting_analysis": "Waiting for analysis...",
        "analysis_placeholder": "Image capture or AI output will appear here.",
        "toggle_performance_hint": "Press P to toggle performance mode.",
        "performance_mode_on": "Performance mode is enabled.",
        "performance_mode_off": "Performance mode is disabled.",
        "choose_image_title": "Choose images from your computer",
        "unknown": "Unknown",
        "settings_title": "System settings",
        "settings_subtitle": "Customize language and appearance with a VS Code inspired dark UI.",
        "general_info": "General information",
        "general_intro": "This screen manages realtime YOLO interface preferences.",
        "current_view": "Current view",
        "language_label": "Language",
        "theme_label": "Appearance",
        "latest_image": "Latest image",
        "latest_label": "Latest label",
        "guide_title": "Quick guide",
        "guide_line_1": "Use the controls on the right to switch language or theme.",
        "guide_line_2": "Dark mode stays the preferred default to match VS Code.",
        "guide_line_3": "Return to the camera screen from the sidebar.",
        "appearance_title": "Preferences",
        "language_system": "System language",
        "theme_system": "Interface theme",
        "language_vi": "Vietnamese",
        "language_en": "English",
        "theme_dark": "Dark",
        "theme_light": "Light",
        "theme_auto": "System",
        "theme_active_note": "Dark mode is the default choice for the VS Code look.",
        "camera_view_name": "Realtime camera",
        "back_to_camera": "Back to camera",
    },
}

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
        if names:
            names[0] = sample_name
        else:
            names = [sample_name]
    elif isinstance(names, dict):
        names[0] = sample_name
    else:
        names = {0: sample_name}
    config["names"] = names
    save_yaml(TRAINING_DATA_YAML, config)


@lru_cache(maxsize=16)
def _load_unicode_font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates_by_weight = {
        "semibold": (
            Path("C:/Windows/Fonts/seguisb.ttf"),
            Path("C:/Windows/Fonts/Inter-SemiBold.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/Roboto-Medium.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ),
        "medium": (
            Path("C:/Windows/Fonts/seguisb.ttf"),
            Path("C:/Windows/Fonts/Inter-Medium.ttf"),
            Path("C:/Windows/Fonts/Roboto-Medium.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ),
        "regular": (
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/Inter-Regular.ttf"),
            Path("C:/Windows/Fonts/Roboto-Regular.ttf"),
            Path("C:/Windows/Fonts/tahoma.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ),
    }
    for candidate in candidates_by_weight.get(weight, candidates_by_weight["regular"]):
        if not candidate.exists():
            continue
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _font_title(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return _load_unicode_font(size, "semibold")


def _font_section(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return _load_unicode_font(size, "semibold")


def _font_body(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return _load_unicode_font(size, "regular")


def _font_body_medium(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return _load_unicode_font(size, "medium")


def _tr(key: str) -> str:
    language = CURRENT_LANGUAGE if CURRENT_LANGUAGE in TEXTS else "vi"
    return TEXTS.get(language, TEXTS["vi"]).get(key, TEXTS["vi"].get(key, key))


def _apply_theme_palette(theme: str) -> None:
    global LIGHT_BG
    global LIGHT_BG_SOFT
    global LIGHT_PANEL_FILL
    global LIGHT_PANEL_OUTLINE
    global LIGHT_PANEL_OUTLINE_SOFT
    global LIGHT_TEXT_PRIMARY
    global LIGHT_TEXT_SECONDARY
    global LIGHT_TEXT_TERTIARY
    global LIGHT_ACCENT
    global LIGHT_SUCCESS
    global LIGHT_WARNING

    active_theme = "dark" if theme == "system" else theme
    if active_theme == "light":
        LIGHT_BG = (245, 250, 255)
        LIGHT_BG_SOFT = (235, 243, 255)
        LIGHT_PANEL_FILL = (255, 255, 255)
        LIGHT_PANEL_OUTLINE = (49, 88, 209)
        LIGHT_PANEL_OUTLINE_SOFT = (186, 215, 255)
        LIGHT_TEXT_PRIMARY = (10, 18, 35)
        LIGHT_TEXT_SECONDARY = (49, 72, 106)
        LIGHT_TEXT_TERTIARY = (107, 134, 181)
        LIGHT_ACCENT = (18, 105, 255)
        LIGHT_SUCCESS = (16, 185, 129)
        LIGHT_WARNING = (245, 158, 11)
        return

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


def _load_ui_preferences() -> None:
    global CURRENT_LANGUAGE
    global CURRENT_THEME
    if not UI_SETTINGS_PATH.exists():
        _apply_theme_palette(CURRENT_THEME)
        return
    try:
        data = load_yaml(UI_SETTINGS_PATH) or {}
    except Exception:
        data = {}
    language = str(data.get("language", CURRENT_LANGUAGE)).lower()
    theme = str(data.get("theme", CURRENT_THEME)).lower()
    if language in TEXTS:
        CURRENT_LANGUAGE = language
    if theme in {"dark", "light", "system"}:
        CURRENT_THEME = theme
    SIDEBAR_SELECTED_VIEW = "add_image"
    _apply_theme_palette(CURRENT_THEME)


def _save_ui_preferences() -> None:
    save_yaml(
        UI_SETTINGS_PATH,
        {
            "language": CURRENT_LANGUAGE,
            "theme": CURRENT_THEME,
        },
    )


def _set_language(language: str) -> None:
    global CURRENT_LANGUAGE
    if language not in TEXTS:
        return
    CURRENT_LANGUAGE = language
    _save_ui_preferences()


def _set_theme(theme: str) -> None:
    global CURRENT_THEME
    if theme not in {"dark", "light", "system"}:
        return
    CURRENT_THEME = theme
    _apply_theme_palette(theme)
    _save_ui_preferences()


def _assistant_lines_for_preparation(remaining_seconds: float, motion_score: float, status: str) -> list[str]:
    return [
        f"Đếm ngược: {int(np.ceil(remaining_seconds))} giây",
        f"Rung/lắc: {motion_score:.2f}",
        status,
        f"Cần ổn định liên tục {STABLE_FRAMES_REQUIRED} frame.",
    ]


def _assistant_lines_for_name_prompt(sample_name: str, detection_count: int) -> list[str]:
    return [
        f"Số nhãn: {detection_count}",
        f"Đặt tên: {sample_name or '(để trống sẽ dùng tên mặc định)'}",
        "Enter để lưu | Esc để thoát",
        "Space để chụp lại",
    ]


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
        self.frame_index = 0
        self.detect_interval = 1
        self.inference_thread: threading.Thread | None = None
        self.inference_stop_event = threading.Event()
        self.inference_ready_event = threading.Event()
        self.inference_lock = threading.Lock()
        self.pending_inference_frame: np.ndarray | None = None
        self.pending_inference_error: Exception | None = None

    def initialize(self) -> None:
        last_error: Exception | None = None
        for runtime in [self.runtime, *list(iter_fallback_configs(self.runtime))]:
            try:
                self._stop_inference_worker()
                self.runtime = runtime
                self.loaded_model, self.runtime.resolved_device = load_yolo_model(runtime)
                self.release()
                self.capture = self._open_capture()
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.runtime.camera_width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.runtime.camera_height)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.capture.isOpened():
                    raise RuntimeError("Không mở được camera.")
                self.consecutive_read_failures = 0
                self.last_frame_ts = time.perf_counter()
                self.frame_index = 0
                self.detect_interval = {"high": 1, "medium": 2, "low": 4}.get(self.runtime.profile_name, 1)
                self.last_detections = []
                self.pending_inference_error = None
                self.pending_inference_frame = None
                self._start_inference_worker()
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
        raise RuntimeError(f"Không khởi tạo được detector. Lỗi cuối: {last_error}")

    def read_and_detect(self) -> tuple[bool, Any, list[DetectionRecord], float]:
        if self.capture is None or self.loaded_model is None:
            raise RuntimeError("Detector chua duoc khoi tao.")
        if self.pending_inference_error is not None:
            exc = self.pending_inference_error
            self.pending_inference_error = None
            logger.warning("Inference failed on %s: %s", self.runtime.primary_model_name, exc)
            self.recovery_count += 1
            self.last_error_message = str(exc)
            self.last_status_message = "Suy luan bi loi, he thong dang tu phuc hoi va thu cau hinh an toan hon."
            self.initialize()
            return False, None, [], 0.0

        ok, frame = self.capture.read()
        if not ok:
            self.consecutive_read_failures += 1
            self.last_error_message = "Không đọc được frame từ camera."
            self.last_status_message = f"Mat frame camera ({self.consecutive_read_failures}/{self.max_consecutive_read_failures})."
            if self.consecutive_read_failures >= self.max_consecutive_read_failures:
                raise RuntimeError("Camera lien tuc khong tra ve frame.")
            return False, None, [], 0.0
        self.consecutive_read_failures = 0
        self.last_raw_frame = frame.copy()
        self.frame_index += 1
        if self.frame_index == 1 and not self.last_detections:
            try:
                self.last_detections = self._predict_frame(frame)
            except Exception as exc:
                logger.warning("Inference failed on %s: %s", self.runtime.primary_model_name, exc)
                self.recovery_count += 1
                self.last_error_message = str(exc)
                self.last_status_message = "Suy luan bi loi, he thong dang tu phuc hoi va thu cau hinh an toan hon."
                self.initialize()
                return False, None, [], 0.0
        else:
            self._queue_inference(frame)
        detections = list(self.last_detections)
        processed_frame = draw_detection_results(
            image=frame,
            detections=detections,
            box_thickness=self.runtime.box_thickness,
            label_font_scale=self.runtime.label_font_scale,
        )
        self.last_status_message = f"Dang nhan dien on dinh voi {len(detections)} doi tuong."
        return True, processed_frame, detections, self._update_fps()

    def _queue_inference(self, frame: np.ndarray) -> None:
        should_submit = self.frame_index == 1 or (self.frame_index % self.detect_interval) == 1
        if not should_submit:
            return
        with self.inference_lock:
            self.pending_inference_frame = frame.copy()
        self.inference_ready_event.set()

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
            imgsz=self.runtime.imgsz,
            conf=self.runtime.conf,
            device=self.runtime.resolved_device,
            half=self.runtime.use_half,
            max_det=self.runtime.max_det,
            verbose=False,
            stream=False,
        )
        return self._parse_results(results)

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
        self._stop_inference_worker()
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
        self.last_status_message = f"Da luu mau train: {image_path.name} va {label_path.name} ({len(detections)} nhan)."
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
        }

    def _open_capture(self) -> cv2.VideoCapture:
        if hasattr(cv2, "CAP_DSHOW"):
            return cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(self.camera_index)


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


def _compose_camera_frame(
    frame: np.ndarray,
    title: str,
    lines: list[str],
    padding: int = CAMERA_BOTTOM_PADDING,
    top_right_text: str = "",
) -> np.ndarray:
    if padding <= 0:
        return frame
    frame_height, frame_width = frame.shape[:2]
    panel = Image.new("RGB", (frame_width, padding), color=LIGHT_BG)
    draw = ImageDraw.Draw(panel)
    draw.rectangle([(0, 0), (frame_width - 1, padding - 1)], outline=(214, 210, 202), width=1)
    if title or lines:
        title_font = _load_unicode_font(28)
        draw.text((26, 18), title, fill=(28, 28, 28), font=title_font)
        if top_right_text:
            top_font = _load_unicode_font(18)
            box = draw.textbbox((0, 0), top_right_text, font=top_font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
            badge_width = max(132, text_width + 34)
            badge_left = frame_width - badge_width - 24
            badge_top = 20
            badge_bottom = 50
            draw.rounded_rectangle(
                [(badge_left, badge_top), (badge_left + badge_width, badge_bottom)],
                radius=14,
                fill=(240, 235, 224),
                outline=(214, 210, 202),
                width=1,
            )
            text_x = badge_left + (badge_width - text_width) / 2
            text_y = badge_top + ((badge_bottom - badge_top) - text_height) / 2 - box[1]
            draw.text((text_x, text_y), top_right_text, fill=(52, 52, 52), font=top_font)
        draw.line([(24, 60), (frame_width - 24, 60)], fill=(226, 221, 212), width=1)
        if title == "Tài nguyên hệ thống":
            _draw_resource_panel(panel, draw, frame_width, padding, lines)
        elif title == CAPTURE_PANEL_TITLE and lines and lines[0].startswith("Đếm ngược:"):
            _draw_capture_prep_panel(panel, draw, frame_width, padding, lines)
        elif title == CAPTURE_PANEL_TITLE:
            _draw_name_prompt_panel(panel, draw, frame_width, padding, lines)
    panel_bgr = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    canvas = np.zeros((frame_height + padding, frame_width, 3), dtype=frame.dtype)
    canvas[:frame_height, :frame_width] = frame
    canvas[frame_height:, :frame_width] = panel_bgr
    return canvas


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    used_words = 1
    for word in words[1:]:
        candidate = f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font)
        if (box[2] - box[0]) <= max_width:
            current = candidate
            used_words += 1
            continue
        lines.append(current)
        current = word
        used_words += 1
        if len(lines) >= max_lines - 1:
            break
    remaining_words = words[len(" ".join(lines).split()) :] if lines else words
    final_line = " ".join(remaining_words) if remaining_words else current
    box = draw.textbbox((0, 0), final_line, font=font)
    while (box[2] - box[0]) > max_width and len(final_line) > 4:
        final_line = final_line[:-4].rstrip() + "..."
        box = draw.textbbox((0, 0), final_line, font=font)
    lines.append(final_line)
    return lines[:max_lines]


def _draw_antialiased_rounded_rectangle(
    image: Image.Image,
    rect: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int] | None = None,
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return
    scale = 4
    outline_width = max(1, width * scale)
    offset = outline_width // 2
    temp = Image.new("RGBA", (max(1, w * scale), max(1, h * scale)), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    scaled_rect = (offset, offset, w * scale - 1 - offset, h * scale - 1 - offset)
    if fill is not None:
        temp_draw.rounded_rectangle(scaled_rect, radius=radius * scale, fill=(*fill, 255))
    if outline is not None and width > 0:
        temp_draw.rounded_rectangle(
            scaled_rect,
            radius=radius * scale,
            outline=(*outline, 255),
            width=outline_width,
        )
    down = temp.resize((w, h), resample=Image.LANCZOS)
    image.paste(down, (x1, y1), down)


def _draw_smooth_card(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    radius: int,
    fill: tuple[int, int, int] | None = None,
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    x1, y1, x2, y2 = rect
    fill = LIGHT_PANEL_FILL if fill is None else fill
    outline = LIGHT_PANEL_OUTLINE if outline is None else outline
    inset_outline = (48, 48, 48) if LIGHT_BG[0] < 40 else (245, 247, 250)
    _draw_antialiased_rounded_rectangle(image, rect, radius=radius, fill=fill, outline=outline, width=width)
    inset = max(2, width + 2)
    if (x2 - x1) > inset * 2 and (y2 - y1) > inset * 2:
        _draw_antialiased_rounded_rectangle(
            image,
            (x1 + inset, y1 + inset, x2 - inset, y2 - inset),
            radius=max(6, radius - inset),
            outline=inset_outline,
            width=1,
        )


def _recent_data_items(limit: int = 5) -> tuple[list[str], list[str]]:
    image_items: list[Path] = []
    label_items: list[Path] = []
    for directory, bucket, patterns in (
        (SAMPLE_IMAGE_DIR, image_items, ("*.jpg", "*.jpeg", "*.png", "*.bmp")),
        (SAMPLE_LABEL_DIR, label_items, ("*.txt",)),
    ):
        if not directory.exists():
            continue
        for pattern in patterns:
            bucket.extend(directory.glob(pattern))
    image_names = [path.name for path in sorted(image_items, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]]
    label_names = [path.name for path in sorted(label_items, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]]
    return image_names, label_names


def _recent_history_items(limit: int = 6) -> list[str]:
    entries: list[tuple[float, str]] = []
    for directory, prefix, patterns in (
        (SAMPLE_IMAGE_DIR, "Ảnh", ("*.jpg", "*.jpeg", "*.png", "*.bmp")),
    ):
        if not directory.exists():
            continue
        for pattern in patterns:
            for path in directory.glob(pattern):
                try:
                    entries.append((path.stat().st_mtime, f"{prefix}  {path.name}"))
                except OSError:
                    continue
    return [label for _mtime, label in sorted(entries, key=lambda item: item[0], reverse=True)[:limit]]


def _latest_sample_pair() -> tuple[str | None, str | None]:
    images, labels = _recent_data_items(limit=1)
    return (images[0] if images else None, labels[0] if labels else None)


def _draw_sidebar_icon(
    image: Image.Image,
    kind: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    scale = 4
    icon_width = 18
    icon_height = 16
    temp = Image.new("RGBA", (icon_width * scale, icon_height * scale), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    ox = 0
    oy = 0
    if kind == "add_image":
        temp_draw.rounded_rectangle(
            [
                (ox + 1 * scale, oy + 1 * scale),
                (ox + 17 * scale, oy + 15 * scale),
            ],
            radius=4 * scale,
            outline=(*color, 255),
            width=2 * scale,
        )
        temp_draw.polygon(
            [
                (ox + 2 * scale, oy + 12 * scale),
                (ox + 7 * scale, oy + 7 * scale),
                (ox + 10 * scale, oy + 10 * scale),
                (ox + 14 * scale, oy + 5 * scale),
                (ox + 16 * scale, oy + 12 * scale),
            ],
            fill=None,
            outline=(*color, 255),
            width=2 * scale,
        )
        temp_draw.ellipse(
            [
                (ox + 12 * scale, oy + 3 * scale),
                (ox + 15 * scale, oy + 6 * scale),
            ],
            outline=(*color, 255),
            width=2 * scale,
        )
    elif kind == "history_images":
        temp_draw.rounded_rectangle(
            [
                (ox + 1 * scale, oy + 2 * scale),
                (ox + 17 * scale, oy + 15 * scale),
            ],
            radius=4 * scale,
            outline=(*color, 255),
            width=2 * scale,
        )
        temp_draw.line(
            [(ox + 5 * scale, oy + 15 * scale), (ox + 5 * scale, oy + 18 * scale)],
            fill=(*color, 255),
            width=2 * scale,
        )
        temp_draw.line(
            [(ox + 13 * scale, oy + 15 * scale), (ox + 13 * scale, oy + 18 * scale)],
            fill=(*color, 255),
            width=2 * scale,
        )
        temp_draw.line(
            [(ox + 5 * scale, oy + 18 * scale), (ox + 13 * scale, oy + 18 * scale)],
            fill=(*color, 255),
            width=2 * scale,
        )
    elif kind == "settings":
        temp_draw.ellipse(
            [
                (ox + 4 * scale, oy + 3 * scale),
                (ox + 14 * scale, oy + 13 * scale),
            ],
            outline=(*color, 255),
            width=2 * scale,
        )
        temp_draw.ellipse(
            [
                (ox + 7 * scale, oy + 6 * scale),
                (ox + 11 * scale, oy + 10 * scale),
            ],
            outline=(*color, 255),
            width=2 * scale,
        )
        for x1, y1, x2, y2 in (
            (8, 0, 10, 4),
            (8, 12, 10, 16),
            (0, 7, 4, 9),
            (14, 7, 18, 9),
            (3, 2, 6, 5),
            (12, 11, 15, 14),
            (12, 2, 15, 5),
            (3, 11, 6, 14),
        ):
            temp_draw.rounded_rectangle(
                [
                    (ox + x1 * scale, oy + y1 * scale),
                    (ox + x2 * scale, oy + y2 * scale),
                ],
                radius=2 * scale,
                fill=(*color, 255),
            )
    else:
        temp_draw.arc(
            [
                (ox + 1 * scale, oy + 1 * scale),
                (ox + 17 * scale, oy + 17 * scale),
            ],
            start=20,
            end=320,
            fill=(*color, 255),
            width=2 * scale,
        )
        temp_draw.line(
            [
                (ox + 9 * scale, oy + 4 * scale),
                (ox + 9 * scale, oy + 9 * scale),
                (ox + 13 * scale, oy + 11 * scale),
            ],
            fill=(*color, 255),
            width=2 * scale,
        )
    down = temp.resize((icon_width, icon_height), resample=Image.LANCZOS)
    image.paste(down, (x, y), down)


def _draw_mic_icon(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    center_y: int,
    color: tuple[int, int, int],
) -> None:
    mic_body = (center_x - 5, center_y - 10, center_x + 5, center_y + 4)
    draw.rounded_rectangle(mic_body, radius=5, outline=color, width=2)
    draw.arc(
        [
            (center_x - 10, center_y - 3),
            (center_x + 10, center_y + 13),
        ],
        start=200,
        end=340,
        fill=color,
        width=2,
    )
    draw.line((center_x, center_y + 4, center_x, center_y + 11), fill=color, width=2)
    draw.line((center_x - 6, center_y + 11, center_x + 6, center_y + 11), fill=color, width=2)


def _scale_int(value: int, scale: float) -> int:
    return int(round(value * scale))


def _scale_box(rect: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    return tuple(_scale_int(value, scale) for value in rect)


def _sidebar_mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    global SIDEBAR_SELECTED_VIEW
    if event != cv2.EVENT_LBUTTONUP:
        return
    layout_width = max(1, CURRENT_LAYOUT_SIZE[0])
    layout_height = max(1, CURRENT_LAYOUT_SIZE[1])
    display_width = max(1, CURRENT_DISPLAY_SIZE[0])
    display_height = max(1, CURRENT_DISPLAY_SIZE[1])
    layout_x = int(round(x * layout_width / display_width))
    layout_y = int(round(y * layout_height / display_height))
    for action, rect in INTERACTIVE_RECTS.items():
        x1, y1, x2, y2 = rect
        if x1 > layout_x or layout_x > x2 or y1 > layout_y or layout_y > y2:
            continue
        if action == "sidebar:add_image":
            SIDEBAR_SELECTED_VIEW = "add_image"
            _save_ui_preferences()
            _import_files_from_machine(
                title=_tr("choose_image_title"),
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All files", "*.*"),
                ],
                destination_dir=SAMPLE_IMAGE_DIR,
            )
            return
        if action == "sidebar:settings":
            SIDEBAR_SELECTED_VIEW = "settings"
            _save_ui_preferences()
            return
        if action == "settings:close":
            SIDEBAR_SELECTED_VIEW = "add_image"
            _save_ui_preferences()
            return
        if action.startswith("language:"):
            _set_language(action.split(":", 1)[1])
            return
        if action.startswith("theme:"):
            _set_theme(action.split(":", 1)[1])
            return


def _import_files_from_machine(
    title: str,
    filetypes: list[tuple[str, str]],
    destination_dir: Path,
) -> int:
    ensure_project_directories()
    destination_dir.mkdir(parents=True, exist_ok=True)
    root: Tk | None = None
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilenames(title=title, filetypes=filetypes)
        count = 0
        for raw_path in selected:
            source = Path(raw_path)
            if not source.is_file():
                continue
            target = destination_dir / source.name
            shutil.copy2(source, target)
            count += 1
        if count:
            logger.info("Đã thêm %s file vào %s", count, destination_dir)
        return count
    except Exception as exc:
        logger.warning("Không mở được hộp chọn file: %s", exc)
        return 0
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _draw_data_sidebar(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    global INTERACTIVE_RECTS
    global SIDEBAR_BUTTON_RECTS
    SIDEBAR_BUTTON_RECTS = {}
    for key in [item for item in list(INTERACTIVE_RECTS) if item.startswith("sidebar:")]:
        INTERACTIVE_RECTS.pop(key, None)
    _draw_smooth_card(image, draw, (left, top, left + width, top + height), radius=28, fill=LIGHT_BG, width=2)
    title_font = _font_title(31)
    section_font = _font_section(19)
    item_font = _font_body_medium(15)
    hint_font = _font_body(13)
    button_font = _font_section(17)

    draw.text((left + SIDEBAR_OUTER_PADDING, top + 18), _tr("app_name"), fill=LIGHT_TEXT_PRIMARY, font=title_font)
    draw.text(
        (left + SIDEBAR_OUTER_PADDING, top + 58),
        _tr("app_tagline"),
        fill=LIGHT_TEXT_SECONDARY,
        font=section_font,
    )

    data_card = (
        left + SIDEBAR_OUTER_PADDING,
        top + 88,
        left + width - SIDEBAR_OUTER_PADDING,
        top + height - 96,
    )

    _draw_smooth_card(image, draw, data_card, radius=24, width=2)

    images, _labels = _recent_data_items()
    history_items = _recent_history_items(limit=4)
    button_specs = [
        ("add_image", _tr("add_image"), LIGHT_PANEL_FILL, LIGHT_TEXT_PRIMARY),
    ]
    button_left = data_card[0] + 16
    button_right = data_card[2] - 16
    button_top = data_card[1] + 16
    button_height = 48
    for index, (view, label, fill, text_fill) in enumerate(button_specs):
        y1 = button_top + index * (button_height + 10)
        y2 = y1 + button_height
        active = SIDEBAR_SELECTED_VIEW == view
        _draw_smooth_card(
            image,
            draw,
            (button_left, y1, button_right, y2),
            radius=18,
            fill=LIGHT_ACCENT if active else fill,
            outline=LIGHT_ACCENT if active else LIGHT_PANEL_OUTLINE_SOFT,
            width=2,
        )
        icon_color = (255, 255, 255) if active else LIGHT_TEXT_PRIMARY
        _draw_sidebar_icon(image, view, button_left + 18, y1 + 15, icon_color)
        draw.text(
            (button_left + 54, y1 + 11),
            label,
            fill=(255, 255, 255) if active else LIGHT_TEXT_PRIMARY,
            font=button_font,
        )
        SIDEBAR_BUTTON_RECTS[view] = (button_left, y1, button_right, y2)
        INTERACTIVE_RECTS[f"sidebar:{view}"] = (button_left, y1, button_right, y2)

    summary_y = button_top + len(button_specs) * (button_height + 12) + 8
    draw.text((data_card[0] + 16, summary_y), _tr("current_data"), fill=LIGHT_TEXT_PRIMARY, font=section_font)
    draw.text(
        (data_card[0] + 16, summary_y + 32),
        f"{_tr('images_count')}: {len(images)}",
        fill=LIGHT_TEXT_SECONDARY,
        font=item_font,
    )

    y = summary_y + 72
    sections = [
        (_tr("image_history"), [f"{_tr('images_count')}  {name}" for name in images[:3]], _tr("no_recent_images")),
        (_tr("all_history"), history_items, _tr("no_recent_activity")),
    ]
    for section_title, items, empty_text in sections:
        draw.text((data_card[0] + 16, y), section_title, fill=LIGHT_TEXT_PRIMARY, font=item_font)
        card_top = y + 24
        card_bottom = card_top + 98
        history_rect = (data_card[0] + 12, card_top, data_card[2] - 12, card_bottom)
        _draw_smooth_card(
            image,
            draw,
            history_rect,
            radius=18,
            fill=LIGHT_BG_SOFT,
            outline=LIGHT_PANEL_OUTLINE,
            width=2,
        )
        line_y = card_top + 16
        rendered = items if items else [empty_text]
        for name in rendered[:3]:
            wrapped_item = _wrap_text(draw, f"• {name}", hint_font, history_rect[2] - history_rect[0] - 24, 2)
            for line in wrapped_item:
                draw.text((history_rect[0] + 12, line_y), line, fill=LIGHT_TEXT_SECONDARY, font=hint_font)
                line_y += 20
            line_y += 4
        y = card_bottom + 20

    settings_rect = (left + 18, top + height - 72, left + width - 18, top + height - 18)
    settings_active = SIDEBAR_SELECTED_VIEW == "settings"
    _draw_smooth_card(
        image,
        draw,
        settings_rect,
        radius=22,
        fill=LIGHT_ACCENT if settings_active else LIGHT_PANEL_FILL,
        outline=LIGHT_ACCENT if settings_active else LIGHT_PANEL_OUTLINE,
        width=2,
    )
    settings_icon_color = (255, 255, 255) if settings_active else LIGHT_TEXT_PRIMARY
    _draw_sidebar_icon(image, "settings", settings_rect[0] + 18, settings_rect[1] + 15, settings_icon_color)
    draw.text(
        (settings_rect[0] + 54, settings_rect[1] + 11),
        _tr("settings"),
        fill=(255, 255, 255) if settings_active else LIGHT_TEXT_PRIMARY,
        font=button_font,
    )
    INTERACTIVE_RECTS["sidebar:settings"] = settings_rect


def _draw_resource_sidebar(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    height: int,
    resource_lines: list[str],
    fps_text: str,
    status_lines: list[str],
) -> None:
    title_font = _font_title(31)
    section_font = _font_section(19)
    label_font = _font_body_medium(15)
    body_font = _font_body(15)
    _draw_smooth_card(image, draw, (left, top, left + width, top + height), radius=28, fill=LIGHT_BG, width=1)
    draw.text((left + 18, top + 18), _tr("statistics"), fill=LIGHT_TEXT_PRIMARY, font=title_font)
    top_card_top = top + 72
    top_card_bottom = top_card_top + 280
    top_card = (left + 16, top_card_top, left + width - 16, top_card_bottom)
    bottom_card_top = top_card_bottom + 12
    bottom_card = (left + 16, bottom_card_top, left + width - 16, top + height - 16)
    for card in (top_card, bottom_card):
        _draw_smooth_card(image, draw, card, radius=22, width=1)

    draw.text(
        (top_card[0] + 14, top_card[1] + 18),
        _tr("resource_stats"),
        fill=LIGHT_TEXT_PRIMARY,
        font=section_font,
    )
    row_step = 46
    for index, raw_line in enumerate(resource_lines[:4]):
        row_y = top_card[1] + 68 + index * row_step
        parts = raw_line.split(None, 1)
        label = parts[0] if parts else ""
        draw.text((top_card[0] + 12, row_y), f"{label}", fill=LIGHT_TEXT_SECONDARY, font=label_font)
        progress_width = top_card[2] - top_card[0] - 28
        pct = None
        raw_value = parts[1] if len(parts) > 1 else ""
        if "%" in raw_value:
            try:
                pct = max(0.0, min(100.0, float(raw_value.split("%", 1)[0].split()[-1])))
            except Exception:
                pct = None
        if pct is not None:
            bar_left = top_card[0] + 12
            bar_top = row_y + 18
            bar_right = bar_left + progress_width
            bar_bottom = bar_top + 8
            track_fill = (60, 60, 64) if LIGHT_BG[0] < 40 else (232, 235, 240)
            draw.rounded_rectangle((bar_left, bar_top, bar_right, bar_bottom), radius=4, fill=track_fill)
            fill_width = int(progress_width * pct / 100.0)
            if fill_width > 8:
                draw.rounded_rectangle((bar_left, bar_top, bar_left + fill_width, bar_bottom), radius=4, fill=LIGHT_ACCENT)

    draw.text(
        (bottom_card[0] + 16, bottom_card[1] + 14),
        _tr("status"),
        fill=LIGHT_TEXT_PRIMARY,
        font=section_font,
    )
    status_y = bottom_card[1] + 48
    available_width = bottom_card[2] - bottom_card[0] - 32
    for raw_line in status_lines[:4]:
        wrapped_lines = _wrap_text(draw, raw_line, body_font, available_width, 2)
        for line in wrapped_lines:
            draw.text((bottom_card[0] + 18, status_y), line, fill=LIGHT_TEXT_SECONDARY, font=body_font)
            status_y += 22
        status_y += 4

    if fps_text:
        box = draw.textbbox((0, 0), fps_text, font=section_font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        badge_width = max(92, text_width + 24)
        badge_height = 34
        badge_left = bottom_card[2] - badge_width - 14
        badge_top = bottom_card[3] - badge_height - 14
        _draw_smooth_card(image, draw, (badge_left, badge_top, badge_left + badge_width, badge_top + badge_height), radius=18, width=1)
        draw.text(
            (
                badge_left + (badge_width - text_width) / 2,
                badge_top + ((badge_height - text_height) / 2) - box[1],
            ),
            fps_text,
            fill=LIGHT_TEXT_PRIMARY,
            font=section_font,
        )


def _draw_chat_box(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    height: int,
    title: str,
    lines: list[str],
) -> None:
    body_font = _font_body(16)
    status_font = _font_body_medium(15)
    input_left = left + 18
    input_top = top + 10
    input_right = left + width - 18
    input_bottom = top + 70
    _draw_smooth_card(image, draw, (input_left, input_top, input_right, input_bottom), radius=34, width=1)

    plus_box_left = input_left + 18
    plus_box_top = input_top + 13
    plus_size = 24
    plus_cx = plus_box_left + plus_size // 2
    plus_cy = plus_box_top + plus_size // 2
    draw.line((plus_cx - 7, plus_cy, plus_cx + 7, plus_cy), fill=LIGHT_TEXT_PRIMARY, width=2)
    draw.line((plus_cx, plus_cy - 7, plus_cx, plus_cy + 7), fill=LIGHT_TEXT_PRIMARY, width=2)

    draw.text((input_left + 60, input_top + 16), _tr("ask_anything"), fill=LIGHT_TEXT_TERTIARY, font=body_font)

    mic_center_x = input_right - 42
    mic_center_y = input_top + 28
    draw.ellipse(
        [
            (mic_center_x - 17, mic_center_y - 17),
            (mic_center_x + 17, mic_center_y + 17),
        ],
        outline=LIGHT_PANEL_OUTLINE_SOFT,
        width=1,
        fill=LIGHT_PANEL_FILL,
    )
    _draw_mic_icon(draw, mic_center_x, mic_center_y, LIGHT_TEXT_PRIMARY)
    status = lines[0] if lines else _tr("analysis")
    wrapped_status = _wrap_text(draw, status, status_font, width - 44, 2)
    detail_y = input_bottom + 14
    for line in wrapped_status:
        draw.text((input_left + 4, detail_y), line, fill=LIGHT_TEXT_PRIMARY, font=status_font)
        detail_y += 20
    detail_font = _font_body(14)
    for detail in lines[1:4]:
        wrapped_detail = _wrap_text(draw, detail, detail_font, width - 44, 2)
        for line in wrapped_detail:
            draw.text((input_left + 4, detail_y), line, fill=LIGHT_TEXT_SECONDARY, font=detail_font)
            detail_y += 22
        detail_y += 2


def _draw_setting_option(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    label: str,
    selected: bool,
) -> None:
    _draw_smooth_card(
        image,
        draw,
        rect,
        radius=16,
        fill=LIGHT_ACCENT if selected else LIGHT_BG_SOFT,
        outline=LIGHT_ACCENT if selected else LIGHT_PANEL_OUTLINE,
        width=2,
    )
    font = _font_body_medium(16)
    text_fill = (255, 255, 255) if selected else LIGHT_TEXT_PRIMARY
    box = draw.textbbox((0, 0), label, font=font)
    draw.text(
        (rect[0] + 16, rect[1] + ((rect[3] - rect[1] - (box[3] - box[1])) / 2) - box[1]),
        label,
        fill=text_fill,
        font=font,
    )


def _draw_wrapped_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    max_lines: int,
    line_height: int = 20,
) -> int:
    lines = _wrap_text(draw, text, font, max_width, max_lines)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def _register_sidebar_interactive_rects(left: int, top: int, width: int, height: int) -> None:
    global SIDEBAR_BUTTON_RECTS
    SIDEBAR_BUTTON_RECTS = {}
    for key in [item for item in list(INTERACTIVE_RECTS) if item.startswith("sidebar:")]:
        INTERACTIVE_RECTS.pop(key, None)

    data_card = (
        left + SIDEBAR_OUTER_PADDING,
        top + 88,
        left + width - SIDEBAR_OUTER_PADDING,
        top + height - 96,
    )
    button_left = data_card[0] + 16
    button_right = data_card[2] - 16
    button_top = data_card[1] + 16
    button_height = 48
    for index, view in enumerate(("add_image",)):
        y1 = button_top + index * (button_height + 10)
        y2 = y1 + button_height
        rect = (button_left, y1, button_right, y2)
        SIDEBAR_BUTTON_RECTS[view] = rect
        INTERACTIVE_RECTS[f"sidebar:{view}"] = rect

    settings_rect = (left + 18, top + height - 72, left + width - 18, top + height - 18)
    INTERACTIVE_RECTS["sidebar:settings"] = settings_rect


def _register_settings_interactive_rects(left: int, top: int, width: int, height: int) -> None:
    for key in [
        item
        for item in list(INTERACTIVE_RECTS)
        if item.startswith("language:") or item.startswith("theme:") or item.startswith("settings:")
    ]:
        INTERACTIVE_RECTS.pop(key, None)

    close_rect = (left + width - 228, top + 18, left + width - 24, top + 66)
    INTERACTIVE_RECTS["settings:close"] = close_rect

    content_top = top + 76
    content_bottom = top + height - 24
    column_gap = 18
    left_width = int(width * 0.42)
    left_card = (left + 20, content_top, left + 20 + left_width, content_bottom)
    right_card = (left_card[2] + column_gap, content_top, left + width - 20, content_bottom)

    block_y = right_card[1] + 58
    option_gap = 12
    option_height = 50
    block_y += 30
    right_inner_width = right_card[2] - right_card[0] - 36
    language_width = (right_inner_width - option_gap) // 2
    for index, code in enumerate(("vi", "en")):
        option_rect = (
            right_card[0] + 18 + index * (language_width + option_gap),
            block_y,
            right_card[0] + 18 + index * (language_width + option_gap) + language_width,
            block_y + option_height,
        )
        INTERACTIVE_RECTS[f"language:{code}"] = option_rect

    block_y += option_height + 28
    block_y += 30
    theme_width = (right_inner_width - (option_gap * 2)) // 3
    for index, code in enumerate(("dark", "light", "system")):
        option_rect = (
            right_card[0] + 18 + index * (theme_width + option_gap),
            block_y,
            right_card[0] + 18 + index * (theme_width + option_gap) + theme_width,
            block_y + option_height,
        )
        INTERACTIVE_RECTS[f"theme:{code}"] = option_rect


def _cached_panel_image(
    cache_name: str,
    key: tuple[object, ...],
    refresh_seconds: float,
    render_fn,
) -> Image.Image:
    cache = PANEL_CACHE[cache_name]
    now = time.perf_counter()
    cached_key = cache.get("key")
    cached_image = cache.get("image")
    cached_ts = float(cache.get("ts", 0.0))
    if cached_image is not None and cached_key == key and (now - cached_ts) < refresh_seconds:
        return cached_image.copy()
    image = render_fn()
    cache["image"] = image.copy()
    cache["key"] = key
    cache["ts"] = now
    return image


def _cached_panel_array(
    cache_name: str,
    key: tuple[object, ...],
    refresh_seconds: float,
    render_fn,
) -> np.ndarray:
    cache = PANEL_CACHE[cache_name]
    now = time.perf_counter()
    cached_key = cache.get("key")
    cached_image = cache.get("image")
    cached_ts = float(cache.get("ts", 0.0))
    if cached_image is not None and cached_key == key and (now - cached_ts) < refresh_seconds:
        return np.array(cached_image, copy=True)
    image = render_fn()
    cache["image"] = np.array(image, copy=True)
    cache["key"] = key
    cache["ts"] = now
    return image


def _runtime_refresh_profile(profile_name: str) -> dict[str, float]:
    if profile_name == "high":
        return {
            "metrics": 0.45,
            "sidebar": 1.2,
            "resource": 0.45,
            "chat": 0.25,
            "settings": 0.45,
        }
    if profile_name == "medium":
        return {
            "metrics": 0.8,
            "sidebar": 2.2,
            "resource": 1.0,
            "chat": 0.55,
            "settings": 1.0,
        }
    return {
        "metrics": 1.4,
        "sidebar": 3.0,
        "resource": 1.5,
        "chat": 0.9,
        "settings": 1.5,
    }


def _render_data_sidebar_panel(width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), color=LIGHT_BG)
    draw = ImageDraw.Draw(panel)
    _register_sidebar_interactive_rects(0, 0, width, height)
    _draw_data_sidebar(panel, draw, 0, 0, width, height)
    return panel


def _render_resource_sidebar_panel(
    width: int,
    height: int,
    resource_lines: list[str],
    fps_text: str,
    status_lines: list[str],
) -> Image.Image:
    panel = Image.new("RGB", (width, height), color=LIGHT_BG)
    draw = ImageDraw.Draw(panel)
    _draw_resource_sidebar(panel, draw, 0, 0, width, height, resource_lines, fps_text, status_lines)
    return panel


def _render_chat_box_panel(width: int, height: int, title: str, lines: list[str]) -> Image.Image:
    panel = Image.new("RGB", (width, height), color=LIGHT_BG)
    draw = ImageDraw.Draw(panel)
    _draw_chat_box(panel, draw, 0, 0, width, height, title, lines)
    return panel


def _render_settings_screen_panel(
    width: int,
    height: int,
    resource_lines: list[str],
    fps_text: str,
    status_lines: list[str],
) -> Image.Image:
    panel = Image.new("RGB", (width, height), color=LIGHT_BG)
    draw = ImageDraw.Draw(panel)
    _register_settings_interactive_rects(0, 0, width, height)
    _draw_settings_screen(panel, draw, 0, 0, width, height, resource_lines, fps_text, status_lines)
    return panel


def _render_live_background_panel(
    frame_width: int,
    frame_height: int,
    total_width: int,
    total_height: int,
    resource_lines: list[str],
    fps_text: str,
    status_lines: list[str],
    chat_title: str,
    chat_lines: list[str],
    refresh_profile: dict[str, float],
) -> np.ndarray:
    layout = np.full((total_height, total_width, 3), LIGHT_BG, dtype=np.uint8)
    sidebar_key = (SIDEBAR_SELECTED_VIEW, CURRENT_LANGUAGE, CURRENT_THEME, SIDEBAR_WIDTH, total_height)
    sidebar_panel = _cached_panel_image(
        "sidebar",
        sidebar_key,
        refresh_profile["sidebar"],
        lambda: _render_data_sidebar_panel(SIDEBAR_WIDTH, total_height),
    )
    sidebar_bgr = cv2.cvtColor(np.array(sidebar_panel), cv2.COLOR_RGB2BGR)
    layout[:, 0:SIDEBAR_WIDTH] = sidebar_bgr

    camera_left = SIDEBAR_WIDTH + LAYOUT_GAP
    resource_left = camera_left + frame_width + LAYOUT_GAP
    resource_key = (CURRENT_THEME, RIGHT_PANEL_WIDTH, total_height, tuple(resource_lines), fps_text, tuple(status_lines))
    resource_panel = _cached_panel_image(
        "resource",
        resource_key,
        refresh_profile["resource"],
        lambda: _render_resource_sidebar_panel(RIGHT_PANEL_WIDTH, total_height, resource_lines, fps_text, status_lines),
    )
    resource_bgr = cv2.cvtColor(np.array(resource_panel), cv2.COLOR_RGB2BGR)
    layout[:, resource_left : resource_left + RIGHT_PANEL_WIDTH] = resource_bgr

    chat_key = (CURRENT_LANGUAGE, CURRENT_THEME, frame_width, BOTTOM_CHAT_HEIGHT, chat_title, tuple(chat_lines))
    chat_panel = _cached_panel_image(
        "chat",
        chat_key,
        refresh_profile["chat"],
        lambda: _render_chat_box_panel(frame_width, BOTTOM_CHAT_HEIGHT, chat_title, chat_lines),
    )
    chat_bgr = cv2.cvtColor(np.array(chat_panel), cv2.COLOR_RGB2BGR)
    chat_top = frame_height + LAYOUT_GAP
    layout[chat_top : chat_top + BOTTOM_CHAT_HEIGHT, camera_left : camera_left + frame_width] = chat_bgr
    return layout


def _draw_settings_screen(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    height: int,
    resource_lines: list[str],
    fps_text: str,
    status_lines: list[str],
) -> None:
    for key in [
        item
        for item in list(INTERACTIVE_RECTS)
        if item.startswith("language:") or item.startswith("theme:") or item.startswith("settings:")
    ]:
        INTERACTIVE_RECTS.pop(key, None)

    title_font = _font_title(30)
    section_font = _font_section(20)
    body_font = _font_body(15)
    value_font = _font_body_medium(16)

    workspace = (left, top, left + width, top + height)
    _draw_smooth_card(image, draw, workspace, radius=28, fill=LIGHT_BG, width=2)

    title_x = left + 24
    title_y = top + 20
    draw.text((title_x, title_y), _tr("settings_title"), fill=LIGHT_TEXT_PRIMARY, font=title_font)
    close_rect = (left + width - 228, top + 18, left + width - 24, top + 66)
    _draw_setting_option(image, draw, close_rect, _tr("back_to_camera"), False)
    INTERACTIVE_RECTS["settings:close"] = close_rect

    content_top = top + 76
    content_bottom = top + height - 24
    column_gap = 18
    left_width = int(width * 0.42)
    left_card = (left + 20, content_top, left + 20 + left_width, content_bottom)
    right_card = (left_card[2] + column_gap, content_top, left + width - 20, content_bottom)
    _draw_smooth_card(image, draw, left_card, radius=24, width=2)
    _draw_smooth_card(image, draw, right_card, radius=24, width=2)

    draw.text((left_card[0] + 18, left_card[1] + 16), _tr("general_info"), fill=LIGHT_TEXT_PRIMARY, font=section_font)
    info_y = left_card[1] + 54
    info_y = _draw_wrapped_block(
        draw,
        left_card[0] + 18,
        info_y,
        _tr("general_intro"),
        body_font,
        LIGHT_TEXT_SECONDARY,
        left_card[2] - left_card[0] - 36,
        3,
    )

    latest_image, latest_label = _latest_sample_pair()
    info_items = [
        (_tr("current_view"), _tr("camera_view_name") if SIDEBAR_SELECTED_VIEW == "add_image" else _tr("settings")),
        (_tr("language_label"), _tr("language_vi") if CURRENT_LANGUAGE == "vi" else _tr("language_en")),
        (_tr("theme_label"), {"dark": _tr("theme_dark"), "light": _tr("theme_light"), "system": _tr("theme_auto")}.get(CURRENT_THEME, _tr("theme_dark"))),
        (_tr("latest_image"), latest_image or _tr("unknown")),
        (_tr("latest_label"), latest_label or _tr("unknown")),
    ]
    card_y = info_y + 10
    for label, value in info_items:
        item_rect = (left_card[0] + 16, card_y, left_card[2] - 16, card_y + 74)
        _draw_smooth_card(image, draw, item_rect, radius=16, fill=LIGHT_BG_SOFT, outline=LIGHT_PANEL_OUTLINE, width=1)
        draw.text((item_rect[0] + 14, item_rect[1] + 10), label, fill=LIGHT_TEXT_TERTIARY, font=body_font)
        _draw_wrapped_block(
            draw,
            item_rect[0] + 14,
            item_rect[1] + 34,
            value,
            value_font,
            LIGHT_TEXT_PRIMARY,
            item_rect[2] - item_rect[0] - 28,
            2,
            18,
        )
        card_y += 84

    draw.text((right_card[0] + 18, right_card[1] + 16), _tr("appearance_title"), fill=LIGHT_TEXT_PRIMARY, font=section_font)
    block_y = right_card[1] + 58
    option_gap = 12
    option_height = 50

    draw.text((right_card[0] + 18, block_y), _tr("language_system"), fill=LIGHT_TEXT_SECONDARY, font=value_font)
    block_y += 30
    right_inner_width = right_card[2] - right_card[0] - 36
    language_width = (right_inner_width - option_gap) // 2
    language_options = [("vi", _tr("language_vi")), ("en", _tr("language_en"))]
    for index, (code, label) in enumerate(language_options):
        option_rect = (
            right_card[0] + 18 + index * (language_width + option_gap),
            block_y,
            right_card[0] + 18 + index * (language_width + option_gap) + language_width,
            block_y + option_height,
        )
        _draw_setting_option(image, draw, option_rect, label, CURRENT_LANGUAGE == code)
        INTERACTIVE_RECTS[f"language:{code}"] = option_rect

    block_y += option_height + 28
    draw.text((right_card[0] + 18, block_y), _tr("theme_system"), fill=LIGHT_TEXT_SECONDARY, font=value_font)
    block_y += 30
    theme_options = [("dark", _tr("theme_dark")), ("light", _tr("theme_light")), ("system", _tr("theme_auto"))]
    theme_width = (right_inner_width - (option_gap * 2)) // 3
    for index, (code, label) in enumerate(theme_options):
        option_rect = (
            right_card[0] + 18 + index * (theme_width + option_gap),
            block_y,
            right_card[0] + 18 + index * (theme_width + option_gap) + theme_width,
            block_y + option_height,
        )
        _draw_setting_option(image, draw, option_rect, label, CURRENT_THEME == code)
        INTERACTIVE_RECTS[f"theme:{code}"] = option_rect

    block_y += option_height + 20
    summary_rect = (right_card[0] + 18, block_y, right_card[2] - 18, right_card[3] - 18)
    _draw_smooth_card(image, draw, summary_rect, radius=18, fill=LIGHT_BG_SOFT, outline=LIGHT_PANEL_OUTLINE, width=1)
    draw.text((summary_rect[0] + 14, summary_rect[1] + 12), _tr("resource_stats"), fill=LIGHT_TEXT_PRIMARY, font=value_font)
    stat_y = summary_rect[1] + 40
    summary_font = _font_body(14)
    for raw_line in [*resource_lines[:4], fps_text]:
        stat_y = _draw_wrapped_block(
            draw,
            summary_rect[0] + 14,
            stat_y,
            raw_line,
            summary_font,
            LIGHT_TEXT_SECONDARY,
            summary_rect[2] - summary_rect[0] - 28,
            2,
            18,
        )
        stat_y += 2


def _scaled_camera_frame(frame: np.ndarray) -> np.ndarray:
    scale = CAMERA_PREVIEW_SCALE
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _compose_desktop_layout(
    frame: np.ndarray,
    resource_lines: list[str],
    fps_text: str,
    status_lines: list[str],
    chat_title: str,
    chat_lines: list[str],
    refresh_profile: dict[str, float],
) -> np.ndarray:
    global CURRENT_LAYOUT_SIZE
    frame = _scaled_camera_frame(frame)
    frame_height, frame_width = frame.shape[:2]
    total_width = SIDEBAR_WIDTH + LAYOUT_GAP + frame_width + LAYOUT_GAP + RIGHT_PANEL_WIDTH
    total_height = frame_height + LAYOUT_GAP + BOTTOM_CHAT_HEIGHT
    CURRENT_LAYOUT_SIZE = (total_width, total_height)
    for key in [item for item in list(INTERACTIVE_RECTS) if not item.startswith("sidebar:")]:
        INTERACTIVE_RECTS.pop(key, None)

    camera_left = SIDEBAR_WIDTH + LAYOUT_GAP
    camera_top = 0
    _register_sidebar_interactive_rects(0, 0, SIDEBAR_WIDTH, total_height)
    if SIDEBAR_SELECTED_VIEW == "settings":
        layout = Image.new("RGB", (total_width, total_height), color=LIGHT_BG)
        draw = ImageDraw.Draw(layout)
        sidebar_key = (SIDEBAR_SELECTED_VIEW, CURRENT_LANGUAGE, CURRENT_THEME, SIDEBAR_WIDTH, total_height)
        sidebar_panel = _cached_panel_image(
            "sidebar",
            sidebar_key,
            refresh_profile["sidebar"],
            lambda: _render_data_sidebar_panel(SIDEBAR_WIDTH, total_height),
        )
        layout.paste(sidebar_panel, (0, 0))
        settings_width = total_width - camera_left
        settings_key = (
            CURRENT_LANGUAGE,
            CURRENT_THEME,
            settings_width,
            total_height,
            tuple(resource_lines),
            fps_text,
            tuple(status_lines),
        )
        settings_panel = _cached_panel_image(
            "settings",
            settings_key,
            refresh_profile["settings"],
            lambda: _render_settings_screen_panel(settings_width, total_height, resource_lines, fps_text, status_lines),
        )
        _register_settings_interactive_rects(camera_left, 0, settings_width, total_height)
        layout.paste(settings_panel, (camera_left, 0))
        return cv2.cvtColor(np.array(layout), cv2.COLOR_RGB2BGR)

    live_background_key = (
        CURRENT_LANGUAGE,
        CURRENT_THEME,
        SIDEBAR_SELECTED_VIEW,
        frame_width,
        frame_height,
        total_width,
        total_height,
        tuple(resource_lines),
        fps_text,
        tuple(status_lines),
        chat_title,
        tuple(chat_lines),
    )
    layout = _cached_panel_array(
        "live_background",
        live_background_key,
        min(refresh_profile["sidebar"], refresh_profile["resource"], refresh_profile["chat"]),
        lambda: _render_live_background_panel(
            frame_width,
            frame_height,
            total_width,
            total_height,
            resource_lines,
            fps_text,
            status_lines,
            chat_title,
            chat_lines,
            refresh_profile,
        ),
    )
    layout[camera_top : camera_top + frame_height, camera_left : camera_left + frame_width] = frame
    return layout


def _compose_camera_only_layout(frame: np.ndarray) -> np.ndarray:
    global CURRENT_LAYOUT_SIZE
    CURRENT_LAYOUT_SIZE = (frame.shape[1], frame.shape[0])
    INTERACTIVE_RECTS.clear()
    for cache in PANEL_CACHE.values():
        cache["ts"] = 0.0
    return _scaled_camera_frame(frame)


def _preferred_desktop_layout_size(frame: np.ndarray) -> tuple[int, int]:
    scaled = _scaled_camera_frame(frame)
    frame_height, frame_width = scaled.shape[:2]
    total_width = SIDEBAR_WIDTH + LAYOUT_GAP + frame_width + LAYOUT_GAP + RIGHT_PANEL_WIDTH
    total_height = frame_height + LAYOUT_GAP + BOTTOM_CHAT_HEIGHT
    return total_width, total_height


def _draw_resource_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    frame_width: int,
    padding: int,
    lines: list[str],
    scale: float = 1.0,
) -> None:
    label_font = _font_body_medium(_scale_int(14, scale))
    value_font = _font_body_medium(_scale_int(18, scale))
    left = _scale_int(22, scale)
    top = _scale_int(78, scale)
    gap = _scale_int(12, scale)
    card_width = (frame_width - (left * 2) - (gap * 3)) // 4
    card_height = _scale_int(48, scale)
    card_fill = LIGHT_PANEL_FILL
    card_outline = LIGHT_PANEL_OUTLINE

    for index, raw_line in enumerate(lines[:4]):
        parts = raw_line.split(None, 1)
        label = parts[0] if parts else ""
        value = parts[1] if len(parts) > 1 else "Không rõ"
        x1 = left + index * (card_width + gap)
        y1 = top
        x2 = x1 + card_width
        y2 = y1 + card_height
        _draw_smooth_card(
            image,
            draw,
            (x1, y1, x2, y2),
            radius=_scale_int(16, scale),
            fill=card_fill,
            outline=card_outline,
            width=2,
        )
        draw.text((x1 + _scale_int(16, scale), y1 + _scale_int(8, scale)), label, fill=LIGHT_TEXT_SECONDARY, font=label_font)
        draw.text((x1 + _scale_int(16, scale), y1 + _scale_int(23, scale)), value, fill=LIGHT_TEXT_PRIMARY, font=value_font)


def _draw_capture_prep_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    frame_width: int,
    padding: int,
    lines: list[str],
) -> None:
    big_font = _font_title(28)
    value_font = _font_body_medium(16)
    body_font = _font_body(15)
    small_font = _font_body(13)

    countdown_value = lines[0].split(":", 1)[1].strip() if ":" in lines[0] else lines[0]
    motion_value = lines[1].split(":", 1)[1].strip() if ":" in lines[1] else lines[1]

    left_chip = (24, 72, 182, 120)
    middle_chip = (194, 72, 370, 120)
    status_card = (384, 72, frame_width - 24, 130)

    for rect in (left_chip, middle_chip, status_card):
        _draw_smooth_card(image, draw, rect, radius=16, fill=LIGHT_PANEL_FILL, outline=LIGHT_PANEL_OUTLINE, width=2)

    draw.text((left_chip[0] + 14, left_chip[1] + 8), "Đếm ngược", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    draw.text((left_chip[0] + 14, left_chip[1] + 24), countdown_value, fill=LIGHT_TEXT_PRIMARY, font=big_font)

    draw.text((middle_chip[0] + 14, middle_chip[1] + 8), "Rung/lắc", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    draw.text((middle_chip[0] + 14, middle_chip[1] + 27), motion_value, fill=LIGHT_TEXT_PRIMARY, font=value_font)

    draw.text((status_card[0] + 14, status_card[1] + 8), "Trạng thái", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    wrapped_status = _wrap_text(draw, lines[2], body_font, status_card[2] - status_card[0] - 28, 2)
    wrapped_hint = _wrap_text(draw, lines[3], small_font, status_card[2] - status_card[0] - 28, 2)
    for index, line in enumerate(wrapped_status):
        draw.text(
            (status_card[0] + 14, status_card[1] + 24 + (index * 20)),
            line,
            fill=LIGHT_TEXT_PRIMARY,
            font=body_font,
        )
    hint_y = status_card[1] + 60
    for line in wrapped_hint:
        draw.text((status_card[0] + 14, hint_y), line, fill=LIGHT_TEXT_SECONDARY, font=small_font)
        hint_y += 16


def _draw_name_prompt_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    frame_width: int,
    padding: int,
    lines: list[str],
) -> None:
    body_font = _font_body(15)
    small_font = _font_body(13)
    value_font = _font_body_medium(18)

    count_chip = (24, 72, 170, 120)
    name_chip = (184, 72, frame_width - 24, 120)

    for rect in (count_chip, name_chip):
        _draw_smooth_card(image, draw, rect, radius=16, fill=LIGHT_PANEL_FILL, outline=LIGHT_PANEL_OUTLINE, width=2)

    draw.text((count_chip[0] + 14, count_chip[1] + 8), "Số nhãn", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    draw.text(
        (count_chip[0] + 14, count_chip[1] + 24),
        lines[0].split(":", 1)[1].strip() if ":" in lines[0] else lines[0],
        fill=LIGHT_TEXT_PRIMARY,
        font=value_font,
    )

    draw.text((name_chip[0] + 14, name_chip[1] + 8), "Tên mẫu", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    wrapped_name = _wrap_text(draw, lines[1], body_font, name_chip[2] - name_chip[0] - 28, 2)
    for index, line in enumerate(wrapped_name):
        draw.text(
            (name_chip[0] + 14, name_chip[1] + 24 + (index * 18)),
            line,
            fill=LIGHT_TEXT_PRIMARY,
            font=body_font,
        )

    draw.text((28, 126), "Enter lưu | Backspace xóa | Esc hủy", fill=LIGHT_TEXT_SECONDARY, font=small_font)


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


def _lock_window_controls(window_name: str) -> None:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return
        style = int(user32.GetWindowLongW(hwnd, GWL_STYLE))
        style &= ~WS_MINIMIZEBOX
        style &= ~WS_MAXIMIZEBOX
        style &= ~WS_THICKFRAME
        user32.SetWindowLongW(hwnd, GWL_STYLE, style)
        user32.SetWindowPos(
            hwnd,
            0,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
        )
    except Exception:
        return


def _set_window_arrow_cursor(window_name: str) -> None:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return
        arrow_cursor = user32.LoadCursorW(None, IDC_ARROW)
        if not arrow_cursor:
            return
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            user32.SetClassLongPtrW(hwnd, GCLP_HCURSOR, arrow_cursor)
        else:
            user32.SetClassLongW(hwnd, GCLP_HCURSOR, arrow_cursor)
        user32.SetCursor(arrow_cursor)
    except Exception:
        return


def _get_window_client_size(window_name: str) -> tuple[int, int] | None:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return None

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return None
        return width, height
    except Exception:
        return None


def _resize_to_window(frame: np.ndarray, window_name: str) -> np.ndarray:
    client_size = _get_window_client_size(window_name)
    if not client_size:
        return frame
    width, height = client_size
    if width == frame.shape[1] and height == frame.shape[0]:
        return frame
    interpolation = cv2.INTER_AREA if width < frame.shape[1] or height < frame.shape[0] else cv2.INTER_LANCZOS4
    try:
        return cv2.resize(frame, (width, height), interpolation=interpolation)
    except Exception:
        return frame


def _should_render_camera_only(window_name: str, preferred_size: tuple[int, int]) -> bool:
    client_size = _get_window_client_size(window_name)
    if not client_size:
        return False
    width, height = client_size
    preferred_width, preferred_height = preferred_size
    return (
        width < (preferred_width - CAMERA_ONLY_MIN_MARGIN)
        or height < (preferred_height - CAMERA_ONLY_MIN_MARGIN)
    )


def _panel_lines_for_live_view(detector: CameraDetector, hardware: Any) -> list[str]:
    usage = get_live_usage_snapshot()

    def _percent(name: str) -> float | None:
        value = usage.get(name)
        return float(value) if value is not None else None

    def _format_percent(name: str) -> str:
        value = _percent(name)
        return f"{value:.1f}%" if value is not None else _tr("unknown")

    def _ram_line() -> str:
        percent = _percent("ram_usage_percent")
        total = getattr(hardware, "ram_gb", None)
        if percent is not None and total:
            used = total * percent / 100.0
            return f"{percent:.1f}% | {used:.1f}/{total:.1f} GB"
        return _format_percent("ram_usage_percent")

    def _vram_line() -> str:
        percent = _percent("vram_usage_percent")
        total = getattr(hardware, "vram_gb", None)
        if percent is not None and total:
            used = total * percent / 100.0
            return f"{percent:.1f}% | {used:.1f}/{total:.1f} GB"
        return _format_percent("vram_usage_percent")

    def _gpu_line() -> str:
        gpu_percent = _format_percent("gpu_usage_percent")
        gpu_name = getattr(hardware, "gpu_name", _tr("unknown"))
        return f"{gpu_percent} | {gpu_name}"

    return [
        f"CPU {_format_percent('cpu_usage_percent')}",
        f"RAM {_ram_line()}",
        f"GPU {_gpu_line()}",
        f"VRAM {_vram_line()}",
    ]


def _fps_panel_line(fps: float) -> str:
    if fps <= 0:
        return f"FPS   {_tr('unknown')}"
    return f"FPS   {fps:.1f}"


def _is_window_maximized(window_name: str) -> bool:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return False
        return bool(user32.IsZoomed(hwnd))
    except Exception:
        return False


def _maximize_window(window_name: str) -> None:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return
        user32.ShowWindow(hwnd, 3)
    except Exception:
        return


def _compose_compact_camera_frame(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    compact = _scaled_camera_frame(frame)
    if not lines:
        return compact
    overlay = compact.copy()
    box_height = 34 + min(3, len(lines)) * 22
    cv2.rectangle(overlay, (18, 18), (min(compact.shape[1] - 18, 420), 18 + box_height), (5, 33, 54), -1)
    cv2.addWeighted(overlay, 0.68, compact, 0.32, 0, compact)
    text_y = 42
    cv2.putText(compact, title, (34, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (232, 247, 255), 2, cv2.LINE_AA)
    text_y += 26
    for line in lines[:3]:
        cv2.putText(compact, line, (34, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (188, 224, 244), 1, cv2.LINE_AA)
        text_y += 22
    return compact


_load_ui_preferences()


def run_camera_session(runtime: RuntimeConfig, camera_index: int = 0) -> None:
    global CURRENT_DISPLAY_SIZE
    detector = CameraDetector(runtime=runtime, camera_index=camera_index)
    hardware = detect_hardware()
    detector.initialize()
    capture_prep: CapturePreparationState | None = None
    naming_mode = False
    typed_name = ""
    frozen_frame: np.ndarray | None = None
    frozen_detections: list[DetectionRecord] = []
    window_positioned = False
    last_metrics_refresh = 0.0
    resource_lines: list[str] = []
    fps_text = _fps_panel_line(0.0)
    refresh_profile = _runtime_refresh_profile(runtime.profile_name)
    performance_mode = runtime.profile_name == "low"
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, _sidebar_mouse_callback)
        _set_window_arrow_cursor(WINDOW_NAME)

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
                if ready and detector.last_raw_frame is not None:
                    capture_prep = None
                    naming_mode = True
                    typed_name = _next_sample_sequence_name()
                    frozen_frame = detector.last_raw_frame.copy()
                    frozen_detections = list(detector.last_detections)
                    detector.last_status_message = "Khung hinh da on dinh. Hay dat ten de luu."
            elif naming_mode and frozen_frame is not None:
                display_frame = draw_detection_results(
                    image=frozen_frame.copy(),
                    detections=frozen_detections,
                    box_thickness=runtime.box_thickness,
                    label_font_scale=runtime.label_font_scale,
                )

            now = time.perf_counter()
            if not resource_lines or (now - last_metrics_refresh) >= refresh_profile["metrics"]:
                resource_lines = _panel_lines_for_live_view(detector, hardware)
                fps_text = _fps_panel_line(_fps)
                last_metrics_refresh = now
            chat_title = "Khung chat"
            if capture_prep is not None:
                chat_title = CAPTURE_PANEL_TITLE
                prep_lines = _assistant_lines_for_preparation(
                    remaining_seconds=remaining,
                    motion_score=capture_prep.motion_score,
                    status=capture_prep.status,
                )
                chat_lines = ["Đang chụp", *prep_lines]
            elif naming_mode:
                chat_title = CAPTURE_PANEL_TITLE
                prompt_lines = _assistant_lines_for_name_prompt(typed_name, len(frozen_detections))
                chat_lines = ["Đặt tên", *prompt_lines]
            else:
                chat_lines = [_tr("analysis"), _tr("press_t_hint"), _tr("toggle_performance_hint")]
            status_lines = (
                chat_lines
                if capture_prep is not None or naming_mode
                else [_tr("waiting_analysis"), _tr("analysis_placeholder")]
            )

            preferred_layout_size = _preferred_desktop_layout_size(display_frame)
            if performance_mode or _should_render_camera_only(WINDOW_NAME, preferred_layout_size):
                composed = _compose_camera_only_layout(display_frame)
            else:
                composed = _compose_desktop_layout(
                    frame=display_frame,
                    resource_lines=resource_lines,
                    fps_text=fps_text,
                    status_lines=status_lines,
                    chat_title=chat_title,
                    chat_lines=chat_lines,
                    refresh_profile=refresh_profile,
                )

            composed = _resize_to_window(composed, WINDOW_NAME)
            CURRENT_DISPLAY_SIZE = (composed.shape[1], composed.shape[0])
            cv2.imshow(WINDOW_NAME, composed)
            if not window_positioned:
                cv2.resizeWindow(WINDOW_NAME, composed.shape[1], composed.shape[0])
                _maximize_window(WINDOW_NAME)
                _set_window_arrow_cursor(WINDOW_NAME)
                window_positioned = True
            key = cv2.waitKey(1) & 0xFF

            if naming_mode:
                if key == ord(" "):
                    naming_mode = False
                    typed_name = ""
                    frozen_frame = None
                    frozen_detections = []
                    capture_prep = _start_capture_preparation()
                    detector.last_status_message = "Bat dau chup lai mau train."
                    continue
                typed_name, should_save, should_cancel = _handle_name_input(typed_name, key)
                if should_cancel:
                    naming_mode, frozen_frame, frozen_detections, typed_name = _reset_capture_flow()
                    detector.last_status_message = "Da huy luu mau train."
                    continue
                if should_save and frozen_frame is not None:
                    image_path, label_path = detector.save_training_sample(
                        frame=frozen_frame,
                        detections=frozen_detections,
                        sample_name=typed_name,
                    )
                    logger.info("Da luu %s va %s", image_path.name, label_path.name)
                    naming_mode, frozen_frame, frozen_detections, typed_name = _reset_capture_flow()
                continue

            if capture_prep is not None:
                if key == 27:
                    capture_prep = None
                    detector.last_status_message = "Da huy che do chup mau train."
                continue

            if key in (ord("t"), ord("T")):
                capture_prep = CapturePreparationState(stable_since=time.perf_counter())
                detector.last_status_message = "Bat dau dem nguoc 5 giay de chup mau train."
                continue
            if key in (ord("p"), ord("P")):
                performance_mode = not performance_mode
                detector.last_status_message = _tr("performance_mode_on") if performance_mode else _tr("performance_mode_off")
                continue
            if key == 27:
                break
    finally:
        detector.release()
        cv2.destroyAllWindows()
