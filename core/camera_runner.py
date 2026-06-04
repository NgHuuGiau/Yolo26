from __future__ import annotations

import ctypes
import shutil
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
WINDOW_MARGIN = 16
SIDEBAR_WIDTH = 320
LAYOUT_GAP = 16
SIDEBAR_OUTER_PADDING = 18
RIGHT_PANEL_WIDTH = 340
BOTTOM_CHAT_HEIGHT = 128
CAMERA_PREVIEW_SCALE = 0.86
GWL_STYLE = -16
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_THICKFRAME = 0x00040000
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020
SIDEBAR_BUTTON_RECTS: dict[str, tuple[int, int, int, int]] = {}
SIDEBAR_SELECTED_VIEW = "add_image"
LIGHT_BG = (238, 241, 245)
LIGHT_BG_SOFT = (249, 250, 251)
LIGHT_PANEL_FILL = (255, 255, 255)
LIGHT_PANEL_OUTLINE = (209, 213, 219)
LIGHT_PANEL_OUTLINE_SOFT = (229, 231, 235)
LIGHT_TEXT_PRIMARY = (17, 24, 39)
LIGHT_TEXT_SECONDARY = (71, 85, 105)
LIGHT_TEXT_TERTIARY = (148, 163, 184)
LIGHT_ACCENT = (79, 70, 229)
LIGHT_SUCCESS = (16, 185, 129)
LIGHT_WARNING = (245, 158, 11)

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
                    raise RuntimeError("Không mở được camera.")
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
        raise RuntimeError(f"Không khởi tạo được detector. Lỗi cuối: {last_error}")

    def read_and_detect(self) -> tuple[bool, Any, list[DetectionRecord], float]:
        if self.capture is None or self.loaded_model is None:
            raise RuntimeError("Detector chua duoc khoi tao.")

        ok, frame = self.capture.read()
        if not ok:
            self.consecutive_read_failures += 1
            self.last_error_message = "Không đọc được frame từ camera."
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
            _draw_resource_panel(draw, frame_width, padding, lines)
        elif title == CAPTURE_PANEL_TITLE and lines and lines[0].startswith("Đếm ngược:"):
            _draw_capture_prep_panel(draw, frame_width, padding, lines)
        elif title == CAPTURE_PANEL_TITLE:
            _draw_name_prompt_panel(draw, frame_width, padding, lines)
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


def _draw_smooth_card(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    radius: int,
    fill: tuple[int, int, int] = LIGHT_PANEL_FILL,
    outline: tuple[int, int, int] = LIGHT_PANEL_OUTLINE,
    width: int = 1,
) -> None:
    x1, y1, x2, y2 = rect
    shadow_rect = (x1 + 2, y1 + 4, x2 + 2, y2 + 4)
    draw.rounded_rectangle(shadow_rect, radius=radius, fill=(222, 226, 232))
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)
    inset = max(2, width + 2)
    x1, y1, x2, y2 = rect
    if (x2 - x1) > inset * 2 and (y2 - y1) > inset * 2:
        draw.rounded_rectangle(
            (x1 + inset, y1 + inset, x2 - inset, y2 - inset),
            radius=max(6, radius - inset),
            outline=(245, 247, 250),
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


def _draw_sidebar_icon(draw: ImageDraw.ImageDraw, kind: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    if kind == "add_image":
        draw.rounded_rectangle([(x, y + 2), (x + 18, y + 16)], radius=4, outline=color, width=2)
        draw.polygon([(x + 3, y + 13), (x + 8, y + 8), (x + 11, y + 11), (x + 14, y + 7), (x + 16, y + 13)], outline=color, fill=None, width=2)
        draw.ellipse([(x + 12, y + 4), (x + 15, y + 7)], outline=color, width=2)
    elif kind == "history_images":
        draw.rounded_rectangle([(x + 1, y + 2), (x + 17, y + 15)], radius=4, outline=color, width=2)
        draw.line([(x + 5, y + 15), (x + 5, y + 18)], fill=color, width=2)
        draw.line([(x + 13, y + 15), (x + 13, y + 18)], fill=color, width=2)
        draw.line([(x + 5, y + 18), (x + 13, y + 18)], fill=color, width=2)
    else:
        draw.arc([(x + 1, y + 1), (x + 17, y + 17)], start=20, end=320, fill=color, width=2)
        draw.line([(x + 9, y + 4), (x + 9, y + 9), (x + 13, y + 11)], fill=color, width=2)


def _sidebar_mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    global SIDEBAR_SELECTED_VIEW
    if event != cv2.EVENT_LBUTTONUP:
        return
    for view, rect in SIDEBAR_BUTTON_RECTS.items():
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            SIDEBAR_SELECTED_VIEW = view
            if view == "add_image":
                _import_files_from_machine(
                    title="Chọn ảnh từ máy",
                    filetypes=[
                        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                        ("All files", "*.*"),
                    ],
                    destination_dir=SAMPLE_IMAGE_DIR,
                )
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


def _draw_data_sidebar(draw: ImageDraw.ImageDraw, left: int, top: int, width: int, height: int) -> None:
    global SIDEBAR_BUTTON_RECTS
    SIDEBAR_BUTTON_RECTS = {}
    _draw_smooth_card(draw, (left, top, left + width, top + height), radius=28, fill=LIGHT_BG, width=2)
    title_font = _font_title(31)
    section_font = _font_section(19)
    item_font = _font_body_medium(15)
    hint_font = _font_body(13)
    button_font = _font_section(17)

    draw.text((left + SIDEBAR_OUTER_PADDING, top + 18), "Yolo", fill=LIGHT_TEXT_PRIMARY, font=title_font)
    draw.text(
        (left + SIDEBAR_OUTER_PADDING, top + 58),
        "AI Blood & Urine Analyzer",
        fill=LIGHT_TEXT_SECONDARY,
        font=section_font,
    )

    data_card = (
        left + SIDEBAR_OUTER_PADDING,
        top + 88,
        left + width - SIDEBAR_OUTER_PADDING,
        top + height - 18,
    )

    _draw_smooth_card(draw, data_card, radius=24, width=2)

    images, _labels = _recent_data_items()
    history_items = _recent_history_items(limit=4)
    button_specs = [
        ("add_image", "Thêm ảnh", (255, 255, 255), (28, 28, 28)),
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
            draw,
            (button_left, y1, button_right, y2),
            radius=18,
            fill=LIGHT_TEXT_PRIMARY if active else fill,
            outline=LIGHT_PANEL_OUTLINE if active else LIGHT_PANEL_OUTLINE_SOFT,
            width=2,
        )
        icon_color = (255, 255, 255) if active else LIGHT_TEXT_PRIMARY
        _draw_sidebar_icon(draw, view, button_left + 16, y1 + 10, icon_color)
        draw.text(
            (button_left + 50, y1 + 11),
            label,
            fill=(255, 255, 255) if active else LIGHT_TEXT_PRIMARY,
            font=button_font,
        )
        SIDEBAR_BUTTON_RECTS[view] = (button_left, y1, button_right, y2)

    summary_y = button_top + len(button_specs) * (button_height + 12) + 8
    draw.text((data_card[0] + 16, summary_y), "Dữ liệu hiện có", fill=LIGHT_TEXT_PRIMARY, font=section_font)
    draw.text((data_card[0] + 16, summary_y + 32), f"Ảnh: {len(images)}", fill=LIGHT_TEXT_SECONDARY, font=item_font)

    y = summary_y + 72
    sections = [
        ("Lịch sử ảnh", [f"Ảnh  {name}" for name in images[:3]], "Chưa có ảnh gần đây"),
        ("Lịch sử toàn bộ", history_items, "Chưa có hoạt động gần đây"),
    ]
    for section_title, items, empty_text in sections:
        draw.text((data_card[0] + 16, y), section_title, fill=LIGHT_TEXT_PRIMARY, font=item_font)
        card_top = y + 24
        card_bottom = card_top + 98
        history_rect = (data_card[0] + 12, card_top, data_card[2] - 12, card_bottom)
        _draw_smooth_card(draw, history_rect, radius=18, fill=LIGHT_BG_SOFT, outline=LIGHT_PANEL_OUTLINE_SOFT, width=1)
        line_y = card_top + 12
        rendered = items if items else [empty_text]
        for name in rendered[:3]:
            wrapped_item = _wrap_text(draw, f"• {name}", hint_font, history_rect[2] - history_rect[0] - 24, 2)
            for line in wrapped_item:
                draw.text((history_rect[0] + 12, line_y), line, fill=LIGHT_TEXT_SECONDARY, font=hint_font)
                line_y += 20
            line_y += 4
        y = card_bottom + 20

def _draw_resource_sidebar(
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
    _draw_smooth_card(draw, (left, top, left + width, top + height), radius=28, fill=LIGHT_BG, width=1)
    draw.text((left + 18, top + 18), "Thống kê", fill=LIGHT_TEXT_PRIMARY, font=title_font)
    top_card = (left + 16, top + 64, left + width - 16, top + 264)
    bottom_card = (left + 16, top + 270, left + width - 16, top + height - 16)
    for card in (top_card, bottom_card):
        _draw_smooth_card(draw, card, radius=22, width=1)

    draw.text((top_card[0] + 16, top_card[1] + 14), "Thống kê tài nguyên", fill=LIGHT_TEXT_PRIMARY, font=section_font)
    row_y = top_card[1] + 48
    row_step = 44
    for raw_line in resource_lines[:4]:
        parts = raw_line.split(None, 1)
        label = parts[0] if parts else ""
        draw.text((top_card[0] + 18, row_y), f"{label}", fill=LIGHT_TEXT_SECONDARY, font=label_font)
        progress_width = top_card[2] - top_card[0] - 36
        pct = None
        raw_value = parts[1] if len(parts) > 1 else ""
        if "%" in raw_value:
            try:
                pct = max(0.0, min(100.0, float(raw_value.split("%", 1)[0].split()[-1])))
            except Exception:
                pct = None
        if pct is not None:
            bar_left = top_card[0] + 18
            bar_top = row_y + 22
            bar_right = bar_left + progress_width
            bar_bottom = bar_top + 8
            draw.rounded_rectangle((bar_left, bar_top, bar_right, bar_bottom), radius=4, fill=(232, 235, 240))
            fill_width = int(progress_width * pct / 100.0)
            if fill_width > 8:
                draw.rounded_rectangle((bar_left, bar_top, bar_left + fill_width, bar_bottom), radius=4, fill=LIGHT_ACCENT)
        row_y += row_step

    draw.text((bottom_card[0] + 16, bottom_card[1] + 14), "Trạng thái", fill=LIGHT_TEXT_PRIMARY, font=section_font)
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
        _draw_smooth_card(
            draw,
            (badge_left, badge_top, badge_left + badge_width, badge_top + badge_height),
            radius=18,
            width=1,
        )
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
    _draw_smooth_card(
        draw,
        (input_left, input_top, input_right, input_bottom),
        radius=34,
        width=1,
    )

    plus_box_left = input_left + 18
    plus_box_top = input_top + 13
    plus_size = 24
    plus_cx = plus_box_left + plus_size // 2
    plus_cy = plus_box_top + plus_size // 2
    draw.line((plus_cx - 7, plus_cy, plus_cx + 7, plus_cy), fill=LIGHT_TEXT_PRIMARY, width=2)
    draw.line((plus_cx, plus_cy - 7, plus_cx, plus_cy + 7), fill=LIGHT_TEXT_PRIMARY, width=2)

    draw.text((input_left + 60, input_top + 16), "Hỏi bất cứ điều gì...", fill=(156, 163, 175), font=body_font)

    mic_center_x = input_right - 42
    mic_center_y = input_top + 28
    draw.ellipse(
        [(mic_center_x - 17, mic_center_y - 17), (mic_center_x + 17, mic_center_y + 17)],
        outline=LIGHT_PANEL_OUTLINE_SOFT,
        width=1,
        fill=LIGHT_PANEL_FILL,
    )
    mic_body = (mic_center_x - 4, mic_center_y - 8, mic_center_x + 4, mic_center_y + 4)
    draw.rounded_rectangle(mic_body, radius=4, outline=LIGHT_TEXT_PRIMARY, width=2)
    draw.line((mic_center_x, mic_center_y + 4, mic_center_x, mic_center_y + 10), fill=LIGHT_TEXT_PRIMARY, width=2)
    draw.arc(
        [(mic_center_x - 8, mic_center_y - 4), (mic_center_x + 8, mic_center_y + 10)],
        start=200,
        end=340,
        fill=LIGHT_TEXT_PRIMARY,
        width=2,
    )
    draw.line((mic_center_x - 5, mic_center_y + 10, mic_center_x + 5, mic_center_y + 10), fill=LIGHT_TEXT_PRIMARY, width=2)
    status = lines[0] if lines else "Phân tích"
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
) -> np.ndarray:
    frame = _scaled_camera_frame(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)
    total_width = SIDEBAR_WIDTH + LAYOUT_GAP + frame_image.width + LAYOUT_GAP + RIGHT_PANEL_WIDTH
    total_height = frame_image.height + LAYOUT_GAP + BOTTOM_CHAT_HEIGHT
    layout = Image.new("RGB", (total_width, total_height), color=LIGHT_BG)
    draw = ImageDraw.Draw(layout)

    _draw_data_sidebar(
        draw,
        0,
        0,
        SIDEBAR_WIDTH,
        total_height,
    )

    camera_left = SIDEBAR_WIDTH + LAYOUT_GAP
    camera_top = 0
    layout.paste(frame_image, (camera_left, camera_top))

    resource_left = camera_left + frame_image.width + LAYOUT_GAP
    _draw_resource_sidebar(
        draw,
        resource_left,
        0,
        RIGHT_PANEL_WIDTH,
        total_height,
        resource_lines,
        fps_text,
        status_lines,
    )
    _draw_chat_box(
        draw,
        camera_left,
        frame_image.height + LAYOUT_GAP,
        frame_image.width,
        BOTTOM_CHAT_HEIGHT,
        chat_title,
        chat_lines,
    )
    return cv2.cvtColor(np.array(layout), cv2.COLOR_RGB2BGR)


def _draw_resource_panel(draw: ImageDraw.ImageDraw, frame_width: int, padding: int, lines: list[str]) -> None:
    label_font = _font_body_medium(14)
    value_font = _font_body_medium(18)
    left = 22
    top = 78
    gap = 12
    card_width = (frame_width - (left * 2) - (gap * 3)) // 4
    card_height = 48
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
        _draw_smooth_card(draw, (x1, y1, x2, y2), radius=16, fill=card_fill, outline=card_outline, width=2)
        draw.text((x1 + 16, y1 + 8), label, fill=LIGHT_TEXT_SECONDARY, font=label_font)
        draw.text((x1 + 16, y1 + 23), value, fill=LIGHT_TEXT_PRIMARY, font=value_font)


def _draw_capture_prep_panel(draw: ImageDraw.ImageDraw, frame_width: int, padding: int, lines: list[str]) -> None:
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
        _draw_smooth_card(draw, rect, radius=16, fill=LIGHT_PANEL_FILL, outline=LIGHT_PANEL_OUTLINE, width=2)

    draw.text((left_chip[0] + 14, left_chip[1] + 8), "Đếm ngược", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    draw.text((left_chip[0] + 14, left_chip[1] + 24), countdown_value, fill=LIGHT_TEXT_PRIMARY, font=big_font)

    draw.text((middle_chip[0] + 14, middle_chip[1] + 8), "Rung/lắc", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    draw.text((middle_chip[0] + 14, middle_chip[1] + 27), motion_value, fill=LIGHT_TEXT_PRIMARY, font=value_font)

    draw.text((status_card[0] + 14, status_card[1] + 8), "Trạng thái", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    wrapped_status = _wrap_text(draw, lines[2], body_font, status_card[2] - status_card[0] - 28, 2)
    wrapped_hint = _wrap_text(draw, lines[3], small_font, status_card[2] - status_card[0] - 28, 2)
    for index, line in enumerate(wrapped_status):
        draw.text((status_card[0] + 14, status_card[1] + 24 + (index * 20)), line, fill=LIGHT_TEXT_PRIMARY, font=body_font)
    hint_y = status_card[1] + 60
    for line in wrapped_hint:
        draw.text((status_card[0] + 14, hint_y), line, fill=LIGHT_TEXT_SECONDARY, font=small_font)
        hint_y += 16


def _draw_name_prompt_panel(draw: ImageDraw.ImageDraw, frame_width: int, padding: int, lines: list[str]) -> None:
    body_font = _font_body(15)
    small_font = _font_body(13)
    value_font = _font_body_medium(18)

    count_chip = (24, 72, 170, 120)
    name_chip = (184, 72, frame_width - 24, 120)

    for rect in (count_chip, name_chip):
        _draw_smooth_card(draw, rect, radius=16, fill=LIGHT_PANEL_FILL, outline=LIGHT_PANEL_OUTLINE, width=2)

    draw.text((count_chip[0] + 14, count_chip[1] + 8), "Số nhãn", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    draw.text((count_chip[0] + 14, count_chip[1] + 24), lines[0].split(":", 1)[1].strip() if ":" in lines[0] else lines[0], fill=LIGHT_TEXT_PRIMARY, font=value_font)

    draw.text((name_chip[0] + 14, name_chip[1] + 8), "Tên mẫu", fill=LIGHT_TEXT_SECONDARY, font=small_font)
    wrapped_name = _wrap_text(draw, lines[1], body_font, name_chip[2] - name_chip[0] - 28, 2)
    for index, line in enumerate(wrapped_name):
        draw.text((name_chip[0] + 14, name_chip[1] + 24 + (index * 18)), line, fill=LIGHT_TEXT_PRIMARY, font=body_font)

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


def _panel_lines_for_live_view(detector: CameraDetector, hardware: Any) -> list[str]:
    usage = get_live_usage_snapshot()

    def _percent(name: str) -> float | None:
        value = usage.get(name)
        return float(value) if value is not None else None

    def _format_percent(name: str) -> str:
        value = _percent(name)
        return f"{value:.1f}%" if value is not None else "Không rõ"

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
        gpu_name = getattr(hardware, "gpu_name", "Không rõ")
        return f"{gpu_percent} | {gpu_name}"

    return [
        f"CPU {_format_percent('cpu_usage_percent')}",
        f"RAM {_ram_line()}",
        f"GPU {_gpu_line()}",
        f"VRAM {_vram_line()}",
    ]


def _fps_panel_line(fps: float) -> str:
    if fps <= 0:
        return "FPS   Không rõ"
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


def run_camera_session(runtime: RuntimeConfig, camera_index: int = 0) -> None:
    detector = CameraDetector(runtime=runtime, camera_index=camera_index)
    hardware = detect_hardware()
    detector.initialize()
    capture_prep: CapturePreparationState | None = None
    naming_mode = False
    typed_name = ""
    frozen_frame: np.ndarray | None = None
    frozen_detections: list[DetectionRecord] = []
    window_positioned = False
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, _sidebar_mouse_callback)

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

            resource_lines = _panel_lines_for_live_view(detector, hardware)
            fps_text = _fps_panel_line(_fps)
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
                chat_lines = ["Phân tích", "Nhấn T để bắt đầu chụp mẫu.", "Giữ camera ổn định khi chụp."]
            status_lines = (
                chat_lines
                if capture_prep is not None or naming_mode
                else ["Đang chờ phân tích...", "Kết quả chụp ảnh hoặc AI sẽ hiển thị tại đây."]
            )

            if _is_window_maximized(WINDOW_NAME):
                composed = _compose_desktop_layout(
                    frame=display_frame,
                    resource_lines=resource_lines,
                    fps_text=fps_text,
                    status_lines=status_lines,
                    chat_title=chat_title,
                    chat_lines=chat_lines,
                )
            else:
                compact_lines: list[str] = []
                compact_title = ""
                if capture_prep is not None or naming_mode:
                    compact_title = chat_title
                    compact_lines = chat_lines
                composed = _compose_compact_camera_frame(display_frame, compact_title, compact_lines)
            cv2.imshow(WINDOW_NAME, composed)
            if not window_positioned:
                cv2.resizeWindow(WINDOW_NAME, composed.shape[1], composed.shape[0])
                _maximize_window(WINDOW_NAME)
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
            if key == 27:
                break
    finally:
        detector.release()
        cv2.destroyAllWindows()
