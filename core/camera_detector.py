from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Tuple

import cv2

from core.fallback_manager import iter_fallback_configs
from core.model_selector import RuntimeConfig
from core.yolo_loader import LoadedModel, load_yolo_model
from utils.logger import get_logger
from utils.visualization import draw_detection_results


logger = get_logger(__name__)


@dataclass
class DetectionRecord:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


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
        self.last_status_message = "Sẵn sàng khởi tạo camera."
        self.last_error_message = ""
        self.active_runtime_summary = ""

    def initialize(self) -> None:
        last_error: Exception | None = None
        runtime_candidates = [self.runtime, *list(iter_fallback_configs(self.runtime))]
        for runtime in runtime_candidates:
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
                self.last_status_message = (
                    "Đã khởi tạo camera thành công. "
                    f"Đang chạy với {self.active_runtime_summary}."
                )
                logger.info("Detector initialized with %s", self.runtime.summary())
                return
            except Exception as exc:
                last_error = exc
                self.last_error_message = str(exc)
                self.last_status_message = "Khởi tạo runtime thất bại, đang thử fallback."
                logger.warning("Runtime failed, trying fallback: %s", exc)
                self.release()
        raise RuntimeError(f"Không khởi tạo được detector. Lỗi cuối: {last_error}")

    def read_and_detect(self) -> Tuple[bool, Any, List[DetectionRecord], float]:
        if self.capture is None or self.loaded_model is None:
            raise RuntimeError("Detector chưa được khởi tạo.")
        ok, frame = self.capture.read()
        if not ok:
            self.consecutive_read_failures += 1
            self.last_error_message = "Không đọc được frame từ camera."
            self.last_status_message = (
                f"Mất frame camera ({self.consecutive_read_failures}/{self.max_consecutive_read_failures})."
            )
            if self.consecutive_read_failures >= self.max_consecutive_read_failures:
                raise RuntimeError("Camera liên tục không trả về frame.")
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
            self.last_status_message = (
                "Suy luận bị lỗi, hệ thống đang tự phục hồi và thử cấu hình an toàn hơn."
            )
            self.initialize()
            return False, None, [], 0.0

        detections = self._parse_results(results)
        frame = draw_detection_results(
            image=frame,
            detections=detections,
            box_thickness=self.runtime.box_thickness,
            label_font_scale=self.runtime.label_font_scale,
        )
        now = time.perf_counter()
        current_fps = 1.0 / max(now - self.last_frame_ts, 1e-6)
        self.last_frame_ts = now
        if self.smoothed_fps == 0.0:
            self.smoothed_fps = current_fps
        else:
            self.smoothed_fps = (self.smoothed_fps * 0.85) + (current_fps * 0.15)
        self.last_status_message = f"Đang nhận diện ổn định với {len(detections)} đối tượng."
        return True, frame, detections, self.smoothed_fps

    def _parse_results(self, results: list) -> List[DetectionRecord]:
        parsed: List[DetectionRecord] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                parsed.append(
                    DetectionRecord(
                        label=names.get(cls_id, str(cls_id)),
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return parsed

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.last_status_message = "Camera đã dừng."

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


def run_camera_session(runtime: RuntimeConfig, camera_index: int = 0) -> None:
    detector = CameraDetector(runtime=runtime, camera_index=camera_index)
    detector.initialize()
    try:
        while True:
            ok, frame, _detections, _fps = detector.read_and_detect()
            if not ok:
                continue
            cv2.imshow("YOLO Realtime Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        detector.release()
        cv2.destroyAllWindows()
