from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class OverlayInfo:
    fps: Optional[float]
    device: str
    model_name: str
    imgsz: int
    object_count: int = 0
    top_labels: str = "None"


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height))


def add_runtime_overlay(frame: np.ndarray, overlay: OverlayInfo) -> np.ndarray:
    lines = [
        f"Device: {overlay.device}",
        f"Model: {overlay.model_name}",
        f"imgsz: {overlay.imgsz}",
        f"Objects: {overlay.object_count}",
        f"Labels: {overlay.top_labels}",
    ]
    if overlay.fps is not None:
        lines.insert(0, f"FPS: {overlay.fps:.1f}")

    max_width = 0
    line_height = 30
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            2,
        )
        max_width = max(max_width, text_width)
        line_height = max(line_height, text_height + 12)

    panel_height = (line_height * len(lines)) + 18
    cv2.rectangle(frame, (8, 8), (max_width + 28, panel_height), (25, 25, 25), -1)
    cv2.rectangle(frame, (8, 8), (max_width + 28, panel_height), (0, 220, 255), 2)

    for index, line in enumerate(lines):
        y = 32 + (index * line_height)
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame
