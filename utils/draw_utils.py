from __future__ import annotations

import hashlib
from typing import Iterable

import cv2
import numpy as np


_COLOR_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 220, 0),
    (255, 170, 0),
    (0, 200, 255),
    (255, 80, 80),
    (180, 90, 255),
    (255, 220, 80),
    (80, 255, 180),
    (220, 120, 40),
)


def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    index = digest[0] % len(_COLOR_PALETTE)
    return _COLOR_PALETTE[index]


def _draw_text_with_background(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    text_color: tuple[int, int, int],
    background_color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    top_left = (x, max(0, y - text_height - baseline - 4))
    bottom_right = (x + text_width + 6, min(image.shape[0] - 1, y + 4))
    cv2.rectangle(image, top_left, bottom_right, background_color, -1)
    cv2.putText(image, text, (x + 3, y - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_detection_results(
    image: np.ndarray,
    detections: Iterable,
    box_thickness: int = 2,
    label_font_scale: float = 0.8,
) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        box_color = _color_for_label(detection.label)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, max(2, box_thickness))
        label_text = f"{detection.label} {detection.confidence:.2f}"
        label_y = max(18, y1 - 6)
        _draw_text_with_background(
            image=image,
            text=label_text,
            origin=(max(0, x1), label_y),
            font_scale=max(0.6, label_font_scale),
            text_color=(0, 0, 0),
            background_color=box_color,
            thickness=1,
        )
        coordinate_text = f"({x1},{y1}) ({x2},{y2})"
        text_y = min(image.shape[0] - 2, y2 + 20)
        _draw_text_with_background(
            image=image,
            text=coordinate_text,
            origin=(max(0, x1), text_y),
            font_scale=max(0.5, label_font_scale * 0.7),
            text_color=(0, 0, 0),
            background_color=box_color,
            thickness=1,
        )
    return image
