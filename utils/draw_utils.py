from __future__ import annotations

import hashlib
from typing import Iterable

import cv2
import numpy as np


_COLOR_PALETTE: tuple[tuple[int, int, int], ...] = (
    (46, 125, 255),
    (255, 159, 67),
    (46, 204, 113),
    (235, 87, 87),
    (155, 89, 182),
    (241, 196, 15),
    (38, 198, 218),
    (230, 126, 34),
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
    top_left = (x, max(0, y - text_height - baseline - 8))
    bottom_right = (x + text_width + 14, min(image.shape[0] - 1, y + 8))
    cv2.rectangle(image, top_left, bottom_right, background_color, -1)
    cv2.putText(image, text, (x + 7, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


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
        label_y = max(24, y1 - 10)
        _draw_text_with_background(
            image=image,
            text=label_text,
            origin=(max(0, x1), label_y),
            font_scale=max(0.62, label_font_scale * 0.92),
            text_color=(255, 255, 255),
            background_color=box_color,
            thickness=max(1, box_thickness - 1),
        )
    return image
