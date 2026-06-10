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


def _clamp_bbox_to_image(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    image_height, image_width = image_shape[:2]
    if image_height <= 0 or image_width <= 0:
        return None
    x1, y1, x2, y2 = [int(value) for value in bbox]
    left = max(0, min(x1, x2, image_width - 1))
    top = max(0, min(y1, y2, image_height - 1))
    right = max(0, min(max(x1, x2), image_width - 1))
    bottom = max(0, min(max(y1, y2), image_height - 1))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def draw_detection_results(
    image: np.ndarray,
    detections: Iterable,
    box_thickness: int = 2,
    label_font_scale: float = 0.8,
    motion_trails: dict[int, list[tuple[int, int]]] | None = None,
) -> np.ndarray:
    trail_overlay = image.copy()
    drew_trail = False
    detection_list = list(detections)
    for detection in detection_list:
        if not motion_trails:
            continue
        trail_points = motion_trails.get(getattr(detection, "track_id", -1), [])
        if len(trail_points) < 2:
            continue
        drew_trail = True
        box_color = _color_for_label(detection.label)
        for index in range(1, len(trail_points)):
            start = trail_points[index - 1]
            end = trail_points[index]
            segment_ratio = index / max(1, len(trail_points) - 1)
            thickness = max(1, int(round(max(1, box_thickness) * (0.6 + (segment_ratio * 0.8)))))
            cv2.line(trail_overlay, start, end, box_color, thickness, cv2.LINE_AA)
            cv2.circle(trail_overlay, end, max(1, thickness // 2), box_color, -1, cv2.LINE_AA)
    if drew_trail:
        image = cv2.addWeighted(trail_overlay, 0.30, image, 0.70, 0.0)
    for detection in detection_list:
        clamped_bbox = _clamp_bbox_to_image(detection.bbox, image.shape)
        if clamped_bbox is None:
            continue
        x1, y1, x2, y2 = clamped_bbox
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
