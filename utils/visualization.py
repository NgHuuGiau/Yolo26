from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def draw_detection_results(
    image: np.ndarray,
    detections: Iterable,
    box_thickness: int = 2,
    label_font_scale: float = 0.8,
) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        box_color = (0, 220, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, max(2, box_thickness))
    return image
