from __future__ import annotations

import unittest

import numpy as np

from core.camera_runner import DetectionRecord
from utils.draw_utils import _clamp_bbox_to_image, _color_for_label, draw_detection_results


class VisualizationTests(unittest.TestCase):
    def test_draw_detection_results_renders_box_and_coordinate_text(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        detections = [DetectionRecord(class_id=0, label="person", confidence=0.91, bbox=(20, 30, 100, 80))]

        output = draw_detection_results(image.copy(), detections, box_thickness=2, label_font_scale=0.8)

        self.assertGreater(int(output.sum()), 0)
        self.assertTrue(np.any(output[79:105, 20:120] != 0))

    def test_color_for_label_is_stable_and_varies_by_label(self) -> None:
        person_color = _color_for_label("person")
        helmet_color = _color_for_label("helmet")

        self.assertEqual(person_color, _color_for_label("person"))
        self.assertNotEqual(person_color, helmet_color)

    def test_draw_detection_results_clamps_out_of_bounds_bbox(self) -> None:
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        detections = [DetectionRecord(class_id=0, label="face", confidence=0.88, bbox=(-10, -5, 100, 60))]

        output = draw_detection_results(image.copy(), detections)

        self.assertGreater(int(output.sum()), 0)

    def test_clamp_bbox_to_image_returns_none_for_invalid_box(self) -> None:
        self.assertIsNone(_clamp_bbox_to_image((20, 20, 20, 30), (40, 50, 3)))
        self.assertEqual(_clamp_bbox_to_image((30, 10, 5, 25), (40, 50, 3)), (5, 10, 30, 25))


if __name__ == "__main__":
    unittest.main()
