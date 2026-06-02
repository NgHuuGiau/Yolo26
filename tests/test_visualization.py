from __future__ import annotations

import unittest

import numpy as np

from core.camera_runner import DetectionRecord
from utils.draw_utils import _color_for_label, draw_detection_results


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


if __name__ == "__main__":
    unittest.main()
